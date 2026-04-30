import json
from collections import defaultdict
from pathlib import Path
from langchain.tools import tool

from platform import python_version
import os
import re
import uuid
import json

import pandas as pd
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import clickhouse_connect
import connections as con

load_dotenv()

MESSAGES_COLLECTION = "mailkb_messages"
THREADS_COLLECTION = "mailkb_threads"

SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# 3. Настройка коллекций
# ---------------------------------------------------

def ensure_collection(collection_name: str, recreate: bool = False) -> QdrantVectorStore:
    collections = con.client_qdrant.get_collections().collections
    existing_names = {c.name for c in collections}

    if recreate and collection_name in existing_names:
        con.client_qdrant.delete_collection(collection_name=collection_name)
        existing_names.remove(collection_name)

    if collection_name not in existing_names:
        con.client_qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=con.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=con.client_qdrant,
        collection_name=collection_name,
        embedding=con.embeddings,
    )


# recreate=True только если хочешь пересобрать с нуля
messages_qv = ensure_collection(MESSAGES_COLLECTION, recreate=False)
threads_qv = ensure_collection(THREADS_COLLECTION, recreate=False)



# ---------------------------------------------------
# 4. Утилиты
# ---------------------------------------------------

RE_PREFIX = re.compile(r'^\s*(re|fw|fwd|aw|ответ):\s*', flags=re.IGNORECASE)

def normalize_subject(subj: str) -> str:
    s = (subj or "").strip()
    while True:
        ns = RE_PREFIX.sub('', s).strip()
        if ns == s:
            break
        s = ns
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def split_addrs(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    return [p.strip() for p in re.split(r'[;,]', str(x)) if p.strip()]

def participants_list(row) -> list[str]:
    people = (
        split_addrs(row.get("from_addr")) +
        split_addrs(row.get("to_addr")) +
        split_addrs(row.get("cc_addr")) +
        split_addrs(row.get("bcc_addr"))
    )
    return sorted(set(people))

def safe_json_loads(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None

# ---------------------------------------------------
# 5. SQL: распарсенные письма из emails_unique
# ---------------------------------------------------

def load_join_batch(limit: int, offset: int) -> pd.DataFrame:
    query = f"""
    SELECT
        e.id,
        e.thread_key,
        e.message_id,
        e.subject,
        e.from_addr,
        e.to_addr,
        e.cc_addr,
        e.bcc_addr,
        e.sent_at_utc,
        e.folder,
        p.parsed_json
    FROM mailkb.emails_unique e
    INNER JOIN mailkb.mail_parsed p
        ON e.id = p.email_id
    ORDER BY e.sent_at_utc ASC, e.id ASC
    LIMIT {limit} OFFSET {offset}
    """
    return con.client_clickhouse.query_df(query)

def load_all_joined_df() -> pd.DataFrame:
    query = """
    SELECT
        e.id,
        e.thread_key,
        e.message_id,
        e.subject,
        e.from_addr,
        e.to_addr,
        e.cc_addr,
        e.bcc_addr,
        e.sent_at_utc,
        e.folder,
        p.parsed_json
    FROM mailkb.emails_unique e
    INNER JOIN mailkb.mail_parsed p
        ON e.id = p.email_id
    ORDER BY e.sent_at_utc ASC, e.id ASC
    """
    return con.client_clickhouse.query_df(query)

    def build_message_docs(df: pd.DataFrame) -> list[Document]:
        docs: list[Document] = []

        for row in df.to_dict("records"):
            parsed = func.safe_json_loads(row.get("parsed_json"))
            if not parsed:
                continue

            parsed_emails = parsed.get("emails", [])
            if not isinstance(parsed_emails, list):
                continue

            subj = (row.get("subject") or "").strip()
            participants = func.participants_list(row)

            for msg in parsed_emails:
                if not isinstance(msg, dict):
                    continue

                email_body = (msg.get("email_body") or "").strip()
                if not email_body:
                    continue

                topic = (msg.get("topic") or subj or "").strip()
                msg_date = msg.get("date")
                mail_query_number = msg.get("mail_query_number")
                keywords = msg.get("key_words") or []
                thread_key = row.get("thread_key") or func.normalize_subject(subj)

                meta = {
                    "email_id": row.get("id"),
                    "message_id": row.get("message_id"),
                    "subject": subj,
                    "topic": topic,
                    "date": str(msg_date) if msg_date is not None else str(row.get("sent_at_utc")),
                    "mail_query_number": int(mail_query_number) if str(
                        mail_query_number).isdigit() else mail_query_number,
                    "keywords": keywords,
                    "sent_at_utc": str(row.get("sent_at_utc")),
                    "folder": row.get("folder"),
                    "participants": participants,
                    "thread_key": thread_key,
                    "source_type": "message",
                }

                docs.append(Document(page_content=email_body, metadata=meta))

        return docs
