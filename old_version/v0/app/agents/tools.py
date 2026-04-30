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
import functions as func

load_dotenv()

MESSAGES_COLLECTION = "mailkb_messages"
THREADS_COLLECTION = "mailkb_threads"

SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# 2. Имена коллекций
# ---------------------------------------------------

MESSAGES_COLLECTION = "mailkb_messages"
THREADS_COLLECTION = "mailkb_threads"


@tool
def search_project_threads(project_hint: str, limit: int = 30) -> str:
    """
    Найти релевантные обсуждения проекта в коллекции mailkb_threads.
    """
    docs = func.threads_qv.similarity_search(project_hint, k=limit)

    results = []
    seen = set()

    for doc in docs:
        md = doc.metadata or {}
        thread_key = md.get("thread_key")

        if not thread_key or thread_key in seen:
            continue
        seen.add(thread_key)

        results.append({
            "thread_key": thread_key,
            "subject": md.get("subject"),
            "participants": md.get("participants") or [],
            "keywords": md.get("keywords") or [],
            "message_count": md.get("message_count"),
            "first_date": md.get("first_date"),
            "last_date": md.get("last_date"),
            "snippet": (doc.page_content[:800] + "…") if doc.page_content else None,
        })

    return json.dumps(
        {"project_hint": project_hint, "threads": results},
        ensure_ascii=False,
        indent=2
    )


def _escape_ch_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


@tool
def get_project_corpus_batch(
        project_hint: str,
        offset: int = 0,
        batch_size: int = 50,
        thread_limit: int = 30
) -> str:
    """
    Собрать батч корпуса проекта:
    1) найти релевантные thread_key в mailkb_threads
    2) достать все сообщения по этим thread_key
    3) дедуплицировать и отсортировать
    4) вернуть батч по offset/batch_size
    """
    thread_docs = func.threads_qv.similarity_search(project_hint, k=thread_limit)

    thread_keys = []
    seen = set()

    for doc in thread_docs:
        md = doc.metadata or {}
        tk = md.get("thread_key")
        if tk and tk not in seen:
            seen.add(tk)
            thread_keys.append(tk)

    if not thread_keys:
        return json.dumps({
            "project_hint": project_hint,
            "offset": offset,
            "batch_size": batch_size,
            "batch_len": 0,
            "has_more": False,
            "thread_keys": [],
            "total_messages": 0,
            "batch": []
        }, ensure_ascii=False, indent=2)

    quoted_keys = ", ".join(f"'{_escape_ch_string(tk)}'" for tk in thread_keys)

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
    WHERE e.thread_key IN ({quoted_keys})
    ORDER BY e.sent_at_utc ASC, e.id ASC
    """



    df = con.client_clickhouse.query_df(query)
    docs = func.build_message_docs(df)

    deduped = []
    seen_msg = set()

    for d in docs:
        dedup_key = (
            f"{d.metadata.get('email_id')}::"
            f"{d.metadata.get('mail_query_number')}::"
            f"{d.metadata.get('date')}::"
            f"{d.page_content[:120]}"
        )
        if dedup_key in seen_msg:
            continue
        seen_msg.add(dedup_key)
        deduped.append(d)

    deduped.sort(
        key=lambda d: (
            d.metadata.get("date") or d.metadata.get("sent_at_utc") or "",
            str(d.metadata.get("mail_query_number") or "")
        )
    )

    batch_docs = deduped[offset: offset + batch_size]

    batch = []
    for d in batch_docs:
        md = d.metadata or {}
        batch.append({
            "thread_key": md.get("thread_key"),
            "email_id": md.get("email_id"),
            "message_id": md.get("message_id"),
            "subject": md.get("subject"),
            "topic": md.get("topic"),
            "sent_at_utc": md.get("date") or md.get("sent_at_utc"),
            "participants": md.get("participants") or [],
            "keywords": md.get("keywords") or [],
            "folder": md.get("folder"),
            "snippet": d.page_content[:800] if d.page_content else None,
        })

    return json.dumps({
        "project_hint": project_hint,
        "offset": offset,
        "batch_size": batch_size,
        "batch_len": len(batch),
        "has_more": offset + batch_size < len(deduped),
        "thread_keys": thread_keys,
        "total_messages": len(deduped),
        "batch": batch
    }, ensure_ascii=False, indent=2)


@tool
def save_summary(summary_text: str, batch_id: int) -> str:
    """
    Сохранить summary батча в summaries/summary_batch_{batch_id}.txt
    """
    p = SUMMARY_DIR / f"summary_batch_{batch_id}.txt"
    p.write_text(summary_text, encoding="utf-8")
    return f"saved_to={str(p)}"


@tool
def load_all_summaries() -> str:
    """
    Прочитать все summary_batch_*.txt и вернуть объединённый текст.
    """
    files = sorted(SUMMARY_DIR.glob("summary_batch_*.txt"))

    if not files:
        return "NO_SUMMARIES_FOUND"

    texts = []
    for file in files:
        content = file.read_text(encoding="utf-8")
        texts.append(f"===== {file.name} =====\n{content}\n")

    return "\n".join(texts)


@tool
def get_project_corpus_batch(
        project_hint: str,
        offset: int = 0,
        batch_size: int = 50,
        thread_limit: int = 30
) -> str:
    """
    Собрать батч корпуса проекта:
    1) найти релевантные thread_key в mailkb_threads
    2) достать все сообщения по этим thread_key
    3) дедуплицировать и отсортировать
    4) вернуть батч по offset/batch_size
    """
    thread_docs = func.threads_qv.similarity_search(project_hint, k=thread_limit)

    thread_keys = []
    seen = set()

    for doc in thread_docs:
        md = doc.metadata or {}
        tk = md.get("thread_key")
        if tk and tk not in seen:
            seen.add(tk)
            thread_keys.append(tk)

    if not thread_keys:
        return json.dumps({
            "project_hint": project_hint,
            "offset": offset,
            "batch_size": batch_size,
            "batch_len": 0,
            "has_more": False,
            "thread_keys": [],
            "total_messages": 0,
            "batch": []
        }, ensure_ascii=False, indent=2)

    quoted_keys = ", ".join(f"'{_escape_ch_string(tk)}'" for tk in thread_keys)

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
    WHERE e.thread_key IN ({quoted_keys})
    ORDER BY e.sent_at_utc ASC, e.id ASC
    """

    df = con.client_clickhouse.query_df(query)
    docs = con.build_message_docs(df)

    deduped = []
    seen_msg = set()

    for d in docs:
        dedup_key = (
            f"{d.metadata.get('email_id')}::"
            f"{d.metadata.get('mail_query_number')}::"
            f"{d.metadata.get('date')}::"
            f"{d.page_content[:120]}"
        )
        if dedup_key in seen_msg:
            continue
        seen_msg.add(dedup_key)
        deduped.append(d)

    deduped.sort(
        key=lambda d: (
            d.metadata.get("date") or d.metadata.get("sent_at_utc") or "",
            str(d.metadata.get("mail_query_number") or "")
        )
    )

    batch_docs = deduped[offset: offset + batch_size]

    batch = []
    for d in batch_docs:
        md = d.metadata or {}
        batch.append({
            "thread_key": md.get("thread_key"),
            "email_id": md.get("email_id"),
            "message_id": md.get("message_id"),
            "subject": md.get("subject"),
            "topic": md.get("topic"),
            "sent_at_utc": md.get("date") or md.get("sent_at_utc"),
            "participants": md.get("participants") or [],
            "keywords": md.get("keywords") or [],
            "folder": md.get("folder"),
            "snippet": d.page_content[:800] if d.page_content else None,
        })

    return json.dumps({
        "project_hint": project_hint,
        "offset": offset,
        "batch_size": batch_size,
        "batch_len": len(batch),
        "has_more": offset + batch_size < len(deduped),
        "thread_keys": thread_keys,
        "total_messages": len(deduped),
        "batch": batch
    }, ensure_ascii=False, indent=2)


@tool
def save_summary(summary_text: str, batch_id: int) -> str:
    """
    Сохранить summary батча в summaries/summary_batch_{batch_id}.txt
    """
    p = SUMMARY_DIR / f"summary_batch_{batch_id}.txt"
    p.write_text(summary_text, encoding="utf-8")
    return f"saved_to={str(p)}"


@tool
def load_all_summaries() -> str:
    """
    Прочитать все summary_batch_*.txt и вернуть объединённый текст.
    """
    files = sorted(SUMMARY_DIR.glob("summary_batch_*.txt"))

    if not files:
        return "NO_SUMMARIES_FOUND"

    texts = []
    for file in files:
        content = file.read_text(encoding="utf-8")
        texts.append(f"===== {file.name} =====\n{content}\n")

    return "\n".join(texts)