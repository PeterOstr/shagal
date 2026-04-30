import json
import re
import uuid
from collections import defaultdict
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from .infra import get_clickhouse_client, get_messages_store, get_threads_store


RE_PREFIX = re.compile(r"^\s*(re|fw|fwd|aw|ответ):\s*", flags=re.IGNORECASE)


def normalize_subject(subj: str) -> str:
    s = (subj or "").strip()
    while True:
        ns = RE_PREFIX.sub("", s).strip()
        if ns == s:
            break
        s = ns
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def split_addrs(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    return [p.strip() for p in re.split(r"[;,]", str(x)) if p.strip()]


def participants_list(row: dict[str, Any]) -> list[str]:
    people = (
        split_addrs(row.get("from_addr"))
        + split_addrs(row.get("to_addr"))
        + split_addrs(row.get("cc_addr"))
        + split_addrs(row.get("bcc_addr"))
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
    return get_clickhouse_client().query_df(query)


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
    return get_clickhouse_client().query_df(query)


def _coerce_mail_query_number(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    return int(s) if s.isdigit() else value


def build_message_docs(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []

    for row in df.to_dict("records"):
        parsed = safe_json_loads(row.get("parsed_json"))
        if not parsed:
            continue

        parsed_emails = parsed.get("emails", [])
        if not isinstance(parsed_emails, list):
            continue

        subj = (row.get("subject") or "").strip()
        participants = participants_list(row)

        for msg in parsed_emails:
            if not isinstance(msg, dict):
                continue

            email_body = (msg.get("email_body") or "").strip()
            if not email_body:
                continue

            topic = (msg.get("topic") or subj or "").strip()
            msg_date = msg.get("date")
            mail_query_number = _coerce_mail_query_number(msg.get("mail_query_number"))
            keywords = msg.get("key_words") or []
            thread_key = row.get("thread_key") or normalize_subject(subj)

            meta = {
                "email_id": row.get("id"),
                "message_id": row.get("message_id"),
                "subject": subj,
                "topic": topic,
                "date": str(msg_date) if msg_date is not None else str(row.get("sent_at_utc")),
                "mail_query_number": mail_query_number,
                "keywords": keywords,
                "sent_at_utc": str(row.get("sent_at_utc")),
                "folder": row.get("folder"),
                "participants": participants,
                "thread_key": thread_key,
                "source_type": "message",
            }

            docs.append(Document(page_content=email_body, metadata=meta))

    return docs


def make_message_ids(docs: list[Document]) -> list[str]:
    ids = []
    for d in docs:
        raw = (
            f"{d.metadata.get('email_id')}::"
            f"{d.metadata.get('mail_query_number')}::"
            f"{d.metadata.get('date')}::"
            f"{d.page_content[:100]}"
        )
        uid = uuid.uuid5(uuid.NAMESPACE_URL, raw)
        ids.append(str(uid))
    return ids


def upload_message_docs(docs: list[Document], qv: QdrantVectorStore) -> int:
    if not docs:
        return 0

    ids = make_message_ids(docs)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    qv.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(docs)


def dedupe_message_docs(docs: list[Document]) -> list[Document]:
    deduped: list[Document] = []
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
            str(d.metadata.get("mail_query_number") or ""),
        )
    )
    return deduped


def build_thread_docs(df: pd.DataFrame) -> list[Document]:
    message_docs = dedupe_message_docs(build_message_docs(df))
    grouped: dict[str, list[Document]] = defaultdict(list)

    for doc in message_docs:
        thread_key = doc.metadata.get("thread_key")
        if thread_key:
            grouped[thread_key].append(doc)

    thread_docs: list[Document] = []

    for thread_key, docs in grouped.items():
        docs.sort(
            key=lambda d: (
                d.metadata.get("date") or d.metadata.get("sent_at_utc") or "",
                str(d.metadata.get("mail_query_number") or ""),
            )
        )

        subject = next((d.metadata.get("subject") for d in docs if d.metadata.get("subject")), "")
        first_date = docs[0].metadata.get("date") or docs[0].metadata.get("sent_at_utc")
        last_date = docs[-1].metadata.get("date") or docs[-1].metadata.get("sent_at_utc")

        participants = sorted(
            {
                p
                for d in docs
                for p in (d.metadata.get("participants") or [])
                if p
            }
        )

        keywords = sorted(
            {
                kw
                for d in docs
                for kw in (d.metadata.get("keywords") or [])
                if kw
            }
        )

        chunks = []
        for d in docs[:50]:
            topic = d.metadata.get("topic") or d.metadata.get("subject") or ""
            date_value = d.metadata.get("date") or d.metadata.get("sent_at_utc") or ""
            body = (d.page_content or "").strip().replace("\n", " ")
            chunks.append(f"[{date_value}] {topic}\n{body[:1200]}")

        page_content = "\n\n".join(chunks)

        meta = {
            "thread_key": thread_key,
            "subject": subject,
            "participants": participants,
            "keywords": keywords,
            "message_count": len(docs),
            "first_date": first_date,
            "last_date": last_date,
            "source_type": "thread",
        }

        thread_docs.append(Document(page_content=page_content, metadata=meta))

    return thread_docs


def make_thread_ids(docs: list[Document]) -> list[str]:
    ids = []
    for d in docs:
        raw = (
            f"{d.metadata.get('thread_key')}::"
            f"{d.metadata.get('message_count')}::"
            f"{d.metadata.get('first_date')}::"
            f"{d.metadata.get('last_date')}"
        )
        uid = uuid.uuid5(uuid.NAMESPACE_URL, raw)
        ids.append(str(uid))
    return ids


def upload_thread_docs(docs: list[Document], qv: QdrantVectorStore) -> int:
    if not docs:
        return 0

    ids = make_thread_ids(docs)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    qv.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(docs)


def ingest_messages(batch_size: int = 1000, recreate: bool = False) -> int:
    qv = get_messages_store(recreate=recreate)

    offset = 0
    total_docs = 0
    total_rows = 0

    while True:
        df_batch = load_join_batch(limit=batch_size, offset=offset)
        if df_batch.empty:
            break

        docs = build_message_docs(df_batch)
        inserted = upload_message_docs(docs, qv)

        total_docs += inserted
        total_rows += len(df_batch)

        print(
            f"[messages] rows_processed={total_rows}, "
            f"docs_inserted_total={total_docs}, "
            f"last_batch_rows={len(df_batch)}, "
            f"last_batch_docs={inserted}"
        )

        offset += len(df_batch)

    print(f"[messages] DONE: rows_processed={total_rows}, docs_inserted_total={total_docs}")
    return total_docs


def ingest_threads(recreate: bool = False) -> int:
    """
    MVP-реализация:
    - читаем весь join
    - строим агрегированные документы по thread_key
    - загружаем в отдельную collection
    """
    qv = get_threads_store(recreate=recreate)

    df = load_all_joined_df()
    docs = build_thread_docs(df)
    inserted = upload_thread_docs(docs, qv)

    print(
        f"[threads] DONE: rows_loaded={len(df)}, "
        f"thread_docs_inserted={inserted}"
    )
    return inserted