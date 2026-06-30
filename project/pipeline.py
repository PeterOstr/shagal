import hashlib
import json
import mailbox
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import List

import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import (
    ATTACH_DIR,
    BATCH,
    CHUNK_SIZE,
    LLM_MODEL,
    MBOX_DIR,
    MESSAGES_COLLECTION,
    SAVE_ATTACHMENTS,
)
from infra import (
    _get_llm,
    build_structured_agent,
    ensure_collection,
    get_clickhouse_client,
)


# =========================================================
# 1. import mbox -> ClickHouse
# =========================================================

def decode_mime(value):
    if not value:
        return ""

    parts = decode_header(value)
    result = ""

    for text, encoding in parts:
        if isinstance(text, bytes):
            try:
                result += text.decode(encoding or "utf-8", errors="ignore")
            except Exception:
                result += text.decode("utf-8", errors="ignore")
        else:
            result += str(text)

    return result


def parse_addrs(value):
    if not value:
        return []
    decoded = decode_mime(value)
    return [addr for _, addr in getaddresses([decoded]) if addr]


def parse_date(value):
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def iter_mbox_files(base):
    for mbox_path in Path(base).rglob("mbox"):
        if mbox_path.is_file():
            yield mbox_path


def extract_body(msg):
    body_text = ""
    body_html = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue

            ctype = part.get_content_type()

            try:
                payload = part.get_payload(decode=True)
                if not payload:
                    continue

                charset = part.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="ignore")
            except Exception:
                continue

            if ctype == "text/plain" and not body_text:
                body_text = content
            elif ctype == "text/html" and not body_html:
                body_html = content
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="ignore")

                if msg.get_content_type() == "text/html":
                    body_html = content
                else:
                    body_text = content
        except Exception:
            pass

    return body_text, body_html


def import_mbox_to_clickhouse():
    client = get_clickhouse_client()

    emails_rows = []
    attach_rows = []

    ensure_dir(ATTACH_DIR)

    for mbox_path in iter_mbox_files(MBOX_DIR):
        print("Processing:", mbox_path)

        folder_name = str(mbox_path.parent.relative_to(MBOX_DIR))
        mbox = mailbox.mbox(str(mbox_path))

        for msg in tqdm(mbox, desc=folder_name):
            stable_id = str(uuid.uuid4())

            message_id = str((msg.get("Message-ID") or msg.get("Message-Id") or "").strip())
            subject = str(decode_mime(msg.get("Subject", "")) or "")

            from_addr = parse_addrs(msg.get("From"))
            to_addr = parse_addrs(msg.get("To"))
            cc_addr = parse_addrs(msg.get("Cc"))
            bcc_addr = parse_addrs(msg.get("Bcc"))

            raw_date_value = msg.get("Date")
            sent_raw = "" if raw_date_value is None else str(raw_date_value)
            sent_utc = parse_date(raw_date_value)

            thread_id = str(
                msg.get("Thread-Index")
                or msg.get("References")
                or msg.get("In-Reply-To")
                or ""
            )

            body_text, body_html = extract_body(msg)

            if not body_text and body_html:
                body_text = BeautifulSoup(body_html, "lxml").get_text("\n")

            body_text = "" if body_text is None else str(body_text)
            body_html = "" if body_html is None else str(body_html)

            if not body_text and not subject:
                continue

            emails_rows.append([
                stable_id,
                message_id,
                thread_id,
                subject,
                from_addr,
                to_addr,
                cc_addr,
                bcc_addr,
                sent_utc,
                sent_raw,
                folder_name,
                body_text,
                body_html
            ])

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_disposition() != "attachment":
                        continue

                    fname = part.get_filename() or "attachment"
                    fname = decode_mime(fname)
                    fname = fname.replace("\\", "_").replace("/", "_")

                    data = part.get_payload(decode=True)
                    if not data:
                        continue

                    size = len(data)
                    fpath = ""

                    if SAVE_ATTACHMENTS:
                        try:
                            dest_dir = Path(ATTACH_DIR) / folder_name / stable_id
                            ensure_dir(dest_dir)

                            dest_fp = dest_dir / fname
                            dest_fp.write_bytes(data)
                            fpath = str(dest_fp)
                        except Exception:
                            pass

                    attach_rows.append([
                        stable_id,
                        fname,
                        fpath,
                        int(size)
                    ])

            if len(emails_rows) >= BATCH:
                client.insert(
                    "mailkb.emails",
                    emails_rows,
                    column_names=[
                        "id",
                        "message_id",
                        "thread_id",
                        "subject",
                        "from_addr",
                        "to_addr",
                        "cc_addr",
                        "bcc_addr",
                        "sent_at_utc",
                        "sent_at_raw",
                        "folder",
                        "body_text",
                        "body_html"
                    ]
                )
                emails_rows.clear()

            if len(attach_rows) >= BATCH:
                client.insert(
                    "mailkb.attachments",
                    attach_rows,
                    column_names=[
                        "email_id",
                        "filename",
                        "path",
                        "size_bytes"
                    ]
                )
                attach_rows.clear()

    if emails_rows:
        client.insert(
            "mailkb.emails",
            emails_rows,
            column_names=[
                "id",
                "message_id",
                "thread_id",
                "subject",
                "from_addr",
                "to_addr",
                "cc_addr",
                "bcc_addr",
                "sent_at_utc",
                "sent_at_raw",
                "folder",
                "body_text",
                "body_html"
            ]
        )

    if attach_rows:
        client.insert(
            "mailkb.attachments",
            attach_rows,
            column_names=[
                "email_id",
                "filename",
                "path",
                "size_bytes"
            ]
        )


# =========================================================
# 2. dedup emails
# =========================================================

def dedup_thread(df_thread):
    rows = df_thread.sort_values(
        by="body_text",
        key=lambda x: x.str.len(),
        ascending=False
    )

    kept = []

    for _, row in rows.iterrows():
        body = row["body_text"]
        duplicate = False

        for k in kept:
            if body in k["body_text"]:
                duplicate = True
                break

        if not duplicate:
            kept.append(row)

    return pd.DataFrame(kept)


def deduplicate_emails():
    client = get_clickhouse_client()
    offset = 0

    while True:
        query = f"""
        SELECT
            id,
            thread_key,
            message_id,
            subject,
            from_addr,
            to_addr,
            sent_at_utc,
            folder,
            body_text
        FROM mailkb.emails
        WHERE body_text IS NOT NULL
        ORDER BY thread_key, sent_at_utc
        LIMIT {CHUNK_SIZE}
        OFFSET {offset}
        """

        df = client.query_df(query)

        if df.empty:
            break

        print("Loaded rows:", len(df))

        result = []

        for thread_key, df_thread in df.groupby("thread_key"):
            deduped = dedup_thread(df_thread)
            result.append(deduped)

        df_result = pd.concat(result)
        rows = df_result.values.tolist()

        client.insert(
            "mailkb.emails_unique",
            rows,
            column_names=df_result.columns.tolist()
        )

        print("Inserted:", len(rows))
        offset += CHUNK_SIZE


# =========================================================
# 3. clean email body
# =========================================================

class CleanBody(BaseModel):
    body_clean: str = Field(
        ...,
        description="Top-level email text without quoted history, headers, or signatures"
    )


class CleanItem(BaseModel):
    raw_md5: str = Field(description="MD5 of the original raw body")
    body_clean: str = Field(description="Cleaned top-level email text")


class CleanBatch(BaseModel):
    items: List[CleanItem]


CLEAN_PROMPT = """
You clean email bodies.

Task:
Extract ONLY the new message written by the sender.

Remove:
- quoted history
- repeated headers (From, Sent, Subject)
- signatures
- reply separators

Keep only the actual message content.

Return JSON matching the schema.
""".strip()

CLEAN_BATCH_PROMPT = """
You clean email bodies.

Task:
For each email body, extract ONLY the new message written by the sender.

Remove:
- quoted history
- repeated headers like From/Sent/Subject
- signatures, phone blocks, company footers

Return JSON matching the schema.

Important:
- DO NOT invent content.
- Return one item per input email with the same raw_md5.
""".strip()

clean_single_agent = build_structured_agent(CleanBody)
clean_batch_agent = build_structured_agent(CleanBatch)


def body_md5(text):
    text = (text or "").strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_email_body(text: str):
    result = clean_single_agent.invoke([
        SystemMessage(CLEAN_PROMPT),
        HumanMessage(text)
    ])
    return result.body_clean


def clean_email_bodies_batch(md5_and_text: List[tuple[str, str]]) -> dict[str, str]:
    blocks = []
    for md5, body in md5_and_text:
        blocks.append(f"raw_md5: {md5}\nbody:\n{body}")

    user_content = "\n\n---\n\n".join(blocks)

    result = clean_batch_agent.invoke([
        SystemMessage(CLEAN_BATCH_PROMPT),
        HumanMessage(user_content)
    ])

    batch_obj: CleanBatch = result

    out = {}
    for item in batch_obj.items:
        out[item.raw_md5] = item.body_clean

    return out


def clean_email_bodies_from_db(fetch_batch: int = 30, llm_batch: int = 5):
    client = get_clickhouse_client()

    print("Loading cache...")
    cached_md5 = set(r[0] for r in client.query("""
        SELECT raw_md5 FROM mailkb.llm_body_clean_cache
    """).result_rows)
    print("Cached:", len(cached_md5))

    while True:
        rows = client.query("""
            SELECT id, body_text
            FROM mailkb.emails
            WHERE body_text IS NOT NULL AND body_text != ''
            ORDER BY rand()
            LIMIT %(limit)s
        """, {"limit": fetch_batch}).result_rows

        if not rows:
            print("No rows found, stop.")
            break

        to_process = []
        batch_meta = []

        for email_id, body in rows:
            md5 = body_md5(body)
            if md5 in cached_md5:
                continue

            to_process.append((md5, body))
            batch_meta.append((email_id, md5, body))

        if not to_process:
            print("All fetched rows already cached; fetching next...")
            continue

        for i in range(0, len(to_process), llm_batch):
            chunk = to_process[i:i + llm_batch]
            chunk_meta = batch_meta[i:i + llm_batch]

            start = time.time()
            insert_batch = []

            try:
                cleaned_map = clean_email_bodies_batch(chunk)
                latency_ms = int((time.time() - start) * 1000)

                for email_id, md5, body in chunk_meta:
                    cleaned = cleaned_map.get(md5, "")

                    insert_batch.append([
                        md5,
                        "v1",
                        LLM_MODEL,
                        "success",
                        cleaned,
                        "",
                        0,
                        0,
                        latency_ms
                    ])
                    cached_md5.add(md5)
                    print("Processing:", email_id)

            except Exception as e:
                for email_id, md5, body in chunk_meta:
                    insert_batch.append([
                        md5,
                        "v1",
                        LLM_MODEL,
                        "failed",
                        "",
                        str(e),
                        0,
                        0,
                        0
                    ])
                    print("Processing:", email_id)

            if insert_batch:
                client.insert(
                    "mailkb.llm_body_clean_cache",
                    insert_batch,
                    column_names=[
                        "raw_md5",
                        "parser_version",
                        "model_name",
                        "status",
                        "body_clean",
                        "error",
                        "tokens_in",
                        "tokens_out",
                        "latency_ms"
                    ]
                )
                print("Inserted:", len(insert_batch))


# =========================================================
# 4. parse emails -> mail_parsed
# =========================================================

class MailInfo(BaseModel):
    topic: str = Field(description="Topic of the email")
    email_body: str = Field(description="Clean email body")
    date: str = Field(description="Date of the email YYYY-MM-DD")
    mail_query_number: int = Field(description="Order number of the email in the chain")
    key_words: List[str] = Field(description="Important keywords")


class ParsedEmailResult(BaseModel):
    email_id: str = Field(description="Original email id from the input batch")
    emails: List[MailInfo] = Field(description="Parsed content for this email")


class ParsedEmailBatch(BaseModel):
    items: List[ParsedEmailResult] = Field(description="Parsed results for all emails in the batch")


parse_agent = build_structured_agent(ParsedEmailBatch)

structured_llm = _get_llm().with_structured_output(ParsedEmailBatch)


def chunk_list(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def build_parse_batch_prompt(batch_rows):
    parts = []

    for idx, row in enumerate(batch_rows, start=1):
        body = (row["body_text"] or "")[:15000]

        parts.append(
            f"""
EMAIL #{idx}
email_id: {row["id"]}

EMAIL BODY:
{body}
""".strip()
        )

    return f"""
Начни обработку нескольких писем.

Для КАЖДОГО письма:
- извлеки структуру переписки
- верни результат отдельно
- ОБЯЗАТЕЛЬНО укажи email_id
- не смешивай письма между собой

Верни результат в формате ParsedEmailBatch.

{chr(10).join(parts)}
"""


def call_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            print(f"Retry {attempt+1} due to error:", e)
            time.sleep(1.5 * (attempt + 1))

    raise Exception("LLM failed after retries")


def process_parse_batch(batch_rows):
    prompt = build_parse_batch_prompt(batch_rows)

    try:
        structured = call_with_retry(prompt)

        parsed_rows = []
        returned_ids = set()

        for item in structured.items:
            returned_ids.add(item.email_id)
            parsed_rows.append({
                "email_id": item.email_id,
                "parsed_json": item.model_dump_json()
            })

        requested_ids = {row["id"] for row in batch_rows}
        missing_ids = requested_ids - returned_ids

        error_rows = [
            {"email_id": mid, "error": "missing_in_response"}
            for mid in missing_ids
        ]

        return parsed_rows, error_rows

    except Exception as e:
        return [], [
            {"email_id": row["id"], "error": str(e)}
            for row in batch_rows
        ]


def parse_emails_from_db(limit: int = 50, batch_size: int = 3, max_workers: int = 6):
    client = get_clickhouse_client()

    query = f"""
    SELECT
        id,
        sent_at_utc,
        body_text
    FROM mailkb.emails_unique

    LEFT ANTI JOIN mailkb.mail_parsed
    ON emails_unique.id = mail_parsed.email_id

    WHERE
        body_text IS NOT NULL
        AND length(body_text) > 30

    ORDER BY sent_at_utc DESC
    LIMIT {limit}
    """

    df = client.query_df(query)
    rows = df.to_dict("records")
    batches = list(chunk_list(rows, batch_size))

    all_success = []
    all_errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_parse_batch, batch) for batch in batches]

        for future in as_completed(futures):
            success_rows, error_rows = future.result()
            all_success.extend(success_rows)
            all_errors.extend(error_rows)

    print("success:", len(all_success))
    print("errors:", len(all_errors))

    if all_success:
        insert_rows = [[r["email_id"], r["parsed_json"]] for r in all_success]
        client.insert(
            "mailkb.mail_parsed",
            insert_rows,
            column_names=["email_id", "parsed_json"]
        )

    return {
        "success_count": len(all_success),
        "error_count": len(all_errors),
        "errors": all_errors,
    }


# =========================================================
# 5. indexing parsed messages -> Qdrant messages
# =========================================================

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


def load_join_batch(limit: int, offset: int) -> pd.DataFrame:
    client = get_clickhouse_client()

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
    return client.query_df(query)


def load_all_joined_df() -> pd.DataFrame:
    client = get_clickhouse_client()

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
    return client.query_df(query)


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
            mail_query_number = msg.get("mail_query_number")
            keywords = msg.get("key_words") or []
            thread_key = row.get("thread_key") or normalize_subject(subj)

            meta = {
                "email_id": row.get("id"),
                "message_id": row.get("message_id"),
                "subject": subj,
                "topic": topic,
                "date": str(msg_date) if msg_date is not None else str(row.get("sent_at_utc")),
                "mail_query_number": int(mail_query_number) if str(mail_query_number).isdigit() else mail_query_number,
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


def upload_message_docs(docs: list[Document], qv):
    if not docs:
        return 0

    ids = make_message_ids(docs)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    qv.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )
    return len(docs)


def index_messages(batch_size: int = 1000, recreate: bool = False):
    messages_qv = ensure_collection(MESSAGES_COLLECTION, recreate=recreate)

    offset = 0
    total_docs = 0
    total_rows = 0

    while True:
        df_batch = load_join_batch(limit=batch_size, offset=offset)
        if df_batch.empty:
            break

        docs = build_message_docs(df_batch)
        inserted = upload_message_docs(docs, messages_qv)

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


def index_threads(*args, **kwargs):
    raise NotImplementedError(
        "В присланных ноутбуках нет реализации build_thread_docs / ingest_threads. "
        "Поиск по mailkb_threads в коде есть, а построения thread-индекса нет."
    )