# app/ingestion/indexer.py

import re
import uuid
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------------------
# PREPROCESSING UTILS
# -----------------------------------------------------

RE_PREFIX = re.compile(r'^\s*(re|fw|fwd):\s*', flags=re.IGNORECASE)
RE_QUOTED = re.compile(r"(?m)^(>+).*$")
RE_HDR = re.compile(
    r"(?:^|\n)(from:|sent:|to:|subject:).*(?:\n.*){0,20}",
    re.IGNORECASE
)


def normalize_subject(subj: str) -> str:
    s = subj or ""
    while True:
        ns = RE_PREFIX.sub('', s).strip()
        if ns == s:
            break
        s = ns
    s = re.sub(r'\s+', ' ', s)
    return s.lower()


def split_addresses(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [p.strip() for p in re.split(r'[;,]', str(x)) if p.strip()]


def clean_text(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r\n", "\n")
    t = RE_HDR.sub("\n", t)
    t = RE_QUOTED.sub("", t)
    t = "\n".join([ln.strip() for ln in t.split("\n") if ln.strip()])
    return t


# -----------------------------------------------------
# INDEXER
# -----------------------------------------------------

class EmailIndexer:

    def __init__(
        self,
        vector_repo,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ):
        self.vector_repo = vector_repo
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # -------------------------
    # PUBLIC ENTRYPOINT
    # -------------------------

    def index_dataframe(self, df: pd.DataFrame):

        docs = self._build_documents(df)
        cleaned = self._clean_documents(docs)
        chunks = self._chunk_documents(cleaned)

        if not chunks:
            return

        ids = self._generate_ids(chunks)

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.vector_repo.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

    # -------------------------
    # INTERNAL LOGIC
    # -------------------------

    def _build_documents(self, df: pd.DataFrame) -> List[Document]:

        documents: List[Document] = []

        for _, row in df.iterrows():

            subject = (row.get("subject") or "").strip()
            body = (row.get("body_text") or "").strip()

            if not body:
                body = (row.get("body_html") or "").strip()

            text = (subject + "\n\n" + body).strip()
            if not text:
                continue

            participants = (
                split_addresses(row.get("from_addr")) +
                split_addresses(row.get("to_addr")) +
                split_addresses(row.get("cc_addr")) +
                split_addresses(row.get("bcc_addr"))
            )

            participants = sorted(set(participants))
            norm_subj = normalize_subject(subject)
            thread_key = f"{norm_subj}||{';'.join(participants)}"

            metadata = {
                "row_id": row.get("id"),
                "message_id": row.get("message_id"),
                "subject": subject,
                "sent_at_utc": str(row.get("sent_at_utc")),
                "folder": row.get("folder"),
                "participants": participants,
                "thread_key": thread_key,
            }

            documents.append(
                Document(page_content=text, metadata=metadata)
            )

        return documents

    def _clean_documents(self, docs: List[Document]) -> List[Document]:

        cleaned = []

        for d in docs:
            txt = clean_text(d.page_content)
            if not txt:
                continue

            cleaned.append(
                Document(
                    page_content=txt,
                    metadata=d.metadata
                )
            )

        return cleaned

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

    def _generate_ids(self, chunks: List[Document]) -> List[str]:

        counters = {}
        ids = []

        for chunk in chunks:

            msg = (
                chunk.metadata.get("message_id") or
                chunk.metadata.get("row_id") or
                "noid"
            )

            i = counters.get(msg, 0)
            counters[msg] = i + 1

            raw = f"{msg}::chunk_{i}"
            uid = uuid.uuid5(uuid.NAMESPACE_URL, raw)

            ids.append(str(uid))

        return ids
