import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CLICKHOUSE_DATABASE
from infra import get_clickhouse_client
from pipeline import (
    clean_email_bodies_from_db,
    deduplicate_emails,
    import_mbox_to_clickhouse,
    index_messages,
    parse_emails_from_db,
)
from retrieval import (
    clear_summaries,
    get_project_corpus_batch,
    run_batch_analysis,
    run_global_analysis,
    search_project_threads,
)

app = FastAPI(title="mailkb backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SQL_DIR = Path(__file__).parent / "sql"


class CleanBodiesRequest(BaseModel):
    fetch_batch: int = 30
    llm_batch: int = 5


class ParseRequest(BaseModel):
    limit: int = 50
    batch_size: int = 3
    max_workers: int = 6


class IndexMessagesRequest(BaseModel):
    batch_size: int = 1000
    recreate: bool = False


class SearchThreadsRequest(BaseModel):
    project_hint: str
    limit: int = 10


class CorpusBatchRequest(BaseModel):
    project_hint: str
    offset: int = 0
    batch_size: int = 50
    thread_limit: int = 30


class AnalysisRequest(BaseModel):
    project_hint: str


def _run_sql_file(client, path: Path):
    sql = path.read_text(encoding="utf-8").strip()
    if not sql:
        return {"file": path.name, "status": "skipped (empty)"}
    client.query(sql)
    return {"file": path.name, "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/pipeline/init-db")
def api_init_db():
    if not SQL_DIR.is_dir():
        raise HTTPException(404, f"SQL directory not found: {SQL_DIR}")
    client = get_clickhouse_client()
    client.query(f"CREATE DATABASE IF NOT EXISTS {CLICKHOUSE_DATABASE}")
    files = sorted(SQL_DIR.glob("*.sql"))
    results = []
    for path in files:
        try:
            results.append(_run_sql_file(client, path))
        except Exception as e:
            results.append({"file": path.name, "status": f"error: {e}"})
    return {"status": "ok", "results": results}


@app.post("/pipeline/import-mbox")
def api_import_mbox():
    import_mbox_to_clickhouse()
    return {"status": "ok"}


@app.post("/pipeline/dedup")
def api_dedup():
    deduplicate_emails()
    return {"status": "ok"}


@app.post("/pipeline/clean-bodies")
def api_clean_bodies(payload: CleanBodiesRequest):
    clean_email_bodies_from_db(
        fetch_batch=payload.fetch_batch,
        llm_batch=payload.llm_batch,
    )
    return {"status": "ok"}


@app.post("/pipeline/parse")
def api_parse(payload: ParseRequest):
    result = parse_emails_from_db(
        limit=payload.limit,
        batch_size=payload.batch_size,
        max_workers=payload.max_workers,
    )
    return {"status": "ok", "result": result}


@app.post("/pipeline/index-messages")
def api_index_messages(payload: IndexMessagesRequest):
    index_messages(
        batch_size=payload.batch_size,
        recreate=payload.recreate,
    )
    return {"status": "ok"}


@app.post("/search/threads")
def api_search_threads(payload: SearchThreadsRequest):
    result = search_project_threads.invoke({
        "project_hint": payload.project_hint,
        "limit": payload.limit,
    })
    return json.loads(result)


@app.post("/search/corpus-batch")
def api_corpus_batch(payload: CorpusBatchRequest):
    result = get_project_corpus_batch.invoke({
        "project_hint": payload.project_hint,
        "offset": payload.offset,
        "batch_size": payload.batch_size,
        "thread_limit": payload.thread_limit,
    })
    return json.loads(result)


@app.post("/analysis/batch")
def api_batch_analysis(payload: AnalysisRequest):
    result = run_batch_analysis(payload.project_hint)
    return {"status": "ok", "result": result}


@app.post("/analysis/global")
def api_global_analysis(payload: AnalysisRequest):
    result = run_global_analysis(payload.project_hint)
    return {"status": "ok", "result": result}


@app.post("/summaries/clear")
def api_clear_summaries():
    clear_summaries()
    return {"status": "ok"}