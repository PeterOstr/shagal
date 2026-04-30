import json

from fastapi import APIRouter, HTTPException

from app.deps import check_connections
from app.schemas import (
    CorpusBatchRequest,
    HealthResponse,
    IngestMessagesRequest,
    IngestThreadsRequest,
    RunBatchAgentRequest,
    RunGlobalAgentRequest,
    SearchThreadsRequest,
)
from mailkb.agents import build_batch_agent, build_global_agent
from mailkb.ingestion import ingest_messages, ingest_threads
from mailkb.tools import (
    clear_summaries,
    get_project_corpus_batch,
    search_project_threads,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.get("/health/deep")
def deep_health():
    return check_connections()


@router.post("/ingest/messages")
def api_ingest_messages(payload: IngestMessagesRequest):
    try:
        inserted = ingest_messages(
            batch_size=payload.batch_size,
            recreate=payload.recreate,
        )
        return {
            "status": "ok",
            "inserted_docs": inserted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/threads")
def api_ingest_threads(payload: IngestThreadsRequest):
    try:
        inserted = ingest_threads(recreate=payload.recreate)
        return {
            "status": "ok",
            "inserted_docs": inserted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/threads")
def api_search_threads(payload: SearchThreadsRequest):
    try:
        raw = search_project_threads.invoke(
            {
                "project_hint": payload.project_hint,
                "limit": payload.limit,
            }
        )
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/corpus/batch")
def api_get_corpus_batch(payload: CorpusBatchRequest):
    try:
        raw = get_project_corpus_batch.invoke(
            {
                "project_hint": payload.project_hint,
                "offset": payload.offset,
                "batch_size": payload.batch_size,
                "thread_limit": payload.thread_limit,
            }
        )
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/batch")
def api_run_batch_agent(payload: RunBatchAgentRequest):
    try:
        agent = build_batch_agent(model=payload.model)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Начни обработку проекта {payload.project_hint} батчами. "
                            "Делай summary каждого батча и сохраняй их через save_summary. "
                            "Итоговый отчёт не делай."
                        ),
                    }
                ]
            }
        )
        return {
            "status": "ok",
            "result": result["messages"][-1].content,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/global")
def api_run_global_agent(payload: RunGlobalAgentRequest):
    try:
        agent = build_global_agent(model=payload.model)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Сделай итоговый отчёт по проекту {payload.project_hint} "
                            "на основе сохранённых batch summaries."
                        ),
                    }
                ]
            }
        )
        return {
            "status": "ok",
            "result": result["messages"][-1].content,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summaries/clear")
def api_clear_summaries():
    try:
        clear_summaries()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))