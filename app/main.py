# app/main.py

from app.core.logging_config import setup_logging
setup_logging()


from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException

from app.db.run_repo import RunRepository
from app.db.clickhouse_repo import ClickhouseRepository
from app.db.vector_repo import VectorRepository
from app.ingestion.indexer import EmailIndexer
from app.services.report_service import ReportService


app = FastAPI(title="Enterprise Mail AI")


# ----------------------------
# Simple DI (manual)
# ----------------------------

run_repo = RunRepository(db_path="runs.db")

clickhouse_repo = ClickhouseRepository()

# ВАЖНО: режим коллекции выбирай как тебе нужно:
# - recreate
# - create_if_not_exists
# - use_existing
vector_repo = VectorRepository(
    collection_name="mailkb_emails",
    mode="create_if_not_exists",
)

email_indexer = EmailIndexer(vector_repo=vector_repo)

service = ReportService(
    run_repo=run_repo,
    clickhouse_repo=clickhouse_repo,
    vector_repo=vector_repo,
    email_indexer=email_indexer,
    artifacts_root=Path("artifacts"),
)


# ----------------------------
# Endpoints
# ----------------------------

@app.post("/index")
def start_indexing(
    background_tasks: BackgroundTasks,
    limit: int = 1000,
    batch_size: int = 500,
):
    run_id = service.create_run(project_hint="(indexing)", run_type="index")
    background_tasks.add_task(service.execute_index, run_id, limit, batch_size)
    return {"run_id": run_id, "status": "started", "limit": limit, "batch_size": batch_size}


@app.post("/report/batch")
def start_batch_report(
    project_hint: str,
    background_tasks: BackgroundTasks,
):
    run_id = service.create_run(project_hint=project_hint, run_type="batch")
    background_tasks.add_task(service.execute_batch_report, run_id, project_hint)
    return {"run_id": run_id, "status": "started", "project_hint": project_hint}


@app.post("/report/final")
def start_final_report(
    project_hint: str,
    background_tasks: BackgroundTasks,
):
    run_id = service.create_run(project_hint=project_hint, run_type="final")
    background_tasks.add_task(service.execute_final_report, run_id, project_hint)
    return {"run_id": run_id, "status": "started", "project_hint": project_hint}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = run_repo.get_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="run_not_found")
    return r


@app.get("/runs")
def list_runs():
    return run_repo.list_runs()


@app.get("/health")
def health():
    status = {
        "status": "ok",
        "services": {}
    }

    # Проверка ClickHouse
    try:
        clickhouse_repo.fetch_emails(limit=1, offset=0)
        status["services"]["clickhouse"] = "ok"
    except Exception as e:
        status["services"]["clickhouse"] = "error"
        status["status"] = "degraded"

    # Проверка Qdrant
    try:
        vector_repo.client.get_collections()
        status["services"]["qdrant"] = "ok"
    except Exception:
        status["services"]["qdrant"] = "error"
        status["status"] = "degraded"

    return status

