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
from app.agents.orchestrator_agent import OrchestratorAgent


app = FastAPI(title="Enterprise Mail AI")


# =========================================================
# Manual DI
# =========================================================

run_repo = RunRepository(db_path="runs.db")

clickhouse_repo = ClickhouseRepository()

vector_repo = VectorRepository(
    collection_name="mailkb_emails",
    mode="create_if_not_exists",
)

email_indexer = EmailIndexer(vector_repo=vector_repo)

report_service = ReportService(
    run_repo=run_repo,
    clickhouse_repo=clickhouse_repo,
    vector_repo=vector_repo,
    email_indexer=email_indexer,
    artifacts_root=Path("artifacts"),
)

orchestrator_agent = OrchestratorAgent(report_service)
report_service.set_orchestrator(orchestrator_agent)


# =========================================================
# INDEX
# =========================================================

@app.post("/index")
def start_indexing(
    background_tasks: BackgroundTasks,
    limit: int = 1000,
    batch_size: int = 500,
):
    run_id = report_service.create_run("(indexing)", "index")

    background_tasks.add_task(
        report_service.execute_index,
        run_id,
        limit,
        batch_size
    )

    return {
        "run_id": run_id,
        "status": "started",
        "limit": limit,
        "batch_size": batch_size,
    }


# =========================================================
# BATCH
# =========================================================

@app.post("/report/batch")
def start_batch_report(
    project_hint: str,
    background_tasks: BackgroundTasks,
):
    run_id = report_service.create_run(project_hint, "batch")

    background_tasks.add_task(
        report_service.execute_batch_report,
        run_id,
        project_hint
    )

    return {
        "run_id": run_id,
        "status": "started",
        "project_hint": project_hint,
    }


# =========================================================
# FINAL
# =========================================================

@app.post("/report/final")
def start_final_report(
    project_hint: str,
    background_tasks: BackgroundTasks,
):
    run_id = report_service.create_run(project_hint, "final")

    background_tasks.add_task(
        report_service.execute_final_report,
        run_id,
        project_hint
    )

    return {
        "run_id": run_id,
        "status": "started",
        "project_hint": project_hint,
    }


# =========================================================
# FULL (deterministic backend orchestration)
# =========================================================

@app.post("/report/full")
def full_report(
    project_hint: str,
    background_tasks: BackgroundTasks,
):
    run_id = report_service.create_run(project_hint, "full")

    background_tasks.add_task(
        report_service.execute_full_report,
        run_id,
        project_hint
    )

    return {
        "run_id": run_id,
        "status": "started"
    }


# =========================================================
# ORCHESTRATOR (AI facade, async)
# =========================================================

@app.post("/report/orchestrate")
def orchestrate(
    message: str,
    background_tasks: BackgroundTasks,
):
    run_id = report_service.create_run("(orchestrator)", "orchestrator")

    background_tasks.add_task(
        report_service.execute_orchestrator,
        run_id,
        message
    )

    return {
        "run_id": run_id,
        "status": "started"
    }


# =========================================================
# RUNS
# =========================================================

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = run_repo.get_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="run_not_found")
    return r


@app.get("/runs")
def list_runs():
    return run_repo.list_runs()


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    status = {
        "status": "ok",
        "services": {}
    }

    try:
        clickhouse_repo.fetch_emails(limit=1, offset=0)
        status["services"]["clickhouse"] = "ok"
    except Exception:
        status["services"]["clickhouse"] = "error"
        status["status"] = "degraded"

    try:
        vector_repo.client.get_collections()
        status["services"]["qdrant"] = "ok"
    except Exception:
        status["services"]["qdrant"] = "error"
        status["status"] = "degraded"

    return status
