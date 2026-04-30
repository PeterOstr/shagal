# app/main.py
from batch_agent import create_my_agent


# from app.core.logging_config import setup_logging
# setup_logging()
#
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException

#
# from app.db.run_repo import RunRepository
# from app.db.clickhouse_repo import ClickhouseRepository
# from app.db.vector_repo import VectorRepository
# from app.ingestion.indexer import EmailIndexer
# from app.services.report_service import ReportService
# from app.agents.orchestrator_agent import OrchestratorAgent


app = FastAPI(title="Enterprise Mail AI")



@app.get("/health")
def health():

    status = {
        "status": "ok",
        "services": {}
    }

    # try:
    #     clickhouse_repo.fetch_emails(limit=1, offset=0)
    #     status["services"]["clickhouse"] = "ok"
    # except Exception:
    #     status["services"]["clickhouse"] = "error"
    #     status["status"] = "degraded"
    #
    # try:
    #     vector_repo.client.get_collections()
    #     status["services"]["qdrant"] = "ok"
    # except Exception:
    #     status["services"]["qdrant"] = "error"
    #     status["status"] = "degraded"

    return status



@app.get("/agent1")
def agent1():
    create_my_agent()


    return create_my_agent

