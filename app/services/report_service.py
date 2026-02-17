# app/services/report_service.py

from __future__ import annotations

from threading import Thread
import time
import logging
from pathlib import Path
import pandas as pd

from app.pipeline.email_pipeline import EmailSummaryPipeline

def extract_content(result) -> str:
    """
    Универсальный извлекатель текста из ответа LLM.
    Защищает от разных форматов (dict, pydantic, ChatCompletion).
    """

    if result is None:
        return ""

    # LangChain message format
    if hasattr(result, "messages"):
        return result.messages[-1].content

    # dict format
    if isinstance(result, dict) and "messages" in result:
        return result["messages"][-1].content

    # OpenAI / DeepSeek ChatCompletion
    if hasattr(result, "choices"):
        return result.choices[0].message.content

    return ""


logger = logging.getLogger(__name__)


class ReportService:
    """
    Service layer.

    Отвечает за:
    - управление run-ами
    - запуск индексации
    - запуск batch summary
    - запуск финального отчета
    - запуск orchestrator facade
    """

    def __init__(
        self,
        run_repo,
        clickhouse_repo,
        vector_repo,
        email_indexer,
        artifacts_root: Path = Path("artifacts"),
    ):
        self.run_repo = run_repo
        self.clickhouse_repo = clickhouse_repo
        self.vector_repo = vector_repo
        self.email_indexer = email_indexer
        self.artifacts_root = artifacts_root
        self.orchestrator_agent = None

    # =====================================================
    # INTERNAL
    # =====================================================

    def set_orchestrator(self, orchestrator_agent):
        self.orchestrator_agent = orchestrator_agent

    def _artifact_dir(self, run_id: str) -> Path:
        path = self.artifacts_root / f"run_{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =====================================================
    # RUN CREATION
    # =====================================================

    def create_run(self, project_hint: str, run_type: str) -> str:
        run_id = self.run_repo.create_run(project_hint, run_type)
        logger.info(f"[{run_id}] Run created. type={run_type}")
        return run_id

    # =====================================================
    # INDEX
    # =====================================================

    def execute_index(
        self,
        run_id: str,
        limit: int = 1000,
        batch_size: int = 500,
    ):
        logger.info(f"[{run_id}] Starting indexing")

        try:
            self.run_repo.update_status(run_id, "running")

            offset = 0
            total = 0

            while total < limit:

                take = min(batch_size, limit - total)

                df: pd.DataFrame = self.clickhouse_repo.fetch_emails(
                    limit=take,
                    offset=offset,
                )

                if df is None or df.empty:
                    break

                self.email_indexer.index_dataframe(df)

                offset += len(df)
                total += len(df)

                logger.info(f"[{run_id}] Indexed batch size={len(df)}")

            self.run_repo.update_status(run_id, "completed")
            logger.info(f"[{run_id}] Index completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            logger.exception(f"[{run_id}] Index failed")
            raise

    # =====================================================
    # BATCH
    # =====================================================

    def execute_batch_report(self, run_id: str, project_hint: str):

        logger.info(f"[{run_id}] Starting batch report")

        try:
            self.run_repo.update_status(run_id, "running")

            artifact_dir = self._artifact_dir(run_id)

            pipeline = EmailSummaryPipeline(
                vector_repo=self.vector_repo,
                artifact_dir=artifact_dir,
                batch_model="deepseek-chat",
                global_model="deepseek-chat",
            )

            pipeline.run_batch_processing(project_hint)

            self.run_repo.update_status(run_id, "completed")
            logger.info(f"[{run_id}] Batch completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            logger.exception(f"[{run_id}] Batch failed")
            raise

    # =====================================================
    # FINAL
    # =====================================================

    def execute_final_report(self, run_id: str, project_hint: str):

        logger.info(f"[{run_id}] Starting final report")

        try:
            self.run_repo.update_status(run_id, "running")

            artifact_dir = self._artifact_dir(run_id)

            pipeline = EmailSummaryPipeline(
                vector_repo=self.vector_repo,
                artifact_dir=artifact_dir,
                batch_model="deepseek-chat",
                global_model="deepseek-chat",
            )

            result = pipeline.run_global_summary(project_hint)

            report_text = extract_content(result)

            (artifact_dir / "final_report.txt").write_text(
                report_text,
                encoding="utf-8",
            )

            self.run_repo.update_status(run_id, "completed")
            logger.info(f"[{run_id}] Final completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            logger.exception(f"[{run_id}] Final failed")
            raise

    # =====================================================
    # FULL (Backend orchestration)
    # =====================================================

    def execute_full_report(self, run_id: str, project_hint: str):

        logger.info(f"[{run_id}] Starting full workflow")

        try:
            self.run_repo.update_status(run_id, "running")

            # Выполняем шаги без изменения статуса внутри
            self._run_batch_internal(run_id, project_hint)
            self._run_final_internal(run_id, project_hint)

            self.run_repo.update_status(run_id, "completed")
            logger.info(f"[{run_id}] Full workflow completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            logger.exception(f"[{run_id}] Full workflow failed")
            raise

    def _run_batch_internal(self, run_id: str, project_hint: str):

        artifact_dir = self._artifact_dir(run_id)

        pipeline = EmailSummaryPipeline(
            vector_repo=self.vector_repo,
            artifact_dir=artifact_dir,
            batch_model="deepseek-chat",
            global_model="deepseek-chat",
        )

        pipeline.run_batch_processing(project_hint)

    def _run_final_internal(self, run_id: str, project_hint: str):

        artifact_dir = self._artifact_dir(run_id)

        pipeline = EmailSummaryPipeline(
            vector_repo=self.vector_repo,
            artifact_dir=artifact_dir,
            batch_model="deepseek-chat",
            global_model="deepseek-chat",
        )

        result = pipeline.run_global_summary(project_hint)

        report_text = extract_content(result)

        (artifact_dir / "final_report.txt").write_text(
            report_text,
            encoding="utf-8",
        )

    # =====================================================
    # ORCHESTRATOR
    # =====================================================

    def execute_orchestrator(self, run_id: str, message: str):

        if not self.orchestrator_agent:
            raise RuntimeError("Orchestrator not configured")

        logger.info(f"[{run_id}] Starting orchestrator")

        try:
            self.run_repo.update_status(run_id, "running")

            result = self.orchestrator_agent.invoke(message)
            response = result["messages"][-1].content

            artifact_dir = self._artifact_dir(run_id)
            (artifact_dir / "orchestrator_response.txt").write_text(
                response,
                encoding="utf-8",
            )

            self.run_repo.update_status(run_id, "completed")
            logger.info(f"[{run_id}] Orchestrator completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            logger.exception(f"[{run_id}] Orchestrator failed")
            raise

    def execute_batch_report_async(self, run_id, project_hint):
        Thread(target=self.execute_batch_report, args=(run_id, project_hint)).start()

    def execute_final_report_async(self, run_id, project_hint):
        Thread(target=self.execute_final_report, args=(run_id, project_hint)).start()

    def execute_full_report_async(self, run_id, project_hint):
        Thread(target=self.execute_full_report, args=(run_id, project_hint)).start()
