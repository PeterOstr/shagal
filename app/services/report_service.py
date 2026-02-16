# app/services/report_service.py

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from app.pipeline.email_pipeline import EmailSummaryPipeline


logger = logging.getLogger(__name__)


class ReportService:
    """
    Service layer.

    Отвечает за:
    - управление run-ами
    - запуск индексации
    - запуск batch summary
    - запуск финального отчета

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

    # =====================================================
    # INTERNAL HELPERS
    # =====================================================

    def _artifact_dir(self, run_id: str) -> Path:
        """
        Каждый run получает свою директорию.
        """
        path = self.artifacts_root / f"run_{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =====================================================
    # RUN CREATION
    # =====================================================

    def create_run(self, project_hint: str, run_type: str) -> str:
        run_id = self.run_repo.create_run(project_hint, run_type)
        logger.info(f"[{run_id}] Run created. type={run_type} project={project_hint}")
        return run_id

    # =====================================================
    # INDEXING
    # =====================================================

    def execute_index(
        self,
        run_id: str,
        limit: int = 1000,
        batch_size: int = 500
    ):
        """
        Индексация писем из ClickHouse в Qdrant.
        """

        start_time = time.perf_counter()

        logger.info(
            f"[{run_id}] Starting indexing. "
            f"limit={limit} batch_size={batch_size}"
        )

        try:
            self.run_repo.update_status(run_id, "running")

            offset = 0
            total = 0

            while total < limit:

                take = min(batch_size, limit - total)

                df: pd.DataFrame = self.clickhouse_repo.fetch_emails(
                    limit=take,
                    offset=offset
                )

                if df is None or df.empty:
                    logger.info(f"[{run_id}] No more emails to index.")
                    break

                self.email_indexer.index_dataframe(df)

                offset += len(df)
                total += len(df)

                logger.info(
                    f"[{run_id}] Indexed batch. "
                    f"batch_size={len(df)} total={total}"
                )

            duration = round(time.perf_counter() - start_time, 2)

            self.run_repo.update_status(run_id, "completed")

            logger.info(
                f"[{run_id}] Indexing completed successfully. "
                f"duration={duration}s total_indexed={total}"
            )

        except Exception as e:
            duration = round(time.perf_counter() - start_time, 2)

            self.run_repo.update_status(run_id, "failed")

            logger.exception(
                f"[{run_id}] Indexing failed. "
                f"duration={duration}s error={str(e)}"
            )

            raise

    # =====================================================
    # BATCH SUMMARY
    # =====================================================

    def execute_batch_report(
        self,
        run_id: str,
        project_hint: str
    ):
        """
        Batch summaries (Qdrant search → summary files).
        """

        start_time = time.perf_counter()

        logger.info(
            f"[{run_id}] Starting batch report. "
            f"project={project_hint}"
        )

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

            duration = round(time.perf_counter() - start_time, 2)

            self.run_repo.update_status(run_id, "completed")

            logger.info(
                f"[{run_id}] Batch report completed successfully. "
                f"project={project_hint} duration={duration}s"
            )

        except Exception as e:
            duration = round(time.perf_counter() - start_time, 2)

            self.run_repo.update_status(run_id, "failed")

            logger.exception(
                f"[{run_id}] Batch report failed. "
                f"project={project_hint} duration={duration}s error={str(e)}"
            )

            raise

    # =====================================================
    # FINAL REPORT
    # =====================================================

    def execute_final_report(
        self,
        run_id: str,
        project_hint: str
    ):
        """
        Итоговый отчет (load summaries → final_report.txt).
        """

        start_time = time.perf_counter()

        logger.info(
            f"[{run_id}] Starting final report. "
            f"project={project_hint}"
        )

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

            report_text = ""
            if result and result.get("messages"):
                report_text = result["messages"][-1].content

            output_file = artifact_dir / "final_report.txt"
            output_file.write_text(report_text, encoding="utf-8")

            duration = round(time.perf_counter() - start_time, 2)

            self.run_repo.update_status(run_id, "completed")

            logger.info(
                f"[{run_id}] Final report completed successfully. "
                f"project={project_hint} duration={duration}s"
            )

        except Exception as e:
            duration = round(time.perf_counter() - start_time, 2)

            self.run_repo.update_status(run_id, "failed")

            logger.exception(
                f"[{run_id}] Final report failed. "
                f"project={project_hint} duration={duration}s error={str(e)}"
            )

            raise
