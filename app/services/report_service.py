# app/services/report_service.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from app.pipeline.email_pipeline import EmailSummaryPipeline


class ReportService:
    """
    Service layer: управляет runs + вызывает ingestion/pipeline.
    FastAPI вызывает сервис, сервис вызывает пайплайн.
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

    # ----------------------------
    # helpers
    # ----------------------------

    def _artifact_dir(self, run_id: str) -> Path:
        # каждый run пишет в свой каталог
        p = self.artifacts_root / f"run_{run_id}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def create_run(self, project_hint: str, run_type: str) -> str:
        return self.run_repo.create_run(project_hint=project_hint, run_type=run_type)

    # ----------------------------
    # EXECUTERS (for BackgroundTasks)
    # ----------------------------

    def execute_index(self, run_id: str, limit: int = 1000, batch_size: int = 500):
        """
        Индексация последних писем из ClickHouse -> Qdrant.
        """
        try:
            self.run_repo.update_status(run_id, "running")

            offset = 0
            total = 0

            while total < limit:
                take = min(batch_size, limit - total)
                df = self.clickhouse_repo.fetch_emails(limit=take, offset=offset)
                if df is None or df.empty:
                    break

                self.email_indexer.index_dataframe(df)

                offset += len(df)
                total += len(df)

            self.run_repo.update_status(run_id, "completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            raise

    def execute_batch_report(self, run_id: str, project_hint: str):
        """
        Batch summaries (семантический поиск по Qdrant -> summary files)
        """
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

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            raise

    def execute_final_report(self, run_id: str, project_hint: str):
        """
        Final report (load summaries -> final report text)
        """
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

            # сохраним финальный отчёт как артефакт
            report_text = result["messages"][-1].content if result and result.get("messages") else ""
            out = artifact_dir / "final_report.txt"
            out.write_text(report_text, encoding="utf-8")

            self.run_repo.update_status(run_id, "completed")

        except Exception:
            self.run_repo.update_status(run_id, "failed")
            raise
