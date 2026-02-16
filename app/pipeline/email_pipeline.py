# app/pipeline/email_pipeline.py

from __future__ import annotations

from pathlib import Path

from app.agents.batch_agent import BatchAgentFactory
from app.agents.global_agent import GlobalAgentFactory
import math
import logging

logger = logging.getLogger(__name__)


class EmailSummaryPipeline:
    """

    """

    def __init__(
        self,
        vector_repo,
        artifact_dir: Path,
        batch_model: str = "deepseek-chat",
        global_model: str = "deepseek-chat",
    ):
        self.vector_repo = vector_repo
        self.artifact_dir = artifact_dir

        self.batch_agent = BatchAgentFactory(
            vector_repo=vector_repo,
            artifact_dir=artifact_dir,
            model=batch_model,
        ).create()

        self.global_agent = GlobalAgentFactory(
            vector_repo=vector_repo,
            artifact_dir=artifact_dir,
            model=global_model,
        ).create()

    def run_batch_processing(self, project_hint: str, batch_size: int = 50):

        offset = 0
        batch_id = 1

        # получаем примерный объём через similarity search
        docs = self.vector_repo.similarity_search(project_hint, k=500)

        total_docs = len(docs)
        total_batches = math.ceil(total_docs / batch_size)

        logger.info(
            f"[BATCH] project={project_hint} "
            f"total_docs={total_docs} "
            f"batch_size={batch_size} "
            f"total_batches={total_batches}"
        )

        while True:

            batch_docs = docs[offset: offset + batch_size]

            if not batch_docs:
                break

            logger.info(
                f"[BATCH] Processing batch {batch_id}/{total_batches}"
            )

            # --- генерируем summary ---
            summary = self._summarize_batch(batch_docs)

            # --- сохраняем ---
            output_file = self.artifact_dir / f"summary_batch_{batch_id}.txt"
            output_file.write_text(summary, encoding="utf-8")

            offset += batch_size
            batch_id += 1

        logger.info(
            f"[BATCH] Completed all batches. total_batches={total_batches}"
        )


    def run_global_summary(self, project_hint: str):
        return self.global_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Начни обработку summaries и сделай итоговый отчет по проекту {project_hint}."
                    )
                }
            ]
        })
