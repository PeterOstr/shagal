# app/pipeline/email_pipeline.py

from __future__ import annotations

from pathlib import Path

from app.agents.batch_agent import BatchAgentFactory
from app.agents.global_agent import GlobalAgentFactory


class EmailSummaryPipeline:
    """
    Pipeline не знает про FastAPI.
    Pipeline знает только как:
      - запустить batch агента
      - запустить global агента
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

    def run_batch_processing(self, project_hint: str):
        return self.batch_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Начни обработку проекта {project_hint} батчами. "
                        f"Делай summary каждого батча и сохраняй через save_summary. "
                        f"Итоговый отчёт НЕ делай."
                    )
                }
            ]
        })

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
