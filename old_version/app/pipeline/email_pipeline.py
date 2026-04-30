from __future__ import annotations

from pathlib import Path
import math
import logging
from typing import List

from openai import OpenAI
import os

logger = logging.getLogger(__name__)


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ===============================
# ПРОМПТЫ (оставляем интеллект)
# ===============================

BATCH_PROMPT = """
Ты — аналитик переписки по проектам.

Проанализируй ТОЛЬКО этот блок переписки.

Сделай:
- ключевые темы
- что обсуждалось
- промежуточные выводы
- важные решения
- проблемы, если есть

НЕ делай глобальных выводов по проекту.
НЕ повторяй текст дословно.
"""


GLOBAL_PROMPT = """
Ты — эксперт по агрегированию больших массивов информации.

Тебе переданы summaries отдельных батчей.

Сформируй единый структурированный отчет:

1. Краткое резюме по проекту
2. Основные темы
3. Ключевые решения
4. Проблемы и риски
5. Повторяющиеся инциденты
6. Финальное резюме

Не переписывай батчи дословно.
Сделай смысловое сжатие.
"""


# ===============================
# PIPELINE
# ===============================

class EmailSummaryPipeline:

    def __init__(
        self,
        vector_repo,
        artifact_dir: Path,
        batch_model: str = "deepseek-chat",
        global_model: str = "deepseek-chat",
    ):
        self.vector_repo = vector_repo
        self.artifact_dir = artifact_dir
        self.batch_model = batch_model
        self.global_model = global_model

    # ----------------------------
    # 1️⃣ BATCH PROCESSING
    # ----------------------------

    def run_batch_processing(
        self,
        project_hint: str,
        batch_size: int = 50,
        max_docs: int = 500,
    ):

        docs = self.vector_repo.similarity_search(project_hint, k=max_docs)

        total_docs = len(docs)
        total_batches = math.ceil(total_docs / batch_size)

        logger.info(
            f"[BATCH] project={project_hint} "
            f"total_docs={total_docs} "
            f"batch_size={batch_size} "
            f"total_batches={total_batches}"
        )

        for batch_id in range(total_batches):

            start = batch_id * batch_size
            end = start + batch_size
            batch_docs = docs[start:end]

            if not batch_docs:
                break

            logger.info(
                f"[BATCH] Processing batch {batch_id+1}/{total_batches}"
            )

            batch_text = "\n\n".join(
                [d.page_content for d in batch_docs]
            )

            response = client.chat.completions.create(
                model=self.batch_model,
                messages=[
                    {"role": "system", "content": BATCH_PROMPT},
                    {
                        "role": "user",
                        "content": batch_text
                    },
                ],
                temperature=0.2,
            )

            summary = response.choices[0].message.content

            output_file = (
                self.artifact_dir
                / f"summary_batch_{batch_id+1}.txt"
            )
            output_file.write_text(summary, encoding="utf-8")

        logger.info("[BATCH] Completed all batches")

    # ----------------------------
    # 2️⃣ GLOBAL SUMMARY
    # ----------------------------

    def run_global_summary(self, project_hint: str):

        summaries = self._load_all_summaries()

        if not summaries:
            logger.warning("[GLOBAL] No summaries found")
            return None

        combined = "\n\n".join(summaries)

        logger.info("[GLOBAL] Generating final report")

        response = client.chat.completions.create(
            model=self.global_model,
            messages=[
                {"role": "system", "content": GLOBAL_PROMPT},
                {"role": "user", "content": combined},
            ],
            temperature=0.2,
        )

        return response

    # ----------------------------
    # HELPER
    # ----------------------------

    def _load_all_summaries(self) -> List[str]:

        summaries = []

        for file in sorted(self.artifact_dir.glob("summary_batch_*.txt")):
            summaries.append(file.read_text(encoding="utf-8"))

        return summaries
