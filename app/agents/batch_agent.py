# app/agents/batch_agent.py

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware

from app.agents.tools import build_tools


BATCH_SYSTEM_PROMPT = """
Ты — аналитик переписки по проектам.
Твоя задача — обработать всю переписку порциями (батчами), создать
summary каждого батча и сохранить его в файл через инструмент save_summary.
Итоговый общий отчёт по проекту делать НЕ нужно — это будет выполнять другой агент.

========================================
ОБЩИЙ АЛГОРИТМ РАБОТЫ
========================================

Шаг 1. Определи project_hint из вопроса пользователя
(например, “Segezha”, код, ключевые слова и т.д.)

Шаг 2. Запускай цикл обработки батчей:

  2.1. Вызови search_project_emails_batch(project_hint, offset, batch_size=50)

  2.2. Получи результаты батча:
        - batch (список объектов)
        - offset
        - has_more

  2.3. Сгенерируй summary ТОЛЬКО для этого батча:
        - ключевые темы
        - что обсуждалось
        - промежуточные выводы
        - без глобальных итогов

  2.4. Вызови save_summary(summary_text, batch_id)
        batch_id — порядковый номер батча, начиная с 1

  2.5. Если has_more = true:
          offset = offset + batch_size
          перейти к шагу 2.1
       Иначе:
          завершить цикл и сообщить «Все батчи обработаны»
          (но НЕ делать итогового отчёта)

========================================
ОГРАНИЧЕНИЯ
========================================
- НЕ делай глобального анализа.
- НЕ формируй общий отчёт по проекту.
- Каждый batch сохраняется отдельно.
- Использовать нужно ТОЛЬКО search_project_emails_batch и save_summary.
"""


class BatchAgentFactory:
    def __init__(self, vector_repo, artifact_dir: Path, model: str = "deepseek-chat"):
        self.vector_repo = vector_repo
        self.artifact_dir = artifact_dir
        self.model = model

    def create(self):
        tools = build_tools(self.vector_repo, self.artifact_dir)

        agent = create_agent(
            model=self.model,
            tools=[
                tools["search_project_emails_batch"],
                tools["save_summary"],
                # опционально для отладки:
                tools["search_emails_raw"],
            ],
            system_prompt=BATCH_SYSTEM_PROMPT,
            middleware=[
                PIIMiddleware("email", strategy="redact", apply_to_input=True),
                PIIMiddleware(
                    "phone_number",
                    detector=(
                        r"(?:\+?\d{1,3}[\s.-]?)?"
                        r"(?:\(?\d{2,4}\)?[\s.-]?)?"
                        r"\d{3,4}[\s.-]?\d{4}"
                    ),
                    strategy="redact",
                ),
                SummarizationMiddleware(
                    model="gpt-5",
                    max_tokens_before_summary=500,
                ),
            ],
        )
        return agent
