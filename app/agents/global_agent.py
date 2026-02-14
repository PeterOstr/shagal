# app/agents/global_agent.py

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware

from app.agents.tools import build_tools


GLOBAL_SYSTEM_PROMPT = """
Ты — эксперт по агрегированию больших массивов информации.

Твоя задача — взять summary отдельных батчей (их подготовил другой агент),
и создать единый, структурированный глобальный отчет.

Алгоритм:
1) Вызови load_all_summaries().
2) Проведи анализ:
   - ключевые темы,
   - решения,
   - риски,
   - нерешённые вопросы,
   - повторяющиеся инциденты.
3) Выдай ОДИН финальный отчёт:

Структура:
1. Краткое резюме по проекту.
2. Основные темы.
3. Ключевые решения.
4. Проблемы и риски.
5. Повторяющиеся инциденты.
6. Финальное резюме.

Не переписывай батчи дословно — делай смысловое сжатие.
"""


class GlobalAgentFactory:
    def __init__(self, vector_repo, artifact_dir: Path, model: str = "deepseek-chat"):
        self.vector_repo = vector_repo
        self.artifact_dir = artifact_dir
        self.model = model

    def create(self):
        tools = build_tools(self.vector_repo, self.artifact_dir)

        agent = create_agent(
            model=self.model,
            tools=[
                tools["load_all_summaries"],
                # опционально:
                tools["search_emails_raw"],
            ],
            system_prompt=GLOBAL_SYSTEM_PROMPT,
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
                    max_tokens_before_summary=1000,
                ),
            ],
        )
        return agent
