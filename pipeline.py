import json
from pathlib import Path
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware


SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)


class EmailSummaryPipeline:
    def __init__(self, qv, model="gpt-5"):
        """
        qv — это QdrantVectorStore, который ты создаёшь в основном ноутбуке.
        model — имя модели (например 'gpt-5').
        """
        self.qv = qv
        self.model = model

        # -------------------------------------------------
        # Register tools bound to this instance
        # -------------------------------------------------
        self.search_project_emails_batch_tool = tool(self.search_project_emails_batch)
        self.save_summary_tool = tool(self.save_summary)
        self.load_all_summaries_tool = tool(self.load_all_summaries)

        # -------------------------------------------------
        # Batch agent (agent #1)
        # -------------------------------------------------
        self.batch_agent = create_agent(
            model=model,
            tools=[
                self.search_project_emails_batch_tool,
                self.save_summary_tool
            ],
            system_prompt=self.BATCH_SYSTEM_PROMPT(),
            middleware=[
                PIIMiddleware("email", strategy="redact", apply_to_input=True),
                PIIMiddleware(
                    "phone_number",
                    detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
                    strategy="redact",
                ),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=1000,
                ),
            ]
        )

        # -------------------------------------------------
        # Global summary agent (agent #2)
        # -------------------------------------------------
        self.global_agent = create_agent(
            model=model,
            tools=[self.load_all_summaries_tool],
            system_prompt=self.GLOBAL_SYSTEM_PROMPT(),
            middleware=[
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=2000,
                ),
            ],
        )

    # =====================================================
    #  TOOLS (INSTANCE METHODS)
    # =====================================================

    def search_project_emails_batch(self, project_hint: str, offset: int = 0, batch_size: int = 50) -> str:
        """
        Возвращает batch из QdrantVectorStore (self.qv).
        """
        docs = self.qv.similarity_search(project_hint, k=offset + batch_size)
        docs = docs[offset: offset + batch_size]

        batch = []
        for d in docs:
            md = d.metadata or {}
            batch.append({
                "subject": md.get("subject"),
                "snippet": d.page_content[:600],
                "sent_at_utc": md.get("sent_at_utc"),
                "thread_key": md.get("thread_key"),
            })

        return json.dumps({
            "project_hint": project_hint,
            "offset": offset,
            "batch_size": batch_size,
            "batch_len": len(batch),
            "has_more": len(batch) == batch_size,
            "batch": batch
        }, ensure_ascii=False)

    def save_summary(self, summary_text: str, batch_id: int) -> str:
        """
        Сохраняет summary батча.
        """
        path = SUMMARY_DIR / f"summary_batch_{batch_id}.txt"
        path.write_text(summary_text, encoding="utf-8")
        return f"saved_to={path}"

    def load_all_summaries(self) -> str:
        """
        Читает все summary_batch_*.txt.
        """
        texts = []
        for file in sorted(SUMMARY_DIR.glob("summary_batch_*.txt")):
            content = file.read_text(encoding="utf-8")
            texts.append(f"===== {file.name} =====\n{content}\n")

        if not texts:
            return "NO_SUMMARIES_FOUND"

        return "\n".join(texts)

    # =====================================================
    #  SYSTEM PROMPTS
    # =====================================================

    def BATCH_SYSTEM_PROMPT(self):
        return """
Ты — аналитик переписки.

Твоя задача — обработать переписку по проекту батчами.
Каждый батч нужно резюмировать и сохранить через save_summary.
Итоговый отчёт НЕ ДЕЛАТЬ.

АЛГОРИТМ:

1) Определи project_hint из запроса.
2) offset = 0, batch_id = 1

3) Вызови search_project_emails_batch(project_hint, offset, batch_size=50)

4) Создай summary ТОЛЬКО для этого батча.

5) Вызови save_summary(summary_text, batch_id).

6) Если has_more = true:
       offset = offset + 50
       batch_id = batch_id + 1
       вернись к шагу 3.
   Иначе:
       сообщи пользователю "Все батчи обработаны".

Требования:
- НЕ делай глобальный отчет.
- НЕ агрегируй темы между батчами.
- Каждый batch сохраняется отдельно.
"""

    def GLOBAL_SYSTEM_PROMPT(self):
        return """
Ты — эксперт по агрегированию больших массивов информации.

Твоя задача — взять summary отдельных батчей (их подготовил другой агент),
и создать единый, структурированный глобальный отчет.

Алгоритм:

1) Вызови load_all_summaries().
2) Проведи глубокий анализ:
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
5. Повторяющиеся инциденты (объединённо).
6. Финальное резюме.

Не переписывай батчи дословно — делай смысловое сжатие.
"""

    # =====================================================
    #  PUBLIC METHODS
    # =====================================================

    def run_batch_processing(self, project_hint: str):
        """
        Запускает обработку батчей.
        """
        return self.batch_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Начни обработку проекта '{project_hint}' батчами. "
                        f"Сохраняй summary каждого батча. "
                        f"Итоговый отчет НЕ делай."
                    )
                }
            ]
        })

    def run_global_summary(self, project_hint: str):
        """
        Делает итоговый отчёт.
        """
        return self.global_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Сделай итоговый отчёт по проекту '{project_hint}'."
                }
            ]
        })
