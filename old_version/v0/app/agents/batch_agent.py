from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
import tools as tls

SYSTEM_PROMPT_BATCH = """
Ты — аналитик проектной переписки.

Твоя задача — обработать корпус сообщений по проекту батчами.
Корпус уже собран из релевантных обсуждений проекта.

Работай строго по циклу:

1. Определи project_hint из запроса пользователя.
2. Вызови get_project_corpus_batch(project_hint, offset, batch_size=50).
3. Получи batch.
4. На основе только этого batch сделай summary.
5. Сохрани summary через save_summary(summary_text, batch_id).
6. Если has_more = true — увеличь offset на batch_size и продолжай.
7. Если has_more = false — сообщи, что все батчи обработаны.

Ограничения:
- Не делай глобального отчёта.
- Не объединяй выводы между батчами.
- Каждый summary должен описывать только текущий batch.
- Не пропускай save_summary.
- Не используй никакие другие инструменты кроме get_project_corpus_batch и save_summary.

Для каждого батча извлекай:
1. Основные темы
2. Что обсуждалось
3. Решения / договорённости
4. Проблемы / риски / открытые вопросы
5. Явные задачи
6. Вероятные задачи
7. Явных ответственных
8. Вероятных ответственных
9. Краткий вывод по батчу

Важно:
- Если задача или ответственный не сформулированы явно, можешь аккуратно реконструировать их,
  но обязательно явно разделяй:
  - "Явно указано"
  - "Вероятно следует из контекста"
"""

def create_my_agent():
    agent_batch = create_agent(
        model="deepseek-chat",
        tools=[
            tls.get_project_corpus_batch,
            tls.save_summary,
        ],
        system_prompt=SYSTEM_PROMPT_BATCH,
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
                model="deepseek-chat",
                max_tokens_before_summary=1200,
            ),
        ],
    )
    return agent_batch
