from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware

from .config import DEFAULT_AGENT_MODEL
from .tools import get_project_corpus_batch, load_all_summaries, save_summary


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
""".strip()


SYSTEM_PROMPT_GLOBAL = """
Ты — эксперт по агрегированию проектной переписки.

Твоя задача:
1. Вызвать load_all_summaries().
2. На основе всех summary батчей сделать единый итоговый отчёт по проекту.

Структура итогового отчёта:
1. Краткое резюме проекта
2. Основные темы обсуждений
3. Ключевые решения и к чему пришли
4. Явные задачи и ответственные
5. Вероятные задачи и вероятные ответственные
6. Проблемы, риски, открытые вопросы
7. Повторяющиеся инциденты / узкие места
8. Итоговый вывод

Важно:
- Не переписывай summaries батчей дословно.
- Делай смысловую агрегацию.
- Если задача/ответственный восстановлены по контексту, помечай это отдельно.
- Если есть противоречия между батчами, укажи их.
""".strip()


def _common_middleware(max_tokens_before_summary: int):
    return [
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
            model=DEFAULT_AGENT_MODEL,
            max_tokens_before_summary=max_tokens_before_summary,
        ),
    ]


def build_batch_agent(model: str = DEFAULT_AGENT_MODEL):
    return create_agent(
        model=model,
        tools=[get_project_corpus_batch, save_summary],
        system_prompt=SYSTEM_PROMPT_BATCH,
        middleware=_common_middleware(max_tokens_before_summary=1200),
    )


def build_global_agent(model: str = DEFAULT_AGENT_MODEL):
    return create_agent(
        model=model,
        tools=[load_all_summaries],
        system_prompt=SYSTEM_PROMPT_GLOBAL,
        middleware=_common_middleware(max_tokens_before_summary=2000),
    )