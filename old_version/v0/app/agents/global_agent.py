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
"""

agent_global = create_agent(
    model="deepseek-chat",
    tools=[load_all_summaries],
    system_prompt=SYSTEM_PROMPT_GLOBAL,
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
            max_tokens_before_summary=2000,
        ),
    ],
)