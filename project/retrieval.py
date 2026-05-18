import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
from langchain.tools import tool

from config import (
    LLM_MODEL,
    MESSAGES_COLLECTION,
    SUMMARY_DIR,
    THREADS_COLLECTION,
)
from infra import ensure_collection, get_clickhouse_client
from pipeline import build_message_docs


threads_qv = ensure_collection(THREADS_COLLECTION, recreate=False)
messages_qv = ensure_collection(MESSAGES_COLLECTION, recreate=False)


@tool
def search_project_threads(project_hint: str, limit: int = 30) -> str:
    """
    Найти релевантные обсуждения проекта в коллекции mailkb_threads.
    """
    docs = threads_qv.similarity_search(project_hint, k=limit)

    results = []
    seen = set()

    for doc in docs:
        md = doc.metadata or {}
        thread_key = md.get("thread_key")

        if not thread_key or thread_key in seen:
            continue
        seen.add(thread_key)

        results.append({
            "thread_key": thread_key,
            "subject": md.get("subject"),
            "participants": md.get("participants") or [],
            "keywords": md.get("keywords") or [],
            "message_count": md.get("message_count"),
            "first_date": md.get("first_date"),
            "last_date": md.get("last_date"),
            "snippet": (doc.page_content[:800] + "…") if doc.page_content else None,
        })

    return json.dumps(
        {"project_hint": project_hint, "threads": results},
        ensure_ascii=False,
        indent=2
    )


def _escape_ch_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


@tool
def get_project_corpus_batch(
    project_hint: str,
    offset: int = 0,
    batch_size: int = 50,
    thread_limit: int = 30
) -> str:
    """
    Собрать батч корпуса проекта:
    1) найти релевантные thread_key в mailkb_threads
    2) достать все сообщения по этим thread_key
    3) дедуплицировать и отсортировать
    4) вернуть батч по offset/batch_size
    """
    client = get_clickhouse_client()

    thread_docs = threads_qv.similarity_search(project_hint, k=thread_limit)

    thread_keys = []
    seen = set()

    for doc in thread_docs:
        md = doc.metadata or {}
        tk = md.get("thread_key")
        if tk and tk not in seen:
            seen.add(tk)
            thread_keys.append(tk)

    if not thread_keys:
        return json.dumps({
            "project_hint": project_hint,
            "offset": offset,
            "batch_size": batch_size,
            "batch_len": 0,
            "has_more": False,
            "thread_keys": [],
            "total_messages": 0,
            "batch": []
        }, ensure_ascii=False, indent=2)

    quoted_keys = ", ".join(f"'{_escape_ch_string(tk)}'" for tk in thread_keys)

    query = f"""
    SELECT
        e.id,
        e.thread_key,
        e.message_id,
        e.subject,
        e.from_addr,
        e.to_addr,
        e.cc_addr,
        e.bcc_addr,
        e.sent_at_utc,
        e.folder,
        p.parsed_json
    FROM mailkb.emails_unique e
    INNER JOIN mailkb.mail_parsed p
        ON e.id = p.email_id
    WHERE e.thread_key IN ({quoted_keys})
    ORDER BY e.sent_at_utc ASC, e.id ASC
    """

    df = client.query_df(query)
    docs = build_message_docs(df)

    deduped = []
    seen_msg = set()

    for d in docs:
        dedup_key = (
            f"{d.metadata.get('email_id')}::"
            f"{d.metadata.get('mail_query_number')}::"
            f"{d.metadata.get('date')}::"
            f"{d.page_content[:120]}"
        )
        if dedup_key in seen_msg:
            continue
        seen_msg.add(dedup_key)
        deduped.append(d)

    deduped.sort(
        key=lambda d: (
            d.metadata.get("date") or d.metadata.get("sent_at_utc") or "",
            str(d.metadata.get("mail_query_number") or "")
        )
    )

    batch_docs = deduped[offset: offset + batch_size]

    batch = []
    for d in batch_docs:
        md = d.metadata or {}
        batch.append({
            "thread_key": md.get("thread_key"),
            "email_id": md.get("email_id"),
            "message_id": md.get("message_id"),
            "subject": md.get("subject"),
            "topic": md.get("topic"),
            "sent_at_utc": md.get("date") or md.get("sent_at_utc"),
            "participants": md.get("participants") or [],
            "keywords": md.get("keywords") or [],
            "folder": md.get("folder"),
            "snippet": d.page_content[:800] if d.page_content else None,
        })

    return json.dumps({
        "project_hint": project_hint,
        "offset": offset,
        "batch_size": batch_size,
        "batch_len": len(batch),
        "has_more": offset + batch_size < len(deduped),
        "thread_keys": thread_keys,
        "total_messages": len(deduped),
        "batch": batch
    }, ensure_ascii=False, indent=2)


@tool
def save_summary(summary_text: str, batch_id: int) -> str:
    p = SUMMARY_DIR / f"summary_batch_{batch_id}.txt"
    p.write_text(summary_text, encoding="utf-8")
    return f"saved_to={str(p)}"


@tool
def load_all_summaries() -> str:
    files = sorted(SUMMARY_DIR.glob("summary_batch_*.txt"))

    if not files:
        return "NO_SUMMARIES_FOUND"

    texts = []
    for file in files:
        content = file.read_text(encoding="utf-8")
        texts.append(f"===== {file.name} =====\n{content}\n")

    return "\n".join(texts)


def clear_summaries():
    for f in SUMMARY_DIR.glob("summary_batch_*.txt"):
        f.unlink()


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


def build_batch_agent():
    return create_agent(
        model=LLM_MODEL,
        tools=[get_project_corpus_batch, save_summary],
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
                model=LLM_MODEL,
                max_tokens_before_summary=1200,
            ),
        ],
    )


def build_global_agent():
    return create_agent(
        model=LLM_MODEL,
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
                model=LLM_MODEL,
                max_tokens_before_summary=2000,
            ),
        ],
    )


def run_batch_analysis(project_hint: str):
    agent_batch = build_batch_agent()
    result_batch = agent_batch.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Начни обработку проекта {project_hint} батчами. "
                    "Делай summary каждого батча и сохраняй их через save_summary. "
                    "Итоговый отчёт не делай."
                )
            }
        ]
    })
    return result_batch["messages"][-1].content


def run_global_analysis(project_hint: str):
    agent_global = build_global_agent()
    result_global = agent_global.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Сделай итоговый отчёт по проекту {project_hint} на основе сохранённых batch summaries."
            }
        ]
    })
    return result_global["messages"][-1].content