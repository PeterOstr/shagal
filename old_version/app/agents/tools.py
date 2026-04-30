# app/agents/tools.py

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from langchain.tools import tool


def build_tools(vector_repo, artifact_dir: Path):
    """
    Возвращает набор LangChain tools, привязанных к конкретному vector_repo и artifact_dir.
    artifact_dir должен быть уникальным для run_id (чтобы summaries не смешивались).
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summaries_dir = artifact_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    @tool
    def search_project_emails_batch(project_hint: str, offset: int = 0, batch_size: int = 50) -> str:
        """
        Возвращает batch семантического поиска по Qdrant.
        Для batch-агента: батч -> summary -> save -> следующий батч.

        Output JSON:
        {
          "project_hint": "...",
          "offset": 0,
          "batch_size": 50,
          "batch_len": 50,
          "has_more": true/false,
          "batch": [
            {"subject": "...", "snippet": "...", "sent_at_utc": "...", "thread_key": "..."}
          ]
        }
        """
        docs = vector_repo.similarity_search(project_hint, k=offset + batch_size)
        docs = docs[offset: offset + batch_size]

        batch = []
        for d in docs:
            md = d.metadata or {}
            batch.append({
                "subject": md.get("subject"),
                "snippet": (d.page_content or "")[:600],
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

    @tool
    def save_summary(summary_text: str, batch_id: int) -> str:
        """
        Сохраняет резюме батча.
        """
        p = summaries_dir / f"summary_batch_{batch_id:04d}.txt"
        p.write_text(summary_text, encoding="utf-8")
        return f"saved_to={str(p)}"

    @tool
    def load_all_summaries() -> str:
        """
        Читает все файлы summary_batch_*.txt из summaries_dir и возвращает объединённый текст.
        """
        files = sorted(summaries_dir.glob("summary_batch_*.txt"))
        if not files:
            return "NO_SUMMARIES_FOUND"

        texts = []
        for file in files:
            content = file.read_text(encoding="utf-8")
            texts.append(f"===== {file.name} =====\n{content}\n")

        return "\n".join(texts)

    @tool
    def search_emails_raw(query: str, limit: int = 50) -> str:
        """
        Общий поиск (без группировки) — удобно для отладки.
        """
        docs = vector_repo.similarity_search(query, k=limit)
        data = []
        for doc in docs:
            md = doc.metadata or {}
            data.append({
                "thread_key": md.get("thread_key"),
                "row_id": md.get("row_id"),
                "message_id": md.get("message_id"),
                "subject": md.get("subject"),
                "sent_at_utc": md.get("sent_at_utc"),
                "folder": md.get("folder"),
                "participants": md.get("participants") or [],
                "snippet": ((doc.page_content or "")[:400] + "…") if doc.page_content else None,
            })
        return json.dumps({"query": query, "results": data}, ensure_ascii=False, indent=2)

    return {
        "search_project_emails_batch": search_project_emails_batch,
        "save_summary": save_summary,
        "load_all_summaries": load_all_summaries,
        "search_emails_raw": search_emails_raw,
        "artifact_dir": artifact_dir,
        "summaries_dir": summaries_dir,
    }
