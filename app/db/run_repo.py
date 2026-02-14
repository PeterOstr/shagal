# app/db/run_repo.py

import sqlite3
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict


class RunRepository:

    def __init__(self, db_path: str = "runs.db"):
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )
        self._create_table()

    # ----------------------------------------
    # INIT
    # ----------------------------------------

    def _create_table(self):

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            project_hint TEXT,
            type TEXT,
            status TEXT,
            created_at TEXT,
            finished_at TEXT
        )
        """)

        self.conn.commit()

    # ----------------------------------------
    # CREATE
    # ----------------------------------------

    def create_run(self, project_hint: str, run_type: str) -> str:

        run_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        self.conn.execute(
            """
            INSERT INTO runs
            (id, project_hint, type, status, created_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, project_hint, run_type, "pending", now, None)
        )

        self.conn.commit()
        return run_id

    # ----------------------------------------
    # UPDATE
    # ----------------------------------------

    def update_status(self, run_id: str, status: str):

        finished_at = None

        if status in ("completed", "failed"):
            finished_at = datetime.utcnow().isoformat()

        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?
            WHERE id = ?
            """,
            (status, finished_at, run_id)
        )

        self.conn.commit()

    # ----------------------------------------
    # READ
    # ----------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict]:

        cursor = self.conn.execute(
            "SELECT * FROM runs WHERE id = ?",
            (run_id,)
        )

        row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "project_hint": row[1],
            "type": row[2],
            "status": row[3],
            "created_at": row[4],
            "finished_at": row[5],
        }

    def list_runs(self):

        cursor = self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        )

        rows = cursor.fetchall()

        return [
            {
                "id": r[0],
                "project_hint": r[1],
                "type": r[2],
                "status": r[3],
                "created_at": r[4],
                "finished_at": r[5],
            }
            for r in rows
        ]
