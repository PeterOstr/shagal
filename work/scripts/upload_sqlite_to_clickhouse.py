# upload_sqlite_to_clickhouse.py
import os, json, sqlite3
from datetime import datetime, timezone
import clickhouse_connect

# ===== Настройки =====
SQLITE_DB = r"E:\outlook\mail_local.db"

CH_HOST = '84.201.160.255'     # прямое подключение к серверу
CH_PORT = 8123                 # HTTP-порт ClickHouse
CH_USER = 'peter'
CH_PASS = '1234'
CH_DB   = 'mailkb'

BATCH = 1000
MAX_ROWS = int(os.getenv('MAX_ROWS', '0'))  # 0 = грузить все

# ===== Утилиты нормализации =====
def to_text(x):
    """bytes/None/прочее -> str (безопасно)."""
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        for enc in ('utf-8', 'cp1251', 'latin-1'):
            try:
                return x.decode(enc, errors='replace')
            except Exception:
                pass
        return x.decode('utf-8', errors='replace')
    return str(x)

def to_str_list(v):
    """Любое значение -> список строк (для Array(String))."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [to_text(e) for e in v]
    if isinstance(v, (bytes, str)):
        return [to_text(v)]
    # JSON из SQLite будет строкой – парсим выше отдельной веткой
    try:
        return [to_text(e) for e in v]
    except Exception:
        return [to_text(v)]

def parse_dt(s: str):
    """ISO-строка -> datetime (UTC) для DateTime64(3,'UTC')."""
    try:
        dt = datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

# ===== Основной код =====
def main():
    print(f"Connecting to ClickHouse http://{CH_HOST}:{CH_PORT} as {CH_USER}, db={CH_DB}")
    client = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASS, database=CH_DB
    )
    # быстрый пинг
    ver = client.query('SELECT version()').result_rows[0][0]
    print("Server version:", ver)

    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()

    # ---- emails ----
    cur.execute("SELECT COUNT(*) FROM emails")
    total = cur.fetchone()[0]
    if MAX_ROWS and MAX_ROWS < total:
        print(f"Всего писем в SQLite: {total}. Загрузим только первые {MAX_ROWS} (MAX_ROWS).")
        total = MAX_ROWS
    else:
        print("Всего писем к загрузке:", total)

    offs = 0
    while offs < total:
        limit = min(BATCH, total - offs)
        cur.execute("""
            SELECT id, message_id, subject,
                   from_addr_json, to_addr_json, cc_addr_json, bcc_addr_json,
                   sent_at_utc, sent_at_raw, folder, body_text, body_html
            FROM emails
            ORDER BY rowid
            LIMIT ? OFFSET ?
        """, (limit, offs))
        rows = cur.fetchall()
        if not rows:
            break
        offs += len(rows)

        payload = []
        for r in rows:
            # JSON массивы адресов -> list[str]
            from_addrs = to_str_list(json.loads(r[3] or "[]"))
            to_addrs   = to_str_list(json.loads(r[4] or "[]"))
            cc_addrs   = to_str_list(json.loads(r[5] or "[]"))
            bcc_addrs  = to_str_list(json.loads(r[6] or "[]"))

            payload.append([
                to_text(r[0]),                 # id
                to_text(r[1]),                 # message_id
                to_text(r[2]),                 # subject
                from_addrs,                    # from_addr Array(String)
                to_addrs,                      # to_addr
                cc_addrs,                      # cc_addr
                bcc_addrs,                     # bcc_addr
                parse_dt(to_text(r[7])),       # sent_at_utc
                to_text(r[8]),                 # sent_at_raw
                to_text(r[9]),                 # folder
                to_text(r[10]),                # body_text  (bytes -> str)
                to_text(r[11]),                # body_html  (bytes -> str)
            ])

        client.insert(
            "mailkb.emails",
            payload,
            column_names=[
                "id","message_id","subject",
                "from_addr","to_addr","cc_addr","bcc_addr",
                "sent_at_utc","sent_at_raw","folder",
                "body_text","body_html",
            ],
        )
        print(f"Загружено писем: {offs}/{total}")

    # ---- attachments ----
    cur.execute("SELECT COUNT(*) FROM attachments")
    total_a = cur.fetchone()[0]
    if MAX_ROWS and MAX_ROWS < total_a:
        print(f"Всего вложений в SQLite: {total_a}. Загрузим только первые {MAX_ROWS} (MAX_ROWS).")
        total_a = MAX_ROWS
    else:
        print("Всего вложений к загрузке:", total_a)

    offs = 0
    while offs < total_a:
        limit = min(BATCH, total_a - offs)
        cur.execute("""
            SELECT email_id, filename, path, size_bytes
            FROM attachments
            ORDER BY rowid
            LIMIT ? OFFSET ?
        """, (limit, offs))
        rows = cur.fetchall()
        if not rows:
            break
        offs += len(rows)

        # нормализуем строковые поля вложений на всякий
        rows_norm = [[to_text(r[0]), to_text(r[1]), to_text(r[2]), int(r[3] or 0)] for r in rows]

        client.insert(
            "mailkb.attachments",
            rows_norm,
            column_names=["email_id","filename","path","size_bytes"],
        )
        print(f"Загружено вложений: {offs}/{total_a}")

    conn.close()
    print("Готово.")

if __name__ == "__main__":
    main()
