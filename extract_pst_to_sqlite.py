# extract_pst_to_sqlite.py
import json, sqlite3, os
from pathlib import Path
from email.parser import HeaderParser
from email import policy
from email.utils import getaddresses, parsedate_to_datetime
from datetime import timezone, datetime
from tqdm import tqdm

PST_DIR = r"E:\outlook"
ATTACH_DIR = r"E:\outlook\attachments"
DB_PATH = r"E:\outlook\mail_local.db"
SAVE_ATTACHMENTS = True
BATCH = 500

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def parse_addrs(v):
    if not v: return []
    return [addr for _, addr in getaddresses([v]) if addr]

def parse_date_utc(v):
    if not v: return datetime(1970,1,1,tzinfo=timezone.utc)
    try:
        dt = parsedate_to_datetime(v)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime(1970,1,1,tzinfo=timezone.utc)

def main():
    from libratom.lib.core import open_mail_archive

    ensure_dir(Path(ATTACH_DIR))
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS emails(
      id TEXT PRIMARY KEY,
      message_id TEXT,
      subject TEXT,
      from_addr_json TEXT,
      to_addr_json TEXT,
      cc_addr_json TEXT,
      bcc_addr_json TEXT,
      sent_at_utc TEXT,
      sent_at_raw TEXT,
      folder TEXT,
      body_text TEXT,
      body_html TEXT
    );
    CREATE TABLE IF NOT EXISTS attachments(
      email_id TEXT,
      filename TEXT,
      path TEXT,
      size_bytes INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_emails_sent ON emails(sent_at_utc);
    """)
    header_parser = HeaderParser(policy=policy.default)

    emails_buf, atts_buf = [], []
    def flush():
        nonlocal emails_buf, atts_buf
        if emails_buf:
            conn.executemany("""INSERT OR IGNORE INTO emails
            (id,message_id,subject,from_addr_json,to_addr_json,cc_addr_json,bcc_addr_json,
             sent_at_utc,sent_at_raw,folder,body_text,body_html)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", emails_buf)
            emails_buf = []
        if atts_buf:
            conn.executemany("""INSERT INTO attachments(email_id,filename,path,size_bytes)
                                VALUES (?,?,?,?)""", atts_buf)
            atts_buf = []
        conn.commit()

    pst_files = list(Path(PST_DIR).rglob("*.pst"))
    for pst in pst_files:
        print(f"\nPST: {pst}")
        with open_mail_archive(pst) as arc:
            for msg in tqdm(arc.messages(), desc=pst.name):
                th = msg.transport_headers or ""
                h = header_parser.parsestr(th)

                mid   = (h.get("Message-ID") or h.get("Message-Id") or "").strip()
                subj  = (h.get("Subject") or "").strip()
                from_ = parse_addrs(h.get("From"))
                to_   = parse_addrs(h.get("To"))
                cc_   = parse_addrs(h.get("Cc"))
                bcc_  = parse_addrs(h.get("Bcc"))
                rawdt = h.get("Date") or ""
                dt_utc = parse_date_utc(rawdt)

                body_text = getattr(msg, "plain_text_body", None) or ""
                body_html = getattr(msg, "html_body", None) or ""
                folder = getattr(msg, "folder_name", None) or "unknown"

                stable_id = mid or f"{pst.name}::{msg.identifier}"

                emails_buf.append((
                    stable_id, mid, subj,
                    json.dumps(from_, ensure_ascii=False),
                    json.dumps(to_, ensure_ascii=False),
                    json.dumps(cc_, ensure_ascii=False),
                    json.dumps(bcc_, ensure_ascii=False),
                    dt_utc.isoformat(), rawdt, folder, body_text, body_html
                ))

                # вложения
                for att in getattr(msg, "attachments", []):
                    fname = (att.name or f"att_{att.identifier}").replace("\\","_").replace("/","_")
                    size = int(getattr(att, "size", 0) or 0)
                    fpath = ""
                    if SAVE_ATTACHMENTS:
                        try:
                            data = att.read_buffer(size) if size else att.read()
                            dest = Path(ATTACH_DIR)/pst.stem/str(msg.identifier)
                            ensure_dir(dest)
                            fp = dest/fname
                            fp.write_bytes(data or b"")
                            fpath = str(fp)
                        except Exception:
                            pass
                    atts_buf.append((stable_id, fname, fpath, size))

                if len(emails_buf) >= BATCH or len(atts_buf) >= BATCH:
                    flush()

    flush()
    conn.close()
    print(f"Готово. SQLite: {DB_PATH}")

if __name__ == "__main__":
    main()
