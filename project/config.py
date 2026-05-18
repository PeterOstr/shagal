import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ClickHouse
CH_HOST = os.getenv("CH_HOST", "84.201.160.255")
CH_PORT = int(os.getenv("CH_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "peter")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "1234")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "mailkb")

# Raw import
MBOX_DIR = os.getenv("MBOX_DIR", r"E:\outlook\mbox")
ATTACH_DIR = os.getenv("ATTACH_DIR", r"E:\outlook\attachments")
SAVE_ATTACHMENTS = os.getenv("SAVE_ATTACHMENTS", "true").lower() == "true"
BATCH = int(os.getenv("BATCH", "500"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "20000"))

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://localhost:8000/v1")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MESSAGES_COLLECTION = os.getenv("MESSAGES_COLLECTION", "mailkb_messages")
THREADS_COLLECTION = os.getenv("THREADS_COLLECTION", "mailkb_threads")

# Summaries
SUMMARY_DIR = Path(os.getenv("SUMMARY_DIR", "summaries"))
SUMMARY_DIR.mkdir(exist_ok=True)