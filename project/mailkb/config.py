from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

CH_HOST = os.getenv("CH_HOST", "84.201.160.255")
CH_PORT = int(os.getenv("CH_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "peter")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "1234")

EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://localhost:8000/v1")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "Qwen/Qwen3-Embedding-0.6B")

MESSAGES_COLLECTION = os.getenv("MESSAGES_COLLECTION", "mailkb_messages")
THREADS_COLLECTION = os.getenv("THREADS_COLLECTION", "mailkb_threads")

DEFAULT_INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "1000"))
DEFAULT_AGENT_MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")

SUMMARY_DIR = Path(os.getenv("SUMMARY_DIR", "summaries"))
SUMMARY_DIR.mkdir(exist_ok=True)