# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    CLICKHOUSE_HOST = os.getenv("CH_HOST", "localhost")
    CLICKHOUSE_PORT = int(os.getenv("CH_PORT", 8123))
    CLICKHOUSE_USER = os.getenv("CH_USER", "default")
    CLICKHOUSE_PASSWORD = os.getenv("CH_PASSWORD", "")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

settings = Settings()
