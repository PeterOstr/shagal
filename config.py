import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    QDRANT_URL = os.getenv("QDRANT_URL")
    CLICKHOUSE_HOST = os.getenv("CH_HOST")
    CLICKHOUSE_PORT = int(os.getenv("CH_PORT", 8123))
    CLICKHOUSE_USER = os.getenv("CH_USER")
    CLICKHOUSE_PASSWORD = os.getenv("CH_PASSWORD")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

settings = Settings()
