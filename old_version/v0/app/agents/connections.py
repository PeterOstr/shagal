

from platform import python_version
import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import clickhouse_connect

# ---------------------------------------------------
# 1. Подключения
# ---------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
client_qdrant = QdrantClient(url=QDRANT_URL)

CH_HOST = os.getenv("CH_HOST", "84.201.160.255")
CH_PORT = int(os.getenv("CH_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "peter")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "1234")

client_clickhouse = clickhouse_connect.get_client(
    host=CH_HOST,
    port=CH_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD
)

BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://localhost:8000/v1")

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B",
    api_key="not-needed",
    base_url=BASE_URL,
    tiktoken_enabled=False,
)

vec = embeddings.embed_query("test")
EMBEDDING_DIM = len(vec)

print("Python:", python_version())
print("Embedding dim:", EMBEDDING_DIM)