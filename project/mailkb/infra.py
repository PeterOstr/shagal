from functools import lru_cache

import clickhouse_connect
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import (
    CH_HOST,
    CH_PORT,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_USER,
    EMBEDDINGS_BASE_URL,
    EMBEDDINGS_MODEL,
    MESSAGES_COLLECTION,
    QDRANT_URL,
    THREADS_COLLECTION,
)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not set")
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key="not-needed",
        base_url=EMBEDDINGS_BASE_URL,
        tiktoken_enabled=False,
    )


@lru_cache(maxsize=1)
def get_embedding_dim() -> int:
    vec = get_embeddings().embed_query("test")
    return len(vec)


def ensure_collection(collection_name: str, recreate: bool = False) -> QdrantVectorStore:
    client_qdrant = get_qdrant_client()
    embeddings = get_embeddings()
    embedding_dim = get_embedding_dim()

    collections = client_qdrant.get_collections().collections
    existing_names = {c.name for c in collections}

    if recreate and collection_name in existing_names:
        client_qdrant.delete_collection(collection_name=collection_name)
        existing_names.remove(collection_name)

    if collection_name not in existing_names:
        client_qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client_qdrant,
        collection_name=collection_name,
        embedding=embeddings,
    )


def get_messages_store(recreate: bool = False) -> QdrantVectorStore:
    return ensure_collection(MESSAGES_COLLECTION, recreate=recreate)


def get_threads_store(recreate: bool = False) -> QdrantVectorStore:
    return ensure_collection(THREADS_COLLECTION, recreate=recreate)