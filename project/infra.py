from functools import lru_cache

import clickhouse_connect
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import (
    CH_HOST,
    CH_PORT,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_USER,
    DEEPSEEK_API_KEY,
    EMBEDDINGS_BASE_URL,
    EMBEDDINGS_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY,
    QDRANT_URL,
)


@lru_cache(maxsize=1)
def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
    )


@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key="not-needed",
        base_url=EMBEDDINGS_BASE_URL,
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


def _get_llm():
    if "deepseek" in LLM_MODEL.lower():
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )


def build_structured_agent(response_format):
    return _get_llm().with_structured_output(response_format)