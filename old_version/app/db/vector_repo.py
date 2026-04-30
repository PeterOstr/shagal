# app/db/vector_repo.py

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from app.config import settings


class VectorRepository:

    def __init__(self, collection_name="mailkb_emails", mode="create_if_not_exists"):
        """
        mode:
            - recreate
            - create_if_not_exists
            - use_existing
        """
        self.collection_name = collection_name
        self.mode = mode

        self.client = QdrantClient(url=settings.QDRANT_URL)

        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_BASE_URL,
            api_key="not-needed",
            tiktoken_enabled=False,
        )

        self.embedding_dim = self._get_embedding_dim()

        self._setup_collection()

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    # ----------------------------------------------------
    # INTERNAL
    # ----------------------------------------------------

    def _get_embedding_dim(self):
        """Определяем размерность эмбеддинга динамически."""
        vec = self.embeddings.embed_query("test")
        return len(vec)

    def _setup_collection(self):

        if self.mode == "recreate":
            self._recreate_collection()

        elif self.mode == "create_if_not_exists":
            self._create_if_not_exists()

        elif self.mode == "use_existing":
            self._use_existing()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ----------------------------------------------------
    # COLLECTION STRATEGIES
    # ----------------------------------------------------

    def _recreate_collection(self):
        print("Recreating Qdrant collection...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def _create_if_not_exists(self):
        print("Creating Qdrant collection if not exists...")

        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def _use_existing(self):
        print("Using existing Qdrant collection...")
        # Ничего не создаём.
        # Предполагаем, что размерность совпадает.

    # ----------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------

    def add_texts(self, texts, metadatas, ids):
        self.store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

    def similarity_search(self, query, k=50):
        return self.store.similarity_search(query, k=k)
