# rag.py
from typing import List, Optional
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ новый импорт
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(folder_path: str) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents: List[Document] = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents


def split_documents(
        docs: List[Document],
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return splitter.split_documents(docs)


def _resolve_embedding_model(user_value: Optional[str]) -> str:
    """
    Если указан путь к локальной модели и он существует — используем его.
    Иначе — валидный публичный id.
    """
    if user_value and Path(user_value).exists():
        return user_value
    # ✅ правильный публичный идентификатор
    return "intfloat/multilingual-e5-large"


def build_vectorstore(
        chunks: List[Document],
        persist_directory: str = "./chroma_db",
        collection_name: str = "flashcards_collection",
        embedding_model: Optional[str] = None,  # можно передать локальный путь
):
    model_id = _resolve_embedding_model(embedding_model)

    # ✅ новый класс эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name=model_id)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vectorstore


def get_retriever(vectorstore, k: int = 6):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def chunks_to_plaintext(chunks: List[Document]) -> List[str]:
    return [d.page_content for d in chunks]
