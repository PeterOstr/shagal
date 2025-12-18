# main.py
import os
from dotenv import load_dotenv
from typing import List

from cards_schemas import Card
from export import save_jsonl, save_csv
from pipeline import make_card_chain, generate_flashcards_from_chunks
from rag import load_documents, split_documents, build_vectorstore, get_retriever, chunks_to_plaintext

load_dotenv()

FOLDER_PATH = "documents"  # <- папка с .pdf/.docx
PERSIST_DIR = "./chroma_db"
COLLECTION = "flashcards_collection"
EMBED_MODEL = "models/multilingual-e5-large-instruct"

def gen_from_full_corpus(model: str = "gpt-4o-mini") -> List[Card]:
    # 1) загрузить документы
    docs = load_documents(FOLDER_PATH)
    if not docs:
        raise RuntimeError(f"Нет документов в папке: {FOLDER_PATH}")

    # 2) порезать
    chunks_docs = split_documents(docs, chunk_size=2000, chunk_overlap=200)
    chunks_texts = chunks_to_plaintext(chunks_docs)

    # 3) модель
    chain = make_card_chain(model_name=model)

    # 4) сгенерить карточки по всем чанкам
    cards = generate_flashcards_from_chunks(chain, chunks_texts)
    return cards


def gen_by_topic(query: str, top_k: int = 8, model: str = "gpt-4o-mini") -> List[Card]:
    # 1) загрузить документы
    docs = load_documents(FOLDER_PATH)
    if not docs:
        raise RuntimeError(f"Нет документов в папке: {FOLDER_PATH}")

    # 2) порезать
    chunks_docs = split_documents(docs, chunk_size=2000, chunk_overlap=200)

    # 3) построить/обновить векторку
    vs = build_vectorstore(
        chunks=chunks_docs,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_model=EMBED_MODEL,
    )
    retriever = get_retriever(vs, k=top_k)

    # 4) извлечь по теме
    retrieved = retriever.get_relevant_documents(query)
    if not retrieved:
        print("По запросу ничего не нашлось, генерю по всему корпусу.")
        chunks_texts = [d.page_content for d in chunks_docs]
    else:
        chunks_texts = [d.page_content for d in retrieved]

    # 5) модель
    chain = make_card_chain(model_name=model)

    # 6) генерим карточки по выбранным чанкам
    cards = generate_flashcards_from_chunks(chain, chunks_texts)
    return cards


if __name__ == "__main__":
    # ==== ВАРИАНТ A: по всему корпусу ====
    cards_all = gen_from_full_corpus(model="gpt-4o-mini")
    print(f"[FULL] Сгенерировано карточек: {len(cards_all)}")
    save_jsonl(cards_all, "out/cards_full.jsonl")
    save_csv(cards_all, "out/cards_full.csv")

    # ==== ВАРИАНТ B: по теме/запросу ====
    topic = "митохондрии и функции в клетке"  # пример
    cards_topic = gen_by_topic(topic, top_k=8, model="gpt-4o-mini")
    print(f"[TOPIC] «{topic}»: карточек {len(cards_topic)}")
    save_jsonl(cards_topic, "out/cards_topic.jsonl")
    save_csv(cards_topic, "out/cards_topic.csv")

    print("Файлы сохранены в ./out/")
