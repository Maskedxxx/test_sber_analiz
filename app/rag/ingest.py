from __future__ import annotations
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings

@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, str]

def load_docs_from_csv(csv_path: str) -> List[Doc]:
    df = pd.read_csv(csv_path)
    docs: List[Doc] = []
    for i, row in df.iterrows():
        parts = [
            str(row.get("reasoning", "")),
            str(row.get("article_text", "")),
            str(row.get("sphere", "")),
        ]
        text = "\n\n".join([p for p in parts if p and p != "nan"]) or ""
        meta = {
            "source": str(row.get("source", "")),
            "date": str(row.get("date", "")),
            "answer": str(row.get("answer", "")),
        }
        docs.append(Doc(id=str(row.get("id", i)), text=text, meta=meta))
    return docs

def embed_and_persist(docs: List[Doc]) -> None:
    client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
    try:
        client.delete_collection(settings.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=settings.COLLECTION_NAME)

    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    texts = [d.text for d in docs]
    ids = [d.id for d in docs]
    metadatas = [d.meta for d in docs]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

def main() -> None:
    docs = load_docs_from_csv(str(settings.CSV_PATH))
    if len(docs) < 100:
        print(f"[WARN] Документов всего {len(docs)} (<100). ТЗ требует ≥100. Продолжу, но проверьте CSV.")
    embed_and_persist(docs)
    print(f"[OK] Индекс записан в {settings.CHROMA_PATH} (коллекция: {settings.COLLECTION_NAME})")

if __name__ == "__main__":
    main()
