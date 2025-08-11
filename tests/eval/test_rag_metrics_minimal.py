"""
Воспроизводимый минимальный тест качества ретривера без внешних файлов:
строим микрокорпус из 4 документов, считаем Recall@k и MRR@k.
"""
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.rag.retriever import Retriever

def test_recall_mrr_minimal(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_PATH", tmp_path)
    monkeypatch.setattr(settings, "COLLECTION_NAME", "mini")

    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection("mini")

    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    docs = [
        ("1", "Сбербанк — крупнейший банк России.", {}),
        ("2", "Промсвязьбанк — российский банк.", {}),
        ("3", "ВТБ — один из ведущих банков России.", {}),
        ("4", "Газпром — энергетическая компания.", {}),
    ]
    embs = model.encode([d[1] for d in docs], normalize_embeddings=True)
    col.add(ids=[d[0] for d in docs], documents=[d[1] for d in docs], metadatas=[d[2] for d in docs], embeddings=embs)

    r = Retriever()

    qa = [
        {"q": "Информация о Сбербанк", "gold": ["Сбербанк"]},
        {"q": "Информация о Промсвязьбанк", "gold": ["Промсвязьбанк"]},
        {"q": "Информация о ВТБ", "gold": ["ВТБ"]},
    ]

    k = 3
    hits = []
    rr = []
    for item in qa:
        res = r.search(item["q"], top_k=k)["results"]
        docs_texts = [it["document"] for it in res]
        rank = next((i for i, t in enumerate(docs_texts, 1) if any(g in t for g in item["gold"])), 0)
        hits.append(1 if rank > 0 else 0)
        rr.append(0 if rank == 0 else 1 / rank)

    recall_at_k = float(np.mean(hits))
    mrr_at_k = float(np.mean(rr))

    assert recall_at_k >= 0.66
    assert mrr_at_k >= 0.5
