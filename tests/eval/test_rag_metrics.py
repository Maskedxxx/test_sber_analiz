"""
Демонстрационный тест метрик на готовом индексе (оставляем skip, чтобы не зависеть от наличия корпуса).
"""
import numpy as np
from app.rag.retriever import Retriever
import pytest

@pytest.mark.skip(reason="Требует готового индекса и QA")
def test_recall_mrr_demo():
    r = Retriever()
    qa = [
        {"q": "Информация о Сбербанк", "gold": ["Сбербанк"]},
        {"q": "Данные о Промсвязьбанк", "gold": ["Промсвязьбанк"]},
    ]
    k = 5
    hits = []
    rr = []
    for item in qa:
        res = r.search(item["q"], top_k=k)["results"]
        docs = [it["document"] for it in res]
        idx = next((i for i, d in enumerate(docs, 1) if any(g in d for g in item["gold"])), 0)
        hits.append(1 if idx > 0 else 0)
        rr.append(0 if idx == 0 else 1/idx)
    recall_at_k = float(np.mean(hits))
    mrr_at_k = float(np.mean(rr))
    assert recall_at_k >= 0.5
    assert mrr_at_k >= 0.3
