import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.rag.retriever import Retriever

def test_retriever_minimal(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_PATH", tmp_path)
    monkeypatch.setattr(settings, "COLLECTION_NAME", "tmp")

    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection("tmp")

    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    docs = [
        ("1", "Промсвязьбанк — крупнейший банк РФ.", {"source": "synthetic", "date": "2024-01-01"}),
        ("2", "Сбербанк — системно значимый банк в России.", {"source": "synthetic", "date": "2024-01-02"}),
    ]
    embs = model.encode([d[1] for d in docs], normalize_embeddings=True)
    col.add(ids=[d[0] for d in docs], documents=[d[1] for d in docs], metadatas=[d[2] for d in docs], embeddings=embs)

    r = Retriever()
    res = r.search("Информация о Сбербанк", top_k=1)["results"]
    assert res, "нет результатов"
    assert "Сбербанк" in res[0]["document"]
