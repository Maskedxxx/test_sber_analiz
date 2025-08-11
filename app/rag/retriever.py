from __future__ import annotations
from typing import Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings

class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        self.collection = self.client.get_collection(settings.COLLECTION_NAME)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for i in range(len(res["documents"][0])):
            items.append({
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "similarity": float(1 - res["distances"][0][i]),
            })
        return {"query": query, "results": items}
