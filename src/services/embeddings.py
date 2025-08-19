from typing import Iterable, List

from openai import OpenAI
import httpx

from utils.config import config
from services.token_manager import GigaChatTokenManager


class GigaChatEmbeddingFunction:
    """
    Совместима с интерфейсом ChromaDB embedding_function: __call__(List[str]) -> List[List[float]]
    """

    def __init__(self):
        self._tm = GigaChatTokenManager()
        self._client: OpenAI | None = None

    def _client_ensure(self) -> OpenAI:
        token = self._tm.get()
        if self._client is None:
            http_client = httpx.Client(verify=config.gigachat_verify_ssl)
            self._client = OpenAI(api_key=token, base_url=config.gigachat_base_url, http_client=http_client)
        else:
            self._client.api_key = token
        return self._client

    def __call__(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        client = self._client_ensure()
        resp = client.embeddings.create(model=config.embedding_model, input=texts)
        return [item.embedding for item in resp.data]
