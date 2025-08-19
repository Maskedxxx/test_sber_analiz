import os
import base64
from typing import Optional


def _get_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


class Config:
    """Конфигурация приложения из переменных окружения."""

    def __init__(self):
        # Общие настройки данных/хранилища
        self.data_path: str = os.getenv("DATA_PATH", "data/mini_df.csv")
        self.chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name: str = os.getenv("COLLECTION_NAME", "financial_news")

        # Модели по умолчанию
        # Эмбеддинги через Hugging Face (лёгкая мультиязычная модель с поддержкой русского)
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.llm_model: str = os.getenv("LLM_MODEL", "GigaChat-2-Pro")

        # Провайдеры
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

        # GigaChat OAuth/Endpoints
        self.gigachat_auth_key: str = os.getenv("GIGACHAT_AUTH_KEY", "")  # base64(ClientID:ClientSecret)
        self.gigachat_client_id: str = os.getenv("GIGACHAT_CLIENT_ID", "")
        self.gigachat_client_secret: str = os.getenv("GIGACHAT_CLIENT_SECRET", "")
        self.gigachat_scope: str = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.gigachat_auth_url: str = os.getenv(
            "GIGACHAT_AUTH_URL",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        )
        self.gigachat_base_url: str = os.getenv(
            "GIGACHAT_BASE_URL",
            "https://gigachat.devices.sberbank.ru/api/v1",
        )
        # По требованию: verify SSL должно быть false по умолчанию
        self.gigachat_verify_ssl: bool = _get_bool("GIGACHAT_VERIFY_SSL", False)
        # Пересоздание векторной коллекции (при смене эмбеддингов)
        self.rebuild_collection: bool = _get_bool("REBUILD_COLLECTION", False)

        # Если ключ не задан, но заданы ClientID/Secret — формируем Base64
        if not self.gigachat_auth_key and self.gigachat_client_id and self.gigachat_client_secret:
            pair = f"{self.gigachat_client_id}:{self.gigachat_client_secret}".encode("ascii")
            self.gigachat_auth_key = base64.b64encode(pair).decode("ascii")

        # Выбираем активного провайдера
        if self.gigachat_auth_key:
            self.provider: str = "gigachat"
        elif self.openai_api_key:
            self.provider: str = "openai"
        else:
            self.provider: str = "none"

    def validate(self) -> bool:
        """Проверяет обязательные настройки и сообщает выбранного провайдера."""
        if self.provider == "gigachat":
            # Минимальный набор для GigaChat
            missing = []
            if not self.gigachat_auth_key:
                # если не смогли собрать ключ — сообщим недостающие
                if not self.gigachat_client_id:
                    missing.append("GIGACHAT_CLIENT_ID")
                if not self.gigachat_client_secret:
                    missing.append("GIGACHAT_CLIENT_SECRET")
            if missing:
                print(f"❌ Ошибка: отсутствуют переменные окружения: {', '.join(missing)}")
                return False
            print("✅ Провайдер: GigaChat (через OpenAI SDK совместимость)")
            return True
        if self.provider == "openai":
            if not self.openai_api_key:
                print("❌ Ошибка: не указан OPENAI_API_KEY в переменных окружения")
                return False
            print("✅ Провайдер: OpenAI")
            return True
        print("❌ Ошибка: не настроен ни один провайдер (GIGACHAT_AUTH_KEY или OPENAI_API_KEY)")
        return False


# Глобальный экземпляр конфигурации
config = Config()
