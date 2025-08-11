import os
from typing import Optional


class Config:
    """Конфигурация приложения из переменных окружения."""
    
    def __init__(self):
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.data_path: str = os.getenv("DATA_PATH", "data/mini_df.csv")
        self.chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name: str = os.getenv("COLLECTION_NAME", "financial_news")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")
        
    def validate(self) -> bool:
        """Проверяет обязательные настройки."""
        if not self.openai_api_key:
            print("❌ Ошибка: не указан OPENAI_API_KEY в переменных окружения")
            return False
        return True


# Глобальный экземпляр конфигурации
config = Config()