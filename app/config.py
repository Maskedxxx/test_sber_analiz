from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    OLLAMA_HOST: str = Field(default="http://localhost:11434", description="URL Ollama API")
    OLLAMA_MODEL: str = Field(default="qwen2.5:7b-instruct", description="Имя модели в Ollama")

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3", description="HF модель эмбеддингов")

    # Data/Index
    CHROMA_PATH: Path = Field(default=Path("chroma_db"))
    COLLECTION_NAME: str = Field(default="financial_news")
    CSV_PATH: Path = Field(default=Path("data/russian_fin_news/mini_df.csv"))

    # RAG
    TOP_K: int = Field(default=5)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")

settings = Settings()
