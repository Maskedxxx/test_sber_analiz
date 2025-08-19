import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from models.article import FinNewsArticle
from utils.config import config
from utils.logger import logger
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class DataService:
    """Сервис для работы с данными и векторной базой знаний."""
    
    def __init__(self):
        self.collection = None
        self.articles = []
    
    def load_articles(self) -> List[FinNewsArticle]:
        """
        Загружает финансовые новости из CSV файла.
        
        Returns:
            Список объектов FinNewsArticle
        """
        try:
            logger.info(f"🔄 Загрузка данных из {config.data_path}")
            df = pd.read_csv(config.data_path)
            
            # Преобразуем в объекты Pydantic
            self.articles = [FinNewsArticle(**row) for row in df.to_dict('records')]
            
            logger.info(f"✅ Загружено {len(self.articles)} статей")
            return self.articles
            
        except FileNotFoundError:
            logger.error_occurred(
                Exception(f"Файл {config.data_path} не найден"), 
                "загрузка данных"
            )
            raise
        except Exception as e:
            logger.error_occurred(e, "загрузка данных")
            raise
    
    def setup_vector_store(self) -> chromadb.Collection:
        """
        Создает или загружает коллекцию ChromaDB с эмбедингами.

        Returns:
            ChromaDB коллекция
        """
        try:
            logger.info("🔄 Настройка векторной базы данных...")

            # Используем лёгкую модель из Hugging Face (русский поддержан)
            # По умолчанию: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
            logger.info(
                "⬇️ Загрузка модели эмбеддингов из Hugging Face — это может занять 1–5 минут при первом запуске",
                model=config.embedding_model,
            )
            ef = SentenceTransformerEmbeddingFunction(model_name=config.embedding_model)
            logger.info("✅ Модель эмбеддингов загружена")

            # Создаем клиент
            client = chromadb.PersistentClient(path=config.chroma_db_path)

            if config.rebuild_collection:
                logger.info("♻️ Включен REBUILD_COLLECTION=true — пересоздаём коллекцию")
                # Удаляем коллекцию, если существует
                try:
                    client.delete_collection(name=config.collection_name)
                    logger.info("🗑️ Старая коллекция удалена")
                except Exception:
                    logger.info("ℹ️ Старой коллекции не было или уже удалена")

                # Создаём новую коллекцию
                self.collection = client.create_collection(
                    name=config.collection_name,
                    embedding_function=ef,
                )

                if not self.articles:
                    self.load_articles()
                self._add_articles_to_collection()
                logger.info("✅ Коллекция переиндексирована")

            else:
                try:
                    # Пытаемся загрузить существующую коллекцию
                    self.collection = client.get_collection(
                        name=config.collection_name,
                        embedding_function=ef,
                    )
                    logger.info(f"✅ Загружена коллекция с {self.collection.count()} документами")

                except Exception:
                    # Создаем новую коллекцию
                    logger.info("🔄 Создание новой коллекции...")
                    self.collection = client.create_collection(
                        name=config.collection_name,
                        embedding_function=ef,
                    )

                    # Загружаем статьи если они не загружены
                    if not self.articles:
                        self.load_articles()

                    # Добавляем документы в коллекцию
                    self._add_articles_to_collection()
                
            return self.collection
            
        except Exception as e:
            logger.error_occurred(e, "настройка векторной БД")
            raise
    
    def _add_articles_to_collection(self):
        """Добавляет статьи в векторную коллекцию."""
        documents = []
        metadatas = []
        ids = []

        for article in self.articles:
            # Объединяем текст для эмбединга
            document = f"{article.reasoning} {article.article_text} {article.sphere}"
            documents.append(document)

            # Метаданные
            metadata = {
                "id": article.id,
                "answer": article.answer,
                "source": article.source,
                "date": article.date.isoformat(),
                "sphere": article.sphere
            }
            metadatas.append(metadata)
            ids.append(str(article.id))

        # Добавляем в коллекцию
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✅ Добавлено {len(documents)} документов в коллекцию")
    
    def search_articles(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Выполняет векторный поиск по статьям.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Словарь с результатами поиска
        """
        try:
            logger.info(f"🔍 Поиск по запросу: '{query}' (top_k={top_k})")
            
            if not self.collection:
                raise Exception("Коллекция не инициализирована")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Форматируем результаты
            import math

            formatted_results = []
            for i in range(len(results['documents'][0])):
                # Безопасно приводим значения и избегаем NaN/Inf в JSON
                raw_distance = results['distances'][0][i]
                distance = float(raw_distance) if raw_distance is not None else None
                if isinstance(distance, float) and not math.isfinite(distance):
                    distance = None

                similarity = None
                if isinstance(distance, float) and math.isfinite(distance):
                    similarity = 1.0 - distance
                    if not math.isfinite(similarity):
                        similarity = None

                doc_text = str(results['documents'][0][i])
                snippet = doc_text[:300] + ("..." if len(doc_text) > 300 else "")

                result = {
                    "id": str(results['metadatas'][0][i].get('id')),
                    "distance": distance,
                    "similarity": similarity,
                    "answer": results['metadatas'][0][i].get('answer'),
                    "source": results['metadatas'][0][i].get('source'),
                    "date": results['metadatas'][0][i].get('date'),
                    "sphere": results['metadatas'][0][i].get('sphere'),
                    "document": snippet,
                }
                formatted_results.append(result)

            search_result = {
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
            logger.info(f"✅ Найдено {len(formatted_results)} результатов")
            return search_result
            
        except Exception as e:
            logger.error_occurred(e, "поиск статей")
            raise
