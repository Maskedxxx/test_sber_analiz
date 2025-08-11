import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from ..models.article import FinNewsArticle
from ..utils.config import config
from ..utils.logger import logger


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
            
            # Создаем OpenAI embedding function
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.openai_api_key,
                model_name=config.embedding_model
            )

            # Создаем клиент
            client = chromadb.PersistentClient(path=config.chroma_db_path)

            try:
                # Пытаемся загрузить существующую коллекцию
                self.collection = client.get_collection(
                    name=config.collection_name,
                    embedding_function=openai_ef
                )
                logger.info(f"✅ Загружена коллекция с {self.collection.count()} документами")
                
            except Exception:
                # Создаем новую коллекцию
                logger.info("🔄 Создание новой коллекции...")
                self.collection = client.create_collection(
                    name=config.collection_name,
                    embedding_function=openai_ef
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
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    "id": results['metadatas'][0][i]['id'],
                    "distance": results['distances'][0][i],
                    "similarity": 1 - results['distances'][0][i],  # Преобразуем в схожесть
                    "answer": results['metadatas'][0][i]['answer'],
                    "source": results['metadatas'][0][i]['source'],
                    "date": results['metadatas'][0][i]['date'],
                    "sphere": results['metadatas'][0][i]['sphere'],
                    "document": results['documents'][0][i][:300] + "..."
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