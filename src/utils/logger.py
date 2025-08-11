import logging
import sys
from datetime import datetime
from typing import Any


class ChatBotLogger:
    """Простой логгер для чат-бота."""
    
    def __init__(self, name: str = "chatbot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Создаем форматтер
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Добавляем handler для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Избегаем дублирования handlers
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Логирует информационное сообщение."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Логирует ошибку."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.error(message)
    
    def user_query(self, query: str):
        """Логирует запрос пользователя."""
        self.info(f"👤 Пользователь: {query}")
    
    def llm_function_call(self, function_name: str, args: dict):
        """Логирует вызов функции LLM."""
        self.info(f"🔧 Вызов функции: {function_name}", args=args)
    
    def system_response(self, response: str):
        """Логирует ответ системы."""
        self.info(f"🤖 Ответ системы: {response[:100]}...")
    
    def error_occurred(self, error: Exception, context: str = ""):
        """Логирует ошибку с контекстом."""
        self.error(f"❌ Ошибка {context}: {str(error)}")


# Глобальный экземпляр логгера
logger = ChatBotLogger()