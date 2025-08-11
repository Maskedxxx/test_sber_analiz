#!/usr/bin/env python3
"""
Консольный чат-бот для работы с финансовой информацией.

Возможности:
- Поиск в базе российских финансовых новостей (RAG)
- Получение статистики системы (CPU/Memory)
- Текущее время в Москве

Использование:
    python chatbot.py
"""

import sys
import os
from typing import Optional

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.data_service import DataService
from services.llm_service import LLMService
from utils.config import config
from utils.logger import logger


class ChatBot:
    """Главный класс консольного чат-бота."""
    
    def __init__(self):
        self.data_service: Optional[DataService] = None
        self.llm_service: Optional[LLMService] = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Инициализирует все сервисы бота.
        
        Returns:
            True если инициализация прошла успешно
        """
        try:
            logger.info("Запуск чат-бота...")
            
            # Проверяем конфигурацию
            if not config.validate():
                return False
            
            # Инициализируем сервисы
            logger.info("Инициализация сервисов...")
            
            self.data_service = DataService()
            self.data_service.setup_vector_store()
            
            self.llm_service = LLMService(self.data_service)
            
            self.initialized = True
            logger.info("✅ Чат-бот готов к работе!")
            return True
            
        except Exception as e:
            logger.error_occurred(e, "инициализация бота")
            return False
    
    def run(self):
        """Запускает основной цикл работы чат-бота."""
        
        if not self.initialized:
            if not self.initialize():
                print("❌ Ошибка инициализации. Проверьте настройки и попробуйте снова.")
                return
        
        self._print_welcome()
        
        try:
            while True:
                try:
                    # Получаем ввод пользователя
                    user_input = input("\n👤 Ваш вопрос: ").strip()
                    
                    # Проверяем команды выхода
                    if user_input.lower() in ['exit', 'quit', 'выход', 'стоп']:
                        self._print_goodbye()
                        break
                    
                    # Проверяем пустой ввод
                    if not user_input:
                        continue
                    
                    # Обрабатываем запрос
                    print("\nОбрабатываю ваш запрос...")
                    response = self.llm_service.process_query(user_input)
                    
                    # Выводим ответ
                    print(f"\nОтвет:\n{response}")
                    
                except KeyboardInterrupt:
                    self._print_goodbye()
                    break
                except Exception as e:
                    logger.error_occurred(e, "обработка запроса")
                    print(f"\n❌ Произошла ошибка: {str(e)}")
                    print("Попробуйте еще раз или напишите 'exit' для выхода.")
                    
        except Exception as e:
            logger.error_occurred(e, "главный цикл")
            print(f"\n❌ Критическая ошибка: {str(e)}")
    
    def _print_welcome(self):
        """Выводит приветственное сообщение."""
        print("\n" + "="*60)
        print("Добро пожаловать в финансового чат-бота!")
        print("="*60)
        print("\nЯ могу помочь вам с:")
        print("Поиском информации в базе финансовых новостей")
        print("Получением статистики системы (CPU/Memory)")
        print("🕐 Текущим временем в Москве")
        print("\nПримеры вопросов:")
        print("• 'Найди новости о Сбербанке'")
        print("• 'Какая загрузка процессора?'")
        print("• 'Сколько сейчас времени в Москве?'")
        print("\nДля выхода напишите: exit, quit, выход или стоп")
        print("-"*60)
    
    def _print_goodbye(self):
        """Выводит прощальное сообщение."""
        print("\nДо свидания! Спасибо за использование чат-бота.")
        logger.info("🛑 Чат-бот завершил работу")


def main():
    """Главная функция запуска приложения."""
    bot = ChatBot()
    bot.run()


if __name__ == "__main__":
    main()