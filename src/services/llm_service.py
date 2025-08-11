import json
from openai import OpenAI
from typing import Dict, Any, List
from services.data_service import DataService
from services.system_service import SystemService
from utils.config import config
from utils.logger import logger


class LLMService:
    """Сервис для работы с LLM и function calling."""
    
    def __init__(self, data_service: DataService):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.data_service = data_service
        self.system_service = SystemService()
        
        # Определяем доступные инструменты
        self.tools = [
            {
            "type": "function",
            "function": {
                "name": "search_financial_news",
                "description": "Поиск по базе российских финансовых новостей. Используйте для вопросов о финансах, экономике, компаниях, рынках.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос для поиска релевантных финансовых новостей"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Количество результатов (по умолчанию 5)"
                        }
                    },
                    "required": ["query", "top_k"],  # <-- ДОБАВИТЬ top_k
                    "additionalProperties": False
                },
                "strict": True
            }
        },
            {
                "type": "function",
                "function": {
                    "name": "get_system_stats",
                    "description": "Получить статистику загрузки системы: CPU и память. Используйте для вопросов о производительности системы.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_moscow_time",
                    "description": "Получить текущее время в Москве. Используйте для вопросов о времени в Москве.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]
    
    def process_query(self, query: str) -> str:
        """
        Обрабатывает запрос пользователя с возможностью вызова функций.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Ответ LLM с результатами выполнения функций
        """
        try:
            logger.user_query(query)
            
            # Подготавливаем сообщения
            messages = [
                {
                    "role": "system",
                    "content": """Вы - полезный ассистент для работы с финансовой информацией. 
                    
                    Вы можете:
                    - Искать информацию в базе российских финансовых новостей
                    - Показывать статистику загрузки системы  
                    - Сообщать текущее время в Москве

                    Выберите подходящую функцию на основе запроса пользователя. 
                    Отвечайте кратко и по существу на русском языке.
                    Не раскрывайте технические детали работы системы."""
                },
                {
                    "role": "user", 
                    "content": query
                }
            ]

            # Первый вызов LLM
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            # Проверяем, были ли вызваны функции
            if response.choices[0].message.tool_calls:
                return self._handle_function_calls(response, messages)
            else:
                # Если функции не вызывались, возвращаем обычный ответ
                result = response.choices[0].message.content
                logger.system_response(result)
                return result
                
        except Exception as e:
            logger.error_occurred(e, "обработка запроса LLM")
            return "Извините, произошла ошибка при обработке вашего запроса."
    
    def _handle_function_calls(self, response, messages: List[Dict]) -> str:
        """Обрабатывает вызовы функций от LLM."""
        
        # Добавляем сообщение модели в историю
        messages.append(response.choices[0].message)

        # Выполняем вызовы функций
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.llm_function_call(function_name, function_args)

            # Вызываем соответствующую функцию
            function_result = self._execute_function(function_name, function_args)

            # Добавляем результат функции в сообщения
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_result
            })

        # Второй вызов LLM для генерации финального ответа
        final_response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            tools=self.tools
        )

        result = final_response.choices[0].message.content
        logger.system_response(result)
        return result
    
    def _execute_function(self, function_name: str, function_args: Dict[str, Any]) -> str:
        """Выполняет указанную функцию с аргументами."""
        
        try:
            if function_name == "search_financial_news":
                query = function_args["query"]
                top_k = function_args.get("top_k", 5)
                result = self.data_service.search_articles(query, top_k)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif function_name == "get_system_stats":
                result = self.system_service.get_system_stats()
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif function_name == "get_moscow_time":
                result = self.system_service.get_moscow_time()
                return result

            else:
                return "Неизвестная функция"
                
        except Exception as e:
            logger.error_occurred(e, f"выполнение функции {function_name}")
            return f"Ошибка при выполнении функции {function_name}"