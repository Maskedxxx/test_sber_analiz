import json
from typing import Dict, Any, List

from openai import OpenAI
import httpx

from services.data_service import DataService
from services.system_service import SystemService
from services.token_manager import GigaChatTokenManager
from utils.config import config
from utils.logger import logger


class LLMService:
    """Сервис для работы с LLM и Function Calling (с поддержкой GigaChat)."""

    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.system_service = SystemService()

        self._tm: GigaChatTokenManager | None = None
        if config.provider == "gigachat":
            self._tm = GigaChatTokenManager()
            token = self._tm.get()
            http_client = httpx.Client(verify=config.gigachat_verify_ssl)
            self.client = OpenAI(api_key=token, base_url=config.gigachat_base_url, http_client=http_client)
        else:
            # OpenAI fallback (для обратной совместимости)
            self.client = OpenAI(api_key=config.openai_api_key)

        # Описание пользовательских функций в формате GigaChat
        self.functions: List[Dict[str, Any]] = [
            {
                "name": "search_financial_news",
                "description": "Поиск в базе российских финансовых новостей и аналитики. Передавайте ВЕСЬ запрос пользователя в query для семантического поиска.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Полный текст запроса пользователя для семантического поиска"},
                        "top_k": {"type": "integer", "description": "Количество результатов (по умолчанию 5)"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_system_stats",
                "description": "Получить статистику загрузки системы: CPU и память.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "get_moscow_time",
                "description": "Получить текущее время в Москве.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ]

        # Для OpenAI fallback — инструменты в стиле tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_financial_news",
                    "description": "Поиск по базе российских финансовых новостей.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_system_stats",
                    "description": "Получить статистику загрузки системы: CPU и память.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_moscow_time",
                    "description": "Получить текущее время в Москве.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    "strict": True,
                },
            },
        ]

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        """Рекурсивно заменяет невалидные для JSON значения (NaN/Inf) на None и очищает строки."""
        import math
        import re
        
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            # Удаляем проблемные управляющие символы, которые могут ломать JSON
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', value)
            return cleaned
        if isinstance(value, dict):
            return {k: LLMService._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [LLMService._sanitize_for_json(v) for v in value]
        return value

    def _gigachat_recursive_call(self, messages: List[Dict[str, Any]]) -> str:
        """Рекурсивный вызов GigaChat с возможностью повторного вызова функций."""
        self._ensure_token()
        
        response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            functions=self.functions,
        )
        
        msg = response.choices[0].message
        fn_call = getattr(msg, "function_call", None)
        
        if fn_call:
            # Еще один вызов функции
            name = getattr(fn_call, "name", None)
            args = getattr(fn_call, "arguments", None)
            
            if isinstance(args, dict):
                args_parsed = args
            else:
                try:
                    args_parsed = json.loads(args) if args else {}
                except Exception:
                    args_parsed = {}

            logger.llm_function_call(name, args_parsed)
            result = self._dispatch(name, args_parsed)
            result_safe = self._sanitize_for_json(result)
            
            # Фильтруем результат
            if name == "search_financial_news" and "results" in result_safe:
                filtered_results = []
                for item in result_safe["results"]:
                    filtered_results.append({
                        "source": item.get("source", ""),
                        "date": item.get("date", ""), 
                        "sphere": item.get("sphere", ""),
                        "document": item.get("document", "")
                    })
                function_content = json.dumps({
                    "query": result_safe.get("query", ""),
                    "total_results": result_safe.get("total_results", 0),
                    "results": filtered_results
                }, ensure_ascii=False)
            else:
                function_content = json.dumps(result_safe, ensure_ascii=False)
            
            # Добавляем в историю и продолжаем рекурсию
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": name,
                    "arguments": args_parsed,
                },
            })
            messages.append({
                "role": "function", 
                "name": name,
                "content": function_content,
            })
            
            return self._gigachat_recursive_call(messages)
        else:
            # Финальный ответ без вызова функций
            result_text = msg.content or ""
            logger.system_response(result_text)
            return result_text

    def _ensure_token(self):
        if config.provider == "gigachat" and self._tm is not None:
            token = self._tm.get()
            self.client.api_key = token

    def _dispatch(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "search_financial_news":
            query = args.get("query", "")
            top_k = int(args.get("top_k", 5))
            return self.data_service.search_articles(query, top_k)
        if name == "get_system_stats":
            return self.system_service.get_system_stats()
        if name == "get_moscow_time":
            return {"time": self.system_service.get_moscow_time()}
        return {"error": f"Unknown function: {name}"}

    def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя с Function Calling."""
        try:
            logger.user_query(query)
            self._ensure_token()

            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "Вы - ИИ-помощник Сбербанка для работы с финансовой аналитикой.\n\n"
                        "ДОСТУПНЫЕ ФУНКЦИИ:\n"
                        "1. search_financial_news - поиск в базе российских финансовых новостей и аналитики. "
                        "Передавайте ВЕСЬ запрос пользователя в query для максимальной релевантности.\n"
                        "2. get_system_stats - получение статистики системы (CPU, память).\n"
                        "3. get_moscow_time - текущее время в Москве.\n\n"
                        "ВАЖНО для поиска новостей:\n"
                        "- Используйте в query ПОЛНЫЙ текст запроса пользователя\n"
                        "- Для 'найди новости о Сбербанке' используйте query='найди новости о Сбербанке'\n"
                        "- Для 'что происходит с рублем' используйте query='что происходит с рублем'\n\n"
                        "Отвечайте профессионально и по существу на русском языке."
                    ),
                },
                {"role": "user", "content": query},
            ]

            if config.provider == "gigachat":
                # GigaChat использует functions формат
                logger.debug(f"GigaChat request: model={config.llm_model}, functions={len(self.functions)}, messages={len(messages)}")
                
                response = self.client.chat.completions.create(
                    model=config.llm_model,
                    messages=messages,
                    functions=self.functions,
                )
                
                logger.debug(f"GigaChat Response: {response.choices[0].message}")
                logger.debug(f"Function call: {response.choices[0].message.function_call}")
                
                msg = response.choices[0].message
                fn_call = getattr(msg, "function_call", None)
                if fn_call:
                    name = getattr(fn_call, "name", None)
                    args = getattr(fn_call, "arguments", None)
                    
                    # GigaChat возвращает arguments как dict, а не строку
                    if isinstance(args, dict):
                        args_parsed = args
                    else:
                        try:
                            args_parsed = json.loads(args) if args else {}
                        except Exception:
                            args_parsed = {}

                    logger.llm_function_call(name, args_parsed)
                    result = self._dispatch(name, args_parsed)
                    result_safe = self._sanitize_for_json(result)
                    
                    # Фильтруем результат - оставляем только текстовые поля для LLM
                    if name == "search_financial_news" and "results" in result_safe:
                        filtered_results = []
                        for item in result_safe["results"]:
                            filtered_results.append({
                                "source": item.get("source", ""),
                                "date": item.get("date", ""), 
                                "sphere": item.get("sphere", ""),
                                "document": item.get("document", "")
                            })
                        function_content = json.dumps({
                            "query": result_safe.get("query", ""),
                            "total_results": result_safe.get("total_results", 0),
                            "results": filtered_results
                        }, ensure_ascii=False)
                    else:
                        function_content = json.dumps(result_safe, ensure_ascii=False)
                    
                    # Добавляем результат функции в сообщения
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": name,
                            "arguments": args_parsed,
                        },
                    })
                    messages.append({
                        "role": "function", 
                        "name": name,
                        "content": function_content,
                    })
                    
                    # Рекурсивный вызов с functions - модель может снова вызвать функцию или ответить
                    return self._gigachat_recursive_call(messages)

                # Без вызова функций — просто ответ
                result_text = msg.content or ""
                logger.system_response(result_text)
                return result_text
            else:
                # OpenAI tools/tool_calls формат
                response = self.client.chat.completions.create(
                    model=config.llm_model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                )

                if response.choices[0].message.tool_calls:
                    return self._handle_function_calls_openai(response, messages)
                else:
                    result = response.choices[0].message.content
                    logger.system_response(result)
                    return result

        except Exception as e:
            logger.error_occurred(e, "обработка запроса LLM")
            return "Извините, произошла ошибка при обработке вашего запроса."

    def _handle_function_calls_openai(self, response, messages: List[Dict]) -> str:
        """Обработка Function Calling в формате OpenAI (tools/tool_calls)."""
        messages.append(response.choices[0].message)

        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.llm_function_call(function_name, function_args)
            function_result = self._dispatch(function_name, function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_result, ensure_ascii=False),
            })

        final_response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            tools=self.tools,
        )

        result = final_response.choices[0].message.content
        logger.system_response(result)
        return result
