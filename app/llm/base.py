from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMClient(ABC):
    @abstractmethod
    def choose_tool(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Возвращает JSON вида {"tool": str, "args": {...}}"""

    @abstractmethod
    def generate_answer(self, messages: List[Dict[str, str]]) -> str:
        """Генерирует финальный ответ пользователю."""
