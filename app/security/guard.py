from typing import Tuple

_FORBIDDEN_MARKERS = (
    "system prompt",
    "системный промпт",
    "покажи свой промпт",
    "какие функции доступны",
    "раскрой внутреннее устройство",
    "tool calls",
    "инструменты ллм",
    "source code",
    "исходный код",
    "переменные окружения",
)

INJECTION_PATTERNS = (
    "ignore previous",
    "disregard previous",
    "override instructions",
    "пропусти предыдущие",
    "игнорируй предыдущие",
)

def check_request_forbidden(text: str) -> Tuple[bool, str]:
    low = (text or "").lower()
    for m in _FORBIDDEN_MARKERS:
        if m in low:
            return True, "Запрос нарушает политику безопасности (уточнение внутренних деталей запрещено)."
    for m in INJECTION_PATTERNS:
        if m in low:
            return True, "Обнаружена попытка prompt-injection."
    return False, ""
