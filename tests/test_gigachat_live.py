#!/usr/bin/env python3
"""
E2E-тесты, подтверждающие, что приложение:
- Запускается внутри Docker-контейнера
- Может обращаться к API GigaChat (через совместимость OpenAI SDK)
- Умеет вызывать пользовательские функции (Function Calling)

Запуск (из корня репозитория):
  docker compose run --rm -e REBUILD_COLLECTION=true chatbot \
    python tests/test_gigachat_live.py

Примечания:
- Тесты пропускаются, если не настроен GigaChat-провайдер
  (нужны переменные окружения: как минимум GIGACHAT_CLIENT_ID/SECRET или GIGACHAT_AUTH_KEY)
- Первый запуск может занять время (скачивание модели эмбеддингов и построение индекса)
"""

import os
import re
import sys
import json
import time
from typing import Any, Dict, List, Tuple

# Обеспечиваем импорт модулей из src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.data_service import DataService
from services.llm_service import LLMService
from utils.config import config
from utils.logger import logger


def _print(msg: str):
    print(msg, flush=True)


def ensure_dirs():
    os.makedirs('test_results', exist_ok=True)


def main():
    ensure_dirs()

    # 1) Проверяем, что настроен провайдер GigaChat
    if not config.validate():
        _print("❌ Конфигурация невалидна. Провайдер не настроен.")
        sys.exit(1)
    if config.provider != "gigachat":
        _print("⚠️ Пропуск E2E: активен не GigaChat-провайдер."
               " Установите переменные GIGACHAT_* и запустите снова.")
        # Пропускаем без ошибки, чтобы пайплайн не падал в окружениях без ключей
        sys.exit(0)

    # 2) Инициализируем сервисы (может занять время при первом запуске)
    start_init = time.time()
    data = DataService()
    data.setup_vector_store()
    llm = LLMService(data)
    init_time = time.time() - start_init

    results: List[Dict[str, Any]] = []
    captured_calls: List[Tuple[str, Dict[str, Any]]] = []

    # Патчим логгер для фиксации вызовов функций LLM
    original_llm_fc = logger.llm_function_call

    def capture_llm_function_call(name: str, args: dict):
        captured_calls.append((name, args))
        return original_llm_fc(name, args)

    logger.llm_function_call = capture_llm_function_call  # type: ignore

    # Набор запросов, которые должны триггерить функции
    test_prompts = [
        {
            "name": "moscow_time",
            "prompt": "Сколько сейчас времени в Москве?",
            "expect_func": "get_moscow_time",
            "validate": lambda text: bool(re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)) or ("MSK" in text or "Моск" in text),
        },
        {
            "name": "system_stats",
            "prompt": "Покажи статистику системы",
            "expect_func": "get_system_stats",
            "validate": lambda text: ("CPU" in text or "Память" in text or re.search(r"\d+\.?\d*%", text)),
        },
    ]

    # 3) Запускаем запросы к LLM (через GigaChat) и валидируем ответы
    all_ok = True
    for item in test_prompts:
        _print(f"\n🧪 Тест: {item['name']}")
        t0 = time.time()
        try:
            answer = llm.process_query(item["prompt"]) or ""
        except Exception as e:
            answer = f"<error: {e}>"
        dt = time.time() - t0

        ok_text = item["validate"](answer)
        # Проверяем, что модель попробовала вызвать функцию хотя бы один раз
        called_funcs = [n for (n, _a) in captured_calls]
        ok_func = item["expect_func"] in called_funcs

        results.append({
            "test": item["name"],
            "prompt": item["prompt"],
            "response_time_sec": round(dt, 2),
            "ok_text": bool(ok_text),
            "ok_function_called": bool(ok_func),
            "called_funcs": called_funcs,
            "response_preview": (answer or "")[:400],
        })

        # Для итогового статуса: текст должен выглядеть валидным, и хотя бы одна функция должна вызваться
        if not ok_text:
            all_ok = False
            _print(f"❌ Ответ не прошёл валидацию по содержанию: {answer[:160]}")
        if not ok_func:
            all_ok = False
            _print(f"❌ Не зафиксирован вызов функции '{item['expect_func']}' (зафиксировано: {called_funcs})")

    # 4) Сохраняем отчёт
    report = {
        "provider": config.provider,
        "llm_model": config.llm_model,
        "init_time_sec": round(init_time, 2),
        "tests": results,
        "summary": {
            "passed": bool(all_ok),
            "failed_tests": [r["test"] for r in results if not (r["ok_text"] and r["ok_function_called"])],
        },
    }

    with open("test_results/gigachat_live.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _print("\n💾 Отчёт сохранён: test_results/gigachat_live.json")

    # 5) Итоговый код возврата
    if not all_ok:
        _print("\n❌ E2E GigaChat тесты не пройдены")
        sys.exit(1)

    _print("\n✅ E2E GigaChat тесты пройдены")


if __name__ == "__main__":
    main()

