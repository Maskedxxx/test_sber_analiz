#!/usr/bin/env python3
"""
Простой smoke-тест LLM: читает переменные из .env, 
создает клиента (GigaChat или OpenAI) и делает 1 запрос в чат.

Запуск:
  python scripts/smoke_llm.py "Привет!"
  python scripts/smoke_llm.py           # по умолчанию использует стандартный промпт
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_dotenv_if_present(root: Path):
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("\"\'")
        # Не перезаписываем уже существующие переменные окружения
        if key not in os.environ:
            os.environ[key] = val


def main():
    # Грузим .env до импорта конфигурации
    root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(root)

    # Делаем видимым src/
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    # Импорты после загрузки .env
    from openai import OpenAI
    import httpx
    from utils.config import config
    from services.token_manager import GigaChatTokenManager

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Скажи коротко привет и представься."

    if config.provider == "gigachat":
        # Получаем токен и создаём клиента на базовый URL GigaChat
        tm = GigaChatTokenManager()
        token = tm.get()
        http_client = httpx.Client(verify=config.gigachat_verify_ssl)
        client = OpenAI(api_key=token, base_url=config.gigachat_base_url, http_client=http_client)
        print("✅ Провайдер: GigaChat")
    elif config.provider == "openai":
        client = OpenAI(api_key=config.openai_api_key)
        print("✅ Провайдер: OpenAI")
    else:
        print("❌ Не настроен ни один провайдер. Заполните .env (GIGACHAT_* или OPENAI_API_KEY)")
        sys.exit(1)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Вы — полезный ассистент. Отвечайте кратко."},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = client.chat.completions.create(model=config.llm_model, messages=messages)
        msg = resp.choices[0].message
        content = getattr(msg, "content", None) or ""
        print("\nОтвет:\n" + content)
    except Exception as e:
        print(f"❌ Ошибка при обращении к LLM: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

