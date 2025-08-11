# RAG Console Bot (MVP)

Консольный чат-бот, выполняющий:
- Запросы о загрузке системы (CPU/Memory)
- Запросы о текущем времени в Москве
- RAG по корпусу документов (из CSV; 1 строка = 1 документ, без чанкования)

Модель LLM: **Ollama** (`qwen2.5:7b-instruct` по умолчанию). Эмбеддинги: **BAAI/bge-m3**.

## Требования
- Ubuntu 24.04, Docker 28+, Docker Compose v2.33+
- NVIDIA GPU 16GB VRAM (для Ollama). Эмбеддинги могут считаться на CPU.

## Установка
```bash
git clone <repo>
cd rag-console-bot
cp .env.example .env
# положите CSV в data/russian_fin_news/mini_df.csv
```

## Запуск через Docker Compose
```bash
docker compose up -d ollama
make pull  # загрузит модель qwen2.5:7b-instruct

# индексация
docker compose run --rm app python -m app.rag.ingest

# чат
docker compose run --rm --service-ports app python -m app.cli chat
```

> **Примечание по VRAM:** полная (без квантизации) 7B модель может не уместиться в 16GB VRAM при большом контексте.
> При нехватке видеопамяти используйте квантованные варианты (например, `qwen2.5:7b-instruct-q4_0`).

## Локальный запуск без Docker
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
OLLAMA_HOST=http://localhost:11434 python -m app.rag.ingest
python -m app.cli chat
```

## Использование
- Введите естественный запрос. LLM выберет один из инструментов: `get_system_stats`, `get_moscow_time` или `search_financial_news` (RAG).
- Для RAG используются топ-K документов из индекса (без чанкования).

## Тестирование
```bash
pytest -q
```
- `tests/test_retriever_small.py` — sanity-check ретривера на мини-корпусе
- `tests/test_guardrails.py` — отказы на утечки
- `tests/test_tools.py` — базовые проверки инструментов
- `tests/eval/test_rag_metrics_minimal.py` — метрики на микрокорпусе (воспроизводимо)
- `tests/eval/test_rag_metrics.py` — демо-метрики (skip по умолчанию)

## Безопасность
Бот не раскрывает системный промпт и внутренние детали; действует фильтр `app/security/guard.py`.

## Корпус
В данном MVP корпус строится напрямую из CSV: одна строка = один документ (см. `app/rag/ingest.py`).

## Лицензия
MIT
