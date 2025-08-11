# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/

# Создаем директорию для ChromaDB
RUN mkdir -p ./chroma_db

# Устанавливаем переменную окружения для Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Команда по умолчанию
CMD ["python", "src/chatbot.py"]