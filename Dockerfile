FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir .

COPY app /app/app
COPY tests /app/tests
COPY scripts /app/scripts
COPY data /app/data

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.cli", "chat"]