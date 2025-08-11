.PHONY: pull ingest chat test

pull:
	curl -s http://localhost:11434/api/pull -d '{"name": "$(or ${MODEL},qwen2.5:7b-instruct)"}' || true

ingest:
	python -m app.rag.ingest

chat:
	python -m app.cli chat

test:
	pytest -q
