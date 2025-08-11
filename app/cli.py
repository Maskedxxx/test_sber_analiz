from __future__ import annotations
import json
import typer
import structlog
from app.config import settings
from app.logging_setup import setup_logging
from app.security.guard import check_request_forbidden
from app.llm.ollama_client import OllamaJSONClient
from app.rag.retriever import Retriever
from app.rag.prompting import build_final_messages
from app.tools.system import get_system_stats
from app.tools.timezone import get_moscow_time

app = typer.Typer(add_completion=False)
log = structlog.get_logger()

@app.command()
def chat() -> None:
    setup_logging(settings.LOG_LEVEL)
    typer.echo("RAG Console Bot. Напишите вопрос. /exit — выход.")

    llm = OllamaJSONClient()
    retriever = Retriever()

    while True:
        q = typer.prompt("you")
        if q.strip().lower() in {"/exit", "exit", "quit"}:
            break

        forbidden, reason = check_request_forbidden(q)
        if forbidden:
            typer.echo(f"bot: {reason}")
            continue

        tool_choice = llm.choose_tool([{"role": "user", "content": q}])
        tool = tool_choice.get("tool", "search_financial_news")
        args = tool_choice.get("args", {})
        log.info("tool_choice", tool=tool, args=args)

        if tool == "get_system_stats":
            result = get_system_stats()
            messages = [
                {"role": "user", "content": q},
                {"role": "system", "content": f"ДАННЫЕ ИНСТРУМЕНТА: {json.dumps(result, ensure_ascii=False)}"},
            ]
            answer = llm.generate_answer(messages)
            typer.echo(f"bot: {answer}")
            continue

        if tool == "get_moscow_time":
            result = get_moscow_time()
            messages = [
                {"role": "user", "content": q},
                {"role": "system", "content": f"ДАННЫЕ ИНСТРУМЕНТА: {result}"},
            ]
            answer = llm.generate_answer(messages)
            typer.echo(f"bot: {answer}")
            continue

        # default: search_financial_news (RAG)
        top_k = int(args.get("top_k") or settings.TOP_K)
        rag = retriever.search(query=q if not args.get("query") else args["query"], top_k=top_k)
        items = rag["results"]
        messages = build_final_messages(q, items)
        answer = llm.generate_answer(messages)
        typer.echo(f"bot: {answer}")

if __name__ == "__main__":
    app()
