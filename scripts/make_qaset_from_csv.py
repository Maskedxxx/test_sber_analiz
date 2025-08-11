"""
Опционально: создать синтетический QA-набор для демонстрации метрик RAG
по CSV, используя поля sphere/source для генерации простых запросов.
"""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

CSV = Path("data/russian_fin_news/mini_df.csv")
OUT = Path("tests/data/qa.jsonl")

OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV).head(50)
with OUT.open("w", encoding="utf-8") as f:
    for _, r in df.iterrows():
        q = f"Информация о {r.get('sphere','банке')}"
        gold = [str(r.get("sphere", ""))]
        f.write(json.dumps({"q": q, "gold": gold}, ensure_ascii=False) + "\n")
print(f"Saved {OUT}")
