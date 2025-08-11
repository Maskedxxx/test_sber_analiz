from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo

def get_moscow_time() -> str:
    tz = ZoneInfo("Europe/Moscow")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
