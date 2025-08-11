from __future__ import annotations
import psutil
from typing import Dict

def get_system_stats() -> Dict[str, float]:
    cpu_percent = psutil.cpu_percent(interval=0.7)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": float(cpu_percent),
        "memory_percent": float(mem.percent),
        "memory_available_gb": round(mem.available / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
    }
