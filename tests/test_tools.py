from app.tools.system import get_system_stats
from app.tools.timezone import get_moscow_time

def test_system_stats_keys():
    stats = get_system_stats()
    assert {"cpu_percent", "memory_percent", "memory_available_gb", "memory_total_gb"} <= stats.keys()

def test_moscow_time_format():
    t = get_moscow_time()
    assert "MSK" in t or "MSD" in t or "Moscow" in t
