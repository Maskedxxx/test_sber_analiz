import psutil
import pytz
from datetime import datetime
from typing import Dict, Any
from ..utils.logger import logger


class SystemService:
    """Сервис для получения системной информации."""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """
        Получает статистику загрузки системы: CPU и память.
        
        Returns:
            Словарь с информацией о CPU и памяти
        """
        try:
            logger.info("🔄 Получение статистики системы...")
            
            # Получаем статистику CPU (ждем 1 секунду для точности)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Получаем статистику памяти
            memory = psutil.virtual_memory()

            stats = {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2)
            }
            
            logger.info("✅ Статистика системы получена", **stats)
            return stats
            
        except Exception as e:
            logger.error_occurred(e, "получение статистики системы")
            raise
    
    @staticmethod
    def get_moscow_time() -> str:
        """
        Получает текущее время в Москве.
        
        Returns:
            Текущее время в Москве в формате строки
        """
        try:
            logger.info("🔄 Получение времени в Москве...")
            
            moscow_tz = pytz.timezone('Europe/Moscow')
            moscow_time = datetime.now(moscow_tz)
            
            formatted_time = moscow_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            
            logger.info(f"✅ Время в Москве: {formatted_time}")
            return formatted_time
            
        except Exception as e:
            logger.error_occurred(e, "получение времени в Москве")
            raise
    
    @staticmethod
    def format_system_stats_for_display(stats: Dict[str, Any]) -> str:
        """
        Форматирует статистику системы для удобного отображения.
        
        Args:
            stats: Словарь со статистикой системы
            
        Returns:
            Отформатированная строка со статистикой
        """
        return f"""Статистика системы:
        CPU: {stats['cpu_percent']}%
        Память: {stats['memory_percent']}% 
        ({stats['memory_used_gb']} ГБ / {stats['memory_total_gb']} ГБ)
        Доступно: {stats['memory_available_gb']} ГБ"""