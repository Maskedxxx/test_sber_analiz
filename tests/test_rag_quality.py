#!/usr/bin/env python3
"""
Тесты качества RAG системы для финансового чат-бота.

Оценивает:
- Релевантность результатов поиска
- Покрытие тематических запросов  
- Время ответа
- Консистентность результатов
"""

import sys
import os
import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.data_service import DataService
from utils.config import config
from utils.logger import logger


@dataclass
class TestCase:
    """Тестовый случай для проверки качества RAG."""
    query: str
    expected_article_ids: List[int]  # ID статей, которые должны быть в top_k
    expected_spheres: List[str]      # Ожидаемые сферы
    description: str


@dataclass
class TestResult:
    """Результат выполнения теста."""
    test_case: TestCase
    found_article_ids: List[int]
    precision_at_5: float
    recall_at_5: float
    response_time: float
    avg_similarity: float
    passed: bool


class RAGQualityTester:
    """Класс для тестирования качества RAG системы."""
    
    def __init__(self):
        self.data_service = DataService()
        self.test_cases = self._load_test_cases()
        
    def initialize(self) -> bool:
        """Инициализирует тестер."""
        try:
            print("🔄 Инициализация тестера RAG...")
            
            if not config.validate():
                return False
                
            self.data_service.setup_vector_store()
            print("✅ Тестер готов к работе!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            return False
    
    def _load_test_cases(self) -> List[TestCase]:
        """Загружает размеченные тестовые случаи."""
        return [
            TestCase(
                query="новости о Сбербанке",
                expected_article_ids=[1, 15, 23, 45],  # Пример ID, будут обновлены после анализа данных
                expected_spheres=["Финансы", "Банки"],
                description="Поиск новостей о конкретном банке"
            ),
            TestCase(
                query="информация о нефтяной отрасли",
                expected_article_ids=[3, 12, 34],
                expected_spheres=["Энергетика", "Нефть"],
                description="Поиск по отраслевой тематике"
            ),
            TestCase(
                query="курс валют и рубль",
                expected_article_ids=[5, 18, 27],
                expected_spheres=["Финансы", "Валюта"],
                description="Поиск по валютной тематике"
            ),
            TestCase(
                query="фондовый рынок акции",
                expected_article_ids=[7, 19, 29, 41],
                expected_spheres=["Финансы", "Фондовый рынок"],
                description="Поиск по фондовому рынку"
            ),
            TestCase(
                query="технологические компании IT",
                expected_article_ids=[9, 22, 35],
                expected_spheres=["Технологии", "IT"],
                description="Поиск по IT сектору"
            ),
        ]
    
    def run_all_tests(self) -> List[TestResult]:
        """Запускает все тесты и возвращает результаты."""
        print("\n🧪 Запуск тестов качества RAG системы...")
        print("="*60)
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📋 Тест {i}/{len(self.test_cases)}: {test_case.description}")
            print(f"   Запрос: '{test_case.query}'")
            
            result = self._run_single_test(test_case)
            results.append(result)
            
            # Выводим результат теста
            status = "✅ ПРОШЕЛ" if result.passed else "❌ НЕ ПРОШЕЛ"
            print(f"   {status} | Precision@5: {result.precision_at_5:.3f} | "
                  f"Recall@5: {result.recall_at_5:.3f} | "
                  f"Время: {result.response_time:.2f}с")
        
        self._print_summary(results)
        return results
    
    def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Выполняет один тест."""
        start_time = time.time()
        
        try:
            # Выполняем поиск
            search_results = self.data_service.search_articles(test_case.query, top_k=5)
            response_time = time.time() - start_time
            
            # Извлекаем ID найденных статей
            found_article_ids = [result["id"] for result in search_results["results"]]
            
            # Вычисляем метрики
            precision_at_5 = self._calculate_precision_at_k(
                found_article_ids, test_case.expected_article_ids, k=5
            )
            recall_at_5 = self._calculate_recall_at_k(
                found_article_ids, test_case.expected_article_ids, k=5
            )
            
            # Средняя схожесть
            similarities = [result["similarity"] for result in search_results["results"]]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            # Определяем, прошел ли тест (пороги можно настроить)
            passed = precision_at_5 >= 0.2 and recall_at_5 >= 0.1 and response_time <= 10.0
            
            return TestResult(
                test_case=test_case,
                found_article_ids=found_article_ids,
                precision_at_5=precision_at_5,
                recall_at_5=recall_at_5,
                response_time=response_time,
                avg_similarity=avg_similarity,
                passed=passed
            )
            
        except Exception as e:
            print(f"   ❌ Ошибка в тесте: {e}")
            return TestResult(
                test_case=test_case,
                found_article_ids=[],
                precision_at_5=0.0,
                recall_at_5=0.0,
                response_time=time.time() - start_time,
                avg_similarity=0.0,
                passed=False
            )
    
    @staticmethod
    def _calculate_precision_at_k(found_ids: List[int], expected_ids: List[int], k: int) -> float:
        """Вычисляет Precision@K."""
        if not found_ids:
            return 0.0
        
        found_set = set(found_ids[:k])
        expected_set = set(expected_ids)
        
        relevant_found = len(found_set.intersection(expected_set))
        return relevant_found / min(len(found_ids), k)
    
    @staticmethod
    def _calculate_recall_at_k(found_ids: List[int], expected_ids: List[int], k: int) -> float:
        """Вычисляет Recall@K."""
        if not expected_ids:
            return 1.0
        
        found_set = set(found_ids[:k])
        expected_set = set(expected_ids)
        
        relevant_found = len(found_set.intersection(expected_set))
        return relevant_found / len(expected_ids)
    
    def _print_summary(self, results: List[TestResult]):
        """Выводит сводку по всем тестам."""
        print("\n" + "="*60)
        print("СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
        print("="*60)
        
        passed_tests = sum(1 for r in results if r.passed)
        total_tests = len(results)
        
        avg_precision = sum(r.precision_at_5 for r in results) / total_tests
        avg_recall = sum(r.recall_at_5 for r in results) / total_tests
        avg_time = sum(r.response_time for r in results) / total_tests
        avg_similarity = sum(r.avg_similarity for r in results) / total_tests
        
        print(f"✅ Успешных тестов: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"📈 Средний Precision@5: {avg_precision:.3f}")
        print(f"📈 Средний Recall@5: {avg_recall:.3f}")
        print(f"⏱️  Среднее время ответа: {avg_time:.2f} секунд")
        print(f"🎯 Средняя схожесть: {avg_similarity:.3f}")
        
        if passed_tests == total_tests:
            print("\n🎉 Все тесты прошли успешно!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} тестов требуют внимания")
    
    def analyze_collection_for_test_data(self):
        """Анализирует коллекцию для создания тестовых данных."""
        print("\n🔍 Анализ коллекции для создания тестовых данных...")
        
        # Загружаем статьи для анализа
        articles = self.data_service.load_articles()
        
        # Группируем по сферам
        spheres = {}
        for article in articles:
            sphere = article.sphere
            if sphere not in spheres:
                spheres[sphere] = []
            spheres[sphere].append({
                'id': article.id,
                'title': article.article_text[:100] + '...',
                'answer': article.answer
            })
        
        print("\n📋 Статистика по сферам:")
        for sphere, articles_list in spheres.items():
            print(f"  {sphere}: {len(articles_list)} статей")
        
        # Создаем рекомендации для тестовых данных
        print("\n💡 Рекомендации для тестовых запросов:")
        for sphere, articles_list in list(spheres.items())[:5]:
            sample_ids = [str(a['id']) for a in articles_list[:3]]
            print(f"  '{sphere.lower()}' -> ID статей: {sample_ids}")


def main():
    """Главная функция для запуска тестов."""
    tester = RAGQualityTester()
    
    if not tester.initialize():
        print("❌ Не удалось инициализировать тестер")
        return
    
    # Сначала анализируем данные для создания тестов
    tester.analyze_collection_for_test_data()
    
    # Запускаем тесты
    results = tester.run_all_tests()
    
    # Сохраняем результаты в файл
    results_data = []
    for result in results:
        results_data.append({
            'query': result.test_case.query,
            'description': result.test_case.description,
            'precision_at_5': result.precision_at_5,
            'recall_at_5': result.recall_at_5,
            'response_time': result.response_time,
            'avg_similarity': result.avg_similarity,
            'passed': result.passed,
            'found_article_ids': result.found_article_ids
        })
    
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print("\n💾 Результаты сохранены в test_results.json")


if __name__ == "__main__":
    main()