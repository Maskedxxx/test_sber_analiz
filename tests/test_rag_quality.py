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
    expected_article_ids: List[str]  # ID статей (как строки), которые должны быть в top_k
    expected_spheres: List[str]      # Ожидаемые сферы
    description: str


@dataclass
class TestResult:
    """Результат выполнения теста."""
    test_case: TestCase
    found_article_ids: List[str]
    precision_at_5: float
    recall_at_5: float
    response_time: float
    avg_similarity: float  # Косинусная схожесть в диапазоне [-1, 1]
    avg_distance: float    # Косинусное расстояние в диапазоне [0, 2]
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
        """Загружает размеченные тестовые случаи на основе реальных данных."""
        return [
            TestCase(
                query="новости о банках ВТБ Промсвязьбанк Сбербанк",
                expected_article_ids=["5", "16"],  # ID 5 - статья про ВТБ Bank Europe и Промсвязьбанк, ID 16 - статья с упоминанием Сбербанка
                expected_spheres=["Финансы"],
                description="Поиск новостей о банковском секторе"
            ),
            TestCase(
                query="нефть газ энергетика Газпром цены на энергоносители",
                expected_article_ids=["1", "4", "10", "11", "17"],  # Газпром (ID 1), курс рубля и нефть (ID 4), инфляция и энергия в Японии (ID 10), нефть (ID 11), запасы нефти (ID 17) 
                expected_spheres=["Энергетика"],
                description="Поиск по энергетическому сектору"
            ),
            TestCase(
                query="валютный курс рубль доллар юань",
                expected_article_ids=["4"],  # ID 4 - статья про курс рубля к юаню и цены на нефть
                expected_spheres=["Энергетика"],
                description="Поиск по валютной тематике"
            ),
            TestCase(
                query="Мечел убыток EBITDA долги финансовые показатели",
                expected_article_ids=["34", "35"],  # ID 34 - убытки Мечела, ID 35 - реструктуризация долга Мечела
                expected_spheres=["Финансы"],
                description="Поиск по финансовым показателям компаний"
            ),
            TestCase(
                query="программа долгосрочных сбережений инвестиции ЦБ облигации",
                expected_article_ids=["15", "16", "29"],  # ID 15, 16 - программа долгосрочных сбережений, ID 29 - расширение доступа к облигациям
                expected_spheres=["Энергетика", "Финансы/Энергетика"],
                description="Поиск по инвестиционным инструментам"
            ),
            TestCase(
                query="финансовые отчеты и дивиденды Яндекса",
                expected_article_ids=["57", "58", "93", "95"],
                expected_spheres=["Финансы", "Финансы/Энергетика"],
                description="Поиск новостей о финансовых показателях, дивидендной и кадровой политике Яндекса."
            ),
            TestCase(
                query="санкции против России нефть алюминий",
                expected_article_ids=["53", "56", "80", "131", "142"],
                expected_spheres=["Энергетика", "Финансы"],
                description="Поиск информации о санкционном давлении на экономику РФ, в частности на энергетический и металлургический секторы."
            ),
            TestCase(
                query="банковское мошенничество и кибербезопасность",
                expected_article_ids=["40", "42", "49", "79", "92", "156", "158"],
                expected_spheres=["Финансы"],
                description="Поиск новостей о мерах ЦБ и банков по борьбе с финансовым мошенничеством, дропперами и киберугрозами."
            ),
            TestCase(
                query="автомобильная промышленность в России и мире",
                expected_article_ids=["0", "13", "33", "50", "83", "87", "122"],
                expected_spheres=["Финансы"],
                description="Поиск новостей о состоянии автопрома, включая финансовые результаты и производственные планы отечественных и зарубежных компаний."
            ),
            TestCase(
                query="динамика цен на нефть Brent WTI и запасы",
                expected_article_ids=["4", "11", "17", "20", "24", "53", "97", "116", "135", "139"],
                expected_spheres=["Энергетика"],
                description="Поиск новостей о динамике мировых цен на нефть и факторах, влияющих на предложение, таких как уровень запасов и решения ОПЕК+."
            ),
            TestCase(
                query="Газпром и газификация России",
                expected_article_ids=["1", "48", "70", "148"],
                expected_spheres=["Финансы", "Энергетика"],
                description="Поиск новостей, связанных с Газпромом, программой газификации и состоянием внутреннего рынка газа в России."
            ),
            TestCase(
                query="цифровой рубль и новые платежные технологии",
                expected_article_ids=["65", "76"],
                expected_spheres=["Финансы"],
                description="Поиск информации о внедрении новых платежных технологий в России, таких как цифровой рубль и универсальные QR-коды."
            ),
            TestCase(
                query="разморозка активов российских инвесторов",
                expected_article_ids=["71", "72"],
                expected_spheres=["Финансы"],
                description="Поиск информации о мерах Банка России по разморозке заблокированных активов российских инвесторов."
            )
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
            
            # Подробная информация для анализа
            if not result.passed:
                print(f"   📊 Найденные ID: {result.found_article_ids}")
                print(f"   🎯 Ожидаемые ID: {result.test_case.expected_article_ids}")
            elif result.precision_at_5 > 0:
                print(f"   📊 Найденные ID: {result.found_article_ids}")
                print(f"   🎯 Ожидаемые ID: {result.test_case.expected_article_ids}")
        
        self._print_summary(results)
        return results
    
    def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Выполняет один тест."""
        start_time = time.time()
        
        try:
            # Выполняем поиск
            search_results = self.data_service.search_articles(test_case.query, top_k=5)
            response_time = time.time() - start_time
            
            # Извлекаем ID найденных статей (преобразуем в строки для совместимости)
            found_article_ids = [str(result["id"]) for result in search_results["results"]]
            
            # Вычисляем метрики
            precision_at_5 = self._calculate_precision_at_k(
                found_article_ids, test_case.expected_article_ids, k=5
            )
            recall_at_5 = self._calculate_recall_at_k(
                found_article_ids, test_case.expected_article_ids, k=5
            )
            
            # Средняя косинусная схожесть и расстояние
            similarities = [result.get("similarity") for result in search_results.get("results", []) if "similarity" in result]
            distances = [result.get("distance") for result in search_results.get("results", []) if "distance" in result]
            avg_similarity = (sum(similarities) / len(similarities)) if similarities else 0.0
            avg_distance = (sum(distances) / len(distances)) if distances else 0.0
            
            # Определяем, прошел ли тест (пороги можно настроить)
            passed = precision_at_5 >= 0.2 and recall_at_5 >= 0.1 and response_time <= 10.0
            
            return TestResult(
                test_case=test_case,
                found_article_ids=found_article_ids,
                precision_at_5=precision_at_5,
                recall_at_5=recall_at_5,
                response_time=response_time,
                avg_similarity=avg_similarity,
                avg_distance=avg_distance,
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
    def _calculate_precision_at_k(found_ids: List[str], expected_ids: List[str], k: int) -> float:
        """Вычисляет Precision@K."""
        if not found_ids:
            return 0.0
        
        found_set = set(found_ids[:k])
        expected_set = set(expected_ids)
        
        relevant_found = len(found_set.intersection(expected_set))
        return relevant_found / min(len(found_ids), k)
    
    @staticmethod
    def _calculate_recall_at_k(found_ids: List[str], expected_ids: List[str], k: int) -> float:
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
        avg_distance_all = sum(r.avg_distance for r in results) / total_tests
        
        print(f"✅ Успешных тестов: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"📈 Средний Precision@5: {avg_precision:.3f}")
        print(f"📈 Средний Recall@5: {avg_recall:.3f}")
        print(f"⏱️  Среднее время ответа: {avg_time:.2f} секунд")
        print(f"🎯 Среднее косинусное сходство: {avg_similarity:.3f} ([-1..1], больше лучше)")
        print(f"🧭 Среднее косинусное расстояние: {avg_distance_all:.3f} ([0..2], меньше лучше)")
        
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
            'avg_distance': result.avg_distance,
            'passed': result.passed,
            'found_article_ids': result.found_article_ids
        })
    
    with open('test_results/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print("\n💾 Результаты сохранены в test_results/test_results.json")
    
    # Также создаем детальный отчет
    detailed_report = {
        'summary': {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r.passed),
            'avg_precision': sum(r.precision_at_5 for r in results) / len(results),
            'avg_recall': sum(r.recall_at_5 for r in results) / len(results),
            'avg_response_time': sum(r.response_time for r in results) / len(results),
            'avg_similarity': sum(r.avg_similarity for r in results) / len(results),
            'avg_distance': sum(r.avg_distance for r in results) / len(results)
        },
        'detailed_results': results_data,
        'recommendations': []
    }
    
    # Добавляем рекомендации
    failed_tests = [r for r in results if not r.passed]
    if failed_tests:
        detailed_report['recommendations'].append(f"Требуется улучшить качество поиска для {len(failed_tests)} тестов")
        for failed in failed_tests:
            detailed_report['recommendations'].append(f"Тест '{failed.test_case.description}': найдены ID {failed.found_article_ids}, ожидались {failed.test_case.expected_article_ids}")
    
    with open('test_results/detailed_report.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
