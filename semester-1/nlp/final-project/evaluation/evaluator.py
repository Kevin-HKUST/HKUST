import time
import json
import os
from typing import List, Dict, Any
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class SearchEvaluator:
    """搜索性能评估器"""

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.results = {}
        self.timing_data = {}

    def evaluate_test_set(self, test_set_name: str, questions: List[str],
                          descriptions: List[str] = None) -> Dict[str, Any]:
        """评估测试集"""
        print(f"\n🔍 开始评估测试集: {test_set_name}")
        print(f"问题数量: {len(questions)}")

        results = []
        search_times = []
        total_times = []

        for i, question in enumerate(questions, 1):
            print(f"处理问题 {i}/{len(questions)}: {question[:50]}...")

            # 记录搜索时间（到获取信息为止）
            search_start_time = time.time()

            # 执行搜索处理（不包括最终LLM生成）
            intent_result = self.search_engine.intent_recognizer.recognize_intent(question)
            retrieval_results = self.search_engine._retrieve_information(question, intent_result)

            search_end_time = time.time()
            search_duration = search_end_time - search_start_time

            # 记录总处理时间（包括LLM生成）
            total_start_time = time.time()

            final_answer = self.search_engine._generate_answer(question, intent_result, retrieval_results)

            total_end_time = time.time()
            total_duration = total_end_time - total_start_time

            # 合并时间
            total_processing_time = search_duration + total_duration

            result = {
                "question": question,
                "description": descriptions[i - 1] if descriptions else "",
                "intent": intent_result,
                "retrieval_count": retrieval_results.get("count", 0),
                "answer": final_answer,
                "search_time": search_duration,  # 仅搜索时间
                "generation_time": total_duration,  # LLM生成时间
                "total_time": total_processing_time,  # 总时间
                "success": True
            }

            results.append(result)
            search_times.append(search_duration)
            total_times.append(total_processing_time)

            print(f"  - 搜索时间: {search_duration:.2f}s, 总时间: {total_processing_time:.2f}s")

        # 计算统计信息
        stats = self._calculate_statistics(search_times, total_times, results)

        evaluation_result = {
            "test_set": test_set_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "statistics": stats
        }

        self.results[test_set_name] = evaluation_result
        self.timing_data[test_set_name] = {
            "search_times": search_times,
            "total_times": total_times
        }

        print(f"\n✅ 测试集 {test_set_name} 评估完成")
        print(f"平均搜索时间: {stats['mean_search_time']:.2f}s")
        print(f"平均总时间: {stats['mean_total_time']:.2f}s")
        print(f"成功率: {stats['success_rate']:.1f}%")

        return evaluation_result

    def _calculate_statistics(self, search_times: List[float], total_times: List[float],
                              results: List[Dict]) -> Dict[str, Any]:
        """计算统计信息"""
        successful_results = [r for r in results if r.get("success", False)]

        stats = {
            "total_questions": len(results),
            "successful_questions": len(successful_results),
            "success_rate": (len(successful_results) / len(results)) * 100,
            "mean_search_time": sum(search_times) / len(search_times),
            "median_search_time": sorted(search_times)[len(search_times) // 2],
            "min_search_time": min(search_times),
            "max_search_time": max(search_times),
            "mean_total_time": sum(total_times) / len(total_times),
            "median_total_time": sorted(total_times)[len(total_times) // 2],
            "min_total_time": min(total_times),
            "max_total_time": max(total_times),
            "mean_retrieval_count": sum(r.get("retrieval_count", 0) for r in results) / len(results)
        }

        return stats

    def generate_report(self, output_dir: str = "evaluation_results"):
        """生成评估报告"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成详细结果JSON
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # 生成统计摘要
        summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
        self._generate_summary(summary_file)

        # 生成CSV格式的时间数据
        csv_file = os.path.join(output_dir, f"timing_data_{timestamp}.csv")
        self._generate_csv_report(csv_file)

        print(f"\n📊 评估报告已生成:")
        print(f"  - 详细报告: {report_file}")
        print(f"  - 统计摘要: {summary_file}")
        print(f"  - 时间数据: {csv_file}")

        return report_file

    def _generate_summary(self, filename: str):
        """生成统计摘要"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("智能搜索引擎评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for test_set, data in self.results.items():
                stats = data["statistics"]
                f.write(f"测试集: {test_set}\n")
                f.write(f"问题数量: {stats['total_questions']}\n")
                f.write(f"成功率: {stats['success_rate']:.1f}%\n")
                f.write(f"平均搜索时间: {stats['mean_search_time']:.3f} 秒\n")
                f.write(f"搜索时间范围: {stats['min_search_time']:.3f} - {stats['max_search_time']:.3f} 秒\n")
                f.write(f"平均总处理时间: {stats['mean_total_time']:.3f} 秒\n")
                f.write(f"总时间范围: {stats['min_total_time']:.3f} - {stats['max_total_time']:.3f} 秒\n")
                f.write(f"平均检索文档数: {stats['mean_retrieval_count']:.1f}\n")
                f.write("-" * 30 + "\n\n")

            # 跨测试集比较
            if len(self.results) > 1:
                f.write("跨测试集性能比较:\n")
                f.write("测试集\t\t平均搜索时间\t平均总时间\t成功率\n")
                f.write("-" * 60 + "\n")
                for test_set, data in self.results.items():
                    stats = data["statistics"]
                    f.write(f"{test_set}\t\t{stats['mean_search_time']:.3f}s\t\t"
                            f"{stats['mean_total_time']:.3f}s\t\t{stats['success_rate']:.1f}%\n")

    def _generate_csv_report(self, filename: str):
        """生成CSV格式的时间报告"""
        rows = []

        for test_set, data in self.results.items():
            for i, result in enumerate(data["results"]):
                rows.append({
                    "test_set": test_set,
                    "question_id": i + 1,
                    "question": result["question"],
                    "intent": result["intent"]["intent"],
                    "intent_confidence": result["intent"]["confidence"],
                    "retrieval_count": result["retrieval_count"],
                    "search_time": result["search_time"],
                    "generation_time": result["generation_time"],
                    "total_time": result["total_time"],
                    "success": result["success"]
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8-sig')