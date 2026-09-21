import time
from typing import Dict, Any, List
from datetime import datetime
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self.metrics = {
            "query_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "plugin_executions": {},
            "errors": []
        }
        self.start_time = time.time()

    def record_query_time(self, query: str, processing_time: float):
        """记录查询处理时间"""
        self.metrics["query_times"].append({
            "query": query,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        })

        # 保持最近1000个记录
        if len(self.metrics["query_times"]) > 1000:
            self.metrics["query_times"] = self.metrics["query_times"][-1000:]

    def record_cache_hit(self):
        """记录缓存命中"""
        self.metrics["cache_hits"] += 1

    def record_cache_miss(self):
        """记录缓存未命中"""
        self.metrics["cache_misses"] += 1

    def record_plugin_execution(self, plugin_name: str, execution_time: float, success: bool):
        """记录插件执行"""
        if plugin_name not in self.metrics["plugin_executions"]:
            self.metrics["plugin_executions"][plugin_name] = {
                "count": 0,
                "success_count": 0,
                "execution_times": []
            }

        self.metrics["plugin_executions"][plugin_name]["count"] += 1
        self.metrics["plugin_executions"][plugin_name]["execution_times"].append(execution_time)

        if success:
            self.metrics["plugin_executions"][plugin_name]["success_count"] += 1

        # 保持最近100个执行时间记录
        if len(self.metrics["plugin_executions"][plugin_name]["execution_times"]) > 100:
            self.metrics["plugin_executions"][plugin_name]["execution_times"] = \
                self.metrics["plugin_executions"][plugin_name]["execution_times"][-100:]

    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """记录错误"""
        error_record = {
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }

        self.metrics["errors"].append(error_record)

        # 保持最近100个错误记录
        if len(self.metrics["errors"]) > 100:
            self.metrics["errors"] = self.metrics["errors"][-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        query_times = [q["processing_time"] for q in self.metrics["query_times"]]

        metrics = {
            "uptime": time.time() - self.start_time,
            "total_queries": len(self.metrics["query_times"]),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "query_time_stats": self._calculate_query_time_stats(query_times),
            "plugin_metrics": self._calculate_plugin_metrics(),
            "recent_errors": self.metrics["errors"][-10:]  # 最近10个错误
        }

        return metrics

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total == 0:
            return 0.0
        return self.metrics["cache_hits"] / total

    def _calculate_query_time_stats(self, query_times: List[float]) -> Dict[str, float]:
        """计算查询时间统计"""
        if not query_times:
            return {}

        return {
            "mean": statistics.mean(query_times),
            "median": statistics.median(query_times),
            "p95": statistics.quantiles(query_times, n=20)[18] if len(query_times) >= 20 else query_times[-1],
            "min": min(query_times),
            "max": max(query_times)
        }

    def _calculate_plugin_metrics(self) -> Dict[str, Any]:
        """计算插件指标"""
        plugin_metrics = {}

        for plugin_name, data in self.metrics["plugin_executions"].items():
            execution_times = data["execution_times"]

            plugin_metrics[plugin_name] = {
                "total_executions": data["count"],
                "success_rate": data["success_count"] / data["count"] if data["count"] > 0 else 0,
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "median_execution_time": statistics.median(execution_times) if execution_times else 0
            }

        return plugin_metrics


# 全局指标收集器
metrics = MetricsCollector()