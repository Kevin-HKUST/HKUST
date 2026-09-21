from typing import Dict, Any
from core.llm_client import llm_client
import logging

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """响应生成器"""

    def __init__(self):
        self.templates = {
            "weather": self._generate_weather_response,
            "finance": self._generate_finance_response,
            "transport": self._generate_transport_response,
            "calculation": self._generate_calculation_response,
            "multimodal": self._generate_multimodal_response,
            "default": self._generate_default_response
        }

    def generate_response(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终响应"""
        intent = workflow_result.get("intent", {}).get("intent", "default")

        # 选择模板生成响应
        generator = self.templates.get(intent, self.templates["default"])
        response = generator(workflow_result)

        # 添加元数据
        response["metadata"] = {
            "processing_time": workflow_result.get("processing_time", 0),
            "intent": intent,
            "confidence": workflow_result.get("final_answer", {}).get("confidence", 0.5),
            "sources_used": workflow_result.get("final_answer", {}).get("sources_used", {})
        }

        return response

    def _generate_weather_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成天气响应"""
        plugin_results = result.get("plugin_results", {})
        weather_data = plugin_results.get("weather", {})

        if "error" in weather_data:
            return {
                "type": "weather",
                "content": result.get("final_answer", {}).get("answer", "无法获取天气信息"),
                "data": {}
            }

        return {
            "type": "weather",
            "content": result.get("final_answer", {}).get("answer", ""),
            "data": weather_data
        }

    def _generate_finance_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成金融响应"""
        plugin_results = result.get("plugin_results", {})
        finance_data = plugin_results.get("finance", {})

        return {
            "type": "finance",
            "content": result.get("final_answer", {}).get("answer", ""),
            "data": finance_data,
            "timestamp": self._get_timestamp()
        }

    def _generate_transport_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成交通响应"""
        plugin_results = result.get("plugin_results", {})
        transport_data = plugin_results.get("transport", {})

        return {
            "type": "transport",
            "content": result.get("final_answer", {}).get("answer", ""),
            "data": transport_data,
            "routes": transport_data.get("routes", [])
        }

    def _generate_calculation_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成计算响应"""
        answer = result.get("final_answer", {}).get("answer", "")

        return {
            "type": "calculation",
            "content": answer,
            "calculation": self._extract_calculation_result(answer)
        }

    def _generate_multimodal_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成多模态响应"""
        multimodal_context = result.get("multimodal_context", "")

        return {
            "type": "multimodal",
            "content": result.get("final_answer", {}).get("answer", ""),
            "files_processed": len(multimodal_context.split("文件内容:")) - 1 if multimodal_context else 0,
            "analysis": "基于上传文件的分析结果"
        }

    def _generate_default_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成默认响应"""
        return {
            "type": "general",
            "content": result.get("final_answer", {}).get("answer", "无法回答您的问题"),
            "sources": result.get("final_answer", {}).get("sources_used", {})
        }

    def _extract_calculation_result(self, answer: str) -> Dict[str, Any]:
        """从答案中提取计算结果"""
        import re

        # 查找数字结果
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            return {
                "result": numbers[-1],
                "unit": self._infer_unit(answer)
            }

        return {"result": "unknown", "unit": "unknown"}

    def _infer_unit(self, text: str) -> str:
        """推断单位"""
        if "元" in text or "¥" in text or "RMB" in text:
            return "CNY"
        elif "$" in text or "美元" in text:
            return "USD"
        elif "°" in text or "度" in text:
            return "temperature"
        elif "公里" in text or "km" in text:
            return "distance"
        elif "小时" in text or "分钟" in text:
            return "time"
        else:
            return "unitless"

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()