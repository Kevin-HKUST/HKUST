import requests
import json
from typing import Dict, Any, List
from plugins.base_plugin import BasePlugin
from config import Config
import logging
import time
import re
from core.intent_recognizer import IntentRecognizer

logger = logging.getLogger(__name__)


class WeatherPlugin(BasePlugin):

    def __init__(self):
        super().__init__("weather_plugin")
        self.api_key = Config.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}

    # ======================================================
    # ⭐ 统一 run() 接口（保持兼容，但实际逻辑在 execute）
    # ======================================================
    def run(self, query: str, retrieval_results=None):
        return self.execute(query, retrieval_results)

    # ======================================================
    # ⭐ 主执行逻辑（未来天气 → 自动 fallback）
    # ======================================================
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            # 1. 如果是未来天气 → 让 workflow 自动 fallback
            if IntentRecognizer().is_future_weather(query):
                return self.failure("Future weather not supported by weather_plugin")

            # 2. 解析城市
            locations = self._extract_locations(query)
            if not locations:
                return self.failure("无法从查询中提取位置信息")

            location = locations[0]

            # 3. 只处理当前天气（未来天气交给 fallback）
            data = self._get_current_cached(location)

            # 4. 成功返回
            return self.success(
                data=data,
                confidence=0.9,
                metadata={"location": location}
            )

        except Exception as e:
            logger.error(f"Weather plugin error: {e}")
            return self.failure(f"Weather query failed: {str(e)}")

    # ======================================================
    # 城市提取
    # ======================================================
    def _extract_locations(self, query: str) -> List[str]:
        q = query.lower()

        mapping = {
            "香港": "Hong Kong",
            "hong kong": "Hong Kong",
            "hk": "Hong Kong",
            "深圳": "Shenzhen",
            "shenzhen": "Shenzhen",
            "广州": "Guangzhou",
            "beijing": "Beijing",
            "北京": "Beijing",
            "上海": "Shanghai",
            "macau": "Macau",
            "澳门": "Macau",
            "珠海": "Zhuhai",
        }

        for key, c in mapping.items():
            if key in q:
                return [c]

        return ["Hong Kong"]

    # ======================================================
    # 缓存
    # ======================================================
    def _get_current_cached(self, location):
        key = f"current_{location}"
        if key in self.cache and time.time() - self.cache[key]["ts"] < 300:
            return self.cache[key]["data"]

        data = self._get_current_weather(location)
        self.cache[key] = {"ts": time.time(), "data": data}
        return data

    # ======================================================
    # 当前天气 API
    # ======================================================
    def _get_current_weather(self, location: str):
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
                "lang": "zh_cn"
            }
            resp = requests.get(url, params=params, timeout=8)
            data = resp.json()

            # API 错误
            if data.get("cod") != 200:
                return {"error": data.get("message", "天气 API 返回错误")}

            return {
                "type": "current",
                "location": location,
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "wind": data["wind"]["speed"],
                "description": data["weather"][0]["description"],
                "time": data["dt"],
            }

        except Exception as e:
            return {"error": f"API 调用失败：{e}"}

    # ======================================================
    # ⭐ 错误返回 → 用 BasePlugin.failure()
    # ======================================================
    def _handle_error(self, msg: str, type="runtime"):
        return self.failure(msg)