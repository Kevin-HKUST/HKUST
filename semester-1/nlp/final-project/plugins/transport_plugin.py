import requests
import json
import logging
from typing import Dict, Any, List

from plugins.base_plugin import BasePlugin
from config import Config

logger = logging.getLogger(__name__)


class TransportPlugin(BasePlugin):
    """使用 Google Maps API 实现 POI 搜索 + 路线规划"""

    def __init__(self):
        super().__init__("transport_plugin")

        self.google_key = Config.GOOGLE_API_KEY

        # 支持能力
        self.capabilities = ["route_planning", "poi_search"]

        # 常用地点映射表（提高解析能力）
        self.hk_locations = {
            "香港科技大学": "Hong Kong University of Science and Technology",
            "科大": "Hong Kong University of Science and Technology",
            "hkust": "Hong Kong University of Science and Technology",
            "西九龙站": "Hong Kong West Kowloon Station",
            "香港西九龙高铁站": "Hong Kong West Kowloon Station",
            "九龙": "Kowloon Hong Kong",
            "中环": "Central Hong Kong",
            "尖沙咀": "Tsim Sha Tsui Hong Kong",
            "旺角": "Mong Kok Hong Kong",
        }

    # =====================================================================
    # 必须添加的 run() —— WorkflowEngine 调用的是 run()
    # =====================================================================
    def run(self, query: str, retrieval_results=None):
        return self.execute(query)

    # =====================================================================
    # 主执行逻辑
    # =====================================================================
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            q = query.lower()

            # POI 查询（附近的麦当劳 / 最近的肯德基）
            if self._is_poi_query(q):
                return self._handle_poi_query(query)

            # 路线规划（从A到B怎么走）
            if self._is_route_query(q):
                return self._handle_route_query(query)

            # 默认提示
            return self._handle_general(query)

        except Exception as e:
            logger.error(f"Transport plugin error: {e}")
            return self.failure(f"Transport query failed: {str(e)}")

    # =====================================================================
    # 意图检测
    # =====================================================================
    def _is_poi_query(self, q: str):
        return any(k in q for k in ["附近", "最近", "哪里有", "kfc", "肯德基", "麦当劳"])

    def _is_route_query(self, q: str):
        return ("怎么走" in q) or ("怎么去" in q) or ("从" in q and "到" in q)

    # =====================================================================
    # 路线解析
    # =====================================================================
    def _extract_locations(self, query: str):
        if "到" not in query:
            return None, None

        parts = query.split("到", 1)
        start = parts[0].replace("怎么去", "").replace("怎么走", "").replace("从", "").strip()
        end = parts[1].replace("怎么去", "").replace("怎么走", "").strip()

        if not start or not end:
            return None, None

        return start, end

    # =====================================================================
    # Google Geocode（地点 → 经纬度）
    # =====================================================================
    def _google_geocode(self, name: str):
        # 中文常用地名映射
        addr = self.hk_locations.get(name, name)

        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            r = requests.get(url, params={"address": addr, "key": self.google_key}, timeout=5).json()
            results = r.get("results")

            if results:
                loc = results[0]["geometry"]["location"]
                return loc["lat"], loc["lng"]

        except Exception:
            pass

        return None, None

    # =====================================================================
    # 路线规划（使用 Directions API）
    # =====================================================================
    def _handle_route_query(self, query: str):
        start, end = self._extract_locations(query)
        if not start or not end:
            return self.failure("无法解析路线请求")

        lat1, lng1 = self._google_geocode(start)
        lat2, lng2 = self._google_geocode(end)

        if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
            return self.failure("无法识别地点，请提供更准确的位置名称")

        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{lat1},{lng1}",
            "destination": f"{lat2},{lng2}",
            "mode": "transit",
            "language": "zh-HK",
            "key": self.google_key
        }

        try:
            r = requests.get(url, params=params, timeout=8).json()
            routes = r.get("routes", [])

            if not routes:
                return self.failure("未找到路线")

            leg = routes[0]["legs"][0]
            summary = {
                "start": start,
                "end": end,
                "distance": leg["distance"]["text"],
                "duration": leg["duration"]["text"],
                "steps": [step.get("html_instructions", "") for step in leg.get("steps", [])],
            }

            return self.success(
                data=summary,
                confidence=0.9,
                metadata={"subtype": "route"}
            )

        except Exception as e:
            return self.failure(f"路线规划失败: {str(e)}")

    # =====================================================================
    # ⭐⭐⭐ 正确的 POI 查询（修复后的版本）
    # =====================================================================
    def _handle_poi_query(self, query: str):
        keyword = (
            query.replace("附近", "")
                 .replace("哪里有", "")
                 .replace("最近的", "")
                 .replace("最近", "")
                 .strip()
        )

        # 尝试提取地点
        location = None
        for zh, eng in self.hk_locations.items():
            if zh in query:
                location = eng
                break

        # fallback："离 A 最近的 B"
        import re
        if not location:
            m = re.search(r"离(.+?)最近", query)
            if m:
                location = m.group(1).strip()

        # fallback：仍未找到地点 → 使用中环
        if location:
            lat, lng = self._google_geocode(location)
        else:
            lat, lng = 22.2819, 114.1582  # 中环

        if lat is None:
            return self.failure("无法识别地点")

        # Google NearbySearch
        try:
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{lat},{lng}",
                "radius": 2000,
                "keyword": keyword,
                "language": "zh-HK",
                "key": self.google_key
            }

            r = requests.get(url, params=params, timeout=7).json()
            results = r.get("results", [])

            if not results:
                return self.failure("未找到附近地点")

            pois = [{
                "name": item.get("name"),
                "address": item.get("vicinity"),
                "rating": item.get("rating"),
            } for item in results[:3]]

            return self.success(
                data={
                    "location": location,
                    "lat": lat,
                    "lng": lng,
                    "query": keyword,
                    "pois": pois
                },
                confidence=0.9,
                metadata={"subtype": "poi"}
            )

        except Exception as e:
            return self.failure(f"POI 查询失败: {e}")

    # =====================================================================
    # 默认回答
    # =====================================================================
    def _handle_general(self, query: str):
        return self.success(
            data={"msg": "你可以问：附近的KFC？从A到B怎么走？"},
            confidence=0.6,
            metadata={"subtype": "general"}
        )