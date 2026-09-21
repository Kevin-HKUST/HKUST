import requests
import logging
from plugins.base_plugin import BasePlugin
from config import Config

logger = logging.getLogger(__name__)

class WebSearchPlugin(BasePlugin):
    """Web 搜索插件（使用 SerpAPI 或 mock fallback）"""

    def __init__(self):
        super().__init__()
        self.api_key = Config.SERPAPI_KEY  # 如果你有 SerpAPI Key，可以放在这里

    # ⭐⭐ 必加：兼容 workflow_engine.run() 调用方式
    def run(self, query: str, retrieval_results=None):
        return self.execute(query, retrieval_results)

    def execute(self, query: str, retrieval_results=None):
        """执行网络搜索"""

        if not self.api_key:
            logger.warning("[MOCK FALLBACK] 无 API KEY → 使用模拟搜索结果")
            return {
                "content": f"[MOCK] 模拟搜索结果：{query}",
                "metadata": {"source": "plugin", "success": True},
                "confidence": 0.7
            }

        try:
            res = self._call_serpapi(query)
            return {
                "content": res,
                "metadata": {"source": "plugin", "success": True},
                "confidence": 0.9
            }

        except Exception as e:
            logger.error(f"[SerpAPI 错误] {e} → fallback mock")
            return {
                "content": f"[MOCK] 搜索失败，返回模拟结果：{query}",
                "metadata": {"source": "plugin", "success": True},
                "confidence": 0.6
            }

    def _call_serpapi(self, query: str):
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "hl": "zh-cn",
                "gl": "hk",
                "api_key": self.api_key
            }

            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            if "error" in data:
                return f"[SerpAPI Error] {data['error']}"

            organic = data.get("organic_results", [])
            if not organic:
                return f"[SerpAPI] 找不到相关搜索结果：{query}"

            lines = []
            for item in organic[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                lines.append(f"标题：{title}\n摘要：{snippet}\n链接：{link}")

            return "\n\n".join(lines)

        except Exception as e:
            logging.error(f"[SerpAPI Exception] {e}")
            return f"[SerpAPI Exception] {e}"