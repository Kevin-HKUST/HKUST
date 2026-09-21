import re
from typing import Dict, List


class IntentRecognizer:
    """
    超强增强版智能意图识别器
    支持：
        - 金融 Finance
        - 天气 Weather
        - 交通 / POI / 地址 Navigation / Transport
        - Web 搜索（新闻 / 热点）
        - 多模态 multimodal
    全部关键词已扩充：英文 + 繁体中文 + 简体中文
    """

    def __init__(self):

        # ======================================================
        # Finance 关键词（英 / 简 / 繁）
        # ======================================================
        self.finance_keywords = [
            # Chinese
            "股票", "股價", "股价", "市值", "行情", "价格", "價格", "股指",
            "涨跌", "漲跌", "涨幅", "跌幅", "開盤", "开盘", "收盤", "收盘",
            "財報", "财报", "淨利潤", "净利润", "營收", "营收",
            "市盈率", "估值", "基金", "期貨", "期货",
            "比特幣", "比特币", "乙太坊",
            # English
            "stock", "price", "share price", "market cap", "valuation",
            "earnings", "revenue", "financial report",
            "bitcoin", "btc", "eth", "cryptocurrency", "crypto",
            "nasdaq", "dow", "s&p", "forex"
        ]
        self.stock_code_pattern = re.compile(r"\b[A-Z]{2,6}\b")

        # ======================================================
        # Weather 关键词（英 / 简 / 繁 / 粤语常见问法）
        # ======================================================
        self.weather_keywords = [
            # Chinese core
            "天气", "天氣",
            "气温", "氣溫",
            "温度", "濕度", "湿度",
            "降雨", "下雨", "會下雨", "會落雨", "会下雨", "会落雨",
            "雷暴", "暴雨",
            "风速", "風速", "风向", "風向",
            "空气质量", "空氣質量", "pm2.5",
            "紫外线", "紫外線",
            "晴天", "多云", "多雲", "阴天", "陰天",

            # Time-related
            "今天天气", "今日天气", "今日天氣",
            "明天天气", "明日天气", "明日天氣",
            "今天", "今日", "明天", "明日", "後天", "后天",
            "上午", "下午", "晚上", "今晚", "明早", "明晚",

            # English
            "weather", "forecast", "temperature", "humidity",
            "rain", "raining", "sunny", "cloudy", "uv index",
            "storm", "thunderstorm", "wind speed", "wind direction",
            "air quality",

            # Natural questions
            "will it rain", "is it raining", "weather tomorrow",
            "weather today"
        ]

        # ======================================================
        # Transport / Navigation（英 / 简 / 繁）
        # ======================================================
        self.location_keywords = [
            # Chinese core
            "在哪里", "在哪兒", "在哪里", "在哪", "在哪儿",
            "附近", "附近有", "周边", "周邊",
            "距离", "距離", "多远", "多遠", "多久",
            "路线", "路線", "路径", "路徑",
            "路线规划", "路線規劃",
            "怎么走", "怎麼走", "怎么去", "怎麼去",
            "怎么到", "怎麼到",
            "坐地铁", "搭地铁", "坐地鐵", "搭地鐵",
            "公交", "巴士",
            "导航", "導航",

            # English
            "where is", "location", "near me", "nearby",
            "how to go", "how to get", "route", "navigation",
            "distance", "how far", "subway", "metro", "bus"
        ]

        # ======================================================
        # Web search（英 / 简 / 繁）
        # ======================================================
        self.news_keywords = [
            # Chinese
            "新闻", "新聞", "最新消息", "頭條", "头条", "熱點", "热点",
            "最近发生", "最新事件", "近期事件",
            # English
            "news", "breaking", "headline", "latest", "recent event"
        ]

        self.search_keywords = [
            # Chinese
            "搜索", "查一下", "查找", "查詢", "查询",
            "是谁", "是誰", "是什么", "是什麼",
            # English
            "search", "who is", "what is", "look up",
        ] + self.news_keywords

        # ======================================================
        # Multimodal（英 / 简 / 繁）
        # ======================================================
        self.multimodal_keywords = [
            "图片", "圖片", "图像", "圖像", "照片",
            "识别", "識別", "看图", "看圖", "图里是什么", "圖裡是什麼",
            "pdf", "扫描件", "掃描件", "ocr",
            # English
            "image", "picture", "photo", "recognize"
        ]


    # -------------------------------------------------
    # Keyword scoring
    # -------------------------------------------------
    def score_keywords(self, query: str, keywords: List[str], weight=1.0) -> float:
        return sum(weight for k in keywords if k in query)


    # -------------------------------------------------
    # Main logic
    # -------------------------------------------------
    def recognize_intent(self, query: str, has_file: bool = False) -> Dict:
        q = query.lower().strip()

        # =============================
        # 1. 文件 → multimodal
        # =============================
        if has_file:
            return {
                "intent": "multimodal",
                "domains": ["file", "ocr", "vision"],
                "confidence": 0.95,
                "requires_web_search": False
            }

        # =============================
        # 2. 新闻 → 强制 Web
        # =============================
        if any(k in q for k in self.news_keywords):
            return {
                "intent": "web_search",
                "domains": ["web_search"],
                "confidence": 0.9,
                "requires_web_search": True
            }

        # =============================
        # 3. 打分系统
        # =============================
        scores = {
            "finance": 0,
            "weather": 0,
            "transport": 0,
            "web_search": 0,
            "general": 0.1
        }

        scores["finance"] += self.score_keywords(q, self.finance_keywords, 1.2)
        if self.stock_code_pattern.search(q):
            scores["finance"] += 1.5

        scores["weather"] += self.score_keywords(q, self.weather_keywords, 1.3)

        scores["transport"] += self.score_keywords(q, self.location_keywords)

        scores["web_search"] += self.score_keywords(q, self.search_keywords)

        # A→B 路线
        if ("从" in q or "從" in q) and ("到" in q):
            scores["transport"] += 1.5
        if re.search(r"怎么(走|去|到)|怎麼(走|去|到)", q):
            scores["transport"] += 1.2

        # =============================
        # 4. 文本中的多模态触发
        # =============================
        if any(k in q for k in self.multimodal_keywords):
            return {
                "intent": "multimodal",
                "domains": ["file", "ocr", "vision"],
                "confidence": 0.9,
                "requires_web_search": False
            }

        # =============================
        # 5. 选最高分意图
        # =============================
        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best == "finance":
            return {
                "intent": "finance",
                "domains": ["finance"],
                "confidence": min(1.0, 0.7 + best_score * 0.1),
                "requires_web_search": False
            }

        if best == "weather":
            return {
                "intent": "weather",
                "domains": ["weather"],
                "confidence": min(1.0, 0.7 + best_score * 0.1),
                "requires_web_search": False
            }

        if best == "transport":
            return {
                "intent": "transport",
                "domains": ["transport"],
                "confidence": min(1.0, 0.7 + best_score * 0.1),
                "requires_web_search": False
            }

        if best == "web_search":
            return {
                "intent": "web_search",
                "domains": ["web_search"],
                "confidence": min(1.0, 0.55 + best_score * 0.1),
                "requires_web_search": True
            }

        # fallback
        return {
            "intent": "general_knowledge",
            "domains": [],
            "confidence": 0.5,
            "requires_web_search": False
        }


    # 兼容旧接口
    def recognize(self, query: str, has_file: bool = False):
        info = self.recognize_intent(query, has_file)
        return {"intent": info.get("intent", "general_knowledge")}