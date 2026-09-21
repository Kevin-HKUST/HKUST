import logging
import re
import requests
from typing import Dict, Any, List
from plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


class FinancePlugin(BasePlugin):
    """Finance Plugin: 全部行情统一使用 Yahoo Finance API（已修复 HK 误识别 + 全角点问题）"""

    def __init__(self):
        super().__init__("finance_plugin")
        self.capabilities = ["stock_quotes", "crypto_prices", "exchange_rates"]
        self.required_params = []

    # ---------------------------
    # 主执行入口
    # ---------------------------
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            qtype = self._determine_query_type(query)

            if qtype == "stock":
                return self._handle_stock_query(query)
            elif qtype == "crypto":
                return self._handle_crypto_query(query)
            elif qtype == "exchange_rate":
                return self._handle_exchange_rate_query(query)
            else:
                return self._handle_general_finance_query(query)

        except Exception as e:
            logger.error(f"Finance plugin error: {e}")
            return self.failure(str(e))

    # ---------------------------
    # 识别查询类型
    # ---------------------------
    def _determine_query_type(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in ["股票", "股价", "price", "market", "hk", "us", "ss", "sz"]):
            return "stock"
        if any(w in q for w in ["btc", "eth", "比特币", "以太坊", "crypto"]):
            return "crypto"
        if any(w in q for w in ["汇率", "exchange", "usd", "cny", "hkd", "eur"]):
            return "exchange_rate"

        return "general"

    # ---------------------------
    # Yahoo API 获取行情
    # ---------------------------
    def _get_stock_data_yahoo(self, symbol: str) -> Dict[str, Any]:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            resp = requests.get(url, headers=headers, timeout=8)
            data = resp.json()
            results = data.get("quoteResponse", {}).get("result", [])

            if not results:
                return {"error": "无行情数据"}

            q = results[0]

            return {
                "symbol": symbol,
                "price": q.get("regularMarketPrice"),
                "change": q.get("regularMarketChange"),
                "percent": q.get("regularMarketChangePercent"),
                "open": q.get("regularMarketOpen"),
                "high": q.get("regularMarketDayHigh"),
                "low": q.get("regularMarketDayLow"),
                "volume": q.get("regularMarketVolume"),
                "currency": q.get("currency"),
                "market_time": q.get("regularMarketTime"),
                "exchange": q.get("fullExchangeName"),
            }

        except Exception as e:
            logger.error(f"Yahoo API error for {symbol}: {e}")
            return {"error": f"API异常: {e}"}

    # ---------------------------
    # 股票查询
    # ---------------------------
    def _handle_stock_query(self, query: str) -> Dict[str, Any]:
        symbols = self._extract_stock_symbols(query)

        if not symbols:
            return self.failure("无法识别股票代码")

        results = {}
        for sym in symbols[:5]:
            data = self._get_stock_data_yahoo(sym)
            if "error" not in data:
                results[sym] = data

        if not results:
            return self.failure("无法获取股票数据")

        return self.success(
            data={"stocks": results},
            confidence=0.9,
            metadata={"source": "yahoo"}
        )

    # ---------------------------
    # ★ 修正后的股票代码提取（核心：标准化 + HK 修复）
    # ---------------------------
    def _extract_stock_symbols(self, query: str) -> List[str]:

        # 全角 → 半角
        def fullwidth_to_halfwidth(s: str) -> str:
            result = []
            for ch in s:
                code = ord(ch)
                if 0xFF01 <= code <= 0xFF5E:
                    result.append(chr(code - 0xFEE0))
                elif code == 0x3002:
                    result.append(".")
                else:
                    result.append(ch)
            return "".join(result)

        q = fullwidth_to_halfwidth(query).upper()
        found = []

        # 美股
        for m in re.findall(r"\b[A-Z]{1,5}\b", q):
            found.append(m)

        # 港股
        for m in re.findall(r"(\d{4,5})(?:\.HK|HK)", q):
            found.append(m + ".HK")

        # A股
        for m in re.findall(r"(60\d{4}|00\d{4}|30\d{4})", q):
            if m.startswith("6"):
                found.append(m + ".SS")
            else:
                found.append(m + ".SZ")

        # 移除误识别
        filtered = [s for s in found if s not in ["HK"]]

        return list(dict.fromkeys(filtered))

    # ---------------------------
    # Crypto
    # ---------------------------
    def _handle_crypto_query(self, query: str) -> Dict[str, Any]:
        pairs = []
        q = query.lower()

        if "btc" in q or "比特" in q:
            pairs.append("BTC-USD")
        if "eth" in q or "以太" in q:
            pairs.append("ETH-USD")

        if not pairs:
            pairs = ["BTC-USD", "ETH-USD"]

        results = {}
        for p in pairs:
            data = self._get_stock_data_yahoo(p)
            if "error" not in data:
                results[p] = data

        return self.success(
            data={"crypto": results},
            confidence=0.85,
            metadata={"source": "yahoo"}
        )

    # ---------------------------
    # 汇率
    # ---------------------------
    def _handle_exchange_rate_query(self, query: str) -> Dict[str, Any]:
        q = query.lower()

        symbol = "USDCNY=X"
        if "usd" in q and "hkd" in q:
            symbol = "USDHKD=X"
        elif "eur" in q and "usd" in q:
            symbol = "EURUSD=X"

        data = self._get_stock_data_yahoo(symbol)

        if "error" in data:
            return self.failure("无法获取汇率数据")

        return self.success(
            data={"exchange_rate": data},
            confidence=0.85,
            metadata={"pair": symbol}
        )

    # ---------------------------
    # 一般查询
    # ---------------------------
    def _handle_general_finance_query(self, query: str) -> Dict[str, Any]:
        return self.success(
            data={"message": "请提供股票代码，例如：AAPL、00388.HK、600519。"},
            confidence=0.6
        )

    # ---------------------------
    # 统一错误输出 → 使用 BasePlugin.failure()
    # ---------------------------
    def _handle_error(self, msg: str, error_type: str = "runtime_error") -> Dict[str, Any]:
        return self.failure(msg)