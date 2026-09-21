import logging
import re
import time
from datetime import date
from typing import Any, Dict, List

from core.intent_recognizer import IntentRecognizer
from core.intents import KNOWLEDGE, PLUGIN_INTENTS
from core.vector_store import VectorStore
from core.llm_client import LLMClient
from multimodal.image_processor import ImageProcessor

# plugins
from plugins.web_search_plugin import WebSearchPlugin
from plugins.weather_plugin import WeatherPlugin
from plugins.calculator_plugin import CalculatorPlugin
from plugins.finance_plugin import FinancePlugin
from plugins.transport_plugin import TransportPlugin

# [Input: file.png] "question"
_MULTIMODAL_RE = re.compile(r'\s*\[Input:\s*(.+?)\s*\]\s*"(.*)"')


class WorkflowEngine:
    """Routes a query to a plugin or the vector store, then asks the LLM to
    synthesise an answer from whatever was retrieved.

    The pipeline is split into `retrieve()` and `synthesize()` so callers that
    need to time the two phases separately (see evaluation.evaluator) can drive
    them independently. `run()` is the convenience path that does both.
    """

    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.vector_store = VectorStore()
        self.llm = LLMClient()
        self.img = ImageProcessor()

        # Keys must be exactly the plugin intents; anything registered here
        # that the recognizer cannot emit is unreachable code.
        self.plugins = {
            "web_search": WebSearchPlugin(),
            "weather": WeatherPlugin(),
            "math": CalculatorPlugin(),
            "finance": FinancePlugin(),
            "transport": TransportPlugin(),
        }
        assert set(self.plugins) == set(PLUGIN_INTENTS), (
            f"plugin registry and core.intents.PLUGIN_INTENTS disagree: "
            f"{set(self.plugins) ^ set(PLUGIN_INTENTS)}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, query: str, images: List[Any] = None) -> Dict[str, Any]:
        query, images = self._parse_multimodal(query, images)
        query = self._append_image_text(query, images)
        retrieved = self.retrieve(query, images)
        return self.synthesize(query, retrieved)

    def execute_workflow(self, query: str, images: List[Any] = None) -> Dict[str, Any]:
        """Alias kept for older call sites."""
        return self.run(query, images)

    def retrieve(self, query: str, images: List[Any] = None) -> Dict[str, Any]:
        """Resolve the query to source material, without calling the LLM."""
        intent_info = self.intent_recognizer.recognize_intent(query, has_file=bool(images))
        intent = intent_info.get("intent")
        logging.info(f"[Workflow] Intent = {intent} (confidence={intent_info.get('confidence')})")

        if intent in PLUGIN_INTENTS:
            out = self._run_plugin(intent, query)
        elif intent == KNOWLEDGE:
            out = self._run_knowledge(query)
        else:
            # multimodal and anything unrecognised fall back to the vector store
            out = self._run_vector_fallback(query)

        out.setdefault("metadata", {})["intent"] = intent_info
        return out

    def synthesize(self, query: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
        """Turn retrieved material into a final answer."""
        start = time.time()
        logging.debug(f"[Workflow] Retrieved = {retrieved}")

        system_prompt = (
            f"你是一个智能助手。当前日期: {date.today().isoformat()}。\n"
            "你必须利用插件输出或图片转文本结果（如有）来回答问题。\n"
            "如果图片解析出现 error，请仅根据文本回答。\n"
        )
        user_prompt = (
            f"用户问题：{query}\n\n"
            f"插件输出/知识内容：\n{retrieved}\n"
        )

        llm_raw = self.llm.chat(system_prompt, user_prompt)
        ans = llm_raw.get("content", "").strip() or "（未能生成回答）"

        return {
            "final_answer": {
                "answer": ans,
                "confidence": retrieved.get("confidence", 0.5),
                "sources_used": retrieved,
            },
            "processing_time": time.time() - start,
        }

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _parse_multimodal(self, query: str, images: List[Any]):
        """Accept both `[Input: file] "question"` and `file | question`."""
        match = _MULTIMODAL_RE.match(query)
        if match:
            path, question = match.group(1).strip(), match.group(2).strip()
            logging.info(f"[Multimodal] Parsed image: {path}")
            return question, [{"path": path}]

        if "|" in query:
            path, question = (part.strip() for part in query.split("|", 1))
            logging.info(f"[Multimodal] Parsed legacy image: {path}")
            return question, [{"path": path}]

        return query, images

    def _append_image_text(self, query: str, images: List[Any]) -> str:
        if not images:
            return query

        path = images[0].get("path")
        if not path:
            return query

        logging.info("[Multimodal] Calling Vision API...")
        ocr_result = self.img.extract_text(path)

        if ocr_result and "error" not in ocr_result:
            logging.info("[Multimodal] OCR appended to query")
            return f"{query}\n\n[Extracted from image {path}]:\n{ocr_result}"

        logging.error(f"[Multimodal] OCR failed: {ocr_result}")
        return query

    # ------------------------------------------------------------------
    # Retrieval strategies
    # ------------------------------------------------------------------
    def _run_plugin(self, name: str, query: str) -> Dict[str, Any]:
        plugin = self.plugins.get(name)
        if not plugin:
            return self._run_web_fallback(query)

        out = plugin.run(query)
        if not out.get("metadata", {}).get("success", True):
            return self._run_web_fallback(query)
        return out

    def _run_web_fallback(self, query: str) -> Dict[str, Any]:
        out = self.plugins["web_search"].run(query)
        if not out.get("metadata", {}).get("success", True):
            return self._run_vector_fallback(query)
        return out

    def _run_vector_fallback(self, query: str) -> Dict[str, Any]:
        res = self.vector_store.search(query)
        return {
            "type": "knowledge" if res else "none",
            "content": res,
            "confidence": 0.5,
            "metadata": {"success": bool(res)},
        }

    def _run_knowledge(self, query: str) -> Dict[str, Any]:
        res = self.vector_store.search(query)
        if not res:
            # An empty local index is not an answer; try the web instead.
            return self._run_web_fallback(query)
        return {
            "type": "knowledge",
            "content": res,
            "confidence": 0.8,
            "metadata": {"success": True},
        }
