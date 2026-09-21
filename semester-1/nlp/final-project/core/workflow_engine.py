import logging
import re
from typing import Dict, Any, List

from core.intent_recognizer import IntentRecognizer
from core.vector_store import VectorStore
from core.llm_client import LLMClient
from multimodal.image_processor import ImageProcessor

# plugins
from plugins.web_search_plugin import WebSearchPlugin
from plugins.weather_plugin import WeatherPlugin
from plugins.calculator_plugin import CalculatorPlugin
from plugins.knowledge_plugin import KnowledgePlugin
from plugins.finance_plugin import FinancePlugin
from plugins.transport_plugin import TransportPlugin


class WorkflowEngine:
    """
    修复版 WorkflowEngine
    - 多模态安全解析
    - Vision OCR 错误保护
    - TransportPlugin 智能触发（避免误触）
    """

    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.vector_store = VectorStore()
        self.llm = LLMClient()
        self.img = ImageProcessor()

        self.plugins = {
            "web_search": WebSearchPlugin(),
            "weather": WeatherPlugin(),
            "math": CalculatorPlugin(),
            "knowledge": KnowledgePlugin(),
            "finance": FinancePlugin(),
            "transport": TransportPlugin(),
        }

    # ============================================================
    def execute_workflow(self, query: str, images: List[Any] = None):
        return self.run(query, images)

    # ============================================================
    def run(self, query: str, images: List[Any] = None) -> Dict[str, Any]:
        logging.info(f"[Workflow] Raw Input = {query}")

        # --------------------------------------------------------
        # 多模态格式：[Input: xx.png] "question"
        # --------------------------------------------------------
        image_path = None
        match = re.match(r'\s*\[Input:\s*(.+?)\s*\]\s*"(.*)"', query)
        if match:
            image_path = match.group(1).strip()
            query = match.group(2).strip()
            images = [{"path": image_path}]
            logging.info(f"[Multimodal] Parsed image: {image_path}, question={query}")

        # --------------------------------------------------------
        # 旧格式 filename | question
        # --------------------------------------------------------
        if not image_path and "|" in query:
            img_path, q = query.split("|", 1)
            image_path = img_path.strip()
            query = q.strip()
            images = [{"path": image_path}]
            logging.info(f"[Multimodal] Parsed legacy image: {image_path}, question={query}")

        # --------------------------------------------------------
        # 图片 OCR/Tagging
        # --------------------------------------------------------
        if images:
            img_path = images[0].get("path")
            if img_path:
                logging.info("[Multimodal] Calling Vision API...")
                ocr_result = self.img.extract_text(img_path)

                if ocr_result and "error" not in ocr_result:
                    query += (
                        f"\n\n[Extracted from image {img_path}]:\n"
                        f"{ocr_result}"
                    )
                    logging.info("[Multimodal] OCR appended to query")
                else:
                    logging.error(f"[Multimodal] OCR FAILED: {ocr_result}")

        logging.info(f"[Workflow] Final Query to LLM = {query}")

        # ================ Intent Detection ================
        intent_info = self.intent_recognizer.recognize(query)
        intent = intent_info.get("intent")
        logging.info(f"[Workflow] Intent = {intent}")

        # ================ TransportPlugin 触发条件 ================
        # 确保不误触图像任务
        transport_trigger = (
            not images and (
                "附近" in query
                or "最近" in query
                or "哪里有" in query
                or "kfc" in query.lower()
                or "肯德基" in query
                or "麦当劳" in query
            )
        )

        if transport_trigger:
            logging.info("[Workflow] Force Trigger: transport plugin")
            out = self.plugins["transport"].run(query)
            return self._llm_synthesis(query, out)

        # ================ 插件路径 ================
        if intent in ["weather", "math", "web_search", "finance", "transport"]:
            out = self._run_plugin(intent, query)
            return self._llm_synthesis(query, out)

        if intent == "knowledge":
            return self._llm_synthesis(query, self._run_knowledge(query))

        # ================ 否则向量 fallback ================
        return self._llm_synthesis(query, self._run_vector_fallback(query))

    # ============================================================
    def _run_plugin(self, name: str, query: str):
        plugin = self.plugins.get(name)
        if not plugin:
            return self._run_web_fallback(query)

        out = plugin.run(query)
        if not out.get("metadata", {}).get("success", True):
            return self._run_web_fallback(query)
        return out

    # ============================================================
    def _run_web_fallback(self, query: str):
        web = self.plugins.get("web_search")
        out = web.run(query)

        if not out.get("metadata", {}).get("success", True):
            return self._run_vector_fallback(query)

        return out

    # ============================================================
    def _run_vector_fallback(self, query: str):
        res = self.vector_store.search(query)
        return {
            "type": "knowledge" if res else "none",
            "content": res,
            "confidence": 0.5,
            "metadata": {"success": bool(res)},
        }

    # ============================================================
    def _run_knowledge(self, query: str):
        res = self.vector_store.search(query)
        return {
            "type": "knowledge",
            "content": res,
            "confidence": 0.8,
            "metadata": {"success": True},
        }

    # ============================================================
    def _llm_synthesis(self, query: str, plugin_output: Dict[str, Any]):
        import time
        start = time.time()

        logging.info(f"[DEBUG] Plugin Output = {plugin_output}")

        system_prompt = (
            "你是一个智能助手。当前日期: 2025-12-14。\n"
            "你必须利用插件输出或图片转文本结果（如有）来回答问题。\n"
            "如果图片解析出现 error，请仅根据文本回答。\n"
        )

        user_prompt = (
            f"用户问题：{query}\n\n"
            f"插件输出/知识内容：\n{plugin_output}\n"
        )

        llm_raw = self.llm.chat(system_prompt, user_prompt)
        ans = llm_raw.get("content", "").strip() or "（未能生成回答）"

        return {
            "final_answer": {
                "answer": ans,
                "confidence": plugin_output.get("confidence", 0.5),
                "sources_used": plugin_output,
            },
            "processing_time": time.time() - start,
        }