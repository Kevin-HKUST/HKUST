import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from core.intent_recognizer import IntentRecognizer


class TestIntentRecognition(unittest.TestCase):
    """Intent routing. Runs offline: no network, no API key needed."""

    def setUp(self):
        self.recognizer = IntentRecognizer()

    def test_expected_intents(self):
        cases = [
            ("今天天气怎么样", "weather"),
            ("明天天气如何", "weather"),
            ("NVIDIA股价多少", "finance"),
            ("怎么去机场", "transport"),
            ("1+1等于多少", "math"),
            ("帮我计算 12 * 7", "math"),
            ("最新新闻有哪些", "web_search"),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(self.recognizer.recognize_intent(query)["intent"], expected)

    def test_bare_time_word_does_not_force_weather(self):
        """Regression: bare "今天" used to be a weather keyword, which dragged
        finance and other queries into the weather branch."""
        result = self.recognizer.recognize_intent("今天股价怎么样")
        self.assertEqual(result["intent"], "finance")

    def test_file_input_routes_to_multimodal(self):
        result = self.recognizer.recognize_intent("这是什么", has_file=True)
        self.assertEqual(result["intent"], "multimodal")

    def test_unknown_query_falls_back_to_knowledge(self):
        result = self.recognizer.recognize_intent("介绍一下香港科技大学")
        self.assertEqual(result["intent"], "knowledge")

    def test_recognize_alias_preserves_full_payload(self):
        """Regression: the alias used to drop everything but the label."""
        result = self.recognizer.recognize("今天天气怎么样")
        for key in ("intent", "domains", "confidence", "requires_web_search"):
            self.assertIn(key, result)


class TestKnowledgeSources(unittest.TestCase):
    """Seed-document collection. Offline: no embeddings, no API key."""

    def test_ships_the_fictional_knowledge_base(self):
        from core.knowledge_sources import collect_source_texts

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        texts = collect_source_texts(
            knowledge_dir=os.path.join(root, "data", "knowledge_base"),
            json_path=os.path.join(root, "data", "knowledge.json"),
        )
        self.assertTrue(texts, "data/knowledge_base should seed at least one document")
        self.assertTrue(any("Sereleia" in text for text in texts))

    def test_missing_paths_are_skipped(self):
        from core.knowledge_sources import collect_source_texts

        self.assertEqual(
            collect_source_texts(knowledge_dir="no-such-dir", json_path="no-such.json"),
            [],
        )


class TestIntentContract(unittest.TestCase):
    """Every intent the recognizer emits must be handled by WorkflowEngine."""

    def test_labels_match_engine_branches(self):
        from core.intents import ALL_INTENTS

        recognizer = IntentRecognizer()
        emitted = set()
        for query in ("今天天气怎么样", "NVIDIA股价多少", "怎么去机场",
                      "1+1等于多少", "最新新闻", "介绍一下香港科技大学"):
            emitted.add(recognizer.recognize_intent(query)["intent"])
        emitted.add(recognizer.recognize_intent("这是什么", has_file=True)["intent"])

        unknown = emitted - set(ALL_INTENTS)
        self.assertFalse(unknown, f"labels not declared in core.intents: {unknown}")

    def test_every_plugin_intent_has_a_registered_plugin(self):
        """Regression: "math" was dispatched by the engine but the recognizer
        never emitted it, so CalculatorPlugin was unreachable."""
        from core.intents import PLUGIN_INTENTS

        recognizer = IntentRecognizer()
        probes = {
            "weather": "今天天气怎么样",
            "finance": "NVIDIA股价多少",
            "transport": "怎么去机场",
            "math": "1+1等于多少",
            "web_search": "最新新闻",
        }
        for intent in PLUGIN_INTENTS:
            with self.subTest(intent=intent):
                self.assertIn(intent, probes, f"no probe query for {intent}")
                self.assertEqual(
                    recognizer.recognize_intent(probes[intent])["intent"], intent
                )


@unittest.skipUnless(Config.has_llm_credentials(), "HKGAI_API_KEY not configured")
class TestLLMClient(unittest.TestCase):
    """Network test. Skipped unless credentials are present."""

    def test_chat_returns_content(self):
        from core.llm_client import LLMClient

        result = LLMClient().chat("You are a test assistant.", "Reply with OK.")
        self.assertIn("content", result)


if __name__ == '__main__':
    unittest.main()
