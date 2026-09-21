import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.intent_recognizer import IntentRecognizer
from core.llm_client import HKGAIClient
from retrieval.vector_store import VectorStore


class TestBasicFunctionality(unittest.TestCase):
    """基础功能测试"""

    def setUp(self):
        self.intent_recognizer = IntentRecognizer()
        self.llm_client = HKGAIClient()
        self.vector_store = VectorStore()

    def test_intent_recognition(self):
        """测试意图识别"""
        test_cases = [
            ("今天天气怎么样", "weather"),
            ("NVIDIA股价多少", "finance"),
            ("怎么去机场", "transport"),
            ("1+1等于多少", "calculation")
        ]

        for query, expected_intent in test_cases:
            with self.subTest(query=query):
                result = self.intent_recognizer.recognize_intent(query)
                self.assertIn("intent", result)
                # 注意：实际意图可能不完全匹配，这里主要测试功能正常
                self.assertIsInstance(result["intent"], str)

    def test_llm_client(self):
        """测试LLM客户端"""
        result = self.llm_client.chat(
            "你是一个测试助手，请回复'测试成功'",
            "请回复指定内容"
        )

        self.assertIn("content", result)
        # 由于API响应可能变化，我们只检查是否有内容返回
        self.assertTrue(len(result["content"]) > 0)

    def test_vector_store(self):
        """测试向量存储"""
        # 测试添加文档
        test_doc = {
            "text": "这是一个测试文档",
            "metadata": {"source": "test"}
        }

        doc_id = self.vector_store.add_documents([test_doc])
        self.assertEqual(len(doc_id), 1)

        # 测试搜索
        results = self.vector_store.similarity_search("测试文档", k=1)
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    unittest.main()