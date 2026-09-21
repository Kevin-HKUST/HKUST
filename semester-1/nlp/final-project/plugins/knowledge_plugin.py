from plugins.base_plugin import BasePlugin
import logging

logger = logging.getLogger(__name__)

class KnowledgePlugin(BasePlugin):
    """
    简单知识插件：
    只有当 RAG 没有命中内容时，fallback 到此插件。
    """
    def __init__(self):
        super().__init__("knowledge_plugin")

    def execute(self, query: str, context=None):
        try:
            # 默认行为：直接返回“无本地知识”
            # 工作流会将这个输出交给 LLM 继续回答
            return self.format_output(
                data=f"[KnowledgePlugin] 无本地知识匹配：{query}",
                confidence=0.5,
                metadata={"success": True}
            )

        except Exception as e:
            logger.error(f"KnowledgePlugin error: {e}")
            return self.format_output(
                data=f"[KnowledgePlugin Error] {e}",
                confidence=0.0,
                metadata={"success": False}
            )