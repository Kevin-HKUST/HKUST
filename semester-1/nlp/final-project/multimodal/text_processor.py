import logging
import os

logger = logging.getLogger(__name__)


class TextProcessor:
    """轻量版 TXT 文本提取器（方案 A）"""

    def __init__(self):
        self.supported_formats = [".txt", ".md"]

    def extract_text(self, file_path: str) -> str:
        """提取纯文本"""
        if not os.path.exists(file_path):
            logger.error(f"[TextProcessor] 文件不存在: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            logger.info(f"[TextProcessor] 成功提取文本：{len(text)} 字符")
            return text
        except Exception as e:
            logger.error(f"[TextProcessor] 提取文本失败: {e}")
            return ""

    def analyze_text(self, file_path: str):
        """轻量分析：返回基本结构"""
        content = self.extract_text(file_path)
        return {
            "length": len(content),
            "preview": content[:200],
            "full_text": content
        }

    def is_supported_format(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.supported_formats)