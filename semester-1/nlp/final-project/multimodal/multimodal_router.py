import os
from multimodal.docx_processor import DocxProcessor
from multimodal.pdf_processor import PDFProcessor
from multimodal.image_processor import ImageProcessor
from multimodal.text_processor import TextProcessor


class MultimodalRouter:

    def __init__(self):
        self.docx = DocxProcessor()
        self.pdf = PDFProcessor()
        self.img = ImageProcessor()
        self.txt = TextProcessor()

    def load(self, path: str):
        """自动判断并提取文本"""
        if not os.path.exists(path):
            return {"error": f"文件不存在: {path}"}

        path_lower = path.lower()

        if path_lower.endswith(".docx"):
            text = self.docx.extract_text(path)
            return {"type": "docx", "text": text}

        if path_lower.endswith(".pdf"):
            text = self.pdf.extract_text(path)
            return {"type": "pdf", "text": text}

        if any(path_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]):
            text = self.img.extract_text(path)
            return {"type": "image", "text": text}

        if path_lower.endswith(".txt") or path_lower.endswith(".md"):
            text = self.txt.extract_text(path)
            return {"type": "text", "text": text}

        return {"error": "不支持的文件格式"}