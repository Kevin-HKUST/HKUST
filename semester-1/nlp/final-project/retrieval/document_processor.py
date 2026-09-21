import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF
from config import Config
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """处理目录中的所有文档"""
        documents = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_docs = self.process_file(file_path)
                documents.extend(file_docs)

        logger.info(f"Processed {len(documents)} chunks from {directory_path}")
        return documents

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """处理单个文件"""
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path)
        elif file_path.lower().endswith('.md'):
            return self._process_markdown(file_path)
        elif file_path.lower().endswith('.txt'):
            return self._process_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []

    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """处理PDF文件"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # 分块
            chunks = self._split_text(text)

            # 构建文档对象
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": file_path,
                        "page": i + 1,
                        "type": "pdf"
                    },
                    "id": f"pdf_{os.path.basename(file_path)}_{i}"
                })

            return documents
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []

    def _process_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """处理Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 分块
            chunks = self._split_text(text)

            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": file_path,
                        "type": "markdown"
                    },
                    "id": f"md_{os.path.basename(file_path)}_{i}"
                })

            return documents
        except Exception as e:
            logger.error(f"Error processing Markdown {file_path}: {e}")
            return []

    def _process_text(self, file_path: str) -> List[Dict[str, Any]]:
        """处理文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            chunks = self._split_text(text)

            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": file_path,
                        "type": "text"
                    },
                    "id": f"txt_{os.path.basename(file_path)}_{i}"
                })

            return documents
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def _split_text(self, text: str) -> List[str]:
        """文本分块"""
        # 简单的按句子和长度分块
        sentences = re.split(r'(?<=[。！？\.!?])', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # 如果块太大，强制分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # 按换行符分割
                sub_chunks = [chunk[i:i + self.chunk_size] for i in range(0, len(chunk), self.chunk_size)]
                final_chunks.extend(sub_chunks)

        return final_chunks