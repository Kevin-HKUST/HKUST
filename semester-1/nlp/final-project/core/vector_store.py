import os
import json
import shutil
import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import Config  # 导入配置

logger = logging.getLogger(__name__)

class VectorStore:
    """
    改进版 VectorStore：
    - 兼容旧 search() API
    - 支持 FAISS 持久化存储
    - 支持新增文档 add_documents()
    - 自动加载 data/knowledge.json
    """

    def __init__(self,
                 data_path="data/knowledge.json",
                 db_path="data/faiss_index",
                 chunk_size=300,
                 chunk_overlap=40):

        self.data_path = data_path
        self.db_path = db_path

        # Embedding 模型（移除 mirror 参数，保留缓存目录）
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True
            },
            cache_folder=os.path.join("data", "embedding_cache")  # 缓存目录
        )

        # 切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 尝试加载 FAISS 索引
        self.vectorstore = self._load_index()

        # 如果没有 index，则尝试导入 knowledge.json
        if self.vectorstore is None and os.path.exists(self.data_path):
            self._load_knowledge_json()

    # -----------------------
    # 加载 FAISS 索引
    # -----------------------
    def _load_index(self):
        if os.path.exists(self.db_path):
            try:
                return FAISS.load_local(
                    self.db_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"[VectorStore] 加载 FAISS 失败: {e}")
        return None

    # -----------------------
    # 导入旧版 JSON 知识库
    # -----------------------
    def _load_knowledge_json(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                docs = json.load(f)

            documents = []
            for d in docs:
                text = d.get("text", "")
                if not text:
                    continue
                documents.extend(self.text_splitter.create_documents([text]))

            if not documents:
                logger.warning("[VectorStore] 空 knowledge.json")
                return

            self.vectorstore = FAISS.from_documents(
                documents, self.embedding_model
            )
            self.vectorstore.save_local(self.db_path)

            logger.info(f"[VectorStore] 已加载 {len(documents)} 条知识片段")

        except Exception as e:
            logger.error(f"[VectorStore] JSON 加载失败: {e}")

    # -----------------------
    # 新增文档
    # -----------------------
    def add_documents(self, docs: List[Dict[str, Any]]):
        if not docs:
            return []

        chunks = []
        for d in docs:
            text = d.get("text", "")
            md = d.get("metadata", {})
            if not text:
                continue

            chunks.extend(
                self.text_splitter.create_documents([text], metadatas=[md])
            )

        if not chunks:
            return []

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.save_local(self.db_path)

        return [c.metadata.get("id", "") for c in chunks]

    # -----------------------
    # search() —— 兼容旧系统 API
    # -----------------------
    def search(self, query: str, top_k=3):
        if self.vectorstore is None:
            logger.warning("[VectorStore] 空向量库")
            return []

        try:
            results = self.vectorstore.similarity_search(query, k=top_k)
            return [
                {
                    "text": r.page_content,
                    "metadata": r.metadata,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"[VectorStore] 检索失败: {e}")
            return []

    # -----------------------
    # 删除整个 index
    # -----------------------
    def delete_collection(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.vectorstore = None
        logger.info("[VectorStore] 已删除向量库")