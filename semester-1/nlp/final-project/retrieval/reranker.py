import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder
import logging
from config import Config  # 导入配置

logger = logging.getLogger(__name__)

class Reranker:
    """重排序器"""

    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        try:
            # 配置模型参数（移除 mirror 参数，保留缓存目录）
            self.model = CrossEncoder(
                model_name,
                trust_remote_code=True,
                cache_folder=os.path.join("data", "reranker_cache")  # 缓存目录
            )
            logger.info(f"Loaded reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """对文档进行重排序"""
        if not self.model or not documents:
            return documents[:top_k]

        try:
            # 准备输入对
            pairs = [(query, doc['text']) for doc in documents]

            # 计算分数
            scores = self.model.predict(pairs)

            # 组合分数和文档
            scored_docs = list(zip(scores, documents))

            # 按分数降序排序
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # 返回top_k
            reranked_docs = [doc for score, doc in scored_docs[:top_k]]

            logger.info(f"Reranked {len(documents)} documents, top {top_k} selected")
            return reranked_docs
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]