import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from core.knowledge_sources import KNOWLEDGE_DIR, KNOWLEDGE_JSON, collect_source_texts

logger = logging.getLogger(__name__)

INDEX_DIR = os.path.join("data", "faiss_index")


class VectorStore:
    """LangChain + FAISS store used by the live query path.

    Seeds itself from ``data/knowledge_base`` when no persisted index exists.
    ``rebuild()`` discards the on-disk index and reseeds from those files.
    """

    def __init__(
        self,
        knowledge_dir: str = KNOWLEDGE_DIR,
        json_path: str = KNOWLEDGE_JSON,
        db_path: str = INDEX_DIR,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.knowledge_dir = knowledge_dir
        self.json_path = json_path
        self.db_path = db_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or Config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            cache_folder=os.path.join("data", "embedding_cache"),
        )
        self.vectorstore = self._load_index()
        if self.vectorstore is None:
            self._seed_from_sources()

    def rebuild(self) -> int:
        """Delete the persisted index and rebuild it from source files."""
        self.delete_collection()
        return self._seed_from_sources()

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        if not docs:
            return []

        chunks = []
        for item in docs:
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            if not text:
                continue
            chunks.extend(
                self.text_splitter.create_documents([text], metadatas=[metadata])
            )

        if not chunks:
            return []

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.save_local(self.db_path)
        return [chunk.metadata.get("id", "") for chunk in chunks]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            logger.warning("[VectorStore] empty index")
            return []
        try:
            results = self.vectorstore.similarity_search(query, k=top_k)
            return [
                {"text": result.page_content, "metadata": result.metadata}
                for result in results
            ]
        except Exception as exc:
            logger.error("[VectorStore] search failed: %s", exc)
            return []

    def delete_collection(self) -> None:
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.vectorstore = None
        logger.info("[VectorStore] deleted index")

    def _load_index(self):
        if not os.path.exists(self.db_path):
            return None
        try:
            return FAISS.load_local(
                self.db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error("[VectorStore] failed to load FAISS index: %s", exc)
            return None

    def _seed_from_sources(self) -> int:
        texts = collect_source_texts(self.knowledge_dir, self.json_path)
        if not texts:
            logger.warning(
                "[VectorStore] no seed documents under %s or %s",
                self.knowledge_dir,
                self.json_path,
            )
            return 0

        chunks = self.text_splitter.create_documents(texts)
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.vectorstore.save_local(self.db_path)
        logger.info("[VectorStore] seeded %s chunks", len(chunks))
        return len(chunks)
