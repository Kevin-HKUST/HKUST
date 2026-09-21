import os
import glob
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx
from config import Config  # 导入配置

class RAGEngine:
    def __init__(self, docs_dir="data/docs", embed_dir="data/embeddings", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_dir = docs_dir
        self.embed_dir = embed_dir
        os.makedirs(embed_dir, exist_ok=True)

        self.chunk_size = 400
        self.chunk_overlap = 80

        # embed model（移除 mirror 参数，保留缓存目录）
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"trust_remote_code": True},
            cache_folder=os.path.join(embed_dir, "model_cache")  # 模型缓存目录
        )

        # FAISS index
        self.index_path = os.path.join(embed_dir, "vector.index")
        self.meta_path = os.path.join(embed_dir, "meta.pkl")

        self.index = None
        self.metadata = []

        self._load_or_create_index()

    # ---------------------------------------------------------
    # Load PDF/DOCX/TXT
    # ---------------------------------------------------------
    def load_documents(self):
        texts = []
        files = glob.glob(os.path.join(self.docs_dir, "*"))

        for file in files:
            if file.lower().endswith(".pdf"):
                texts.extend(self._load_pdf(file))
            elif file.lower().endswith(".docx"):
                texts.extend(self._load_docx(file))
            elif file.lower().endswith(".txt"):
                texts.extend(self._load_txt(file))

        return texts

    def _load_pdf(self, path):
        reader = PdfReader(path)
        output = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                output.append(txt)
        return output

    def _load_docx(self, path):
        doc = docx.Document(path)
        return [p.text for p in doc.paragraphs if p.text.strip()]

    def _load_txt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------
    def create_chunks(self, texts):
        chunks = []
        for t in texts:
            words = t.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk = " ".join(words[i:i + self.chunk_size])
                chunks.append(chunk)
        return chunks

    # ---------------------------------------------------------
    # Embedding + FAISS
    # ---------------------------------------------------------
    def _build_index(self, chunks):
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # store
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(chunks, f)

        self.index = index
        self.metadata = chunks

    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            print("[RAG] No index found, need to build ...")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def rebuild(self):
        print("[RAG] Rebuilding index...")

        docs = self.load_documents()
        chunks = self.create_chunks(docs)
        self._build_index(chunks)

        print(f"[RAG] Built {len(chunks)} chunks")

    def query(self, text, top_k=5):
        if not self.index or len(self.metadata) == 0:
            return []

        embedding = self.model.encode([text]).astype("float32")
        D, I = self.index.search(embedding, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results