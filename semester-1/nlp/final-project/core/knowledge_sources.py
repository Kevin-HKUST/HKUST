"""Collect seed texts for the vector store.

Kept free of LangChain / FAISS so tests can exercise it offline.
"""

import json
import os
from typing import List

KNOWLEDGE_DIR = os.path.join("data", "knowledge_base")
KNOWLEDGE_JSON = os.path.join("data", "knowledge.json")


def collect_source_texts(
    knowledge_dir: str = KNOWLEDGE_DIR,
    json_path: str = KNOWLEDGE_JSON,
) -> List[str]:
    """Return seed texts from files that actually ship with the project.

    Reads every ``.md`` / ``.txt`` under ``knowledge_dir`` and, if present,
    ``knowledge.json`` as an optional extra. Missing paths are skipped.
    """
    texts: List[str] = []

    if os.path.isdir(knowledge_dir):
        for name in sorted(os.listdir(knowledge_dir)):
            if not name.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(knowledge_dir, name)
            with open(path, encoding="utf-8") as handle:
                body = handle.read().strip()
            if body:
                texts.append(body)

    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                if text.strip():
                    texts.append(text.strip())

    return texts
