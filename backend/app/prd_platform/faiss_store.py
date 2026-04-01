"""FAISS index build / persist / search."""

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(vectors: list[list[float]], dim: int) -> faiss.Index:
    if not vectors:
        raise ValueError("No vectors to index")
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    return index


def save_index(index: faiss.Index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index | None:
    if not path.is_file():
        return None
    return faiss.read_index(str(path))


def search(index: faiss.Index, query_vec: list[float], top_k: int) -> tuple[np.ndarray, np.ndarray]:
    dim = index.d
    q = np.array([query_vec], dtype="float32")
    faiss.normalize_L2(q)
    return index.search(q, min(top_k, index.ntotal))
