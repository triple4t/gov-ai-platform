"""Retrieve codebase context for a project."""

import logging
from pathlib import Path

from app.core.config import settings
from app.prd_platform.database import (
    get_chunks_by_indices,
    get_faiss_idx_to_chunk,
    get_priority_bootstrap_chunks,
    get_project,
)
from app.prd_platform.embeddings import embed_query
from app.prd_platform.faiss_store import load_index, search

logger = logging.getLogger(__name__)


def _pack_file_blocks(rows: list[tuple[str, str]], budget: int) -> tuple[str, int]:
    """Join ### path blocks until char budget; returns (text, chars_used)."""
    parts: list[str] = []
    total = 0
    for fp, content in rows:
        block = f"### {fp}\n{content}\n"
        if total + len(block) > budget:
            remain = budget - total
            if remain > 200:
                parts.append(block[:remain] + "\n[truncated]\n")
                total = budget
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts), total


def retrieve_context(
    project_id: str,
    query: str,
    *,
    top_k: int | None = None,
    max_chars: int | None = None,
) -> str:
    proj = get_project(project_id)
    if not proj or proj["status"] != "ready":
        return ""
    faiss_path = proj.get("faiss_path")
    if not faiss_path:
        return ""
    index = load_index(Path(faiss_path))
    if index is None or index.ntotal == 0:
        return ""

    qv, _ = embed_query(query[:4000])
    k = top_k if top_k is not None else settings.PRD_PLATFORM_RAG_TOP_K
    scores, idxs = search(index, qv, k)
    flat_idx = [int(i) for i in idxs[0] if i >= 0]
    # Preserve similarity order (SQLite IN order is undefined).
    order_map = {idx: rank for rank, idx in enumerate(flat_idx)}
    idx_to_chunk = get_faiss_idx_to_chunk(project_id, flat_idx)
    sorted_idx = sorted(idx_to_chunk.keys(), key=lambda i: order_map.get(i, 999999))
    rows = [idx_to_chunk[i] for i in sorted_idx if i in idx_to_chunk]

    budget = max_chars if max_chars is not None else settings.PRD_PLATFORM_RAG_MAX_CHARS
    text, _ = _pack_file_blocks(rows, budget)
    return text


def retrieve_context_breadth(
    project_id: str,
    queries: list[str],
    *,
    top_k_per_query: int,
    max_chars: int,
    max_chunks_per_file: int,
    exclude_indices: frozenset[int] | None = None,
) -> str:
    """
    Multi-query vector search, merge by best score per chunk, cap chunks per file, pack by score.
    """
    exclude_indices = exclude_indices or frozenset()
    proj = get_project(project_id)
    if not proj or proj["status"] != "ready":
        return ""
    faiss_path = proj.get("faiss_path")
    if not faiss_path:
        return ""
    index = load_index(Path(faiss_path))
    if index is None or index.ntotal == 0:
        return ""

    merged_scores: dict[int, float] = {}
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        qv, _ = embed_query(q[:4000])
        scores, idxs = search(index, qv, top_k_per_query)
        row_scores = scores[0]
        row_idxs = idxs[0]
        for rank in range(len(row_idxs)):
            ii = int(row_idxs[rank])
            if ii < 0 or ii in exclude_indices:
                continue
            s = float(row_scores[rank])
            merged_scores[ii] = max(merged_scores.get(ii, -1e30), s)

    if not merged_scores:
        return ""

    ordered_idx = sorted(merged_scores.keys(), key=lambda i: merged_scores[i], reverse=True)
    idx_to_chunk = get_faiss_idx_to_chunk(project_id, ordered_idx)

    per_file: dict[str, int] = {}
    rows_ordered: list[tuple[str, str]] = []
    for idx in ordered_idx:
        if idx not in idx_to_chunk:
            continue
        fp, content = idx_to_chunk[idx]
        n = per_file.get(fp, 0)
        if n >= max_chunks_per_file:
            continue
        per_file[fp] = n + 1
        rows_ordered.append((fp, content))

    text, _ = _pack_file_blocks(rows_ordered, max_chars)
    return text


def build_deep_codebase_queries(capability_title: str, answers_block: str) -> list[str]:
    """Varied retrieval queries so application source surfaces, not only one file type."""
    base = (answers_block or "").strip()[:2000]
    ct = (capability_title or "").strip()
    return [
        f"{ct} technical documentation project overview setup modules {base}",
        f"application source code implementation API routes handlers services {base}",
        f"frontend UI components pages React TypeScript JavaScript {base}",
        f"backend server Python FastAPI Flask database models ORM {base}",
        f"configuration docker deployment environment scripts CI {base}",
    ]


def retrieve_deep_codebase_context(
    project_id: str,
    *,
    capability_title: str,
    answers_block: str,
) -> str:
    """
    Priority paths (README, package.json, main/app entrypoints, configs) plus breadth RAG.
    Used for tech_docs, architecture, and code_review.
    """
    max_total = settings.PRD_TECH_DOCS_RAG_MAX_CHARS
    pri_char_cap = min(settings.PRD_TECH_DOCS_PRIORITY_MAX_CHARS, max_total // 2)
    priority_rows = get_priority_bootstrap_chunks(
        project_id,
        settings.PRD_TECH_DOCS_PRIORITY_MAX_CHUNKS,
    )
    excluded = frozenset(idx for idx, _, _ in priority_rows)
    pri_blocks = [(fp, content) for _, fp, content in priority_rows]
    priority_text, used_pri = _pack_file_blocks(pri_blocks, pri_char_cap)

    sep = "\n\n## Additional repository context (vector search)\n\n"
    breadth_budget = max(0, max_total - used_pri - len(sep))

    queries = build_deep_codebase_queries(capability_title, answers_block)
    breadth_text = retrieve_context_breadth(
        project_id,
        queries,
        top_k_per_query=settings.PRD_TECH_DOCS_RAG_TOP_K_PER_QUERY,
        max_chars=breadth_budget,
        max_chunks_per_file=settings.PRD_TECH_DOCS_MAX_CHUNKS_PER_FILE,
        exclude_indices=excluded,
    )

    if priority_text and breadth_text:
        return priority_text + sep + breadth_text
    if priority_text:
        return priority_text
    return breadth_text
