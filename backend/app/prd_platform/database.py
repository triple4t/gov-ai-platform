"""SQLite persistence for PRD platform projects and chunk metadata."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.prd_platform.neo4j_graph import try_delete_project_graph

logger = logging.getLogger(__name__)


def _conn():
    conn = sqlite3.connect(settings.PRD_PLATFORM_DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_prd_db():
    c = _conn()
    try:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                root_path TEXT,
                faiss_path TEXT,
                embedding_dim INTEGER,
                chunk_count INTEGER DEFAULT 0,
                error_message TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id)")
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('chatbot', 'hybrid_rag')),
                rag_document_id TEXT,
                conversation_summary TEXT NOT NULL DEFAULT '',
                rollup_message_index INTEGER NOT NULL DEFAULT 0,
                rag_indexed INTEGER NOT NULL DEFAULT 0,
                rag_last_file_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                client_msg_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta_json TEXT,
                FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE,
                UNIQUE(thread_id, seq)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id)")
        c.commit()
    finally:
        c.close()


def cleanup_expired_projects():
    """Remove projects older than PRD_PLATFORM_RETENTION_DAYS from DB and disk."""
    days = settings.PRD_PLATFORM_RETENTION_DAYS
    if days <= 0:
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    c = _conn()
    try:
        rows = c.execute(
            "SELECT id, root_path, faiss_path FROM projects WHERE created_at < ?",
            (cutoff,),
        ).fetchall()
        for pid, root, faiss in rows:
            try_delete_project_graph(pid)
            delete_project_disk(pid, root, faiss)
            c.execute("DELETE FROM chunks WHERE project_id = ?", (pid,))
            c.execute("DELETE FROM projects WHERE id = ?", (pid,))
        if rows:
            c.commit()
            logger.info("PRD platform: cleaned %d expired project(s)", len(rows))
    finally:
        c.close()


def delete_project_disk(project_id: str, root_path: str | None, faiss_path: str | None):
    import shutil

    base = Path(settings.UPLOAD_DIR) / "prd_projects" / project_id
    if base.is_dir():
        try:
            shutil.rmtree(base, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to rmtree %s: %s", base, e)
    if root_path:
        p = Path(root_path)
        if p.is_dir() and "prd_projects" in str(p):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
    if faiss_path:
        fp = Path(faiss_path)
        if fp.is_file():
            try:
                fp.unlink()
            except Exception:
                pass


def upsert_project(
    project_id: str,
    status: str,
    root_path: str,
    faiss_path: str,
    embedding_dim: int,
    chunk_count: int,
    error_message: str | None = None,
):
    now = datetime.now(timezone.utc).isoformat()
    c = _conn()
    try:
        c.execute(
            """
            INSERT INTO projects
            (id, status, created_at, updated_at, root_path, faiss_path, embedding_dim, chunk_count, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at,
                root_path = excluded.root_path,
                faiss_path = excluded.faiss_path,
                embedding_dim = excluded.embedding_dim,
                chunk_count = excluded.chunk_count,
                error_message = excluded.error_message
            """,
            (project_id, status, now, now, root_path, faiss_path, embedding_dim, chunk_count, error_message),
        )
        c.commit()
    finally:
        c.close()


def update_project_status(project_id: str, status: str, error_message: str | None = None):
    now = datetime.now(timezone.utc).isoformat()
    c = _conn()
    try:
        c.execute(
            "UPDATE projects SET status = ?, updated_at = ?, error_message = ? WHERE id = ?",
            (status, now, error_message, project_id),
        )
        c.commit()
    finally:
        c.close()


def delete_project(project_id: str):
    try_delete_project_graph(project_id)
    c = _conn()
    try:
        row = c.execute(
            "SELECT root_path, faiss_path FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row:
            delete_project_disk(project_id, row[0], row[1])
        c.execute("DELETE FROM chunks WHERE project_id = ?", (project_id,))
        c.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        c.commit()
    finally:
        c.close()


def clear_chunks(project_id: str):
    c = _conn()
    try:
        c.execute("DELETE FROM chunks WHERE project_id = ?", (project_id,))
        c.commit()
    finally:
        c.close()


def insert_chunks(project_id: str, rows: list[tuple[int, str, str]]):
    """rows: (faiss_idx, file_path, content)"""
    c = _conn()
    try:
        c.executemany(
            "INSERT INTO chunks (project_id, faiss_idx, file_path, content) VALUES (?, ?, ?, ?)",
            [(project_id, fi, fp, txt) for fi, fp, txt in rows],
        )
        c.commit()
    finally:
        c.close()


def get_chunks_by_indices(project_id: str, indices: list[int]) -> list[tuple[str, str]]:
    if not indices:
        return []
    c = _conn()
    try:
        qmarks = ",".join("?" * len(indices))
        cur = c.execute(
            f"SELECT file_path, content FROM chunks WHERE project_id = ? AND faiss_idx IN ({qmarks})",
            [project_id, *indices],
        )
        return list(cur.fetchall())
    finally:
        c.close()


def get_faiss_idx_to_chunk(project_id: str, indices: list[int]) -> dict[int, tuple[str, str]]:
    """Map FAISS row index -> (file_path, content)."""
    if not indices:
        return {}
    c = _conn()
    try:
        qmarks = ",".join("?" * len(indices))
        cur = c.execute(
            f"SELECT faiss_idx, file_path, content FROM chunks WHERE project_id = ? AND faiss_idx IN ({qmarks})",
            [project_id, *indices],
        )
        return {int(r[0]): (r[1], r[2]) for r in cur.fetchall()}
    finally:
        c.close()


# SQL fragments for README, manifests, entrypoints (LOWER paths).
_PRIORITY_PATH_CLAUSES = [
    "LOWER(file_path) LIKE '%readme%'",
    "LOWER(file_path) LIKE '%package.json%'",
    "LOWER(file_path) LIKE '%pyproject.toml%'",
    "LOWER(file_path) LIKE '%requirements.txt%'",
    "LOWER(file_path) LIKE '%cargo.toml%'",
    "LOWER(file_path) LIKE '%go.mod%'",
    "LOWER(file_path) LIKE '%dockerfile%'",
    "LOWER(file_path) LIKE '%makefile%'",
    "LOWER(file_path) LIKE '%vite.config%'",
    "LOWER(file_path) LIKE '%webpack.config%'",
    "LOWER(file_path) LIKE '%tsconfig.json%'",
    "LOWER(file_path) = 'main.py' OR LOWER(file_path) LIKE '%/main.py'",
    "LOWER(file_path) = 'app.py' OR LOWER(file_path) LIKE '%/app.py'",
    "LOWER(file_path) LIKE '%/app.tsx' OR LOWER(file_path) LIKE '%/app.jsx'",
]


def get_priority_bootstrap_chunks(project_id: str, max_chunks: int) -> list[tuple[int, str, str]]:
    """
    One chunk per matching file path (lowest faiss_idx), up to max_chunks rows.
    Returns (faiss_idx, file_path, content).
    """
    if max_chunks <= 0:
        return []
    where = " OR ".join(f"({clause})" for clause in _PRIORITY_PATH_CLAUSES)
    sql = f"""
        SELECT faiss_idx, file_path, content FROM chunks
        WHERE project_id = ? AND ({where})
        ORDER BY faiss_idx ASC
    """
    c = _conn()
    try:
        cur = c.execute(sql, (project_id,))
        rows = cur.fetchall()
    finally:
        c.close()

    seen_fp: set[str] = set()
    out: list[tuple[int, str, str]] = []
    for faiss_idx, fp, content in rows:
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        out.append((int(faiss_idx), fp, content))
        if len(out) >= max_chunks:
            break
    return out


def get_project(project_id: str) -> dict | None:
    c = _conn()
    try:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None
    finally:
        c.close()


def delete_chat_thread(thread_id: str) -> bool:
    """Delete thread and messages. Returns True if a row was removed."""
    c = _conn()
    try:
        cur = c.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
        c.commit()
        return cur.rowcount > 0
    finally:
        c.close()


def get_chat_thread_full(thread_id: str) -> dict[str, Any] | None:
    """Return thread row plus messages list (id, role, content, meta?), or None if missing."""
    c = _conn()
    try:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM chat_threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["rag_indexed"] = bool(out.get("rag_indexed", 0))
        cur = c.execute(
            """
            SELECT client_msg_id, role, content, meta_json
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY seq ASC
            """,
            (thread_id,),
        )
        messages: list[dict[str, Any]] = []
        for r in cur.fetchall():
            mid, role, content, meta_raw = r[0], r[1], r[2], r[3]
            item: dict[str, Any] = {
                "id": int(mid) if mid is not None else 0,
                "role": role,
                "content": content or "",
            }
            if meta_raw:
                try:
                    parsed = json.loads(meta_raw)
                    if isinstance(parsed, dict):
                        item["meta"] = parsed
                except json.JSONDecodeError:
                    pass
            messages.append(item)
        out["messages"] = messages
        return out
    finally:
        c.close()


def save_chat_thread(
    thread_id: str,
    mode: str,
    rag_document_id: str | None,
    conversation_summary: str,
    rollup_message_index: int,
    rag_indexed: bool,
    rag_last_file_name: str,
    messages: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    c = _conn()
    try:
        c.execute("BEGIN")
        c.execute(
            """
            INSERT INTO chat_threads
            (id, mode, rag_document_id, conversation_summary, rollup_message_index,
             rag_indexed, rag_last_file_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                mode = excluded.mode,
                rag_document_id = excluded.rag_document_id,
                conversation_summary = excluded.conversation_summary,
                rollup_message_index = excluded.rollup_message_index,
                rag_indexed = excluded.rag_indexed,
                rag_last_file_name = excluded.rag_last_file_name,
                updated_at = excluded.updated_at
            """,
            (
                thread_id,
                mode,
                rag_document_id,
                conversation_summary or "",
                max(0, int(rollup_message_index)),
                1 if rag_indexed else 0,
                rag_last_file_name or "",
                now,
                now,
            ),
        )
        c.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
        for seq, m in enumerate(messages):
            mid = m.get("id")
            try:
                client_id = int(mid) if mid is not None else 0
            except (TypeError, ValueError):
                client_id = 0
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            meta = m.get("meta")
            meta_json = json.dumps(meta) if isinstance(meta, dict) else None
            c.execute(
                """
                INSERT INTO chat_messages
                (thread_id, seq, client_msg_id, role, content, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, seq, client_id, role, content, meta_json),
            )
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()
