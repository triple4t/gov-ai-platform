"""SQLite persistence for PRD platform projects and chunk metadata."""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import settings
from app.prd_platform.neo4j_graph import try_delete_project_graph

logger = logging.getLogger(__name__)


def _conn():
    return sqlite3.connect(settings.PRD_PLATFORM_DB_FILE, check_same_thread=False)


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
