"""Persist PRD import-level CDG to Neo4j.

Schema (project scoped via ``projectId`` for uniform cleanup):

- ``(:Project { id, projectId })`` — ``id`` is the PRD ``project_id``.
- ``(:SourceFile { projectId, path, key })`` — repo-relative file path from :func:`build_cdg`.
- ``(:ExternalDep { projectId, name, key })`` — package names from synthetic ``ext:*`` CDG nodes.
- ``(:SourceFile)-[:IMPORTS { projectId }]->(:SourceFile | :ExternalDep)``

This mirrors :func:`build_cdg` import edges only, not a full AST call graph.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from app.core.config import settings
from app.prd_platform.cdg_builder import build_cdg

logger = logging.getLogger(__name__)

_driver_lock = threading.Lock()
_driver: Any = None

_WRITE_BATCH = 500


def _get_driver():
    global _driver
    with _driver_lock:
        if _driver is None:
            auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            _driver = GraphDatabase.driver(settings.NEO4J_URI, auth=auth)
        return _driver


def _session_kwargs() -> dict[str, Any]:
    if settings.NEO4J_DATABASE:
        return {"database": settings.NEO4J_DATABASE}
    return {}


def delete_project_graph(project_id: str) -> None:
    """Remove all graph elements for ``project_id``. No-op if Neo4j disabled."""
    if not settings.NEO4J_ENABLED:
        return
    drv = _get_driver()
    with drv.session(**_session_kwargs()) as session:
        session.run(
            "MATCH (n) WHERE n.projectId = $pid DETACH DELETE n",
            pid=project_id,
        )


def _neo4j_unreachable(exc: BaseException) -> bool:
    """True when broker is down / refused (avoid multi-page tracebacks for expected dev setups)."""
    try:
        from neo4j.exceptions import ServiceUnavailable as _Neo4jUnavailable
    except ImportError:
        _Neo4jUnavailable = None

    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    for e in chain:
        if isinstance(e, ConnectionRefusedError):
            return True
        if _Neo4jUnavailable is not None and isinstance(e, _Neo4jUnavailable):
            return True
    return False


def try_delete_project_graph(project_id: str) -> None:
    """Like :func:`delete_project_graph` but never raises (logs on failure)."""
    if not settings.NEO4J_ENABLED:
        return
    try:
        delete_project_graph(project_id)
    except Exception as e:
        if _neo4j_unreachable(e):
            logger.warning(
                "Neo4j: skipped delete for project %s — not reachable (%s). "
                "Start the database or set NEO4J_ENABLED=false.",
                project_id,
                e.__class__.__name__,
            )
        else:
            logger.warning("Neo4j: failed to delete graph for project %s", project_id, exc_info=True)


def sync_project_graph_to_neo4j(project_id: str, src_root: Path) -> None:
    """
    Replace Neo4j subgraph for ``project_id`` from a fresh :func:`build_cdg` at ``src_root``.
    Raises on Neo4j errors unless ``NEO4J_SYNC_FAIL_OPEN`` is true (then logs).
    """
    if not settings.NEO4J_ENABLED:
        return
    root = Path(src_root).resolve()
    if not root.is_dir():
        logger.warning(
            "Neo4j: sync skipped for project %s — src root is not a directory: %s",
            project_id,
            root,
        )
        return

    try:
        logger.info("Neo4j: syncing graph for project %s", project_id)
        delete_project_graph(project_id)
        data = build_cdg(root)
        nodes: list[str] = list(data.get("nodes") or [])
        edges: list[list[str]] = [e for e in (data.get("edges") or []) if len(e) >= 2]

        max_nodes = settings.NEO4J_MAX_GRAPH_NODES
        max_edges = settings.NEO4J_MAX_GRAPH_EDGES
        file_paths = sorted({n for n in nodes if not n.startswith("ext:")})
        if max_nodes > 0 and len(file_paths) > max_nodes:
            logger.warning(
                "Neo4j: truncating SourceFile nodes for project %s from %d to %d",
                project_id,
                len(file_paths),
                max_nodes,
            )
            file_paths = file_paths[:max_nodes]
        allowed_files = set(file_paths)

        if max_edges > 0 and len(edges) > max_edges:
            logger.warning(
                "Neo4j: truncating IMPORTS edges for project %s from %d to %d",
                project_id,
                len(edges),
                max_edges,
            )
            edges = edges[:max_edges]

        external_names: set[str] = set()
        for n in nodes:
            if n.startswith("ext:"):
                external_names.add(n[4:] if len(n) > 4 else n)
        for _, dst in edges:
            if dst.startswith("ext:"):
                external_names.add(dst[4:] if len(dst) > 4 else dst)

        def file_key(path: str) -> str:
            return f"{project_id}\u0001{path}"

        def ext_key(name: str) -> str:
            return f"{project_id}\u0001ext\u0001{name}"

        drv = _get_driver()
        with drv.session(**_session_kwargs()) as session:

            def write_all(tx):
                tx.run(
                    "MERGE (p:Project { id: $pid }) SET p.projectId = $pid",
                    pid=project_id,
                )
                file_rows = [
                    {"projectId": project_id, "path": p, "key": file_key(p)} for p in file_paths
                ]
                for i in range(0, len(file_rows), _WRITE_BATCH):
                    batch = file_rows[i : i + _WRITE_BATCH]
                    tx.run(
                        """
                        UNWIND $rows AS row
                        MERGE (f:SourceFile { key: row.key })
                        SET f.projectId = row.projectId, f.path = row.path
                        """,
                        rows=batch,
                    )
                ext_rows = [
                    {"projectId": project_id, "name": name, "key": ext_key(name)}
                    for name in sorted(external_names)
                ]
                for i in range(0, len(ext_rows), _WRITE_BATCH):
                    batch = ext_rows[i : i + _WRITE_BATCH]
                    tx.run(
                        """
                        UNWIND $rows AS row
                        MERGE (e:ExternalDep { key: row.key })
                        SET e.projectId = row.projectId, e.name = row.name
                        """,
                        rows=batch,
                    )
                rel_rows: list[dict[str, str]] = []
                for src, dst in edges:
                    if src not in allowed_files:
                        continue
                    if dst.startswith("ext:"):
                        name = dst[4:] if len(dst) > 4 else dst
                        tk = ext_key(name)
                    else:
                        if dst not in allowed_files:
                            continue
                        tk = file_key(dst)
                    rel_rows.append(
                        {
                            "projectId": project_id,
                            "from_key": file_key(src),
                            "to_key": tk,
                        }
                    )
                for i in range(0, len(rel_rows), _WRITE_BATCH):
                    batch = rel_rows[i : i + _WRITE_BATCH]
                    tx.run(
                        """
                        UNWIND $rows AS row
                        MATCH (a:SourceFile { key: row.from_key })
                        MATCH (b { key: row.to_key })
                        WHERE b:SourceFile OR b:ExternalDep
                        MERGE (a)-[:IMPORTS { projectId: row.projectId }]->(b)
                        """,
                        rows=batch,
                    )

            session.execute_write(write_all)
        logger.info(
            "Neo4j: synced CDG for project %s (%d files, %d externals, %d edges)",
            project_id,
            len(file_paths),
            len(external_names),
            len(edges),
        )
    except Exception as e:
        if settings.NEO4J_SYNC_FAIL_OPEN:
            if _neo4j_unreachable(e):
                logger.warning(
                    "Neo4j: graph sync skipped for project %s — not reachable (%s). "
                    "Start Neo4j or set NEO4J_ENABLED=false.",
                    project_id,
                    e.__class__.__name__,
                )
            else:
                logger.warning(
                    "Neo4j: graph sync failed for project %s (fail-open)",
                    project_id,
                    exc_info=True,
                )
            return
        raise
