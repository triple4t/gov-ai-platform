"""Build a simple import-level dependency graph from source files."""

import re
from pathlib import Path
from typing import Any

# Python: import x / from x.y import z
_PY_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))\s*",
    re.MULTILINE,
)
# JS/TS: import ... from 'x' or require('x')
_JS_FROM = re.compile(r"""import\s+(?:[\w*{}\s,]+\s+from\s+)?['"]([^'"]+)['"]""")
_JS_REQ = re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""")


def _py_modules_from_file(text: str) -> set[str]:
    found: set[str] = set()
    for m in _PY_IMPORT.finditer(text):
        a, b = m.group(1), m.group(2)
        if a:
            found.add(a.split(".")[0])
        if b:
            for part in b.split(","):
                part = part.strip().split(" as ")[0].strip()
                if part:
                    found.add(part.split(".")[0])
    return found


def _js_modules_from_file(text: str) -> set[str]:
    found: set[str] = set()
    for m in _JS_FROM.finditer(text):
        s = m.group(1).strip()
        if s.startswith(".") or s.startswith("/"):
            continue
        found.add(s.split("/")[0].replace("@", ""))
    for m in _JS_REQ.finditer(text):
        s = m.group(1).strip()
        if s.startswith(".") or s.startswith("/"):
            continue
        found.add(s.split("/")[0])
    return found


def build_cdg(root: Path) -> dict[str, Any]:
    """
    Returns { "nodes": [path_str, ...], "edges": [[src, dst], ...] }
    src/dst are relative file paths; external imports point to synthetic nodes "ext:pkg".
    """
    root = root.resolve()
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(x in p.parts for x in ("node_modules", ".git", "__pycache__", ".venv", "venv")):
            continue
        suf = p.suffix.lower()
        if suf not in {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            continue
        try:
            if p.stat().st_size > 512 * 1024:
                continue
        except OSError:
            continue
        files.append(p)

    rel_paths = [str(p.relative_to(root)).replace("\\", "/") for p in files]
    path_set = set(rel_paths)

    def resolve_py_mod(mod: str) -> str | None:
        # try mod as path mod.py or mod/__init__.py
        candidates = [
            f"{mod.replace('.', '/')}.py",
            f"{mod.replace('.', '/')}/__init__.py",
        ]
        for c in candidates:
            if c in path_set:
                return c
        # partial: first segment
        first = mod.split(".")[0]
        for rp in rel_paths:
            if rp.startswith(first + "/") or rp == f"{first}.py":
                return rp
        return None

    edges: list[list[str]] = []
    seen_e = set()

    for p, rel in zip(files, rel_paths):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p.suffix.lower() == ".py":
            mods = _py_modules_from_file(text)
            for m in mods:
                if m in ("os", "sys", "re", "json", "typing", "pathlib", "logging", "datetime"):
                    continue
                tgt = resolve_py_mod(m)
                if tgt is None:
                    tgt = f"ext:{m}"
                key = (rel, tgt)
                if key not in seen_e:
                    seen_e.add(key)
                    edges.append([rel, tgt])
        elif p.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            mods = _js_modules_from_file(text)
            for m in mods:
                if not m or m in ("react", "react-dom"):
                    continue
                tgt = f"ext:{m}"
                key = (rel, tgt)
                if key not in seen_e:
                    seen_e.add(key)
                    edges.append([rel, tgt])

    nodes = sorted(set(rel_paths) | {e[1] for e in edges if e[1].startswith("ext:")})
    return {"nodes": nodes, "edges": edges}
