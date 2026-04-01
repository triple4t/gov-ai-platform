"""ZIP extraction and text file collection."""

import io
import logging
import zipfile
from pathlib import Path

from app.core.config import settings
from app.prd_platform.chunking import chunk_text

logger = logging.getLogger(__name__)

# Not indexed: huge, low-signal for RAG; they crowd out application source in vector search.
DEPENDENCY_LOCKFILE_NAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pipfile.lock",
        "uv.lock",
    }
)


def _is_dependency_lockfile(path: Path) -> bool:
    return path.name.lower() in DEPENDENCY_LOCKFILE_NAMES


SKIP_DIR_NAMES = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "target",
    ".next",
    "coverage",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".sh",
    ".env.example",
    ".gitignore",
    "Dockerfile",
}


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith(".")


def _is_text_file(path: Path) -> bool:
    suf = path.suffix.lower()
    if suf in TEXT_EXTENSIONS:
        return True
    if path.name in ("Dockerfile", "Makefile", "README", "LICENSE"):
        return True
    return False


def extract_and_chunk_zip(
    zip_bytes: bytes,
    dest_root: Path,
) -> list[tuple[str, str]]:
    """
    Extract zip to dest_root, return list of (relative_path, chunk_text).
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    chunks_out: list[tuple[str, str]] = []
    file_count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/").strip("/")
            if not name or name.startswith("__MACOSX"):
                continue
            parts = Path(name).parts
            if any(_should_skip_dir(p) for p in parts):
                continue
            rel = Path(name)
            if not _is_text_file(rel):
                continue
            if _is_dependency_lockfile(rel):
                logger.debug("Skip dependency lockfile from index: %s", name)
                continue
            if info.file_size > settings.PRD_PLATFORM_MAX_FILE_BYTES:
                logger.debug("Skip large file: %s (%s bytes)", name, info.file_size)
                continue
            try:
                raw = zf.read(info)
                text = raw.decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug("Skip unreadable %s: %s", name, e)
                continue

            file_count += 1
            if file_count > settings.PRD_PLATFORM_MAX_FILES:
                raise ValueError(
                    f"Too many text files (limit {settings.PRD_PLATFORM_MAX_FILES}). "
                    "Trim the archive or increase PRD_PLATFORM_MAX_FILES."
                )

            out_path = dest_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(text, encoding="utf-8", errors="replace")
            except OSError:
                pass

            rel_s = str(rel).replace("\\", "/")
            for piece in chunk_text(text, rel_s):
                chunks_out.append((rel_s, piece))

    return chunks_out
