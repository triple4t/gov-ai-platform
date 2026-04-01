"""Split file text into overlapping character chunks."""

from app.core.config import settings


def chunk_text(text: str, file_path: str) -> list[str]:
    max_c = settings.PRD_PLATFORM_CHUNK_CHARS
    overlap = min(settings.PRD_PLATFORM_CHUNK_OVERLAP, max_c // 2)
    if not text or not text.strip():
        return []
    t = text.strip()
    if len(t) <= max_c:
        return [t]
    parts: list[str] = []
    start = 0
    while start < len(t):
        end = min(start + max_c, len(t))
        chunk = t[start:end]
        # try break on newline near end
        if end < len(t):
            nl = chunk.rfind("\n", max_c // 2)
            if nl > max_c // 4:
                chunk = chunk[: nl + 1]
                end = start + len(chunk)
        parts.append(chunk.strip())
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]
