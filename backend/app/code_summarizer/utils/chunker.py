"""Build symbol-level chunks from parsed source for outlines and future indexing."""

from __future__ import annotations

from typing import Any

from app.code_summarizer.utils.code_parser import ParseResult, lines_slice, parse_source


def symbol_chunks(file_path: str, content: str, language: str) -> list[dict[str, Any]]:
    """
    Return chunk dicts: path, kind, name, start_line, end_line, text.
    """
    parsed: ParseResult = parse_source(content, language)
    out: list[dict[str, Any]] = []
    for kind, name, start_line, end_line in parsed.symbols:
        text = lines_slice(content, start_line, end_line)
        if not text.strip():
            continue
        out.append(
            {
                "path": file_path,
                "kind": kind,
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
                "text": text,
            }
        )
    return out
