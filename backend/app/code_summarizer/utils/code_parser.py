"""
Extract structure from Python (AST) and JavaScript/TypeScript (heuristics).

JS/TS parsing is best-effort: brace counting and regexes miss nested templates
and unusual syntax; use for summaries only, not static analysis.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ParseResult:
    language: Literal["python", "javascript", "unknown"]
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    # (name, start_line, end_line) 1-based inclusive for snippets
    symbols: list[tuple[str, str, int, int]] = field(default_factory=list)


def _py_imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module.split(".")[0])
            else:
                for alias in node.names:
                    out.append(alias.name)
    return sorted(set(out))


def parse_python(source: str) -> ParseResult:
    """Parse Python with ast; collect imports, class/function names and line ranges."""
    result = ParseResult(language="python")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    result.imports = _py_imports(tree)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result.classes.append(node.name)
            result.symbols.append(("class", node.name, node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            result.functions.append(node.name)
            result.symbols.append(
                ("function", node.name, node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno)
            )
    return result


# JS: function foo(, async function, class X, const foo = ( or =>
_JS_FUNC = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("
    r"|(?:export\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_JS_ARROW = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
    re.MULTILINE,
)


def _js_brace_end(source: str, open_pos: int) -> int:
    """Find index after matching `}` starting from first `{` at or after open_pos."""
    i = source.find("{", open_pos)
    if i < 0:
        return len(source)
    depth = 0
    in_str: str | None = None
    escape = False
    while i < len(source):
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'", "`"):
            in_str = ch if ch != "`" else "`"
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(source)


def parse_javascript(source: str) -> ParseResult:
    """
    Heuristic extraction of function/class names and import specifiers.
    Does not build a full AST; line ranges are approximate for top-level blocks.
    """
    result = ParseResult(language="javascript")

    # import ... from 'x'
    for m in re.finditer(
        r"""import\s+(?:[\w*{}\s,]+\s+from\s+)?['"]([^'"]+)['"]""",
        source,
    ):
        result.imports.append(m.group(1).split("/")[0].replace("@", ""))
    for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", source):
        result.imports.append(m.group(1).split("/")[0])
    result.imports = sorted(set(result.imports))

    seen_names: set[str] = set()

    for m in _JS_FUNC.finditer(source):
        name = m.group(1) or m.group(2)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        start_line = source[: m.start()].count("\n") + 1
        if m.group(0).strip().startswith("class"):
            result.classes.append(name)
            end_char = _js_brace_end(source, m.end())
            end_line = source[:end_char].count("\n") + 1
            result.symbols.append(("class", name, start_line, end_line))
        else:
            result.functions.append(name)
            end_char = _js_brace_end(source, m.end())
            end_line = source[:end_char].count("\n") + 1
            result.symbols.append(("function", name, start_line, end_line))

    for m in _JS_ARROW.finditer(source):
        name = m.group(1)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        result.functions.append(name)
        start_line = source[: m.start()].count("\n") + 1
        end_char = _js_brace_end(source, m.end())
        end_line = source[:end_char].count("\n") + 1
        result.symbols.append(("function", name, start_line, end_line))

    return result


def parse_source(source: str, language: str) -> ParseResult:
    lang = (language or "").strip().lower()
    if lang in ("python", "py"):
        return parse_python(source)
    if lang in ("javascript", "js", "typescript", "ts", "tsx", "jsx"):
        return parse_javascript(source)
    return ParseResult(language="unknown")


def lines_slice(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    a = max(0, start_line - 1)
    b = min(len(lines), end_line)
    return "\n".join(lines[a:b])
