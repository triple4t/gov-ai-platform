"""SVG diagrams: flow graphs from LLM JSON and CDG layout."""

import base64
import math
from collections import defaultdict
from typing import Any, Literal

LayoutMode = Literal["circle", "flow_lr"]

BOX_H = 28
ROW_GAP = 22
COL_GAP = 64
PAD_L = 40
PAD_R = 40
PAD_T = 44
PAD_B = 36
TITLE_DY = 26
CHAR_W_EST = 6.8
MAX_BOX_W = 240
MIN_BOX_W = 72


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _display_label(node: str) -> str:
    return node if len(node) < 42 else node[:39] + "…"


def _node_box_width(label: str) -> float:
    d = _display_label(label)
    return min(MAX_BOX_W, max(MIN_BOX_W, CHAR_W_EST * len(d) + 28))


def _arrow_defs() -> str:
    """Marker must be defined before any <line marker-end=...> for reliable rendering."""
    return (
        "<defs>"
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>'
        "</marker>"
        "</defs>"
    )


def layout_circle(nodes: list[str], width: float = 900, height: float = 600) -> dict[str, tuple[float, float]]:
    n = len(nodes)
    cx, cy = width / 2, height / 2
    r = min(width, height) * 0.32
    pos: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        ang = 2 * math.pi * i / max(n, 1) - math.pi / 2
        pos[node] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    return pos


def layout_flow_lr(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> tuple[dict[str, tuple[float, float, float, float]], float, float]:
    """
    Left-to-right layered layout (x, y, w, h) per node; returns (boxes, total_w, total_h).
    """
    seen: set[str] = set()
    all_nodes: list[str] = []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            all_nodes.append(n)
    for u, v in edges:
        for x in (u, v):
            if x not in seen:
                seen.add(x)
                all_nodes.append(x)

    if not all_nodes:
        return {}, PAD_L + PAD_R, PAD_T + PAD_B + TITLE_DY

    in_deg: dict[str, int] = {n: 0 for n in all_nodes}
    for u, v in edges:
        if v in in_deg:
            in_deg[v] += 1

    level: dict[str, int] = {n: 0 for n in all_nodes}
    for _ in range(len(all_nodes) + 10):
        for u, v in edges:
            if u in level and v in level:
                nv = level[u] + 1
                if nv > level[v]:
                    level[v] = nv

    m = min(level.values())
    for n in all_nodes:
        level[n] -= m

    by_layer: dict[int, list[str]] = defaultdict(list)
    max_l = 0
    for n in all_nodes:
        L = level[n]
        by_layer[L].append(n)
        max_l = max(max_l, L)

    order_index = {n: i for i, n in enumerate(all_nodes)}
    for L in by_layer:
        by_layer[L].sort(key=lambda n: order_index[n])

    widths = {n: _node_box_width(n) for n in all_nodes}
    pos: dict[str, tuple[float, float, float, float]] = {}
    x_cursor = PAD_L

    for L in range(max_l + 1):
        layer_nodes = by_layer.get(L, [])
        if not layer_nodes:
            continue
        inner_col_w = max(widths[n] for n in layer_nodes)
        col_w = inner_col_w + COL_GAP
        y = PAD_T + TITLE_DY
        for n in layer_nodes:
            w = widths[n]
            x_inner = x_cursor + (col_w - COL_GAP - inner_col_w) / 2 + (inner_col_w - w) / 2
            pos[n] = (x_inner, y, w, BOX_H)
            y += BOX_H + ROW_GAP
        x_cursor += col_w

    total_w = max(x_cursor + PAD_R, 320)
    max_bottom = PAD_T + TITLE_DY
    for x, y, w, h in pos.values():
        max_bottom = max(max_bottom, y + h)
    total_h = max_bottom + PAD_B

    return pos, total_w, total_h


def graph_to_svg(
    nodes: list[str],
    edges: list[tuple[str, str]],
    title: str = "",
    layout: LayoutMode = "circle",
) -> str:
    """Render graph: circle (CDG) or flow_lr (process flow)."""
    lines: list[str] = []

    if layout == "flow_lr":
        boxes, w, h = layout_flow_lr(nodes, edges)
        lines.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
            f'viewBox="0 0 {w:.1f} {h:.1f}">'
        )
        lines.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
        lines.append(_arrow_defs())
        if title:
            lines.append(
                f'<text x="{PAD_L}" y="{PAD_T}" font-size="15" font-weight="600" '
                f'font-family="system-ui,Segoe UI,sans-serif" fill="#1e293b">{_escape(title)}</text>'
            )

        def anchor(n: str, side: str) -> tuple[float, float] | None:
            b = boxes.get(n)
            if not b:
                return None
            x, y, bw, bh = b
            mid = y + bh / 2
            if side == "right":
                return x + bw, mid
            if side == "left":
                return x, mid
            return x + bw / 2, y + bh / 2

        for a, b in edges:
            p1 = anchor(a, "right")
            p2 = anchor(b, "left")
            if not p1 or not p2:
                continue
            x1, y1 = p1
            x2, y2 = p2
            if x2 <= x1 + 2:
                lines.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    'stroke="#94a3b8" stroke-width="1.4" marker-end="url(#arrow)"/>'
                )
            else:
                mx = (x1 + x2) / 2
                d = f"M {x1:.1f},{y1:.1f} C {mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
                lines.append(
                    f'<path d="{d}" fill="none" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#arrow)"/>'
                )

        for node, (x, y, bw, bh) in boxes.items():
            label = _display_label(node)
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="8" '
                'fill="#e8eeff" stroke="#4f46e5" stroke-width="1.2"/>'
            )
            lines.append(
                f'<text x="{x + bw / 2:.1f}" y="{y + bh / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="11" font-family="system-ui,Segoe UI,sans-serif" fill="#1e293b">'
                f"{_escape(label)}</text>"
            )

        lines.append("</svg>")
        return "\n".join(lines)

    # --- circle layout (CDG) ---
    w, h = 920, 640
    pos = layout_circle(nodes, w, h)
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    lines.append(_arrow_defs())
    if title:
        lines.append(
            f'<text x="20" y="28" font-size="16" font-family="system-ui,sans-serif">{_escape(title)}</text>'
        )

    for a, b in edges:
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#94a3b8" stroke-width="1.4" marker-end="url(#arrow)"/>'
        )

    for node, (x, y) in pos.items():
        label = _display_label(node)
        tw = min(220, 12 * len(label) + 24)
        lines.append(
            f'<rect x="{x - tw/2:.1f}" y="{y - 14:.1f}" width="{tw:.1f}" height="28" rx="6" '
            'fill="#e0e7ff" stroke="#6366f1" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" '
            f'font-family="system-ui,sans-serif">{_escape(label)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def svg_to_base64(svg: str) -> str:
    return base64.standard_b64encode(svg.encode("utf-8")).decode("ascii")


def flow_from_llm_json(data: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    """Parse LLM JSON into (node labels, edges).

    Models often emit edges with numeric ``from``/``to`` matching ``id`` while listing human
    ``label`` on each node. Previously those ids became extra ghost nodes (horizontal "1→2→3"
    row) while real labels sat disconnected in column 0. We map ids to labels so edges attach
    to the same strings shown in boxes.

    Schema: ``{ "nodes": [{"id":"1","label":"Step A"}, ...], "edges": [{"from":"1","to":"2"}] }``
    or edges may use labels directly. Also supports ``edges: [["a","b"], ...]`` and plain
    string ``nodes``.
    """
    id_to_label: dict[str, str] = {}
    nodes_ordered: list[str] = []
    seen_label: set[str] = set()

    raw_nodes = data.get("nodes") or []
    for n in raw_nodes:
        if isinstance(n, dict):
            nid = str(n.get("id") or "").strip()
            lab = str(n.get("label") or n.get("id") or "?").strip()
            if nid:
                id_to_label[nid] = lab
            if lab and lab not in seen_label:
                seen_label.add(lab)
                nodes_ordered.append(lab)
        else:
            s = str(n).strip()
            if s and s not in seen_label:
                seen_label.add(s)
                nodes_ordered.append(s)

    def resolve_endpoint(x: str) -> str:
        x = (x or "").strip()
        if x in id_to_label:
            return id_to_label[x]
        return x

    edges: list[tuple[str, str]] = []
    raw_edges = data.get("edges") or []
    for e in raw_edges:
        if isinstance(e, dict):
            a = str(e.get("from") or e.get("source") or "").strip()
            b = str(e.get("to") or e.get("target") or "").strip()
            if a and b:
                ra, rb = resolve_endpoint(a), resolve_endpoint(b)
                edges.append((ra, rb))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            ra = resolve_endpoint(str(e[0]))
            rb = resolve_endpoint(str(e[1]))
            if ra and rb:
                edges.append((ra, rb))

    for a, b in edges:
        for x in (a, b):
            if x not in seen_label:
                seen_label.add(x)
                nodes_ordered.append(x)

    if not nodes_ordered:
        nodes_ordered = ["Start", "Process", "End"]
        edges = [("Start", "Process"), ("Process", "End")]

    return nodes_ordered, edges
