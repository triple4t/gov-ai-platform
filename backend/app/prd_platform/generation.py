"""Run generation per capability (markdown + optional diagram)."""

import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local
from app.prd_platform.cdg_builder import build_cdg
from app.prd_platform.clarifying import get_capability
from app.prd_platform.diagram_gen import flow_from_llm_json, graph_to_svg, svg_to_base64
from app.prd_platform.llm_clients import chat_complete, extract_json_object
from app.prd_platform.rag_service import retrieve_context, retrieve_deep_codebase_context

logger = logging.getLogger(__name__)


def _local_capped_max_tokens(requested: int) -> int:
    """Keep local CPU generations bounded; full 8192 can look 'hung' for tens of minutes."""
    if not llm_provider_is_local():
        return requested
    cap = settings.LOCAL_DOC_GENERATION_MAX_TOKENS
    if cap <= 0:
        return requested
    return min(int(requested), cap)


def _answers_block(answers: dict[str, str]) -> str:
    """Plain labeled lines for LLM context (no markdown bold)."""
    return "\n".join(f"- {k}: {v}" for k, v in answers.items() if v)


def strip_prd_markdown_artifacts(text: str) -> str:
    """Remove common markdown leakage from PRD plain-text output."""
    if not text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        s = line.rstrip()
        if re.match(r"^#{1,6}\s+", s):
            s = re.sub(r"^#{1,6}\s+", "", s).strip()
        if re.match(r"^[-*_]{3,}\s*$", s.strip()):
            continue
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        s = re.sub(r"`([^`]+)`", r"\1", s)
        out.append(s)
    return "\n".join(out).strip()


_MARKDOWN_FILLER_SEARCH = re.compile(
    r"(?is)\b(if you want|if you'?d like|next places to check|let me know|feel free to ask|"
    r"deeper api response examples|missing service modules|happy to (help|provide))\b"
)

_NO_MARKDOWN_CHAT_CLOSING = (
    "OUTPUT BOUNDARY: End at the last substantive section (e.g. Summary, Extension points). "
    "Do not add follow-up offers or chat-style closings—no 'If you want…', 'If you'd like…', "
    "'next places to check', 'let me know', 'feel free to ask', invitations to provide more files, "
    "or a horizontal rule (---) immediately before such filler. "
    "Do not add a standalone 'next steps for the reader' paragraph unless it is factual content "
    "inside a proper titled section (e.g. open items with specific repository paths)."
)


def strip_markdown_doc_filler(text: str) -> str:
    """Drop trailing --- + conversational tail or final filler paragraph from markdown docs."""
    if not text or not text.strip():
        return text
    lines = text.splitlines()
    for j in range(len(lines) - 1, -1, -1):
        if re.match(r"^-{3,}\s*$", lines[j].strip()):
            tail = "\n".join(lines[j + 1 :]).strip()
            if tail and _MARKDOWN_FILLER_SEARCH.search(tail):
                lines = lines[:j]
            break
    while lines:
        last_nz = len(lines) - 1
        while last_nz >= 0 and not lines[last_nz].strip():
            last_nz -= 1
        if last_nz < 0:
            break
        start = last_nz
        while start > 0 and lines[start - 1].strip():
            start -= 1
        para = "\n".join(lines[start : last_nz + 1]).strip()
        if para and _MARKDOWN_FILLER_SEARCH.search(para):
            lines = lines[:start]
            while lines and not lines[-1].strip():
                lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


_PRD_SYSTEM = """You are a senior product manager drafting a Product Requirements Document for a government digital service.

OUTPUT RULES (strict):
- Plain text only. Do NOT use: # headings, ** bold, ` code fences`, horizontal rules (---), numbered markdown lists that use #.
- Do NOT add conversational framing: no "Below is", "Below are", "If you'd like", "I can also", "Let me know", or similar chat filler before or after the document.
- Start with line 1: the official document title only, format: "<Product or initiative name> – Product Requirements Document" using the real product name from the user's answers.
- Line 2 must be blank.
- Then use EXACTLY these section headers, one per line, ALL CAPS, each followed by one blank line then bullet lines only:
  PROBLEM STATEMENT
  SOLUTION OVERVIEW
  KEY FEATURES AND SCOPE
  USER FLOW AND JOURNEY
  FUNCTIONAL REQUIREMENTS
  NON-FUNCTIONAL REQUIREMENTS
  SUCCESS METRICS
  OUT OF SCOPE
  ASSUMPTIONS
  DEPENDENCIES
  RISKS AND EDGE CASES
  OPEN QUESTIONS
- Every content line under a section must start with "- " (hyphen space). Use professional, concise language. Synthesize and expand from the user's inputs and optional repository context; do not paste the questionnaire verbatim.
- If a topic has little user input, still add 1–2 reasonable bullets marked as TBD or inferred, or state "To be confirmed with stakeholders."
"""


def generate(
    capability_id: str,
    answers: dict[str, str],
    project_id: str | None,
) -> dict[str, Any]:
    cap = get_capability(capability_id)
    if not cap:
        raise ValueError("Unknown capability")

    if capability_id == "code_summarizer":
        raise ValueError(
            "Full project code summary uses POST /api/v1/code-summarize/project — run it from the PRD Platform UI."
        )

    if cap["requires_codebase"] and not project_id:
        raise ValueError("This capability requires an indexed project (upload a ZIP first).")

    use_codex = bool(cap.get("use_codex"))
    ab = _answers_block(answers)
    rag = ""
    if project_id:
        if capability_id in ("tech_docs", "architecture", "code_review"):
            rag = retrieve_deep_codebase_context(
                project_id,
                capability_title=cap["title"],
                answers_block=ab,
            )
        else:
            rag = retrieve_context(
                project_id,
                f"{capability_id} {cap['title']} {ab}"[:4000],
            )

    # --- PRD: plain structured text (no markdown); optional RAG ---
    if capability_id == "prd":
        user_p = (
            "User inputs (field id: value):\n"
            f"{ab}\n\n"
            "Relevant repository context (if indexed; may be empty):\n"
            f"{rag[:16000] or '(none)'}"
        )
        # PRD is product prose; keep chat model even when RAG includes code excerpts.
        text = chat_complete(False, _PRD_SYSTEM, user_p, max_tokens=_local_capped_max_tokens(8192))
        text = strip_prd_markdown_artifacts(text)
        return {
            "markdown": text,
            "content_format": "prd_plain",
            "message": "Generation complete.",
        }

    # --- CDG: deterministic graph + optional summary ---
    if capability_id == "cdg":
        from app.prd_platform.database import get_project

        proj = get_project(project_id)
        root = Path(proj["root_path"])
        graph = build_cdg(root)
        nodes = graph["nodes"][:80]
        edges_t = [(a, b) for a, b in graph["edges"] if a in nodes and b in nodes][:200]
        svg = graph_to_svg(
            nodes,
            edges_t,
            title="Code dependency graph (imports)",
            layout="flow_lr",
        )
        summary_sys = "You summarize dependency graphs for engineers. Be concise."
        summary_user = f"Nodes (sample): {nodes[:30]}\nEdges count: {len(edges_t)}\nUser note: {answers.get('scope', '')}"
        summary = chat_complete(use_codex, summary_sys, summary_user, max_tokens=2048)
        md = f"## Code dependency graph\n\n{summary}\n\n_Edges show import-style relationships; `ext:*` are external packages._"
        return {
            "markdown": md,
            "svg_base64": svg_to_base64(svg),
            "message": f"{len(nodes)} nodes, {len(edges_t)} edges (capped for display).",
        }

    # --- Flow diagram: Codex emits JSON, we render SVG ---
    if capability_id == "flow_diagram":
        sys = (
            "You output ONLY valid JSON (no markdown fences) for a left-to-right process flowchart. "
            'Schema: {"nodes":[{"id":"1","label":"Short step title"},...],'
            '"edges":[{"from":"1","to":"2"},...]}. '
            "Each node needs a stable `id` (string) and a readable `label` (what appears in the diagram). "
            "Edges may use either `id` or full `label` in `from`/`to`; prefer chaining the same step order "
            "as the user's flow (e.g. login → create project → upload → process → view). "
            "Do not emit edges only between numeric ids while leaving labels disconnected—edges must connect "
            "the steps of the story (the server maps ids to labels). "
            "Model a clear user/system journey: mostly acyclic, ordered steps. "
            "Prefer a linear or tree-like flow; use at most 18 nodes and avoid dense many-to-many connections. "
            "Reflect the user's scope and optional codebase context."
        )
        user = f"User answers:\n{ab}\n\nCode context (excerpt):\n{rag[:12000] or '(none)'}"
        raw = chat_complete(use_codex, sys, user, max_tokens=4096)
        data = extract_json_object(raw) or {}
        nodes, edges = flow_from_llm_json(data)
        svg = graph_to_svg(nodes, [tuple(e) for e in edges], title="Project flow", layout="flow_lr")
        md = "## Project flow diagram\n\n_SVG generated from model-structured nodes and edges. Verify against your systems._"
        return {
            "markdown": md,
            "svg_base64": svg_to_base64(svg),
            "message": "Flow diagram rendered as SVG.",
        }

    # --- Text-heavy capabilities ---
    templates: dict[str, tuple[str, str]] = {
        "tech_docs": (
            "You write technical documentation for software engineers: overview, setup, key modules, APIs, data flow, extension points. "
            "Use markdown. Ground every major claim in the provided repository excerpts; cite file paths (### headers) when you reference behavior. "
            "Prioritize application source (e.g. src/, backend/, app code) over manifests: do not treat dependency lists or lockfile-style content alone as sufficient for APIs, runtime data flow, or architecture. "
            "If excerpts lack application code for a topic, state that clearly and list which areas of the repo would need to be reviewed—not a generic request to re-upload. "
            + _NO_MARKDOWN_CHAT_CLOSING,
            f"User focus:\n{ab}\n\nRepository excerpts (priority files + broad retrieval; not exhaustive):\n{rag or '(empty)'}",
        ),
        "sop": (
            "You write Standard Operating Procedures: purpose, scope, roles, prerequisites, numbered steps, verification, escalation. "
            "Use markdown. "
            + _NO_MARKDOWN_CHAT_CLOSING,
            f"User input:\n{ab}",
        ),
        "code_review": (
            "You perform an advisory code review. Use markdown sections: Summary, Strengths, Issues (severity), Security, Testing, Suggestions. "
            "Be specific; cite file paths from context when possible. This is not a substitute for human review. "
            "Base findings on application source in the excerpts, not only on manifests or dependency metadata. "
            + _NO_MARKDOWN_CHAT_CLOSING,
            f"User focus:\n{ab}\n\nRepository excerpts (priority files + broad retrieval):\n{rag or '(empty)'}",
        ),
        "architecture": (
            "You describe system architecture: context, containers/components, key flows, data stores, integrations, deployment notes. "
            "Use markdown. Ground in repository context and cite paths. Prefer describing real modules and flows from source excerpts over inferring from dependencies alone. "
            + _NO_MARKDOWN_CHAT_CLOSING,
            f"User viewpoint:\n{ab}\n\nRepository excerpts (priority files + broad retrieval):\n{rag or '(empty)'}",
        ),
    }

    if capability_id not in templates:
        raise ValueError("Capability not implemented")

    sys_p, user_p = templates[capability_id]
    text = chat_complete(use_codex, sys_p, user_p, max_tokens=_local_capped_max_tokens(8192))
    text = strip_markdown_doc_filler(text)

    if capability_id != "architecture":
        return {"markdown": text, "message": "Generation complete."}

    arch_graph_sys = (
        "You output ONLY valid JSON (no markdown fences) for a left-to-right architecture overview. "
        'Schema: {"nodes":[{"id":"1","label":"..."}],"edges":[{"from":"labelA","to":"labelB"}]}. '
        "The `from` and `to` values must match node `label` strings exactly. "
        "Model major elements: actors/users → clients/frontends → APIs/gateways → services/workers → data stores / external systems. "
        "Prefer a mostly acyclic, layered flow; at most 18 nodes; avoid dense many-to-many edges. "
        "Align with the narrative summary and repository context."
    )
    narrative_excerpt = (text[:3000] + "…") if len(text) > 3000 else text
    arch_graph_user = (
        f"User viewpoint:\n{ab}\n\n"
        f"Architecture narrative (excerpt; full doc returned separately):\n{narrative_excerpt}\n\n"
        f"Repository excerpts:\n{rag[:12000] or '(none)'}"
    )
    raw_graph = chat_complete(use_codex, arch_graph_sys, arch_graph_user, max_tokens=4096)
    data = extract_json_object(raw_graph) or {}
    has_nodes = bool(data.get("nodes"))
    has_edges = bool(data.get("edges"))
    if not has_nodes and not has_edges:
        return {"markdown": text, "message": "Generation complete."}

    nodes, edges = flow_from_llm_json(data)
    svg = graph_to_svg(nodes, [tuple(e) for e in edges], title="Architecture overview", layout="flow_lr")
    return {
        "markdown": text,
        "svg_base64": svg_to_base64(svg),
        "message": "Generation complete. Architecture overview diagram included.",
    }
