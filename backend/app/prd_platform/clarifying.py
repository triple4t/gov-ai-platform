"""Per-capability metadata and clarifying questions (API + prompts)."""

from typing import Any

CAPABILITIES: list[dict[str, Any]] = [
    {
        "id": "prd",
        "title": "Product Requirements Document (PRD)",
        "description": "Structured PRD from your answers; optional codebase context if indexed.",
        "requires_codebase": False,
        "use_codex": False,
        # Full PRD inputs. Former "goals/constraints" live in Success Metrics / Non-functional / Dependencies.
        "clarifying_questions": [
            {
                "id": "product_name",
                "label": "Product or initiative name",
                "type": "text",
                "required": True,
                "placeholder": "e.g. Citizen Portal 2.0",
            },
            {
                "id": "audience",
                "label": "Primary users / stakeholders",
                "type": "textarea",
                "required": True,
                "placeholder": "Who uses this and who approves delivery?",
            },
            {
                "id": "problem_statement",
                "label": "Problem Statement",
                "type": "textarea",
                "required": True,
                "placeholder": "What problem does this solve? For whom?",
            },
            {
                "id": "solution_overview",
                "label": "Solution Overview",
                "type": "textarea",
                "required": True,
                "placeholder": "High-level approach and value proposition.",
            },
            {
                "id": "key_features_scope",
                "label": "Key Features / Scope",
                "type": "textarea",
                "required": False,
                "placeholder": "Must-have capabilities and boundaries of the release.",
            },
            {
                "id": "user_flow_journey",
                "label": "User Flow / Journey",
                "type": "textarea",
                "required": False,
                "placeholder": "Main steps users take; channels (web, mobile, kiosk).",
            },
            {
                "id": "functional_requirements",
                "label": "Functional Requirements",
                "type": "textarea",
                "required": True,
                "placeholder": "Behaviors, APIs, data, integrations the system must support.",
            },
            {
                "id": "non_functional_requirements",
                "label": "Non-Functional Requirements",
                "type": "textarea",
                "required": False,
                "placeholder": "Performance, security, compliance, accessibility, hosting (include regulatory constraints here).",
            },
            {
                "id": "success_metrics",
                "label": "Success Metrics",
                "type": "textarea",
                "required": False,
                "placeholder": "KPIs and how you will measure success.",
            },
            {
                "id": "out_of_scope",
                "label": "Out of Scope",
                "type": "textarea",
                "required": False,
                "placeholder": "Explicitly excluded work or future phases.",
            },
            {
                "id": "assumptions",
                "label": "Assumptions",
                "type": "textarea",
                "required": False,
                "placeholder": "What we assume to be true for planning.",
            },
            {
                "id": "dependencies",
                "label": "Dependencies",
                "type": "textarea",
                "required": False,
                "placeholder": "Other teams, systems, vendors, approvals, data sources.",
            },
            {
                "id": "risks_edge_cases",
                "label": "Risks & Edge Cases",
                "type": "textarea",
                "required": False,
                "placeholder": "Risks, failure modes, edge cases to mitigate.",
            },
        ],
    },
    {
        "id": "tech_docs",
        "title": "Technical documentation (from code)",
        "description": "Developer-oriented docs grounded in uploaded repository (RAG + Codex).",
        "requires_codebase": True,
        "use_codex": True,
        "clarifying_questions": [
            {
                "id": "audience",
                "label": "Doc audience",
                "type": "text",
                "required": True,
                "placeholder": "e.g. New backend engineers",
            },
            {
                "id": "depth",
                "label": "Depth",
                "type": "text",
                "required": False,
                "placeholder": "e.g. High-level only / include APIs and data flow",
            },
        ],
    },
    {
        "id": "flow_diagram",
        "title": "Project flow diagram",
        "description": "SVG flow chart from scope you describe, refined with code context if indexed.",
        "requires_codebase": False,
        "use_codex": True,
        "clarifying_questions": [
            {
                "id": "flow_scope",
                "label": "What flow should the diagram cover?",
                "type": "textarea",
                "required": True,
                "placeholder": "e.g. User login → dashboard → form submit → API → DB",
            },
            {
                "id": "actors",
                "label": "Actors / systems",
                "type": "textarea",
                "required": False,
                "placeholder": "e.g. Citizen, API Gateway, Auth service, PostgreSQL",
            },
        ],
    },
    {
        "id": "sop",
        "title": "Standard Operating Procedure (SOP)",
        "description": "Step-by-step operational doc from your inputs.",
        "requires_codebase": False,
        "use_codex": False,
        "clarifying_questions": [
            {
                "id": "sop_topic",
                "label": "SOP topic",
                "type": "text",
                "required": True,
                "placeholder": "e.g. Production deployment checklist",
            },
            {
                "id": "environment",
                "label": "Environment / team context",
                "type": "textarea",
                "required": True,
                "placeholder": "Where does this run? Who executes it?",
            },
        ],
    },
    {
        "id": "code_review",
        "title": "Code review analysis",
        "description": "Advisory review: risks, tests, security hints (requires indexed codebase).",
        "requires_codebase": True,
        "use_codex": True,
        "clarifying_questions": [
            {
                "id": "focus",
                "label": "Focus areas",
                "type": "textarea",
                "required": False,
                "placeholder": "e.g. Security, performance, API design",
            },
        ],
    },
    {
        "id": "architecture",
        "title": "Architecture description",
        "description": "Narrative architecture from repo structure + RAG, plus an auto-generated layered overview diagram (requires codebase).",
        "requires_codebase": True,
        "use_codex": True,
        "clarifying_questions": [
            {
                "id": "viewpoint",
                "label": "Architecture viewpoint",
                "type": "text",
                "required": True,
                "placeholder": "e.g. C4 container / deployment / data flow",
            },
        ],
    },
    {
        "id": "cdg",
        "title": "Code dependency graph (CDG)",
        "description": "Import-based graph (Python/JS/TS) as SVG + short summary.",
        "requires_codebase": True,
        "use_codex": True,
        "clarifying_questions": [
            {
                "id": "scope",
                "label": "Scope note (optional)",
                "type": "text",
                "required": False,
                "placeholder": "e.g. Only backend/app",
            },
        ],
    },
    {
        "id": "code_summarizer",
        "title": "Full project code summary",
        "description": "High-level summary of an indexed ZIP: purpose, tech stack, architecture, key modules, and main flows (RAG). Short, medium, or detailed depth.",
        "requires_codebase": False,
        "use_codex": True,
        "clarifying_questions": [],
    },
]


def get_capability(cap_id: str) -> dict[str, Any] | None:
    for c in CAPABILITIES:
        if c["id"] == cap_id:
            return c
    return None
