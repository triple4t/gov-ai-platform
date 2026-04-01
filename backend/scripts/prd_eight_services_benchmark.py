"""
Benchmark all 8 PRD Platform capabilities × 6 context sizes (10k–60k chars).

Matches UI capability order:
  prd, tech_docs, flow_diagram, sop, code_review, architecture, cdg, code_summarizer

- Capabilities 1–7: POST /api/v1/prd-platform/projects/generate (same as frontend).
- code_summarizer: UI uses POST /api/v1/code-summarize/project (no user text sizing).
  For 10k–60k parity we use POST /api/v1/code-summarize/query with a sized `question`
  (still requires an indexed project_id). See report footnote.

Requires: backend running on port 8000, requests, indexed project for codebase caps.

  set PRD_BENCHMARK_PROJECT_ID=<uuid>   # or pass --project-id

Run from backend folder:
  set PYTHONPATH=.
  venv\\Scripts\\python scripts\\prd_eight_services_benchmark.py --project-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import requests
from dotenv import dotenv_values

# UI order (see app.prd_platform.clarifying.CAPABILITIES)
CAPABILITY_ORDER = [
    "prd",
    "tech_docs",
    "flow_diagram",
    "sop",
    "code_review",
    "architecture",
    "cdg",
    "code_summarizer",
]

SIZES = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000]


def _run_ps(command: str) -> str:
    full = ["powershell", "-NoProfile", "-Command", command]
    return subprocess.check_output(full, text=True).strip()


def _get_server_pid(port: int = 8000) -> int:
    cmd = (
        f"(Get-NetTCPConnection -State Listen -LocalPort {port} "
        "| Select-Object -First 1 -ExpandProperty OwningProcess)"
    )
    out = _run_ps(cmd)
    return int(out) if out else 0


def _get_process_mem_mib(pid: int) -> float:
    if pid <= 0:
        return 0.0
    try:
        out = _run_ps(f"(Get-Process -Id {pid}).WorkingSet64")
        return round(int(out) / (1024 * 1024), 2)
    except Exception:
        return 0.0


def _get_gpu_mem_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return float(sum(vals))
    except Exception:
        return 0.0


def _kv_cache_mib(n_ctx: int, n_layer: int = 40, n_embd: int = 3584, bytes_per_elem: int = 2) -> float:
    total = 2 * n_layer * n_ctx * n_embd * bytes_per_elem
    return round(total / (1024 * 1024), 2)


def _filler(need: int) -> str:
    line = (
        "Benchmark context line: citizen services, APIs, data stores, and compliance checks. "
    )
    if need <= 0:
        return ""
    out = []
    cur = 0
    while cur < need:
        take = min(len(line), need - cur)
        out.append(line[:take])
        cur += take
    return "".join(out)


def _total_answer_chars(answers: dict[str, str]) -> int:
    return sum(len(v) for v in answers.values())


def build_answers(cap_id: str, target_total_chars: int) -> dict[str, str]:
    """Pack ~target_total_chars into capability answers (sum of values)."""
    if cap_id == "prd":
        base = {
            "product_name": "Bench Product",
            "audience": "Engineers",
            "problem_statement": "Benchmark problem statement.",
            "solution_overview": "Benchmark solution overview.",
            "functional_requirements": "",
        }
        used = sum(len(v) for k, v in base.items() if k != "functional_requirements")
        need = max(0, target_total_chars - used)
        base["functional_requirements"] = _filler(need) or "F1."
        return base

    if cap_id == "tech_docs":
        base = {"audience": "Backend engineers", "depth": ""}
        used = len(base["audience"])
        base["depth"] = _filler(max(0, target_total_chars - used)) or "High-level."
        return base

    if cap_id == "flow_diagram":
        return {"flow_scope": _filler(target_total_chars) or "Login flow.", "actors": ""}

    if cap_id == "sop":
        base = {"sop_topic": "Production release", "environment": ""}
        used = len(base["sop_topic"])
        base["environment"] = _filler(max(0, target_total_chars - used)) or "Cloud."
        return base

    if cap_id == "code_review":
        return {"focus": _filler(target_total_chars) or "Security."}

    if cap_id == "architecture":
        return {"viewpoint": _filler(target_total_chars) or "Container view."}

    if cap_id == "cdg":
        return {"scope": _filler(target_total_chars) or "backend/"}

    raise ValueError(f"Unknown capability for answers: {cap_id}")


@dataclass
class Row:
    capability_id: str
    capability_title: str
    case_label: str
    target_context_chars: int
    actual_answer_chars: int
    approx_prompt_tokens: int
    endpoint: str
    http_status: int
    latency_sec: float
    response_chars: int
    server_mem_before_mib: float
    server_mem_after_mib: float
    server_mem_delta_mib: float
    gpu_mem_before_mib: float
    gpu_mem_after_mib: float
    gpu_mem_delta_mib: float
    kv_cache_est_prompt_mib: float
    notes: str


def _ensure_backend_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _cap_title(cap_id: str) -> str:
    try:
        _ensure_backend_on_path()
        from app.prd_platform.clarifying import get_capability

        c = get_capability(cap_id)
        return str(c["title"]) if c else cap_id
    except Exception:
        return cap_id


def run_one_generate(
    base: str,
    cap_id: str,
    answers: dict[str, str],
    project_id: str | None,
    timeout: int,
) -> tuple[int, float, int, str]:
    url = f"{base}/prd-platform/projects/generate"
    t0 = time.perf_counter()
    notes = ""
    try:
        r = requests.post(
            url,
            json={"capability": cap_id, "answers": answers, "project_id": project_id},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        n = 0
        if r.headers.get("content-type", "").startswith("application/json"):
            body = r.json()
            n = len((body.get("markdown") or body.get("message") or ""))
        else:
            n = len(r.text or "")
        return r.status_code, elapsed, n, notes
    except Exception as e:
        return 0, time.perf_counter() - t0, 0, str(e)[:200]


def run_one_code_query(
    base: str,
    project_id: str,
    question: str,
    timeout: int,
) -> tuple[int, float, int, str]:
    url = f"{base}/code-summarize/query"
    t0 = time.perf_counter()
    try:
        r = requests.post(
            url,
            json={
                "project_id": project_id,
                "question": question,
                "detail_level": "short",
            },
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        n = 0
        if r.headers.get("content-type", "").startswith("application/json"):
            body = r.json()
            n = len((body.get("summary") or ""))
        else:
            n = len(r.text or "")
        return r.status_code, elapsed, n, ""
    except Exception as e:
        return 0, time.perf_counter() - t0, 0, str(e)[:200]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--project-id", default=os.getenv("PRD_BENCHMARK_PROJECT_ID", "").strip())
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    root = _ensure_backend_on_path()
    env = dotenv_values(root / ".env")
    n_ctx = int((env.get("LLAMA_N_CTX") or "8192").strip())

    pid = _get_server_pid(args.port)
    if pid <= 0:
        raise SystemExit(f"No process listening on port {args.port}. Start the backend first.")

    project_id = args.project_id or None
    if not project_id:
        raise SystemExit(
            "Indexed project_id required for tech_docs, flow_diagram (optional), "
            "code_review, architecture, cdg, code_summarizer. "
            "Set PRD_BENCHMARK_PROJECT_ID or pass --project-id."
        )

    base = args.base_url.rstrip("/")
    rows: list[Row] = []

    for cap_id in CAPABILITY_ORDER:
        title = _cap_title(cap_id)
        for size in SIZES:
            case_label = f"{size // 1000}k"
            mem_b = _get_process_mem_mib(pid)
            gpu_b = _get_gpu_mem_mib()
            endpoint = ""
            notes = ""
            status = 0
            elapsed = 0.0
            resp_chars = 0

            if cap_id == "code_summarizer":
                endpoint = f"{base}/code-summarize/query"
                q = (
                    _filler(size)
                    + "\n\nIn 5 bullet points: main purpose, tech stack, key folders, main flows, and risks."
                )
                actual_chars = len(q)
                status, elapsed, resp_chars, err = run_one_code_query(
                    base, project_id, q, args.timeout
                )
                if err:
                    notes = err
            else:
                endpoint = f"{base}/prd-platform/projects/generate"
                answers = build_answers(cap_id, size)
                actual_chars = _total_answer_chars(answers)
                status, elapsed, resp_chars, err = run_one_generate(
                    base, cap_id, answers, project_id, args.timeout
                )
                if err:
                    notes = err

            mem_a = _get_process_mem_mib(pid)
            gpu_a = _get_gpu_mem_mib()
            approx_toks = max(1, int(actual_chars / 3.5))
            kv_est = _kv_cache_mib(min(approx_toks, n_ctx))

            if status >= 400 and not notes:
                notes = f"HTTP {status}"
            if approx_toks > n_ctx:
                notes = (notes + "; " if notes else "") + "answers may exceed LLAMA_N_CTX (server may truncate)."

            rows.append(
                Row(
                    capability_id=cap_id,
                    capability_title=title,
                    case_label=case_label,
                    target_context_chars=size,
                    actual_answer_chars=actual_chars,
                    approx_prompt_tokens=approx_toks,
                    endpoint=endpoint,
                    http_status=status,
                    latency_sec=round(elapsed, 2),
                    response_chars=resp_chars,
                    server_mem_before_mib=mem_b,
                    server_mem_after_mib=mem_a,
                    server_mem_delta_mib=round(mem_a - mem_b, 2),
                    gpu_mem_before_mib=gpu_b,
                    gpu_mem_after_mib=gpu_a,
                    gpu_mem_delta_mib=round(gpu_a - gpu_b, 2),
                    kv_cache_est_prompt_mib=kv_est,
                    notes=notes,
                )
            )

    out_dir = root / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"prd_eight_services_benchmark_{ts}.md"
    json_path = out_dir / f"prd_eight_services_benchmark_{ts}.json"
    latest_md = out_dir / "prd_eight_services_benchmark_latest.md"
    latest_json = out_dir / "prd_eight_services_benchmark_latest.json"

    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# PRD Platform — 8 services × 6 context benchmarks")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- API base: `{base}`")
    lines.append(f"- Project ID: `{project_id}`")
    lines.append(f"- Backend PID (port {args.port}): `{pid}`")
    lines.append(f"- `LLAMA_N_CTX` (from `.env`): `{n_ctx}`")
    lines.append(
        f"- Estimated full KV at `n_ctx`: `{_kv_cache_mib(n_ctx)} MiB` "
        "(40 layers, 3584 hidden, fp16 KV — approximate)."
    )
    lines.append("")
    lines.append(
        "**code_summarizer:** UI calls `POST .../code-summarize/project` (no user-sized text). "
        "This benchmark uses `POST .../code-summarize/query` with a sized `question` so cases stay "
        "comparable to 10k–60k char loads."
    )
    lines.append("")

    current_cap: str | None = None
    for cap_id in CAPABILITY_ORDER:
        cap_rows = [r for r in rows if r.capability_id == cap_id]
        title = cap_rows[0].capability_title if cap_rows else cap_id
        lines.append(f"## {title} (`{cap_id}`)")
        lines.append("")
        lines.append(
            "| Case | Target chars | Actual chars | ~Tokens | HTTP | Latency (s) | Response chars | "
            "CPU Δ MiB | GPU Δ MiB | KV est (MiB) | Notes |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in cap_rows:
            lines.append(
                f"| {r.case_label} | {r.target_context_chars} | {r.actual_answer_chars} | {r.approx_prompt_tokens} | "
                f"{r.http_status} | {r.latency_sec:.2f} | {r.response_chars} | {r.server_mem_delta_mib:.2f} | "
                f"{r.gpu_mem_delta_mib:.2f} | {r.kv_cache_est_prompt_mib:.2f} | {r.notes or '-'} |"
            )
        lines.append("")

    lines.append("## Analysis")
    lines.append("")
    lines.append("- Sequential runs; timings include RAG/embeddings where applicable.")
    lines.append("- CPU/GPU memory is sampled before/after each request (not peak within the call).")
    lines.append("- When `LLAMA_N_CTX` is small, large answer payloads are truncated server-side.")

    md_text = "\n".join(lines)
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
