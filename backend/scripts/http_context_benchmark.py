import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from dotenv import dotenv_values


def _run_ps(command: str) -> str:
    full = ["powershell", "-NoProfile", "-Command", command]
    return subprocess.check_output(full, text=True).strip()


def _get_server_pid(port: int = 8000) -> int:
    cmd = f"(Get-NetTCPConnection -State Listen -LocalPort {port} | Select-Object -First 1 -ExpandProperty OwningProcess)"
    out = _run_ps(cmd)
    return int(out) if out else 0


def _get_process_mem_mib(pid: int) -> float:
    if pid <= 0:
        return 0.0
    try:
        out = _run_ps(f"(Get-Process -Id {pid}).WorkingSet64")
        b = int(out)
        return round(b / (1024 * 1024), 2)
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
    # Approximation: 2 (K,V) * n_layer * n_ctx * n_embd * bytes_per_elem
    total = 2 * n_layer * n_ctx * n_embd * bytes_per_elem
    return round(total / (1024 * 1024), 2)


def _build_code_blob(chars: int) -> str:
    line = "def service_flow(user_id: str) -> str:\n    return 'public service delivery and grievance resolution workflow'\n\n"
    out = "# synthetic benchmark content\n\n"
    while len(out) < chars:
        out += line
    return out[:chars]


@dataclass
class Row:
    case: str
    context_chars: int
    approx_prompt_tokens: int
    status_code: int
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dotenv_values(root / ".env")
    n_ctx = int((env.get("LLAMA_N_CTX") or "8192").strip())

    url = "http://127.0.0.1:8000/api/v1/code-summarize/file"
    sizes = [10000, 20000, 30000, 40000, 50000, 60000]
    rows: list[Row] = []

    pid = _get_server_pid(8000)
    if pid <= 0:
        raise RuntimeError("No listening process found on port 8000. Start backend first.")

    for s in sizes:
        content = _build_code_blob(s)
        approx_toks = max(1, int(len(content) / 3.5))
        mem_before = _get_process_mem_mib(pid)
        gpu_before = _get_gpu_mem_mib()
        started = time.perf_counter()
        notes = ""
        status = 0
        resp_chars = 0

        try:
            r = requests.post(
                url,
                json={
                    "content": content,
                    "language": "python",
                    "file_path": f"bench_{s}.py",
                    "detail_level": "short",
                },
                timeout=1200,
            )
            status = r.status_code
            if r.headers.get("content-type", "").startswith("application/json"):
                body = r.json()
                resp_chars = len((body.get("summary") or ""))
            else:
                resp_chars = len(r.text or "")
        except Exception as e:
            notes = f"request failed: {str(e)[:180]}"

        elapsed = time.perf_counter() - started
        mem_after = _get_process_mem_mib(pid)
        gpu_after = _get_gpu_mem_mib()

        if status >= 400 and not notes:
            notes = f"http {status}"
        if approx_toks > n_ctx:
            notes = (notes + "; " if notes else "") + "input likely exceeds configured n_ctx (server may truncate)."

        rows.append(
            Row(
                case=f"{s//1000}k",
                context_chars=s,
                approx_prompt_tokens=approx_toks,
                status_code=status,
                latency_sec=round(elapsed, 2),
                response_chars=resp_chars,
                server_mem_before_mib=mem_before,
                server_mem_after_mib=mem_after,
                server_mem_delta_mib=round(mem_after - mem_before, 2),
                gpu_mem_before_mib=gpu_before,
                gpu_mem_after_mib=gpu_after,
                gpu_mem_delta_mib=round(gpu_after - gpu_before, 2),
                kv_cache_est_prompt_mib=_kv_cache_mib(min(approx_toks, n_ctx)),
                notes=notes,
            )
        )

    out_dir = root / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "context_benchmark_report.md"
    json_path = out_dir / "context_benchmark_results_http.json"

    json_path.write_text(json.dumps([r.__dict__ for r in rows], indent=2), encoding="utf-8")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append("# Context Benchmark Report (10k to 60k)")
    md.append("")
    md.append(f"- Generated at: `{now}`")
    md.append(f"- Endpoint tested: `{url}`")
    md.append(f"- Backend PID on port 8000: `{pid}`")
    md.append(f"- Configured `LLAMA_N_CTX` from `.env`: `{n_ctx}`")
    md.append(f"- Estimated full KV cache at `LLAMA_N_CTX`: `{_kv_cache_mib(n_ctx)} MiB` (assumed 40 layers, 3584 hidden, fp16 KV)")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| Case | Context Size (chars) | Approx Prompt Tokens | HTTP | Latency (s) | Response Size (chars) | Server Mem Before (MiB) | Server Mem After (MiB) | Server Mem Delta (MiB) | GPU Mem Before (MiB) | GPU Mem After (MiB) | GPU Mem Delta (MiB) | KV Est at Prompt (MiB) | Notes |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"| {r.case} | {r.context_chars} | {r.approx_prompt_tokens} | {r.status_code} | {r.latency_sec:.2f} | {r.response_chars} | "
            f"{r.server_mem_before_mib:.2f} | {r.server_mem_after_mib:.2f} | {r.server_mem_delta_mib:.2f} | "
            f"{r.gpu_mem_before_mib:.2f} | {r.gpu_mem_after_mib:.2f} | {r.gpu_mem_delta_mib:.2f} | {r.kv_cache_est_prompt_mib:.2f} | {r.notes or '-'} |"
        )
    md.append("")
    md.append("## Analysis")
    md.append("")
    md.append("- `LLAMA_N_CTX` is lower than the requested 60k in the currently running backend configuration, so higher-size tests may be truncated by server logic.")
    md.append("- Memory values are sampled before and after each request; transient peaks between samples can be higher.")
    md.append("- KV cache values are estimates (not direct llama.cpp counters).")
    md.append("- For strict 60k-token validation, restart backend with high `LLAMA_N_CTX` and a stable llama.cpp build for that window.")

    md_path.write_text("\n".join(md), encoding="utf-8")
    print(str(md_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
