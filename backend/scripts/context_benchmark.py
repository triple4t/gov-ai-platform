import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _bytes_to_mib(v: int) -> float:
    return round(v / (1024 * 1024), 2)


def _get_process_memory_bytes() -> int:
    # Windows working set via psapi (fallbacks to 0 on unsupported systems)
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = k32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.WorkingSetSize)
    except Exception:
        pass
    return 0


def _get_gpu_memory_bytes() -> int:
    # nvidia-smi query in MiB; sums all visible GPUs.
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5)
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return int(sum(vals) * 1024 * 1024)
    except Exception:
        return 0


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


@dataclass
class CaseResult:
    case_name: str
    target_prompt_tokens: int
    actual_prompt_tokens: int
    completion_tokens: int
    duration_sec: float
    tokens_per_sec: float
    process_mem_before_mib: float
    process_mem_after_mib: float
    process_mem_delta_mib: float
    gpu_mem_before_mib: float
    gpu_mem_after_mib: float
    gpu_mem_delta_mib: float
    kv_cache_estimated_mib: float
    status: str
    error: str


def _build_exact_token_text(llm: Any, target_tokens: int) -> str:
    base = (
        "Government services, welfare schemes, identity verification, digital records, "
        "public grievance handling, and policy implementation details are listed below. "
    )
    text = base
    while len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)) < target_tokens:
        text += base

    # Binary trim to match target token count as closely as possible.
    lo, hi = 0, len(text)
    best = text
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        t = len(llm.tokenize(candidate.encode("utf-8"), add_bos=False, special=True))
        if t <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _estimate_kv_cache_mib(meta: dict[str, Any], n_ctx: int) -> float:
    # Approximate llama.cpp KV memory:
    # bytes ~= 2 (K,V) * n_layer * n_ctx * n_embd * bytes_per_elem
    n_layer = _safe_int(
        meta.get("llama.block_count")
        or meta.get("qwen2.block_count")
        or meta.get("qwen3.block_count"),
        0,
    )
    n_embd = _safe_int(
        meta.get("llama.embedding_length")
        or meta.get("qwen2.embedding_length")
        or meta.get("qwen3.embedding_length"),
        0,
    )
    bytes_per_elem = 2  # typical fp16 KV cache
    if n_layer <= 0 or n_embd <= 0:
        return 0.0
    est = 2 * n_layer * n_ctx * n_embd * bytes_per_elem
    return _bytes_to_mib(est)


def _build_report_md(
    out_path: Path,
    model_path: str,
    n_ctx: int,
    kv_est_mib_full_ctx: float,
    rows: list[CaseResult],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Context Benchmark Report (10k to 60k)")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Configured `n_ctx`: `{n_ctx}`")
    lines.append(f"- Estimated KV cache at full `n_ctx`: `{kv_est_mib_full_ctx} MiB`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Case | Target Prompt Tokens | Actual Prompt Tokens | Completion Tokens | Response Time (s) | Output Tok/s | CPU Mem Before (MiB) | CPU Mem After (MiB) | CPU Delta (MiB) | GPU Mem Before (MiB) | GPU Mem After (MiB) | GPU Delta (MiB) | KV Est (MiB) | Status |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for r in rows:
        lines.append(
            f"| {r.case_name} | {r.target_prompt_tokens} | {r.actual_prompt_tokens} | {r.completion_tokens} | "
            f"{r.duration_sec:.2f} | {r.tokens_per_sec:.2f} | {r.process_mem_before_mib:.2f} | {r.process_mem_after_mib:.2f} | "
            f"{r.process_mem_delta_mib:.2f} | {r.gpu_mem_before_mib:.2f} | {r.gpu_mem_after_mib:.2f} | {r.gpu_mem_delta_mib:.2f} | "
            f"{r.kv_cache_estimated_mib:.2f} | {r.status} |"
        )
        if r.error:
            lines.append(f"|  |  |  |  |  |  |  |  |  |  |  |  |  | Error: `{r.error}` |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CPU memory = process working set of benchmark process.")
    lines.append("- GPU memory = `nvidia-smi` sampled before/after each request.")
    lines.append("- KV cache is an estimate using model metadata (`n_layer`, `n_embd`) and fp16 cache assumption.")
    lines.append("- If any case fails, check `LLAMA_N_CTX`, VRAM headroom, and llama.cpp build limits.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    # Force benchmark context window unless user has explicitly set one higher.
    env_ctx = int(os.getenv("LLAMA_N_CTX", "8192"))
    if env_ctx < 60000:
        os.environ["LLAMA_N_CTX"] = "65536"

    from app.core.config import settings
    from app.core.local_llm import get_chat_llama

    llm = get_chat_llama()
    meta = llm.metadata or {}
    n_ctx = int(llm.n_ctx())

    cases = [10000, 20000, 30000, 40000, 50000, 60000]
    kv_est_full = _estimate_kv_cache_mib(meta, n_ctx)
    results: list[CaseResult] = []

    for t in cases:
        name = f"{t//1000}k"
        prompt_text = _build_exact_token_text(llm, t)
        prompt_tokens = len(llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True))
        messages = [
            {"role": "system", "content": "You are a concise benchmarking assistant."},
            {"role": "user", "content": f"Summarize this context in 5 bullet points:\n\n{prompt_text}"},
        ]

        mem_before = _get_process_memory_bytes()
        gpu_before = _get_gpu_memory_bytes()
        started = time.perf_counter()
        status = "ok"
        err = ""
        completion_tokens = 0

        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=128, temperature=0.0)
            usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
            completion_tokens = _safe_int(usage.get("completion_tokens"), 0)
        except Exception as e:
            status = "failed"
            err = str(e).strip().replace("\n", " ")[:300]
        elapsed = time.perf_counter() - started
        mem_after = _get_process_memory_bytes()
        gpu_after = _get_gpu_memory_bytes()

        out_tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        kv_est = _estimate_kv_cache_mib(meta, prompt_tokens)

        results.append(
            CaseResult(
                case_name=name,
                target_prompt_tokens=t,
                actual_prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_sec=elapsed,
                tokens_per_sec=out_tps,
                process_mem_before_mib=_bytes_to_mib(mem_before),
                process_mem_after_mib=_bytes_to_mib(mem_after),
                process_mem_delta_mib=_bytes_to_mib(mem_after - mem_before),
                gpu_mem_before_mib=_bytes_to_mib(gpu_before),
                gpu_mem_after_mib=_bytes_to_mib(gpu_after),
                gpu_mem_delta_mib=_bytes_to_mib(gpu_after - gpu_before),
                kv_cache_estimated_mib=kv_est,
                status=status,
                error=err,
            )
        )

    out_dir = root / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "context_benchmark_report.md"
    json_path = out_dir / "context_benchmark_results.json"

    _build_report_md(
        out_path=report_path,
        model_path=settings.LOCAL_CHAT_GGUF_PATH,
        n_ctx=n_ctx,
        kv_est_mib_full_ctx=kv_est_full,
        rows=results,
    )
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(str(report_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
