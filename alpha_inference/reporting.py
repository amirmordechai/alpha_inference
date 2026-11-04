"""Candidate validation, Vertex GPU management, and articulated report generation."""

from __future__ import annotations

import ast
import base64
import csv
import json
import os
import pprint
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpha_inference.adapters import (
    SERVED_MODEL_NAME,
    build_chat_payload,
    build_frozen_jev_payload,
    fetch_gcp_bearer_token,
    resolve_completions_url,
)
from alpha_inference.evaluator import METRIC_INFO, TraceEvalResult, format_metric, normalize_metric
MACHINE_TYPE = os.environ.get("GPU_MACHINE_TYPE", "g4-standard-48")
ACCELERATOR_TYPE = os.environ.get("GPU_ACCELERATOR_TYPE", "NVIDIA_RTX_PRO_6000")
ACCELERATOR_COUNT = int(os.environ.get("GPU_ACCELERATOR_COUNT", "1"))

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_SEED_FILE = REPO_ROOT / "candidate.py"
RELOAD_PROXY_FILE = Path(__file__).resolve().parent / "reload_proxy.py"

FORBIDDEN_VLLM_FLAGS = (
    "--model",
    "--quantization",
    "--load-format",
    "--dtype",
    "--kv-cache-dtype=fp8",
    "--seed",
    "--speculative-config",
    "--chat-template",
    "--tokenizer",
)
FORBIDDEN_KERNEL_IMPORTS = {
    "socket",
    "urllib",
    "requests",
    "aiohttp",
    "httpx",
    "subprocess",
    "ftplib",
}


@dataclass
class CandidateSpec:
    """Parsed candidate serving configuration & optional custom GPU kernel."""

    name: str
    plain_title: str
    hypothesis: str
    category: str
    vllm_args: list[str]
    env_vars: dict[str, str]
    custom_kernel_code: str = ""
    raw_source: str = ""


def extract_flag_value(args: Sequence[str], flag_name: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag_name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"{flag_name}="):
            return a.split("=", 1)[1]
    return None


def parse_candidate_code(code: str, name: str = "candidate") -> CandidateSpec:
    """Safely execute `candidate.py` source and extract `get_candidate_config()`."""
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(code, "candidate.py", "exec"), namespace)
    fn = namespace.get("get_candidate_config")
    if not callable(fn):
        raise ValueError("Missing callable `get_candidate_config()` in candidate program.")
    cfg = fn()
    if not isinstance(cfg, dict):
        raise ValueError("`get_candidate_config()` must return a Python dict.")
    vllm_args = [str(x) for x in (cfg.get("vllm_args") or [])]
    env_raw = cfg.get("env_vars") or {}
    if not isinstance(env_raw, dict):
        raise ValueError("`env_vars` must be a dict[str, str].")
    hyp = str(cfg.get("hypothesis") or "").strip()
    title = str(cfg.get("plain_title") or hyp[:48] or name).strip()
    cat = str(cfg.get("category") or "vllm_flags").strip()
    if cat not in ("vllm_flags", "cuda_graphs", "backends", "triton_kernels", "baseline"):
        cat = "triton_kernels" if str(cfg.get("custom_kernel_code") or "").strip() else "vllm_flags"
    return CandidateSpec(
        name=name,
        plain_title=title[:56],
        hypothesis=hyp,
        category=cat,
        vllm_args=vllm_args,
        env_vars={str(k): str(v) for k, v in env_raw.items()},
        custom_kernel_code=str(cfg.get("custom_kernel_code") or ""),
        raw_source=code,
    )


def build_seed_program_source(model_override: str = "", served_model_name: str = "") -> str:
    """Load `candidate.py` or build a clean seed program for `--model <hf-id>`."""
    seed_src = CANDIDATE_SEED_FILE.read_text(encoding="utf-8")
    if not model_override.strip():
        return seed_src
    default_spec = parse_candidate_code(seed_src, name="seed")
    default_served = extract_flag_value(default_spec.vllm_args, "--served-model-name") or SERVED_MODEL_NAME
    if model_override.strip() in (default_spec.vllm_args[0], default_served) and not served_model_name.strip():
        return seed_src

    served = served_model_name.strip() or model_override.strip()
    vllm_args = [
        model_override.strip(),
        f"--served-model-name={served}",
        "--host=0.0.0.0",
        "--port=8080",
        "--max-logprobs=16",
        "--enable-prefix-caching",
        "--max-num-seqs=16",
        "--max-model-len=8192",
        "--gpu-memory-utilization=0.90",
    ]
    return (
        '"""Auto-generated seed configuration for custom model."""\n\n'
        "from typing import Any\n\n\n"
        "# EVOLVE-BLOCK-START\n"
        "# IMPORTANT: Keep the function name `get_candidate_config` and return dict keys unchanged.\n"
        "def get_candidate_config() -> dict[str, Any]:\n"
        "    return {\n"
        '        "plain_title": "Out-of-the-box vLLM defaults (batch 16)",\n'
        f'        "hypothesis": "Ground-Zero baseline for {model_override.strip()}",\n'
        '        "category": "vllm_flags",\n'
        f'        "vllm_args": {pprint.pformat(vllm_args, indent=12)},\n'
        '        "env_vars": {"PYTHONUNBUFFERED": "1"},\n'
        '        "custom_kernel_code": "",\n'
        "    }\n"
        "# EVOLVE-BLOCK-END\n"
    )


def validate_candidate_spec(cand: CandidateSpec, baseline: CandidateSpec) -> tuple[bool, str]:
    """Validate untouchable model/precision constraints and custom kernel AST safety."""
    if not cand.vllm_args:
        return False, "vllm_args cannot be empty"
    if cand.vllm_args[0] != baseline.vllm_args[0]:
        return False, f"Model checkpoint must stay {baseline.vllm_args[0]!r}; got {cand.vllm_args[0]!r}"

    req_served = extract_flag_value(baseline.vllm_args, "--served-model-name")
    got_served = extract_flag_value(cand.vllm_args, "--served-model-name")
    if req_served and got_served != req_served:
        return False, f"--served-model-name must stay {req_served!r}; got {got_served!r}"

    base_diff = extract_flag_value(baseline.vllm_args, "--diffusion-config")
    if base_diff:
        cand_diff = extract_flag_value(cand.vllm_args, "--diffusion-config")
        if not cand_diff or "canvas_length" not in cand_diff:
            return False, "--diffusion-config with canvas_length must remain intact"

    max_lp = extract_flag_value(cand.vllm_args, "--max-logprobs")
    if max_lp is None or int(max_lp) < 5:
        return False, f"--max-logprobs must be >= 5; got {max_lp!r}"

    min_len = int(extract_flag_value(baseline.vllm_args, "--max-model-len") or 4096)
    got_len = extract_flag_value(cand.vllm_args, "--max-model-len")
    if got_len is None or int(got_len) < min_len:
        return False, f"--max-model-len must be >= {min_len}; got {got_len!r}"

    for arg in cand.vllm_args[1:]:
        if arg == "--disable-log-requests" or arg.startswith("--disable-log-requests="):
            return False, "Flag '--disable-log-requests' was removed in vLLM v0.31+; use '--disable-uvicorn-access-log' instead."
        for bad in FORBIDDEN_VLLM_FLAGS:
            if arg == bad or arg.startswith(f"{bad}="):
                return False, f"Forbidden model/precision-altering flag: {arg!r}"

    if str(cand.env_vars.get("CUDA_MODULE_LOADING", "")).strip().upper() == "EAGER":
        return False, "CUDA_MODULE_LOADING='EAGER' triggers multi-minute PTX JIT compilation stalls on Blackwell sm_120; omit it."
    if "expandable_segments" in str(cand.env_vars.get("PYTORCH_CUDA_ALLOC_CONF", "")):
        return False, "PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' conflicts with vLLM CUDA graph memory pools; omit it."

    if cand.custom_kernel_code.strip():
        try:
            tree = ast.parse(cand.custom_kernel_code)
        except SyntaxError as exc:
            return False, f"SyntaxError in custom_kernel_code: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_root = alias.name.split(".")[0]
                    if mod_root in FORBIDDEN_KERNEL_IMPORTS:
                        return False, f"Forbidden import {mod_root!r} in custom_kernel_code"
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod_root = node.module.split(".")[0]
                if mod_root in FORBIDDEN_KERNEL_IMPORTS:
                    return False, f"Forbidden import {mod_root!r} in custom_kernel_code"

    return True, "ok"


def deploy_search_vertex_endpoint(
    project_id: str,
    region: str,
    ar_repo: str,
    baseline_cand: CandidateSpec,
    display_name: str,
    skip_mirror: bool = False,
    *,
    upstream_image: str = "",
    machine_type: str = MACHINE_TYPE,
    accelerator_type: str = ACCELERATOR_TYPE,
    accelerator_count: int = ACCELERATOR_COUNT,
) -> dict[str, Any]:
    """Deploy a warm hot-reloadable Vertex AI Dedicated GPU Endpoint matching any GPU/model spec."""
    from alpha_inference.vertex import DEFAULT_UPSTREAM_VLLM_IMAGE, deploy_vertex_worker_or_endpoint

    info = deploy_vertex_worker_or_endpoint(
        cand=baseline_cand,
        project_id=project_id,
        region=region,
        ar_repo=ar_repo,
        display_name=display_name,
        skip_mirror=skip_mirror,
        upstream_image=upstream_image or DEFAULT_UPSTREAM_VLLM_IMAGE,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        hot_reload=True,
    )
    info["deploy_wall_sec"] = info.get("total_deploy_sec", 0.0)
    return info


async def reload_endpoint(
    endpoint_url: str,
    admin_token: str,
    cand: CandidateSpec,
    use_gcp_auth: bool = True,
    startup_timeout_sec: float = 420.0,
) -> dict[str, Any]:
    """Hot-swap `vLLM` flags, env vars, and custom Triton kernels in ~50-65s."""
    url = resolve_completions_url(endpoint_url)
    headers = {"Content-Type": "application/json", "X-AE-Admin-Token": admin_token}
    if use_gcp_auth:
        headers["Authorization"] = f"Bearer {fetch_gcp_bearer_token()}"
    safe_env = {
        k: v
        for k, v in cand.env_vars.items()
        if not (k == "CUDA_MODULE_LOADING" and str(v).strip().upper() == "EAGER")
        and not (k == "PYTORCH_CUDA_ALLOC_CONF" and "expandable_segments" in str(v))
    }
    if use_gcp_auth:
        safe_env.setdefault("HF_HUB_OFFLINE", "1")
        safe_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    safe_args = [a for a in cand.vllm_args if not a.startswith("--disable-log-requests")]
    payload = {
        "model": "__ae_admin_reload__",
        "admin_token": admin_token,
        "vllm_args": safe_args,
        "env_vars": safe_env,
        "custom_kernel_code": cand.custom_kernel_code,
        "startup_timeout_sec": startup_timeout_sec,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=startup_timeout_sec + 45.0)) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                return await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Reload transport/timeout error: {exc}", "log_tail": []}


async def capture_gpu_profile(
    endpoint_url: str,
    admin_token: str,
    sample_req: Mapping[str, Any],
    *,
    task_mode: str = "jev_diffusion",
    served_model_name: str = SERVED_MODEL_NAME,
    use_gcp_auth: bool = True,
) -> list[dict[str, Any]]:
    """Capture top GPU CUDA kernels via `reload_proxy.py`'s Kineto profiler."""
    if not admin_token:
        return []
    url = resolve_completions_url(endpoint_url)
    headers = {"Content-Type": "application/json", "X-AE-Admin-Token": admin_token}
    if use_gcp_auth:
        headers["Authorization"] = f"Bearer {fetch_gcp_bearer_token()}"
    sample_payload = (
        build_frozen_jev_payload(sample_req, served_model_name=served_model_name)
        if task_mode == "jev_diffusion"
        else build_chat_payload(sample_req, served_model_name)
    )
    payload = {
        "model": "__ae_admin_profile__",
        "admin_token": admin_token,
        "sample_payload": sample_payload,
        "num_burst": 8,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90.0)) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                if data.get("ok"):
                    return list(data.get("top_gpu_kernels") or [])
    except Exception:  # noqa: BLE001
        pass
    return []


def inspect_vertex_endpoint(project_id: str, region: str, endpoint_id: str) -> dict[str, Any]:
    """Resolve a live Vertex AI Endpoint ID into its `:rawPredict` URL, hot-reload token, and model args."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    ep = aiplatform.Endpoint(endpoint_name=endpoint_id)
    dns = getattr(ep, "dedicated_endpoint_dns", "") or ""
    if dns:
        url = f"https://{dns}/v1/projects/{project_id}/locations/{region}/endpoints/{ep.name}:rawPredict"
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{ep.name}:rawPredict"

    admin_token = ""
    vllm_args: list[str] = []
    for dm in ep.gca_resource.deployed_models:
        try:
            m = aiplatform.Model(model_name=dm.model)
            cspec = m.gca_resource.container_spec
            env_map = {e.name: e.value for e in (cspec.env or [])}
            if env_map.get("AE_ADMIN_TOKEN"):
                admin_token = env_map["AE_ADMIN_TOKEN"]
            if env_map.get("AE_INITIAL_SPEC_JSON"):
                init_obj = json.loads(env_map["AE_INITIAL_SPEC_JSON"])
                vllm_args = [str(x) for x in (init_obj.get("vllm_args") or [])]
            elif cspec.args:
                vllm_args = [str(x) for x in cspec.args]
            break
        except Exception:  # noqa: BLE001
            continue

    return {
        "endpoint_id": ep.name,
        "endpoint_resource_name": ep.resource_name,
        "dedicated_endpoint_dns": dns,
        "chat_completions_url": url,
        "admin_token": admin_token,
        "has_hot_reload": bool(admin_token),
        "vllm_args": vllm_args,
    }


TSV_COLUMNS = (
    "iteration",
    "candidate_id",
    "status",
    "category",
    "plain_title",
    "p99_latency_ms",
    "p50_latency_ms",
    "ttft_p99_ms",
    "throughput_rps",
    "mean_kl",
    "exact_match_pct",
    "eval_wall_sec",
    "hypothesis",
)


def append_results_tsv(
    tsv_path: Path,
    iteration: int,
    candidate_id: str,
    status: str,
    cand: CandidateSpec,
    res: TraceEvalResult,
    note: str = "",
) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if write_header:
            w.writerow(TSV_COLUMNS)
        w.writerow(
            [
                iteration,
                candidate_id,
                status,
                cand.category,
                cand.plain_title.replace("\t", " "),
                f"{res.p99_latency_ms:.2f}",
                f"{res.p50_latency_ms:.2f}",
                f"{res.ttft_p99_ms:.2f}",
                f"{res.throughput_rps:.2f}",
                f"{res.equivalence.mean_kl_divergence:.5f}",
                f"{res.equivalence.exact_slot_match_pct:.2f}",
                f"{res.eval_wall_sec:.1f}",
                (note or cand.hypothesis).replace("\t", " ").replace("\n", " ")[:240],
            ]
        )


def plot_progress_chart(tsv_path: Path, out_png: Path, metric: str = "p99_latency") -> None:
    if not tsv_path.exists():
        return
    m = normalize_metric(metric)
    col_map = {
        "p99_latency": "p99_latency_ms",
        "latency": "p50_latency_ms",
        "throughput": "throughput_rps",
        "ttft": "ttft_p99_ms",
    }
    col = col_map[m]
    maximize = METRIC_INFO[m]["direction"] == "maximize"

    xs_all, ys_all = [], []
    xs_keep, ys_keep, titles_keep, cats_keep = [], [], [], []
    with tsv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            it = int(row["iteration"])
            val = float(row[col])
            st = row["status"]
            if (maximize and val <= 0.0) or (not maximize and val >= 90000.0):
                continue
            xs_all.append(it)
            ys_all.append(val)
            if st in ("baseline", "keep"):
                xs_keep.append(it)
                ys_keep.append(val)
                titles_keep.append(row["plain_title"])
                cats_keep.append(row["category"])

    if not xs_all:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="-", linewidth=0.6, color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(xs_all, ys_all, s=28, color="#cbd5e1", alpha=0.8, label="Evaluated candidates", zorder=2)
    if xs_keep:
        step_x = [*xs_keep, max(xs_all)] if max(xs_all) > xs_keep[-1] else xs_keep
        step_y = [*ys_keep, ys_keep[-1]] if max(xs_all) > xs_keep[-1] else ys_keep
        ax.step(step_x, step_y, where="post", linewidth=2.4, color="#16a34a", label="Best frontier", zorder=3)
        ax.scatter(xs_keep, ys_keep, s=68, color="#16a34a", edgecolors="white", linewidths=1.2, zorder=4)
        for idx, (xk, yk, tk, ck) in enumerate(zip(xs_keep, ys_keep, titles_keep, cats_keep)):
            if idx == 0 or idx == len(xs_keep) - 1 or len(xs_keep) <= 7:
                tag = "Kernel: " if ck == "triton_kernels" else ""
                offset_y = 10 if idx % 2 == 0 else -14
                ax.annotate(
                    f"{tag}{tk}",
                    (xk, yk),
                    textcoords="offset points",
                    xytext=(4, offset_y),
                    fontsize=8.2,
                    fontweight="bold" if idx == len(xs_keep) - 1 else "normal",
                    color="#0f172a",
                )

    ax.set_xlabel("AlphaEvolve Candidate Evaluation #", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{METRIC_INFO[m]['label']} ({METRIC_INFO[m]['unit']})", fontsize=10.5, fontweight="bold")
    ax.set_title("Google Cloud AlphaEvolve — Inference Optimization Trajectory", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_articulated_summary(
    out_dir: Path,
    *,
    metric: str,
    base_res: TraceEvalResult,
    best_cand: CandidateSpec,
    best_res: TraceEvalResult,
    kept_history: Sequence[Mapping[str, Any]],
    total_evaluated: int,
    spend_usd: float,
    experiment_name: str = "",
    state: str = "COMPLETED",
) -> None:
    """Write a crisp, developer-friendly `summary.md` with Baseline vs. Best and Top Breakthroughs."""
    m = normalize_metric(metric)
    base_v = base_res.metric_value(m)
    best_v = best_res.metric_value(m)
    pct = ((best_v - base_v) / base_v * 100.0) if base_v > 0 else 0.0

    def _delta(b: float, e: float) -> str:
        return f"{((e - b) / b * 100.0):+.1f}%" if b > 0 else "—"

    lines = [
        f"# Alpha Inference Optimization Report (`{state}`)",
        "",
        f"- **Updated (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Official AlphaEvolve Experiment:** `{experiment_name.split('/')[-1] if experiment_name else 'local'}`",
        f"- **Optimized Metric:** `{m}` ({METRIC_INFO[m]['label']})",
        f"- **Candidates Evaluated:** `{total_evaluated}` (`{len(kept_history)}` record-setting breakthroughs)",
        f"- **Estimated Cloud Spend:** `${spend_usd:.2f}`",
        "",
        "## Ground-Zero Baseline vs. Best Evolved",
        "",
        "| Metric | Baseline (`#0`) | Best Evolved | Change |",
        "| :--- | ---: | ---: | :--- |",
        f"| **Target ({METRIC_INFO[m]['label']})** | `{format_metric(base_v, m)}` | **`{format_metric(best_v, m)}`** | **`{pct:+.1f}%`** |",
        f"| **P99 Latency** | `{base_res.p99_latency_ms:.2f} ms` | `{best_res.p99_latency_ms:.2f} ms` | `{_delta(base_res.p99_latency_ms, best_res.p99_latency_ms)}` |",
        f"| **Median (P50) Latency** | `{base_res.p50_latency_ms:.2f} ms` | `{best_res.p50_latency_ms:.2f} ms` | `{_delta(base_res.p50_latency_ms, best_res.p50_latency_ms)}` |",
        f"| **Sustained Throughput** | `{base_res.throughput_rps:.2f} req/s` | `{best_res.throughput_rps:.2f} req/s` | `{_delta(base_res.throughput_rps, best_res.throughput_rps)}` |",
        f"| **KL Divergence vs. Baseline** | `0.00000` | `{best_res.equivalence.mean_kl_divergence:.5f}` | `Exact match: {best_res.equivalence.exact_slot_match_pct:.2f}%` |",
        "",
    ]

    if kept_history:
        lines.extend(
            [
                "## Discovered Breakthroughs",
                "",
                "| Eval # | Category | What AlphaEvolve Changed | P99 Latency | Throughput |",
                "| ---: | :--- | :--- | ---: | ---: |",
            ]
        )
        for row in kept_history:
            lines.append(
                f"| `#{row['iteration']}` | `{row['category']}` | {row['plain_title']} | "
                f"`{row['p99_latency_ms']:.2f} ms` | `{row['throughput_rps']:.2f} req/s` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Reproduce & Roll Out to Vertex AI",
            f"- Inspect exact diff against baseline: `python optimize.py --diff --out-dir {out_dir}`",
            f"- Roll out winning configuration with native `gcloud ai`: `bash {out_dir}/gcloud_rollout.sh`",
            f"- Validate on unseen real-time trace: `python validate.py --deploy-vertex --auto-teardown --out-dir {out_dir}`",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
