#!/usr/bin/env python3
"""Optimize GenAI serving on a Vertex AI Endpoint or GPU VM using Google Cloud AlphaEvolve.

Quickstart:
    # 1. Dry-run check ($0 spend):
    python optimize.py --dry-run

    # 2. Optimize against any live Vertex AI GPU Endpoint (auto-inherits GPU & model; auto-tears down search workers):
    python optimize.py \
        --vertex-endpoint-id "$VERTEX_ENDPOINT_ID" \
        --trace examples/custom_openai_chat/sample_trace.jsonl \
        --metric p99_latency \
        --max-programs 100 \
        --budget-usd 50 \
        --out-dir results/my_workload

    # 3. Optimize an existing GPU VM running `alpha_inference.reload_proxy`:
    python optimize.py \
        --trace examples/custom_openai_chat/sample_trace.jsonl \
        --endpoint http://localhost:8080 --admin-token "$AE_ADMIN_TOKEN" \
        --model google/gemma-4-26B-A4B-it \
        --metric p99_latency \
        --max-programs 40 \
        --budget-usd 25
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import google.auth
import google.auth.transport.requests
import nest_asyncio
from alpha_evolve.client import AlphaEvolveClient
from alpha_evolve.controller import run_controller_loop
from alpha_evolve.experiment import AlphaEvolveExperiment
from dotenv import load_dotenv

from alpha_inference.adapters import TASK_MODE_CHOICES, detect_task_mode, load_jsonl_requests
from alpha_inference.equivalence import EquivalenceReport
from alpha_inference.evaluator import (
    METRIC_CHOICES,
    METRIC_INFO,
    TraceEvalResult,
    evaluate_candidate_on_endpoint,
    format_metric,
    is_better,
    normalize_metric,
    to_alpha_evolve_score,
)
from alpha_inference.reporting import (
    ACCELERATOR_COUNT,
    ACCELERATOR_TYPE,
    MACHINE_TYPE,
    CandidateSpec,
    append_results_tsv,
    build_seed_program_source,
    capture_gpu_profile,
    deploy_search_vertex_endpoint,
    extract_flag_value,
    introspect_endpoint_model,
    parse_candidate_code,
    plot_progress_chart,
    reload_endpoint,
    scrape_engine_metrics,
    summarize_engine_metrics_delta,
    upgrade_candidate_source_for_evolution,
    validate_candidate_spec,
    write_articulated_summary,
)
from alpha_inference.vertex import (
    DEFAULT_UPSTREAM_VLLM_IMAGE,
    build_gcloud_rollout_command,
    ensure_ar_image_mirrored,
    format_candidate_diff,
    inspect_vertex_endpoint,
    teardown_endpoint,
)

logger = logging.getLogger("alpha_inference")

LLM_TOKEN_RATES_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-4-argon": (2.0, 15.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.5-flash": (1.5, 9.0),
    "gemini-2.5-flash": (0.3, 2.5),
}


class BudgetedAlphaEvolveExperiment(AlphaEvolveExperiment):
    """Official `AlphaEvolveExperiment` with a hard USD cloud-spend circuit breaker."""

    def __init__(
        self,
        ae_client: AlphaEvolveClient,
        evaluator_function: Callable,
        max_programs_evaluated: int,
        *,
        budget_usd: float,
        current_spend_fn: Callable[[], float],
        parallel_evaluation: bool = False,
    ) -> None:
        super().__init__(
            ae_client=ae_client,
            evaluator_function=evaluator_function,
            max_programs_evaluated=max_programs_evaluated,
            parallel_evaluation=parallel_evaluation,
        )
        self.budget_usd = budget_usd
        self.current_spend_fn = current_spend_fn
        self.backend_terminal = False

    def stopping_criteria_met(self) -> bool:
        if self.backend_terminal or super().stopping_criteria_met():
            return True
        spent = self.current_spend_fn()
        if spent >= self.budget_usd:
            logger.info("Hard budget cap reached ($%.2f >= $%.2f); stopping evolution cleanly.", spent, self.budget_usd)
            return True
        return False


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    default_trace = (
        "examples/email_triage/monday_peak_3m.jsonl"
        if Path("examples/email_triage/monday_peak_3m.jsonl").exists()
        else "examples/custom_openai_chat/sample_trace.jsonl"
    )

    p = argparse.ArgumentParser(
        description="Optimize GenAI serving on Vertex AI or GPU VMs using official Google Cloud AlphaEvolve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trace", "--data", dest="trace", default=default_trace, help="JSONL production request trace")
    p.add_argument("--equiv-trace", default="", help="Optional separate JSONL file for KL equivalence gate (defaults to first 80 rows of --trace)")
    p.add_argument("--task-mode", choices=TASK_MODE_CHOICES, default="auto", help="OpenAI chat workload mode")
    p.add_argument("--model", default="", help="Optional HuggingFace model override (defaults to candidate.py)")
    p.add_argument("--served-model-name", default="", help="Optional --served-model-name override")
    p.add_argument("--metric", choices=METRIC_CHOICES, default="p99_latency", help="Target metric to optimize")

    # Budget & scale
    p.add_argument("--max-programs", type=int, default=60, help="Maximum candidate programs to generate & evaluate")
    p.add_argument("--workers", type=int, default=4, help="Parallel GPU evaluation workers")
    p.add_argument("--budget-usd", type=float, default=40.0, help="Hard USD cloud-spend cap")
    p.add_argument("--hourly-gpu-usd", type=float, default=2.65, help="Per-GPU hourly rate in USD for budget tracking")
    p.add_argument("--confirm-wins", action="store_true", help="Re-verify new record candidates before promoting")

    # Endpoint / Vertex AI / GPU hardware settings
    p.add_argument("--vertex-endpoint-id", default="", help="Live Vertex AI Endpoint ID(s) created via `gcloud ai` or Model Garden")
    p.add_argument("--endpoint", "--endpoint-url", dest="endpoint", default="", help="Existing reload_proxy URL(s), comma-separated")
    p.add_argument("--admin-token", default="", help="Admin token(s) for reload_proxy hot-swap")
    p.add_argument("--deploy-vertex", action="store_true", help="Provision ephemeral Vertex AI GPU worker(s) for search")
    p.add_argument("--auto-teardown", action="store_true", help="Delete provisioned Vertex GPU worker(s) on exit")
    p.add_argument("--machine-type", default="", help=f"Override Vertex machine type (default: inherit from endpoint or {MACHINE_TYPE})")
    p.add_argument("--accelerator-type", default="", help=f"Override GPU accelerator (default: inherit from endpoint or {ACCELERATOR_TYPE})")
    p.add_argument("--accelerator-count", type=int, default=0, help=f"Override GPU count / TP size (default: inherit or {ACCELERATOR_COUNT})")
    p.add_argument("--vllm-image", default="", help=f"Override vLLM container image (default: inherit or {DEFAULT_UPSTREAM_VLLM_IMAGE})")
    p.add_argument("--skip-mirror", action="store_true", help="Skip Artifact Registry container mirror check")
    p.add_argument("--no-gcp-auth", action="store_true", help="Disable GCP Bearer auth on endpoint HTTP requests")

    # Official Google Cloud AlphaEvolve settings
    p.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID", ""), help="Google Cloud project ID")
    p.add_argument("--ge-app-id", default=os.getenv("GE_APP_ID", ""), help="Gemini Enterprise App / Engine ID")
    p.add_argument("--ae-location", default=os.getenv("AE_LOCATION", "global"), help="Discovery Engine location")
    p.add_argument("--region", default=os.getenv("GCP_REGION", os.getenv("GCP_REGION_GPU", "us-central1")), help="Vertex GPU region")
    p.add_argument("--ar-repo", default=os.getenv("GCP_AR_REPO", "alpha-inference-images"), help="Artifact Registry repo")
    p.add_argument("--model-1", default=os.getenv("AE_MODEL_1", ""), help="Optional primary AlphaEvolve model override (empty = server default)")
    p.add_argument("--model-1-weight", type=float, default=float(os.getenv("AE_MODEL_1_WEIGHT", "1.0")))
    p.add_argument("--model-2", default=os.getenv("AE_MODEL_2", ""), help="Optional secondary AlphaEvolve model")
    p.add_argument("--model-2-weight", type=float, default=float(os.getenv("AE_MODEL_2_WEIGHT", "0.0")))

    p.add_argument("--out-dir", default="examples/email_triage/results", help="Output directory for articulated results")
    p.add_argument(
        "--continue-from",
        default="",
        help="Warm-start & stitch on top of an existing results directory (starts from its best_candidate.py and appends to results.tsv)",
    )
    p.add_argument("--diff", action="store_true", help="Print unified diff between candidate.py and <out-dir>/best_candidate.py")
    p.add_argument("--dry-run", action="store_true", help="Validate trace, seed candidate, and budget with $0 spend")
    return p


def estimate_llm_spend(args: argparse.Namespace, in_toks: int, out_toks: int) -> float:
    """Estimate AlphaEvolve LLM token spend in USD from weighted model rates."""
    m1 = (args.model_1 or "gemini-4-argon").strip()
    w1 = max(0.0, float(args.model_1_weight or 1.0))
    m2 = (args.model_2 or "").strip()
    w2 = max(0.0, float(args.model_2_weight or 0.0)) if (m2 and m2 != m1) else 0.0
    tot_w = max(1e-6, w1 + w2)
    r1_in, r1_out = LLM_TOKEN_RATES_PER_1M.get(m1, (2.0, 15.0))
    r2_in, r2_out = LLM_TOKEN_RATES_PER_1M.get(m2, (2.0, 12.0)) if w2 > 0 else (0.0, 0.0)
    eff_in = (w1 * r1_in + w2 * r2_in) / tot_w
    eff_out = (w1 * r1_out + w2 * r2_out) / tot_w
    return round((in_toks / 1_000_000.0) * eff_in + (out_toks / 1_000_000.0) * eff_out, 4)


def _reconstruct_eval_from_tsv_row(row: dict[str, str], num_requests: int) -> TraceEvalResult:
    eq = EquivalenceReport(
        passed=True,
        num_checked=80,
        num_failed_http=0,
        exact_slot_matches=80,
        total_slots=80,
        exact_slot_match_pct=float(row.get("exact_match_pct") or 100.0),
        max_prob_abs_diff=0.0,
        reason="ok",
        mean_kl_divergence=float(row.get("mean_kl") or 0.0),
    )
    p99 = float(row.get("p99_latency_ms") or 0.0)
    p50 = float(row.get("p50_latency_ms") or 0.0)
    tput = float(row.get("throughput_rps") or 0.0)
    wall = float(row.get("eval_wall_sec") or 180.0)
    return TraceEvalResult(
        candidate_name=f"cand_{int(row.get('iteration', 0)):03d}",
        equivalence=eq,
        p99_latency_ms=p99,
        p50_latency_ms=p50,
        mean_latency_ms=round(0.5 * (p50 + p99), 2),
        ttft_p50_ms=0.0,
        ttft_p99_ms=float(row.get("ttft_p99_ms") or 0.0),
        trace_rps=tput,
        throughput_rps=tput,
        num_requests=num_requests,
        num_failures=0,
        eval_wall_sec=wall,
    )


def _build_problem_description(
    *,
    canon_metric: str,
    base_res: TraceEvalResult,
    best_val: float,
    seed_engine_internals: Mapping[str, float],
    eff_accel: str,
    eff_accel_count: int,
    eff_machine: str,
    model_id: str,
    gpu_profile: list[dict[str, Any]] | None,
    model_introspect: Mapping[str, Any],
    cont_dir: Path | None,
    prior_eval_offset: int,
) -> str:
    problem_guide = (Path(__file__).resolve().parent / "alpha_inference" / "problem.md").read_text(encoding="utf-8")
    prof_summary = json.dumps((gpu_profile or [])[:12], indent=2) if gpu_profile else "[]"
    dir_word = METRIC_INFO[canon_metric]["direction"].upper()
    score_expr = "`+throughput_rps`" if dir_word == "MAXIMIZE" else f"`-{canon_metric}_ms`"

    gpu_detail = f"`{eff_accel}` (`x{eff_accel_count}` on `{eff_machine}`)"
    if model_introspect.get("gpu_name"):
        gpu_detail += f" — `{model_introspect['gpu_name']}` (SM `{model_introspect.get('sm_capability', '?')}`)"
    dims_json = json.dumps(model_introspect.get("model_dims") or {}, indent=2)
    vllm_mod_name = model_introspect.get("vllm_model_module", "")
    hot_src = str(model_introspect.get("hot_layer_source") or "").strip()
    internals_str = json.dumps(dict(seed_engine_internals)) if seed_engine_internals else "{}"

    introspect_section = ""
    if model_introspect.get("model_dims") or hot_src:
        introspect_section = (
            "\n## Live Introspected Model Dimensions & Target `vLLM` Layer Source\n"
            f"- **Target `vLLM` Python Module:** `{vllm_mod_name or 'auto'}`\n"
            f"- **Model Dimensions:**\n```json\n{dims_json}\n```\n"
        )
        if hot_src:
            introspect_section += (
                "- **Active Model Layer Source (target for `register_vllm_patch` `@triton.jit` fusions):**\n"
                f"```python\n{hot_src}\n```\n"
            )

    stage_note = ""
    if cont_dir is not None:
        stage_note = (
            f"- **Warm-Started Continuation (Stage 2 from Eval `#{prior_eval_offset}`):** "
            f"Scheduler & attention flags in the seed program already reached `{best_val:.2f} ms` P99 "
            f"(down from `{base_res.p99_latency_ms:.2f} ms`). Prioritize hardware `FP8` (`--quantization=fp8`, "
            "`--kv-cache-dtype=fp8`), `--kernel-config`, GPU n-gram `--speculative-config`, `moe_tuned_config` "
            "Triton tile tables, `--compilation-config`, and `@triton.jit` layer fusions!\n"
        )

    return (
        f"{problem_guide}\n\n"
        "## Active Experiment Context\n"
        f"- **Optimization Objective:** {dir_word} `{canon_metric}` ({METRIC_INFO[canon_metric]['label']}). "
        f"AlphaEvolve maximizes `fitness_score` = {score_expr}.\n"
        f"- **Ground-Zero Baseline (`#0`):** P99=`{base_res.p99_latency_ms:.2f} ms`, "
        f"P50=`{base_res.p50_latency_ms:.2f} ms`, Peak Throughput=`{base_res.throughput_rps:.2f} req/s`.\n"
        f"- **Current Seed Record:** `{format_metric(best_val, canon_metric)}` | Engine Internals: `{internals_str}`.\n"
        f"{stage_note}"
        f"- **Hardware:** {gpu_detail} serving `{model_id}`.\n"
        f"- **Live Top GPU CUDA Kernels:**\n```json\n{prof_summary}\n```\n"
        f"{introspect_section}"
    )


async def run_optimization(args: argparse.Namespace) -> dict[str, Any]:
    canon_metric = normalize_metric(args.metric)
    seed_source = build_seed_program_source(args.model, args.served_model_name)
    baseline_cand = parse_candidate_code(seed_source, name="cand_000_baseline")
    ok_seed, seed_msg = validate_candidate_spec(baseline_cand, baseline_cand)
    if not ok_seed:
        raise RuntimeError(f"Seed candidate.py failed validation: {seed_msg}")

    served_model = extract_flag_value(baseline_cand.vllm_args, "--served-model-name") or baseline_cand.vllm_args[0]
    trace_path = Path(args.trace)
    if args.equiv_trace:
        equiv_path = Path(args.equiv_trace)
    elif trace_path.name == "monday_peak_3m.jsonl" and (trace_path.parent / "equivalence_golden.jsonl").exists():
        equiv_path = trace_path.parent / "equivalence_golden.jsonl"
    else:
        equiv_path = trace_path
    trace_rows = load_jsonl_requests(trace_path)
    resolved_mode = detect_task_mode(trace_rows, requested_mode=args.task_mode, model_name=baseline_cand.vllm_args[0])
    max_programs = max(2, int(args.max_programs))
    num_workers = max(1, int(args.workers))

    cont_dir = Path(args.continue_from) if args.continue_from.strip() else None
    prior_summary: dict[str, Any] = {}
    prior_rows: list[dict[str, str]] = []
    prior_eval_offset = 0
    prior_spend_usd = 0.0
    warm_seed_source = seed_source
    warm_seed_cand = baseline_cand

    if cont_dir is not None:
        for req_f in ("best_candidate.py", "results.tsv", "optimization_summary.json", "baseline_responses.jsonl"):
            if not (cont_dir / req_f).exists():
                raise FileNotFoundError(f"--continue-from directory {cont_dir} is missing required file {req_f!r}")
        prior_summary = json.loads((cont_dir / "optimization_summary.json").read_text(encoding="utf-8"))
        with (cont_dir / "results.tsv").open("r", encoding="utf-8") as fh:
            prior_rows = list(csv.DictReader(fh, delimiter="\t"))
        if not prior_rows:
            raise RuntimeError(f"Prior results.tsv in {cont_dir} is empty")
        prior_eval_offset = max(int(r["iteration"]) for r in prior_rows)
        prior_spend_usd = float(prior_summary.get("total_spend_usd", 0.0))
        warm_seed_source = upgrade_candidate_source_for_evolution(
            (cont_dir / "best_candidate.py").read_text(encoding="utf-8")
        )
        warm_seed_cand = parse_candidate_code(warm_seed_source, name=f"cand_{prior_eval_offset:03d}_warm_seed")
        ok_ws, ws_msg = validate_candidate_spec(warm_seed_cand, baseline_cand)
        if not ok_ws:
            raise RuntimeError(f"Prior best_candidate.py failed validation: {ws_msg}")

    if args.dry_run:
        dry_out: dict[str, Any] = {
            "dry_run": True,
            "official_alpha_evolve_sdk": True,
            "metric": canon_metric,
            "metric_direction": METRIC_INFO[canon_metric]["direction"],
            "task_mode": resolved_mode,
            "trace_path": str(trace_path),
            "trace_requests": len(trace_rows),
            "equiv_trace_path": str(equiv_path),
            "model": baseline_cand.vllm_args[0],
            "served_model_name": served_model,
            "workers": num_workers,
            "max_programs": max_programs,
            "budget_usd": args.budget_usd,
            "out_dir": args.out_dir,
            "seed_candidate": {k: v for k, v in asdict(warm_seed_cand).items() if k != "raw_source"},
        }
        if cont_dir is not None:
            dry_out["continued_from"] = str(cont_dir)
            dry_out["prior_programs_evaluated"] = prior_eval_offset
            dry_out["prior_spend_usd"] = round(prior_spend_usd, 4)
        return dry_out

    if not args.project_id or not args.ge_app_id:
        raise RuntimeError("Set GCP_PROJECT_ID and GE_APP_ID in .env (or pass --project-id and --ge-app-id).")

    out_dir = Path(args.out_dir)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    if cont_dir is not None:
        if cont_dir.resolve() == out_dir.resolve():
            snap_dir = out_dir / "stage1_snapshot"
            if not snap_dir.exists():
                snap_dir.mkdir(parents=True, exist_ok=True)
                for fn in ("results.tsv", "optimization_summary.json", "summary.md", "best_candidate.py", "progress.png", "unseen_validation.json", "baseline_responses.jsonl"):
                    if (out_dir / fn).exists():
                        shutil.copy2(out_dir / fn, snap_dir / fn)
        else:
            for fn in ("results.tsv", "baseline_responses.jsonl", "best_candidate.py"):
                shutil.copy2(cont_dir / fn, out_dir / fn)
            if (cont_dir / "candidates").exists():
                for cf in (cont_dir / "candidates").glob("*.py"):
                    if not (out_dir / "candidates" / cf.name).exists():
                        shutil.copy2(cf, out_dir / "candidates" / cf.name)
    else:
        (out_dir / "candidates" / "cand_000_baseline.py").write_text(seed_source, encoding="utf-8")

    t_start = time.perf_counter()
    provisioned_endpoint_ids: list[str] = []
    worker_pool: list[tuple[int, str, str]] = []
    ae_token_state = {"input_tokens": 0, "output_tokens": 0, "llm_spend_usd": 0.0}

    def stage_gpu_spend_usd() -> float:
        tracks_vertex_gpus = args.deploy_vertex or any((":rawPredict" in u or "vertexai" in u) for _, u, _ in worker_pool)
        if not tracks_vertex_gpus:
            return 0.0
        hrs = (time.perf_counter() - t_start) / 3600.0
        n_gpus = max(1, len(worker_pool) if worker_pool else num_workers)
        return round(hrs * args.hourly_gpu_usd * n_gpus, 4)

    def stage_spend_usd() -> float:
        return round(stage_gpu_spend_usd() + float(ae_token_state["llm_spend_usd"]), 4)

    def cumulative_spend_usd() -> float:
        return round(prior_spend_usd + stage_spend_usd(), 4)

    auto_teardown_workers = args.auto_teardown

    try:
        # 1. Inspect live Vertex AI Endpoint(s) and provision/attach hot-reload GPU workers
        explicit_toks = [t.strip() for t in args.admin_token.split(",") if t.strip()]
        ep_ids = [eid.strip() for eid in args.vertex_endpoint_id.split(",") if eid.strip()]
        inspected_eps = [inspect_vertex_endpoint(args.project_id, args.region, eid) for eid in ep_ids]

        eff_machine = args.machine_type.strip() or (inspected_eps[0]["machine_type"] if inspected_eps and inspected_eps[0]["machine_type"] else MACHINE_TYPE)
        eff_accel = args.accelerator_type.strip() or (inspected_eps[0]["accelerator_type"] if inspected_eps and inspected_eps[0]["accelerator_type"] else ACCELERATOR_TYPE)
        eff_accel_count = int(args.accelerator_count) if int(args.accelerator_count) > 0 else (int(inspected_eps[0]["accelerator_count"]) if inspected_eps and inspected_eps[0]["accelerator_count"] else ACCELERATOR_COUNT)
        eff_image = args.vllm_image.strip() or (inspected_eps[0]["image_uri"] if inspected_eps and inspected_eps[0]["image_uri"] else DEFAULT_UPSTREAM_VLLM_IMAGE)

        if inspected_eps and not args.model.strip() and inspected_eps[0].get("vllm_args"):
            live_args = inspected_eps[0]["vllm_args"]
            if live_args and live_args[0] != baseline_cand.vllm_args[0]:
                seed_source = build_seed_program_source(live_args[0], args.served_model_name)
                baseline_cand = parse_candidate_code(seed_source, name="cand_000_baseline")
                served_model = extract_flag_value(baseline_cand.vllm_args, "--served-model-name") or baseline_cand.vllm_args[0]
                resolved_mode = detect_task_mode(trace_rows, requested_mode=args.task_mode, model_name=baseline_cand.vllm_args[0])
                (out_dir / "candidates" / "cand_000_baseline.py").write_text(seed_source, encoding="utf-8")

        clone_standard_endpoint = bool(inspected_eps) and not any(ep["has_hot_reload"] for ep in inspected_eps) and not explicit_toks
        should_provision_workers = args.deploy_vertex or clone_standard_endpoint
        auto_teardown_workers = args.auto_teardown or clone_standard_endpoint
        initial_worker_cand = warm_seed_cand if cont_dir is not None else baseline_cand

        if should_provision_workers:
            logger.info(
                "Provisioning %d hot-reload Vertex AI GPU worker(s) (%s x%d on %s) in %s...",
                num_workers, eff_accel, eff_accel_count, eff_machine, args.region,
            )
            if not args.skip_mirror:
                eff_image = ensure_ar_image_mirrored(args.project_id, args.region, args.ar_repo, upstream_image=eff_image)
            ep_results = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        deploy_search_vertex_endpoint,
                        args.project_id,
                        args.region,
                        args.ar_repo,
                        initial_worker_cand,
                        f"ae-inf-w{idx}",
                        True,
                        upstream_image=eff_image,
                        machine_type=eff_machine,
                        accelerator_type=eff_accel,
                        accelerator_count=eff_accel_count,
                    )
                    for idx in range(num_workers)
                ),
                return_exceptions=True,
            )
            ep_meta = []
            for idx, item in enumerate(ep_results):
                if isinstance(item, Exception):
                    logger.warning("Worker w%d failed to deploy: %s", idx, item)
                else:
                    provisioned_endpoint_ids.append(item["endpoint_id"])
                    worker_pool.append((len(worker_pool), item["chat_completions_url"], item["admin_token"]))
                    ep_meta.append({
                        "worker_idx": len(worker_pool) - 1,
                        "deploy_wall_sec": item.get("deploy_wall_sec", 0.0),
                        "machine_type": eff_machine,
                        "accelerator_type": eff_accel,
                        "accelerator_count": eff_accel_count,
                    })
            (out_dir / "search_endpoints.json").write_text(json.dumps(ep_meta, indent=2) + "\n", encoding="utf-8")
        else:
            for idx, ep_info in enumerate(inspected_eps):
                tok = explicit_toks[idx] if idx < len(explicit_toks) else (explicit_toks[0] if explicit_toks else ep_info["admin_token"])
                worker_pool.append((len(worker_pool), ep_info["chat_completions_url"], tok))
            urls = [u.strip() for u in args.endpoint.split(",") if u.strip()]
            if urls:
                toks = explicit_toks if explicit_toks else [""]
                if len(toks) == 1 and len(urls) > 1:
                    toks = toks * len(urls)
                for u, t in zip(urls, toks):
                    worker_pool.append((len(worker_pool), u, t))

        if not worker_pool:
            raise RuntimeError("No active GPU endpoints available. Pass --vertex-endpoint-id, --endpoint, or --deploy-vertex.")

        use_gcp_auth = (not args.no_gcp_auth) and (
            args.deploy_vertex or "prediction.vertexai" in worker_pool[0][1] or ":rawPredict" in worker_pool[0][1]
        )

        # 2. Evaluate Ground-Zero Baseline (#0) or Warm-Start Seed, capture GPU profile & introspect model layers
        _, primary_url, primary_tok = worker_pool[0]
        if not should_provision_workers and primary_tok:
            init_targets = worker_pool[:2] if len(worker_pool) > 1 else worker_pool[:1]
            init_rels = await asyncio.gather(
                *(reload_endpoint(u, t, initial_worker_cand, use_gcp_auth=use_gcp_auth) for _, u, t in init_targets)
            )
            for (_, u, _), rel_item in zip(init_targets, init_rels):
                if not rel_item.get("ok"):
                    raise RuntimeError(f"Initial reload failed on {u}: {rel_item}")

        secondary_url = worker_pool[1][1] if len(worker_pool) > 1 else ""
        baseline_resp_path = out_dir / "baseline_responses.jsonl"
        tsv_path = out_dir / "results.tsv"
        png_path = out_dir / "progress.png"
        sample_req = load_jsonl_requests(equiv_path)[0]
        stage_boundaries: list[dict[str, Any]] = list(prior_summary.get("stage_boundaries") or [])
        seed_engine_internals: dict[str, float] = {}

        if cont_dir is not None:
            stage_num = len(stage_boundaries) + 2
            stage_boundaries.append(
                {
                    "stage": stage_num,
                    "start_iteration": prior_eval_offset + 1,
                    "label": f"Stage {stage_num}: FP8 + Speculative + Triton/MoE",
                }
            )
            base_row = next((r for r in prior_rows if int(r["iteration"]) == 0), prior_rows[0])
            base_res = _reconstruct_eval_from_tsv_row(base_row, len(trace_rows))
            best_keep_rows = [r for r in prior_rows if r["status"] in ("baseline", "keep")]
            best_prior_row = best_keep_rows[-1] if best_keep_rows else base_row
            best_res = _reconstruct_eval_from_tsv_row(best_prior_row, len(trace_rows))
            best_cand = warm_seed_cand
            best_val = float(prior_summary.get("best_value", best_res.metric_value(canon_metric)))
            kept_history = list(prior_summary.get("kept_breakthroughs") or [])

            logger.info("[Warm-Start #%d] Verifying warm seed (%s) & capturing live telemetry...", prior_eval_offset, warm_seed_cand.plain_title)
            m_ws_before = await scrape_engine_metrics(primary_url, primary_tok, use_gcp_auth=use_gcp_auth)
            ws_check = await evaluate_candidate_on_endpoint(
                candidate_name=warm_seed_cand.name,
                endpoint_url=primary_url,
                trace_path=trace_path,
                equiv_path=equiv_path,
                baseline_responses_path=baseline_resp_path,
                record_baseline=False,
                use_gcp_auth=use_gcp_auth,
                task_mode=resolved_mode,
                model_name=served_model,
                metric=canon_metric,
            )
            m_ws_after = await scrape_engine_metrics(primary_url, primary_tok, use_gcp_auth=use_gcp_auth)
            seed_engine_internals = summarize_engine_metrics_delta(m_ws_before, m_ws_after)
            if not ws_check.equivalence.passed:
                raise RuntimeError(f"Warm-start seed failed KL equivalence against Ground-Zero baseline: {ws_check.equivalence.reason}")
        else:
            if tsv_path.exists():
                tsv_path.unlink()
            logger.info("[Eval #0] Measuring Ground-Zero Baseline on %s...", trace_path)
            m_b0 = await scrape_engine_metrics(primary_url, primary_tok, use_gcp_auth=use_gcp_auth)
            base_res = await evaluate_candidate_on_endpoint(
                candidate_name=baseline_cand.name,
                endpoint_url=primary_url,
                trace_path=trace_path,
                equiv_path=equiv_path,
                baseline_responses_path=baseline_resp_path,
                record_baseline=True,
                secondary_baseline_url=secondary_url,
                use_gcp_auth=use_gcp_auth,
                task_mode=resolved_mode,
                model_name=served_model,
                metric=canon_metric,
            )
            m_b1 = await scrape_engine_metrics(primary_url, primary_tok, use_gcp_auth=use_gcp_auth)
            seed_engine_internals = summarize_engine_metrics_delta(m_b0, m_b1)
            if not base_res.equivalence.passed:
                raise RuntimeError(f"Ground-Zero baseline failed equivalence recording: {base_res.equivalence.reason}")
            best_cand = baseline_cand
            best_res = base_res
            best_val = base_res.metric_value(canon_metric)
            (out_dir / "best_candidate.py").write_text(seed_source, encoding="utf-8")
            append_results_tsv(tsv_path, 0, "seed_000", "baseline", baseline_cand, base_res)
            kept_history = []
            logger.info(
                "[Eval #0] Baseline %s = %s | P99=%.2fms | P50=%.2fms | Throughput=%.2f req/s",
                canon_metric, format_metric(best_val, canon_metric),
                base_res.p99_latency_ms, base_res.p50_latency_ms, base_res.throughput_rps,
            )

        gpu_profile, model_introspect = await asyncio.gather(
            capture_gpu_profile(
                primary_url, primary_tok, sample_req,
                task_mode=resolved_mode, served_model_name=served_model, use_gcp_auth=use_gcp_auth,
            ),
            introspect_endpoint_model(primary_url, primary_tok, use_gcp_auth=use_gcp_auth),
        )
        if gpu_profile:
            (out_dir / "gpu_kernel_profile.json").write_text(json.dumps(gpu_profile, indent=2) + "\n", encoding="utf-8")
        if model_introspect:
            (out_dir / "model_introspection.json").write_text(json.dumps(model_introspect, indent=2) + "\n", encoding="utf-8")

        plot_progress_chart(tsv_path, png_path, metric=canon_metric, stage_boundaries=stage_boundaries)
        write_articulated_summary(
            out_dir,
            metric=canon_metric,
            base_res=base_res,
            best_cand=best_cand,
            best_res=best_res,
            kept_history=kept_history,
            total_evaluated=prior_eval_offset,
            spend_usd=cumulative_spend_usd(),
            state="IN_PROGRESS",
        )

        # 3. Connect to Official Google Cloud AlphaEvolve (Discovery Engine API)
        ae_client = AlphaEvolveClient(
            project_id=args.project_id,
            location=args.ae_location,
            collection="default_collection",
            engine=args.ge_app_id,
            assistant="default_assistant",
            base_url="discoveryengine.googleapis.com",
        )
        ae_client._credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        ae_client._auth_request = google.auth.transport.requests.Request()

        gpu_queue: asyncio.Queue[tuple[int, str, str]] = asyncio.Queue()
        for w in worker_pool:
            gpu_queue.put_nowait(w)

        ledger_lock = asyncio.Lock()
        eval_counter = prior_eval_offset
        experiment_ref: BudgetedAlphaEvolveExperiment | None = None

        async def evaluate_alpha_evolve_candidate(program_candidate: dict[str, Any]) -> dict[str, Any]:
            nonlocal eval_counter, best_cand, best_res, best_val, gpu_profile
            if ae_token_state["input_tokens"] == 0 and ae_token_state["output_tokens"] == 0:
                ae_token_state["llm_spend_usd"] = round(float(ae_token_state["llm_spend_usd"]) + 0.15, 4)
            prog_id = str(program_candidate.get("name", "unknown")).rsplit("/", 1)[-1]
            files = (program_candidate.get("content") or {}).get("files") or []
            code = files[0].get("content", "") if files else ""

            async with ledger_lock:
                eval_counter += 1
                it_num = eval_counter

            try:
                cand = parse_candidate_code(code, name=f"cand_{it_num:03d}")
            except Exception as exc:  # noqa: BLE001
                fail_eq = EquivalenceReport(False, 0, 0, 0, 0, 0.0, 1.0, f"Parse error: {exc}")
                fail_res = TraceEvalResult.failed(f"cand_{it_num:03d}", fail_eq)
                dummy = CandidateSpec(f"cand_{it_num:03d}", "Syntax/parse error", str(exc), "invalid", [], {})
                async with ledger_lock:
                    append_results_tsv(tsv_path, it_num, prog_id[:10], "invalid", dummy, fail_res, str(exc))
                return {
                    "scores": {"scores": [{"metric": "fitness_score", "score": -1_000_000.0}]},
                    "insights": {"insights": [{"label": "Syntax / Structure Error", "text": str(exc)[:1200]}]},
                }

            (out_dir / "candidates" / f"cand_{it_num:03d}_{cand.category}.py").write_text(code, encoding="utf-8")
            ok_spec, spec_reason = validate_candidate_spec(cand, baseline_cand)
            if not ok_spec:
                fail_eq = EquivalenceReport(False, 0, 0, 0, 0, 0.0, 1.0, spec_reason)
                fail_res = TraceEvalResult.failed(cand.name, fail_eq)
                async with ledger_lock:
                    append_results_tsv(tsv_path, it_num, prog_id[:10], "invalid", cand, fail_res, spec_reason)
                return {
                    "scores": {"scores": [{"metric": "fitness_score", "score": -950_000.0}]},
                    "insights": {"insights": [{"label": "Constraint Violation", "text": spec_reason}]},
                }

            w_idx, w_url, w_tok = await gpu_queue.get()
            try:
                rel = await reload_endpoint(w_url, w_tok, cand, use_gcp_auth=use_gcp_auth)
                if not rel.get("ok"):
                    err_txt = str(rel.get("error", "vLLM failed to start"))
                    tail_txt = "\n".join((rel.get("log_tail") or [])[-18:])
                    fail_eq = EquivalenceReport(False, 0, 0, 0, 0, 0.0, 1.0, err_txt)
                    fail_res = TraceEvalResult.failed(cand.name, fail_eq, wall_sec=float(rel.get("reload_sec", 0.0)))
                    async with ledger_lock:
                        append_results_tsv(tsv_path, it_num, prog_id[:10], "crash", cand, fail_res, f"BOOT CRASH: {err_txt}")
                    return {
                        "scores": {"scores": [{"metric": "fitness_score", "score": -900_000.0}]},
                        "insights": {
                            "insights": [{"label": "vLLM / Kernel Startup Crash", "text": f"{err_txt}\n{tail_txt}"[:2500]}]
                        },
                    }

                m_before = await scrape_engine_metrics(w_url, w_tok, use_gcp_auth=use_gcp_auth)
                res = await evaluate_candidate_on_endpoint(
                    candidate_name=cand.name,
                    endpoint_url=w_url,
                    trace_path=trace_path,
                    equiv_path=equiv_path,
                    baseline_responses_path=baseline_resp_path,
                    record_baseline=False,
                    use_gcp_auth=use_gcp_auth,
                    task_mode=resolved_mode,
                    model_name=served_model,
                    metric=canon_metric,
                )
                m_after = await scrape_engine_metrics(w_url, w_tok, use_gcp_auth=use_gcp_auth)
                eng_delta = summarize_engine_metrics_delta(m_before, m_after)

                if not res.equivalence.passed:
                    kl_pen = float(max(-800_000.0, -500_000.0 - 10_000.0 * res.equivalence.mean_kl_divergence))
                    async with ledger_lock:
                        append_results_tsv(tsv_path, it_num, prog_id[:10], "invalid", cand, res, f"KL GATE FAIL: {res.equivalence.reason}")
                    return {
                        "scores": {"scores": [{"metric": "fitness_score", "score": kl_pen}]},
                        "insights": {
                            "insights": [
                                {
                                    "label": "Equivalence Gate Failed",
                                    "text": (
                                        f"{res.equivalence.reason} | mean_kl={res.equivalence.mean_kl_divergence:.5f} | "
                                        f"exact_match={res.equivalence.exact_slot_match_pct:.2f}%"
                                    ),
                                }
                            ]
                        },
                    }

                cand_val = res.metric_value(canon_metric)
                is_potential_record = is_better(cand_val, best_val, canon_metric)
                if args.confirm_wins and is_potential_record:
                    res2 = await evaluate_candidate_on_endpoint(
                        candidate_name=cand.name,
                        endpoint_url=w_url,
                        trace_path=trace_path,
                        equiv_path=equiv_path,
                        baseline_responses_path=baseline_resp_path,
                        record_baseline=False,
                        use_gcp_auth=use_gcp_auth,
                        task_mode=resolved_mode,
                        model_name=served_model,
                        metric=canon_metric,
                    )
                    if res2.equivalence.passed:
                        res = TraceEvalResult(
                            candidate_name=res.candidate_name,
                            equivalence=res2.equivalence,
                            p99_latency_ms=round(0.5 * (res.p99_latency_ms + res2.p99_latency_ms), 2),
                            p50_latency_ms=round(0.5 * (res.p50_latency_ms + res2.p50_latency_ms), 2),
                            mean_latency_ms=round(0.5 * (res.mean_latency_ms + res2.mean_latency_ms), 2),
                            ttft_p50_ms=round(0.5 * (res.ttft_p50_ms + res2.ttft_p50_ms), 2),
                            ttft_p99_ms=round(0.5 * (res.ttft_p99_ms + res2.ttft_p99_ms), 2),
                            trace_rps=round(0.5 * (res.trace_rps + res2.trace_rps), 2),
                            throughput_rps=round(0.5 * (res.throughput_rps + res2.throughput_rps), 2),
                            num_requests=res.num_requests + res2.num_requests,
                            num_failures=0,
                            eval_wall_sec=round(res.eval_wall_sec + res2.eval_wall_sec, 2),
                            time_buckets=res2.time_buckets,
                        )
                        cand_val = res.metric_value(canon_metric)
                    else:
                        is_potential_record = False

                status = "discard"
                refresh_prof = False
                async with ledger_lock:
                    if is_better(cand_val, best_val, canon_metric) and (not args.confirm_wins or is_potential_record):
                        status = "keep"
                        best_val = cand_val
                        best_cand = cand
                        best_res = res
                        kept_history.append(
                            {
                                "iteration": it_num,
                                "candidate_id": prog_id[:10],
                                "category": cand.category,
                                "plain_title": cand.plain_title,
                                "p99_latency_ms": res.p99_latency_ms,
                                "p50_latency_ms": res.p50_latency_ms,
                                "throughput_rps": res.throughput_rps,
                                "hypothesis": cand.hypothesis,
                            }
                        )
                        (out_dir / "best_candidate.py").write_text(code, encoding="utf-8")
                        refresh_prof = True

                    append_results_tsv(tsv_path, it_num, prog_id[:10], status, cand, res)
                    plot_progress_chart(tsv_path, png_path, metric=canon_metric, stage_boundaries=stage_boundaries)
                    write_articulated_summary(
                        out_dir,
                        metric=canon_metric,
                        base_res=base_res,
                        best_cand=best_cand,
                        best_res=best_res,
                        kept_history=kept_history,
                        total_evaluated=it_num,
                        spend_usd=cumulative_spend_usd(),
                        experiment_name=(experiment_ref.experiment_name or "") if experiment_ref else "",
                        state="IN_PROGRESS",
                    )

                if refresh_prof:
                    new_prof = await capture_gpu_profile(
                        w_url, w_tok, sample_req,
                        task_mode=resolved_mode, served_model_name=served_model, use_gcp_auth=use_gcp_auth,
                    )
                    if new_prof:
                        gpu_profile = new_prof
                        (out_dir / "gpu_kernel_profile_best.json").write_text(
                            json.dumps(gpu_profile, indent=2) + "\n", encoding="utf-8"
                        )

                logger.info(
                    "[Eval #%03d | w%d] %-7s | [%s] %s | %s=%s (best=%s) | P99=%.1fms | Tput=%.1f rps | stage=$%.2f (tot=$%.2f)",
                    it_num, w_idx, status.upper(), cand.category, cand.plain_title,
                    canon_metric, format_metric(cand_val, canon_metric), format_metric(best_val, canon_metric),
                    res.p99_latency_ms, res.throughput_rps, stage_spend_usd(), cumulative_spend_usd(),
                )

                fitness = to_alpha_evolve_score(cand_val, canon_metric)
                bucket_str = ", ".join(f"t={int(b['start_sec'])}s:p99={b['p99_ms']:.0f}ms" for b in res.time_buckets[:6])
                top_k_str = ", ".join(f"{k['kernel'][:48]} ({k['pct_gpu_time']:.1f}%)" for k in (gpu_profile or [])[:8])
                eng_str = ", ".join(f"{k}={v}" for k, v in eng_delta.items()) if eng_delta else "n/a"
                return {
                    "scores": {"scores": [{"metric": "fitness_score", "score": fitness}]},
                    "insights": {
                        "insights": [
                            {
                                "label": "Serving Telemetry",
                                "text": (
                                    f"Target {canon_metric}={format_metric(cand_val, canon_metric)} "
                                    f"(GroundZero#0={format_metric(base_res.metric_value(canon_metric), canon_metric)}, "
                                    f"CurrentBest={format_metric(best_val, canon_metric)}) | "
                                    f"P99={res.p99_latency_ms:.2f}ms, P50={res.p50_latency_ms:.2f}ms, "
                                    f"TTFT_P99={res.ttft_p99_ms:.2f}ms, PeakThroughput={res.throughput_rps:.2f} req/s | "
                                    f"KL={res.equivalence.mean_kl_divergence:.5f} (exact={res.equivalence.exact_slot_match_pct:.2f}%) | "
                                    f"30s Windows: [{bucket_str}]"
                                ),
                            },
                            {"label": "Engine Internals (ITL / Queue / Decode / SpecDecode)", "text": eng_str},
                            {"label": "Top GPU CUDA Kernels", "text": top_k_str or "No Kineto kernel profile available."},
                        ]
                    },
                }
            finally:
                gpu_queue.put_nowait((w_idx, w_url, w_tok))

        # 4. Build Problem Description & Launch Official AlphaEvolve Loop
        full_problem_description = _build_problem_description(
            canon_metric=canon_metric,
            base_res=base_res,
            best_val=best_val,
            seed_engine_internals=seed_engine_internals,
            eff_accel=eff_accel,
            eff_accel_count=eff_accel_count,
            eff_machine=eff_machine,
            model_id=baseline_cand.vllm_args[0],
            gpu_profile=gpu_profile,
            model_introspect=model_introspect,
            cont_dir=cont_dir,
            prior_eval_offset=prior_eval_offset,
        )

        models_cfg = []
        if args.model_1 and args.model_1.strip():
            models_cfg.append({"name": args.model_1.strip(), "weight": round(max(0.01, args.model_1_weight), 2)})
        if args.model_2 and args.model_2.strip() and args.model_2.strip() != args.model_1.strip() and args.model_2_weight > 0:
            models_cfg.append({"name": args.model_2.strip(), "weight": round(args.model_2_weight, 2)})

        experiment = BudgetedAlphaEvolveExperiment(
            ae_client=ae_client,
            evaluator_function=evaluate_alpha_evolve_candidate,
            max_programs_evaluated=max_programs,
            budget_usd=args.budget_usd,
            current_spend_fn=stage_spend_usd,
            parallel_evaluation=False,
        )
        experiment_ref = experiment
        gen_settings: dict[str, Any] = {"include_full_program_in_prompt": True}
        if models_cfg:
            gen_settings["models"] = models_cfg

        experiment.create_experiment(
            {
                "title": f"alpha_inference_{canon_metric}_{int(time.time())}",
                "problem_description": full_problem_description,
                "program_language": "python",
                "run_settings": {"max_programs": max_programs + 1, "concurrency": len(worker_pool)},
                "generation_settings": gen_settings,
            }
        )
        experiment.create_initial_program(
            {
                "content": {"files": [{"path": "candidate.py", "content": warm_seed_source}]},
                "evaluation": {"scores": {"scores": [{"metric": "fitness_score", "score": to_alpha_evolve_score(best_val, canon_metric)}]}},
            }
        )
        experiment.start_experiment()
        logger.info("Started Official AlphaEvolve Experiment: %s", experiment.experiment_name)

        def _refresh_token_stats(exp_info: dict[str, Any] | None) -> None:
            nonlocal models_cfg
            stats = (exp_info or {}).get("stats") or {}
            in_t = int(stats.get("inputTokenCount") or 0)
            out_t = int(stats.get("outputTokenCount") or 0)
            srv_models = (((exp_info or {}).get("config") or {}).get("generationSettings") or {}).get("models")
            if srv_models and not models_cfg:
                models_cfg = srv_models
            ae_token_state["input_tokens"] = in_t
            ae_token_state["output_tokens"] = out_t
            ae_token_state["llm_spend_usd"] = estimate_llm_spend(args, in_t, out_t)

        async def _auto_resume_watchdog() -> None:
            while not experiment.stopping_criteria_met():
                await asyncio.sleep(30.0)
                if experiment.stopping_criteria_met() or not experiment.experiment_name:
                    break
                try:
                    info = await asyncio.to_thread(ae_client.get_alpha_evolve_experiment, experiment.experiment_name)
                    _refresh_token_stats(info)
                    st = str((info or {}).get("state", "")).strip().upper()
                    if "PAUSED" in st and not experiment.stopping_criteria_met():
                        logger.warning("AlphaEvolve experiment entered PAUSED state; calling resume_experiment()...")
                        await asyncio.to_thread(experiment.resume_experiment)
                    elif st in {"COMPLETED", "SUCCEEDED", "CANCELLED", "FAILED"}:
                        logger.info("AlphaEvolve backend reached terminal state (%s); finishing controller loop.", st)
                        experiment.backend_terminal = True
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Watchdog check skipped: %s", exc)

        watchdog_task = asyncio.create_task(_auto_resume_watchdog())
        try:
            await run_controller_loop(
                experiment,
                num_samplers=min(4, max(1, len(worker_pool))),
                num_evaluators=len(worker_pool),
                idle_timeout_s=600,
            )
        finally:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task, return_exceptions=True)
            if experiment.experiment_name:
                try:
                    final_info = await asyncio.to_thread(ae_client.get_alpha_evolve_experiment, experiment.experiment_name)
                    _refresh_token_stats(final_info)
                except Exception:  # noqa: BLE001
                    pass

        # 5. Finalize articulated outputs + native `gcloud ai` rollout script
        portable_image = (
            eff_image.replace(args.project_id, "${GCP_PROJECT_ID}")
            if args.project_id and args.project_id in eff_image
            else eff_image
        )
        rollout_sh = build_gcloud_rollout_command(
            best_cand,
            project_id="${GCP_PROJECT_ID}",
            region="${GCP_REGION:-us-central1}",
            endpoint_id=ep_ids[0] if ep_ids else "${VERTEX_ENDPOINT_ID}",
            image_uri=portable_image,
            machine_type=eff_machine,
            accelerator_type=eff_accel,
            accelerator_count=eff_accel_count,
        )
        (out_dir / "gcloud_rollout.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + rollout_sh, encoding="utf-8")

        write_articulated_summary(
            out_dir,
            metric=canon_metric,
            base_res=base_res,
            best_cand=best_cand,
            best_res=best_res,
            kept_history=kept_history,
            total_evaluated=eval_counter,
            spend_usd=cumulative_spend_usd(),
            experiment_name=experiment.experiment_name or "",
            state="COMPLETED",
        )
        prev_exp_ids = list(prior_summary.get("prior_experiment_ids") or [])
        if prior_summary.get("alpha_evolve_experiment_id"):
            prev_exp_ids.append(prior_summary["alpha_evolve_experiment_id"])
        run_meta = {
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "alpha_evolve_experiment_id": (experiment.experiment_name or "").rsplit("/", 1)[-1],
            "prior_experiment_ids": prev_exp_ids,
            "stage_boundaries": stage_boundaries,
            "metric": canon_metric,
            "baseline_value": base_res.metric_value(canon_metric),
            "best_value": best_val,
            "delta_pct": round(((best_val - base_res.metric_value(canon_metric)) / base_res.metric_value(canon_metric)) * 100.0, 2)
            if base_res.metric_value(canon_metric) > 0
            else 0.0,
            "baseline_p99_ms": base_res.p99_latency_ms,
            "best_p99_ms": best_res.p99_latency_ms,
            "baseline_throughput_rps": base_res.throughput_rps,
            "best_throughput_rps": best_res.throughput_rps,
            "best_kl_divergence": best_res.equivalence.mean_kl_divergence,
            "best_exact_match_pct": best_res.equivalence.exact_slot_match_pct,
            "programs_evaluated": eval_counter,
            "stage_programs_evaluated": eval_counter - prior_eval_offset,
            "kept_breakthroughs": kept_history,
            "models_used": models_cfg,
            "input_tokens": int(prior_summary.get("input_tokens", 0)) + int(ae_token_state["input_tokens"]),
            "output_tokens": int(prior_summary.get("output_tokens", 0)) + int(ae_token_state["output_tokens"]),
            "gpu_spend_usd": round(float(prior_summary.get("gpu_spend_usd", prior_spend_usd)) + stage_gpu_spend_usd(), 4),
            "llm_token_spend_usd": round(float(prior_summary.get("llm_token_spend_usd", 0.0)) + float(ae_token_state["llm_spend_usd"]), 4),
            "stage_spend_usd": stage_spend_usd(),
            "total_spend_usd": cumulative_spend_usd(),
            "out_dir": str(out_dir),
        }
        (out_dir / "optimization_summary.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        return run_meta
    finally:
        if provisioned_endpoint_ids and auto_teardown_workers:
            logger.info("Tearing down %d provisioned Vertex GPU worker(s)...", len(provisioned_endpoint_ids))
            await asyncio.gather(
                *(
                    asyncio.to_thread(teardown_endpoint, args.project_id, args.region, ep_id)
                    for ep_id in provisioned_endpoint_ids
                ),
                return_exceptions=True,
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    nest_asyncio.apply()
    parser = build_parser()
    args = parser.parse_args()
    if args.diff:
        best_file = Path(args.out_dir) / "best_candidate.py"
        print(format_candidate_diff("candidate.py", best_file), end="")
        return
    res = asyncio.run(run_optimization(args))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
