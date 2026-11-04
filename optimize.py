#!/usr/bin/env python3
"""Optimize GenAI serving on a Vertex AI Endpoint or GPU VM using Google Cloud AlphaEvolve.

Quickstart:
    # 1. Dry-run check ($0 spend):
    python optimize.py --dry-run

    # 2. Optimize on managed Vertex AI Dedicated GPU Endpoints (auto-teardown on exit):
    python optimize.py \
        --trace examples/email_triage/monday_peak_3m.jsonl \
        --equiv-trace examples/email_triage/equivalence_golden.jsonl \
        --deploy-vertex --workers 4 --auto-teardown \
        --metric p99_latency \
        --max-programs 200 \
        --budget-usd 160

    # 3. Optimize an existing GPU VM or Vertex Endpoint running `alpha_inference.reload_proxy`:
    python optimize.py \
        --trace examples/custom_openai_chat/sample_trace.jsonl \
        --endpoint http://localhost:8080 --admin-token my-secret \
        --model Qwen/Qwen2.5-7B-Instruct \
        --metric ttft \
        --max-programs 30 \
        --budget-usd 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    ACCELERATOR_TYPE,
    CandidateSpec,
    append_results_tsv,
    build_seed_program_source,
    capture_gpu_profile,
    deploy_search_vertex_endpoint,
    extract_flag_value,
    inspect_vertex_endpoint,
    parse_candidate_code,
    plot_progress_chart,
    reload_endpoint,
    validate_candidate_spec,
    write_articulated_summary,
)
from deploy import ensure_ar_image_mirrored, teardown_endpoint

logger = logging.getLogger("alpha_inference")


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

    def stopping_criteria_met(self) -> bool:
        if super().stopping_criteria_met():
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
    default_equiv = (
        "examples/email_triage/equivalence_golden.jsonl"
        if Path("examples/email_triage/equivalence_golden.jsonl").exists()
        else ""
    )

    p = argparse.ArgumentParser(
        description="Optimize GenAI serving on Vertex AI or GPU VMs using official Google Cloud AlphaEvolve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trace", "--data", dest="trace", default=default_trace, help="JSONL production request trace")
    p.add_argument("--equiv-trace", default=default_equiv, help="Optional separate JSONL file for KL equivalence gate")
    p.add_argument("--task-mode", choices=TASK_MODE_CHOICES, default="auto", help="auto | chat | jev_diffusion")
    p.add_argument("--model", default="", help="Optional HuggingFace model override (defaults to candidate.py)")
    p.add_argument("--served-model-name", default="", help="Optional --served-model-name override")
    p.add_argument("--metric", choices=METRIC_CHOICES, default="p99_latency", help="Target metric to optimize")

    # Budget & scale
    p.add_argument("--max-programs", type=int, default=60, help="Maximum candidate programs to generate & evaluate")
    p.add_argument("--workers", type=int, default=4, help="Parallel GPU evaluation workers")
    p.add_argument("--budget-usd", type=float, default=40.0, help="Hard USD cloud-spend cap")
    p.add_argument("--hourly-gpu-usd", type=float, default=2.65, help="Per-GPU hourly rate in USD for budget tracking")
    p.add_argument("--confirm-wins", action="store_true", help="Re-verify new record candidates before promoting")

    # Endpoint / Vertex AI settings
    p.add_argument("--vertex-endpoint-id", default="", help="Live Vertex AI Endpoint ID(s), comma-separated")
    p.add_argument("--endpoint-json", default="", help="JSON file emitted by `deploy.py --out-json`")
    p.add_argument("--endpoint", "--endpoint-url", dest="endpoint", default="", help="Existing reload_proxy URL(s), comma-separated")
    p.add_argument("--admin-token", default="", help="Admin token(s) for reload_proxy hot-swap")
    p.add_argument("--deploy-vertex", action="store_true", help="Provision Dedicated Vertex AI GPU Endpoint(s)")
    p.add_argument("--auto-teardown", action="store_true", help="Delete provisioned Vertex GPU Endpoint(s) on exit")
    p.add_argument("--skip-mirror", action="store_true", help="Skip Artifact Registry container mirror check")
    p.add_argument("--no-gcp-auth", action="store_true", help="Disable GCP Bearer auth on endpoint HTTP requests")

    # Official Google Cloud AlphaEvolve settings
    p.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID", ""), help="Google Cloud project ID")
    p.add_argument("--ge-app-id", default=os.getenv("GE_APP_ID", ""), help="Gemini Enterprise App / Engine ID")
    p.add_argument("--ae-location", default=os.getenv("AE_LOCATION", "global"), help="Discovery Engine location")
    p.add_argument("--region", default=os.getenv("GCP_REGION", os.getenv("GCP_REGION_GPU", "us-central1")), help="Vertex GPU region")
    p.add_argument("--ar-repo", default=os.getenv("GCP_AR_REPO", "geap-models"), help="Artifact Registry repo")
    p.add_argument("--model-1", default=os.getenv("AE_MODEL_1", "gemini-3.5-flash"), help="Primary AlphaEvolve model")
    p.add_argument("--model-1-weight", type=float, default=float(os.getenv("AE_MODEL_1_WEIGHT", "0.55")))
    p.add_argument("--model-2", default=os.getenv("AE_MODEL_2", "gemini-3.1-pro-preview"), help="Secondary AlphaEvolve model")
    p.add_argument("--model-2-weight", type=float, default=float(os.getenv("AE_MODEL_2_WEIGHT", "0.45")))

    p.add_argument("--out-dir", default="examples/email_triage/results", help="Output directory for articulated results")
    p.add_argument("--dry-run", action="store_true", help="Validate trace, seed candidate, and budget with $0 spend")
    return p


async def run_optimization(args: argparse.Namespace) -> dict[str, Any]:
    canon_metric = normalize_metric(args.metric)
    seed_source = build_seed_program_source(args.model, args.served_model_name)
    baseline_cand = parse_candidate_code(seed_source, name="cand_000_baseline")
    ok_seed, seed_msg = validate_candidate_spec(baseline_cand, baseline_cand)
    if not ok_seed:
        raise RuntimeError(f"Seed candidate.py failed validation: {seed_msg}")

    served_model = extract_flag_value(baseline_cand.vllm_args, "--served-model-name") or baseline_cand.vllm_args[0]
    trace_path = Path(args.trace)
    equiv_path = Path(args.equiv_trace) if args.equiv_trace else trace_path
    trace_rows = load_jsonl_requests(trace_path)
    resolved_mode = detect_task_mode(trace_rows, requested_mode=args.task_mode, model_name=baseline_cand.vllm_args[0])
    max_programs = max(2, int(args.max_programs))
    num_workers = max(1, int(args.workers))

    if args.dry_run:
        return {
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
            "seed_candidate": {k: v for k, v in asdict(baseline_cand).items() if k != "raw_source"},
        }

    if not args.project_id or not args.ge_app_id:
        raise RuntimeError("Set GCP_PROJECT_ID and GE_APP_ID in .env (or pass --project-id and --ge-app-id).")

    out_dir = Path(args.out_dir)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates" / "cand_000_baseline.py").write_text(seed_source, encoding="utf-8")

    t_start = time.perf_counter()
    provisioned_endpoint_ids: list[str] = []
    worker_pool: list[tuple[int, str, str]] = []
    ae_api_spend_est = 0.0

    def current_spend_usd() -> float:
        tracks_vertex_gpus = args.deploy_vertex or any(
            (":rawPredict" in u or "vertexai" in u) for _, u, _ in worker_pool
        )
        if not tracks_vertex_gpus:
            return round(ae_api_spend_est, 4)
        hrs = (time.perf_counter() - t_start) / 3600.0
        n_gpus = max(1, len(worker_pool) if worker_pool else num_workers)
        return round(hrs * args.hourly_gpu_usd * n_gpus + ae_api_spend_est, 4)

    try:
        # Optional: inspect live --vertex-endpoint-id(s) or --endpoint-json
        explicit_toks = [t.strip() for t in args.admin_token.split(",") if t.strip()]
        if args.endpoint_json:
            ep_obj = json.loads(Path(args.endpoint_json).read_text(encoding="utf-8"))
            if ep_obj.get("chat_completions_url"):
                worker_pool.append((len(worker_pool), str(ep_obj["chat_completions_url"]), str(ep_obj.get("admin_token", ""))))

        ep_ids = [eid.strip() for eid in args.vertex_endpoint_id.split(",") if eid.strip()]
        inspected_eps = [inspect_vertex_endpoint(args.project_id, args.region, eid) for eid in ep_ids]
        if inspected_eps and not args.model.strip() and inspected_eps[0].get("vllm_args"):
            live_args = inspected_eps[0]["vllm_args"]
            if live_args and live_args[0] != baseline_cand.vllm_args[0]:
                seed_source = build_seed_program_source(live_args[0], args.served_model_name)
                baseline_cand = parse_candidate_code(seed_source, name="cand_000_baseline")
                served_model = extract_flag_value(baseline_cand.vllm_args, "--served-model-name") or baseline_cand.vllm_args[0]
                resolved_mode = detect_task_mode(trace_rows, requested_mode=args.task_mode, model_name=baseline_cand.vllm_args[0])
                (out_dir / "candidates" / "cand_000_baseline.py").write_text(seed_source, encoding="utf-8")

        # 1. Provision or attach GPU worker pool
        if args.deploy_vertex:
            logger.info("Provisioning %d parallel Vertex AI Dedicated GPU Endpoint(s) in %s...", num_workers, args.region)
            if not args.skip_mirror:
                ensure_ar_image_mirrored(args.project_id, args.region, args.ar_repo)
            ep_results = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        deploy_search_vertex_endpoint,
                        args.project_id,
                        args.region,
                        args.ar_repo,
                        baseline_cand,
                        f"ae-inf-w{idx}",
                        True,
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
                        "machine_type": "g4-standard-48",
                        "accelerator_type": ACCELERATOR_TYPE,
                    })
            (out_dir / "search_endpoints.json").write_text(json.dumps(ep_meta, indent=2) + "\n", encoding="utf-8")
        else:
            for idx, ep_info in enumerate(inspected_eps):
                tok = explicit_toks[idx] if idx < len(explicit_toks) else (explicit_toks[0] if explicit_toks else ep_info["admin_token"])
                if not tok:
                    raise RuntimeError(
                        f"Vertex Endpoint {ep_info['endpoint_id']} does not have hot-reload enabled yet. "
                        f"Either pass --deploy-vertex to spin up temporary search workers, or enable hot-reload on it via: "
                        f"python deploy.py --endpoint-id {ep_info['endpoint_id']} --hot-reload --deploy"
                    )
                worker_pool.append((len(worker_pool), ep_info["chat_completions_url"], tok))

            urls = [u.strip() for u in args.endpoint.split(",") if u.strip()]
            if urls:
                toks = explicit_toks if explicit_toks else [""]
                if len(toks) == 1 and len(urls) > 1:
                    toks = toks * len(urls)
                for u, t in zip(urls, toks):
                    worker_pool.append((len(worker_pool), u, t))

        if not worker_pool:
            raise RuntimeError("No active GPU endpoints available. Pass --vertex-endpoint-id, --endpoint-json, --endpoint, or --deploy-vertex.")

        use_gcp_auth = (not args.no_gcp_auth) and (
            args.deploy_vertex or "prediction.vertexai" in worker_pool[0][1] or ":rawPredict" in worker_pool[0][1]
        )

        # 2. Evaluate Ground-Zero Baseline (#0) & capture live GPU CUDA kernel profile
        _, primary_url, primary_tok = worker_pool[0]
        if not args.deploy_vertex and primary_tok:
            init_targets = worker_pool[:2] if len(worker_pool) > 1 else worker_pool[:1]
            init_rels = await asyncio.gather(
                *(reload_endpoint(u, t, baseline_cand, use_gcp_auth=use_gcp_auth) for _, u, t in init_targets)
            )
            for (_, u, _), rel_item in zip(init_targets, init_rels):
                if not rel_item.get("ok"):
                    raise RuntimeError(f"Initial reload failed on {u}: {rel_item}")

        secondary_url = worker_pool[1][1] if len(worker_pool) > 1 else ""
        baseline_resp_path = out_dir / "baseline_responses.jsonl"
        tsv_path = out_dir / "results.tsv"
        png_path = out_dir / "progress.png"
        if tsv_path.exists():
            tsv_path.unlink()

        logger.info("[Eval #0] Measuring Ground-Zero Baseline on %s...", trace_path)
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
        if not base_res.equivalence.passed:
            raise RuntimeError(f"Ground-Zero baseline failed equivalence recording: {base_res.equivalence.reason}")

        sample_req = load_jsonl_requests(equiv_path)[0]
        gpu_profile = await capture_gpu_profile(
            primary_url,
            primary_tok,
            sample_req,
            task_mode=resolved_mode,
            served_model_name=served_model,
            use_gcp_auth=use_gcp_auth,
        )
        if gpu_profile:
            (out_dir / "gpu_kernel_profile.json").write_text(json.dumps(gpu_profile, indent=2) + "\n", encoding="utf-8")

        best_cand = baseline_cand
        best_res = base_res
        best_val = base_res.metric_value(canon_metric)
        (out_dir / "best_candidate.py").write_text(seed_source, encoding="utf-8")

        append_results_tsv(tsv_path, 0, "seed_000", "baseline", baseline_cand, base_res)
        plot_progress_chart(tsv_path, png_path, metric=canon_metric)
        kept_history: list[dict[str, Any]] = []
        write_articulated_summary(
            out_dir,
            metric=canon_metric,
            base_res=base_res,
            best_cand=best_cand,
            best_res=best_res,
            kept_history=kept_history,
            total_evaluated=0,
            spend_usd=current_spend_usd(),
            state="IN_PROGRESS",
        )
        logger.info(
            "[Eval #0] Baseline %s = %s | P99=%.2fms | P50=%.2fms | Throughput=%.2f req/s",
            canon_metric,
            format_metric(best_val, canon_metric),
            base_res.p99_latency_ms,
            base_res.p50_latency_ms,
            base_res.throughput_rps,
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
        eval_counter = 0

        async def evaluate_alpha_evolve_candidate(program_candidate: dict[str, Any]) -> dict[str, Any]:
            nonlocal eval_counter, best_cand, best_res, best_val, gpu_profile, ae_api_spend_est
            ae_api_spend_est += 0.04
            prog_id = str(program_candidate.get("name", "unknown")).rsplit("/", 1)[-1]
            files = (program_candidate.get("content") or {}).get("files") or []
            code = files[0].get("content", "") if files else ""

            async with ledger_lock:
                eval_counter += 1
                it_num = eval_counter

            # Tier 1: Fast static & AST validation (0s GPU cost if invalid)
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

            # Acquire dedicated GPU worker exclusively
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
                            "insights": [
                                {"label": "vLLM / Kernel Startup Crash", "text": f"{err_txt}\n{tail_txt}"[:2500]},
                            ]
                        },
                    }

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

                if not res.equivalence.passed:
                    kl_pen = float(max(-800_000.0, -500_000.0 - 10_000.0 * res.equivalence.mean_kl_divergence))
                    async with ledger_lock:
                        append_results_tsv(
                            tsv_path,
                            it_num,
                            prog_id[:10],
                            "invalid",
                            cand,
                            res,
                            f"KL GATE FAIL: {res.equivalence.reason}",
                        )
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

                # Optional confirmation re-run on the same already-warm worker before promoting a new record
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
                    plot_progress_chart(tsv_path, png_path, metric=canon_metric)
                    write_articulated_summary(
                        out_dir,
                        metric=canon_metric,
                        base_res=base_res,
                        best_cand=best_cand,
                        best_res=best_res,
                        kept_history=kept_history,
                        total_evaluated=it_num,
                        spend_usd=current_spend_usd(),
                        experiment_name=experiment.experiment_name or "",
                        state="IN_PROGRESS",
                    )

                if refresh_prof:
                    new_prof = await capture_gpu_profile(
                        w_url,
                        w_tok,
                        sample_req,
                        task_mode=resolved_mode,
                        served_model_name=served_model,
                        use_gcp_auth=use_gcp_auth,
                    )
                    if new_prof:
                        gpu_profile = new_prof
                        (out_dir / "gpu_kernel_profile_best.json").write_text(
                            json.dumps(gpu_profile, indent=2) + "\n", encoding="utf-8"
                        )

                logger.info(
                    "[Eval #%03d | w%d] %-7s | [%s] %s | %s=%s (best=%s) | P99=%.1fms | Tput=%.1f rps | spend=$%.2f",
                    it_num,
                    w_idx,
                    status.upper(),
                    cand.category,
                    cand.plain_title,
                    canon_metric,
                    format_metric(cand_val, canon_metric),
                    format_metric(best_val, canon_metric),
                    res.p99_latency_ms,
                    res.throughput_rps,
                    current_spend_usd(),
                )

                fitness = to_alpha_evolve_score(cand_val, canon_metric)
                bucket_str = ", ".join(
                    f"t={int(b['start_sec'])}s:p99={b['p99_ms']:.0f}ms" for b in res.time_buckets[:6]
                )
                top_k_str = ", ".join(
                    f"{k['kernel'][:48]} ({k['pct_gpu_time']:.1f}%)" for k in (gpu_profile or [])[:6]
                )
                return {
                    "scores": {"scores": [{"metric": "fitness_score", "score": fitness}]},
                    "insights": {
                        "insights": [
                            {
                                "label": "Serving Telemetry",
                                "text": (
                                    f"Target {canon_metric}={format_metric(cand_val, canon_metric)} "
                                    f"(Baseline={format_metric(base_res.metric_value(canon_metric), canon_metric)}, "
                                    f"CurrentBest={format_metric(best_val, canon_metric)}) | "
                                    f"P99={res.p99_latency_ms:.2f}ms, P50={res.p50_latency_ms:.2f}ms, "
                                    f"TTFT_P99={res.ttft_p99_ms:.2f}ms, PeakThroughput={res.throughput_rps:.2f} req/s | "
                                    f"KL={res.equivalence.mean_kl_divergence:.5f} (exact={res.equivalence.exact_slot_match_pct:.2f}%) | "
                                    f"30s Windows: [{bucket_str}]"
                                ),
                            },
                            {
                                "label": "Top GPU CUDA Kernels",
                                "text": top_k_str or "No Kineto kernel profile available.",
                            },
                        ]
                    },
                }
            finally:
                gpu_queue.put_nowait((w_idx, w_url, w_tok))

        # 4. Build Problem Description & Launch Official AlphaEvolve Loop
        problem_guide = (Path(__file__).resolve().parent / "alpha_inference" / "problem.md").read_text(encoding="utf-8")
        prof_summary = json.dumps((gpu_profile or [])[:12], indent=2) if gpu_profile else "[]"
        dir_word = METRIC_INFO[canon_metric]["direction"].upper()
        score_expr = "`+throughput_rps`" if dir_word == "MAXIMIZE" else f"`-{canon_metric}_ms`"
        full_problem_description = (
            f"{problem_guide}\n\n"
            "## Active Experiment Context\n"
            f"- **Optimization Objective:** {dir_word} `{canon_metric}` ({METRIC_INFO[canon_metric]['label']}). "
            f"AlphaEvolve maximizes `fitness_score` = {score_expr}.\n"
            f"- **Ground-Zero Baseline (`#0`):** P99=`{base_res.p99_latency_ms:.2f} ms`, "
            f"P50=`{base_res.p50_latency_ms:.2f} ms`, Peak Throughput=`{base_res.throughput_rps:.2f} req/s`.\n"
            f"- **Hardware:** NVIDIA RTX PRO 6000 (96GB GDDR7, Blackwell sm_120) serving `{baseline_cand.vllm_args[0]}`.\n"
            f"- **Live Ground-Zero Top GPU CUDA Kernels:**\n```json\n{prof_summary}\n```\n"
        )

        models_cfg = []
        if args.model_1:
            models_cfg.append({"name": args.model_1, "weight": round(args.model_1_weight, 2)})
        if args.model_2 and args.model_2 != args.model_1:
            models_cfg.append({"name": args.model_2, "weight": round(args.model_2_weight, 2)})

        experiment = BudgetedAlphaEvolveExperiment(
            ae_client=ae_client,
            evaluator_function=evaluate_alpha_evolve_candidate,
            max_programs_evaluated=max_programs,
            budget_usd=args.budget_usd,
            current_spend_fn=current_spend_usd,
            parallel_evaluation=False,  # Native async evaluator runs concurrently across `num_evaluators` tasks
        )
        experiment.create_experiment(
            {
                "title": f"alpha_inference_{canon_metric}_{int(time.time())}",
                "problem_description": full_problem_description,
                "program_language": "python",
                "run_settings": {
                    "max_programs": max_programs,
                    "concurrency": len(worker_pool),
                },
                "generation_settings": {
                    "include_full_program_in_prompt": True,
                    "models": models_cfg,
                },
            }
        )
        seed_fitness = to_alpha_evolve_score(base_res.metric_value(canon_metric), canon_metric)
        experiment.create_initial_program(
            {
                "content": {"files": [{"path": "candidate.py", "content": seed_source}]},
                "evaluation": {"scores": {"scores": [{"metric": "fitness_score", "score": seed_fitness}]}},
            }
        )
        experiment.start_experiment()
        logger.info("Started Official AlphaEvolve Experiment: %s", experiment.experiment_name)

        async def _auto_resume_watchdog() -> None:
            while not experiment.stopping_criteria_met():
                await asyncio.sleep(180.0)
                if experiment.stopping_criteria_met() or not experiment.experiment_name:
                    break
                try:
                    info = await asyncio.to_thread(
                        ae_client.get_alpha_evolve_experiment, experiment.experiment_name
                    )
                    st = str((info or {}).get("state", ""))
                    if "PAUSED" in st.upper() and not experiment.stopping_criteria_met():
                        logger.warning("AlphaEvolve experiment entered PAUSED state; calling resume_experiment()...")
                        await asyncio.to_thread(experiment.resume_experiment)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Watchdog check skipped: %s", exc)

        watchdog_task = asyncio.create_task(_auto_resume_watchdog())
        try:
            await run_controller_loop(
                experiment,
                num_samplers=1,
                num_evaluators=len(worker_pool),
                idle_timeout_s=2400,
            )
        finally:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task, return_exceptions=True)

        # 5. Finalize articulated outputs
        write_articulated_summary(
            out_dir,
            metric=canon_metric,
            base_res=base_res,
            best_cand=best_cand,
            best_res=best_res,
            kept_history=kept_history,
            total_evaluated=eval_counter,
            spend_usd=current_spend_usd(),
            experiment_name=experiment.experiment_name or "",
            state="COMPLETED",
        )
        run_meta = {
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "alpha_evolve_experiment_id": (experiment.experiment_name or "").rsplit("/", 1)[-1],
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
            "kept_breakthroughs": kept_history,
            "total_spend_usd": current_spend_usd(),
            "out_dir": str(out_dir),
        }
        (out_dir / "optimization_summary.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        return run_meta
    finally:
        if provisioned_endpoint_ids and args.auto_teardown:
            logger.info("Tearing down %d provisioned Vertex GPU Endpoint(s)...", len(provisioned_endpoint_ids))
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
    res = asyncio.run(run_optimization(args))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
