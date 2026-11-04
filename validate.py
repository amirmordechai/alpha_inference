#!/usr/bin/env python3
"""Validate Baseline (`candidate.py`) vs. Best Evolved (`best_candidate.py`) on an unseen 1:1 production trace.

Quickstart:
    # 1. Dry-run check ($0 spend):
    python validate.py --dry-run

    # 2. Deploy standalone Baseline & Best Vertex GPU Endpoints, replay 30-min unseen trace 1:1, and tear down:
    python validate.py \
        --trace examples/email_triage/tuesday_peak_30m.jsonl \
        --deploy-vertex --auto-teardown \
        --out-dir examples/email_triage/results

    # 3. Validate two already-running endpoints:
    python validate.py \
        --trace examples/custom_openai_chat/sample_trace.jsonl \
        --baseline-url http://gpu-base:8080 \
        --best-url http://gpu-best:8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

import deploy
from alpha_inference.adapters import (
    ParsedChatResponse,
    ParsedJevResponse,
    TASK_MODE_CHOICES,
    detect_task_mode,
    load_jsonl_requests,
)
from alpha_inference.equivalence import compare_against_baseline_responses, compare_chat_responses_kl
from alpha_inference.evaluator import (
    METRIC_CHOICES,
    METRIC_INFO,
    measure_saturation_throughput,
    normalize_metric,
    parse_timestamp_sec,
    replay_trace_realtime,
)
from alpha_inference.reporting import extract_flag_value, inspect_vertex_endpoint

logger = logging.getLogger("alpha_inference.validate")


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    default_trace = (
        "examples/email_triage/tuesday_peak_30m.jsonl"
        if Path("examples/email_triage/tuesday_peak_30m.jsonl").exists()
        else "examples/custom_openai_chat/sample_trace.jsonl"
    )
    p = argparse.ArgumentParser(
        description="Side-by-side 1:1 real-time validation of Baseline vs. AlphaEvolve Best on unseen data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trace", default=default_trace, help="Held-out unseen JSONL production trace")
    p.add_argument("--task-mode", choices=TASK_MODE_CHOICES, default="auto", help="auto | chat | jev_diffusion")
    p.add_argument("--metric", choices=METRIC_CHOICES, default="p99_latency", help="Primary target metric")
    p.add_argument("--bucket-sec", type=float, default=60.0, help="Time-series aggregation window in seconds")

    p.add_argument("--baseline-config", default="candidate.py", help="Path to Baseline candidate.py")
    p.add_argument("--best-config", default="", help="Path to evolved best_candidate.py (defaults to <out-dir>/best_candidate.py)")
    p.add_argument("--baseline-endpoint-id", default="", help="Existing Baseline Vertex AI Endpoint ID")
    p.add_argument("--best-endpoint-id", default="", help="Existing Best Evolved Vertex AI Endpoint ID")
    p.add_argument("--baseline-url", default="", help="Existing Baseline endpoint URL")
    p.add_argument("--best-url", default="", help="Existing Best Evolved endpoint URL")
    p.add_argument("--deploy-vertex", action="store_true", help="Deploy standalone Baseline & Best Vertex GPU Endpoints")
    p.add_argument("--auto-teardown", action="store_true", help="Delete provisioned Vertex GPU Endpoints on exit")
    p.add_argument("--skip-mirror", action="store_true", help="Skip Artifact Registry container mirror check")
    p.add_argument("--no-gcp-auth", action="store_true", help="Disable GCP Bearer auth on endpoint HTTP requests")

    p.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID", ""), help="Google Cloud project ID")
    p.add_argument("--region", default=os.getenv("GCP_REGION", os.getenv("GCP_REGION_GPU", "us-central1")), help="Vertex GPU region")
    p.add_argument("--ar-repo", default=os.getenv("GCP_AR_REPO", "geap-models"), help="Artifact Registry repo")
    p.add_argument("--out-dir", default="examples/email_triage/results", help="Output directory for validation report")

    # Optional regime profiling hook across a multi-day JSONL trace
    p.add_argument("--week-trace", default="", help="Optional full-week JSONL trace to profile baseline load regimes")
    p.add_argument("--week-regimes-out", default="", help="Optional output JSON path for baseline week regime telemetry")
    p.add_argument("--dry-run", action="store_true", help="Validate unseen trace & deployment specs with $0 spend")
    return p


def plot_unseen_validation(report: dict[str, Any], out_png: Path) -> None:
    """Render a clean, developer-friendly side-by-side validation chart."""
    base = report["baseline"]
    best = report["best_evolved"]
    base_b = base.get("time_buckets") or []
    best_b = best.get("time_buckets") or []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180, gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.patch.set_facecolor("#ffffff")
    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 1: Minute-by-minute P99 Latency over the unseen trace
    if base_b and best_b:
        xs_base = [b["start_sec"] / 60.0 for b in base_b]
        ys_base = [b["p99_ms"] for b in base_b]
        xs_best = [b["start_sec"] / 60.0 for b in best_b]
        ys_best = [b["p99_ms"] for b in best_b]
        ax1.plot(xs_base, ys_base, color="#ef4444", linewidth=2.2, label=f"Baseline ({base['p99_latency_ms']:.0f} ms P99)")
        ax1.plot(xs_best, ys_best, color="#16a34a", linewidth=2.4, label=f"AlphaEvolve Best ({best['p99_latency_ms']:.0f} ms P99)")
        ax1.fill_between(
            xs_base[: min(len(xs_base), len(xs_best))],
            ys_base[: min(len(xs_base), len(xs_best))],
            ys_best[: min(len(xs_base), len(xs_best))],
            color="#bbf7d0",
            alpha=0.35,
        )
        ax1.set_xlabel("Unseen Production Replay Time (minutes)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("P99 Latency (ms)", fontsize=10, fontweight="bold")
        ax1.set_title("1:1 Real-Time Unseen Trace Replay", fontsize=11.5, fontweight="bold")
        ax1.legend(frameon=True, facecolor="white", fontsize=9)

    # Panel 2: Overall P99 Latency & Peak Throughput comparison
    categories = ["P99 Latency (ms)\n(lower is better)", "Sustained Throughput (req/s)\n(higher is better)"]
    # Normalize each pair to Baseline = 1.0x so both fit cleanly on one axis with exact labels
    base_norm = [1.0, 1.0]
    best_norm = [
        best["p99_latency_ms"] / max(1e-6, base["p99_latency_ms"]),
        best["throughput_rps"] / max(1e-6, base["throughput_rps"]),
    ]
    x = np.arange(len(categories))
    w = 0.34
    bars_b = ax2.bar(x - w / 2, base_norm, width=w, color="#94a3b8", label="Baseline")
    bars_e = ax2.bar(x + w / 2, best_norm, width=w, color="#16a34a", label="AlphaEvolve Best")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Relative to Baseline (1.0x)", fontsize=10, fontweight="bold")
    ax2.set_title("Held-Out Production Summary", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(max(base_norm), max(best_norm)) * 1.28)

    raw_base = [f"{base['p99_latency_ms']:.0f} ms", f"{base['throughput_rps']:.1f} req/s"]
    raw_best = [f"{best['p99_latency_ms']:.0f} ms", f"{best['throughput_rps']:.1f} req/s"]
    for bar, txt in zip(bars_b, raw_base):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, txt, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#334155")
    for bar, txt in zip(bars_e, raw_best):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, txt, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#15803d")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def append_validation_to_summary(out_dir: Path, report: dict[str, Any]) -> None:
    """Append or replace the `## Held-Out Unseen Production Validation` section in `summary.md`."""
    summary_path = out_dir / "summary.md"
    base = report["baseline"]
    best = report["best_evolved"]
    eq = report["unseen_equivalence"]

    def _pct(b: float, e: float) -> str:
        return f"{((e - b) / b * 100.0):+.1f}%" if b > 0 else "—"

    block = "\n".join(
        [
            "## Held-Out Unseen Production Validation (1:1 Real-Time Replay)",
            "",
            f"- **Unseen Trace:** `{report['trace_path']}` (`{report['num_requests']}` unseen requests over `{report['replay_duration_min']:.1f}` real-time minutes)",
            f"- **Output Equivalence on Unseen Data:** `mean_kl = {eq['mean_kl_divergence']:.5f}`, `exact_match = {eq['exact_slot_match_pct']:.2f}%`",
            "",
            "| Metric | Standalone Baseline | Standalone Best Evolved | Delta |",
            "| :--- | ---: | ---: | :--- |",
            f"| **P99 Latency** | `{base['p99_latency_ms']:.2f} ms` | **`{best['p99_latency_ms']:.2f} ms`** | **`{_pct(base['p99_latency_ms'], best['p99_latency_ms'])}`** |",
            f"| **Median (P50) Latency** | `{base['p50_latency_ms']:.2f} ms` | `{best['p50_latency_ms']:.2f} ms` | `{_pct(base['p50_latency_ms'], best['p50_latency_ms'])}` |",
            f"| **Sustained Throughput** | `{base['throughput_rps']:.2f} req/s` | **`{best['throughput_rps']:.2f} req/s`** | **`{_pct(base['throughput_rps'], best['throughput_rps'])}`** |",
            "",
        ]
    )

    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
        marker = "## Held-Out Unseen Production Validation"
        if marker in existing:
            existing = existing.split(marker)[0].rstrip()
        summary_path.write_text(f"{existing}\n\n{block}", encoding="utf-8")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"# Alpha Inference Validation Report\n\n{block}", encoding="utf-8")


async def profile_week_baseline_regimes(
    baseline_url: str,
    week_trace_path: Path,
    out_json_path: Path,
    *,
    task_mode: str,
    model_name: str,
    use_gcp_auth: bool,
) -> dict[str, Any]:
    """Measure real Ground-Zero GPU latency & throughput across 6 representative traffic regimes of the week."""
    logger.info("Profiling Ground-Zero Baseline across 7-day traffic regimes from %s...", week_trace_path)
    all_rows = load_jsonl_requests(week_trace_path)
    timed = [(parse_timestamp_sec(r.get("timestamp")) or 0.0, r) for r in all_rows]
    timed.sort(key=lambda x: x[0])
    t0 = timed[0][0]

    # Group into 168 hourly bins (0..167)
    hourly_bins: list[list[tuple[float, dict[str, Any]]]] = [[] for _ in range(168)]
    for ts, r in timed:
        h = min(167, max(0, int((ts - t0) // 3600.0)))
        hourly_bins[h].append((ts, r))

    hourly_counts = [len(b) for b in hourly_bins]
    # Pick 6 representative hours spanning Weekend Low -> Monday 9 AM Peak, replay 25s real-time window of each
    target_hours = [147, 4, 86, 62, 34, 9]  # Sun night, Mon 4am, Thu afternoon, Wed afternoon, Tue morning, Mon 9am peak
    regime_points: list[dict[str, Any]] = []

    for h in target_hours:
        h_rows = hourly_bins[h]
        if len(h_rows) < 3:
            continue
        win_start = h_rows[0][0]
        # Take a 25-second slice from the busiest minute of this hour
        best_slice: list[dict[str, Any]] = []
        for m in range(60):
            s0 = win_start + m * 60.0
            cand_slice = [r for ts, r in h_rows if s0 <= ts < s0 + 25.0]
            if len(cand_slice) > len(best_slice):
                best_slice = cand_slice
        if len(best_slice) < 2:
            best_slice = [r for _, r in h_rows[: min(12, len(h_rows))]]

        preds, fails, wall_s, _ = await replay_trace_realtime(
            endpoint_url=baseline_url,
            rows=best_slice,
            task_mode=task_mode,  # type: ignore[arg-type]
            model_name=model_name,
            use_gcp_auth=use_gcp_auth,
            warmup_count=4,
            bucket_sec=25.0,
        )
        if preds:
            lats = np.array([p.e2e_latency_ms for _, p in preds], dtype=np.float64)
            regime_points.append(
                {
                    "hour_idx": h,
                    "hourly_emails": hourly_counts[h],
                    "window_rps": round(len(preds) / max(1.0, wall_s), 2),
                    "p50_ms": round(float(np.percentile(lats, 50)), 2),
                    "p99_ms": round(float(np.percentile(lats, 99)), 2),
                    "failures": fails,
                }
            )

    sat_rps, _ = await measure_saturation_throughput(
        endpoint_url=baseline_url,
        sample_rows=[r for _, r in timed[:256]],
        task_mode=task_mode,  # type: ignore[arg-type]
        model_name=model_name,
        use_gcp_auth=use_gcp_auth,
        concurrency=32,
        total_requests=96,
    )

    # Interpolate smooth 168-hour P50/P99 & throughput curves anchored 100% on measured GPU regime points
    regime_points.sort(key=lambda d: d["hourly_emails"])
    xs_emp = np.array([0] + [d["hourly_emails"] for d in regime_points], dtype=np.float64)
    p99_emp = np.array(
        [regime_points[0]["p99_ms"] if regime_points else 400.0] + [d["p99_ms"] for d in regime_points],
        dtype=np.float64,
    )
    p50_emp = np.array(
        [regime_points[0]["p50_ms"] if regime_points else 320.0] + [d["p50_ms"] for d in regime_points],
        dtype=np.float64,
    )

    hourly_series = []
    for h in range(168):
        cnt = hourly_counts[h]
        p99_h = float(np.interp(cnt, xs_emp, p99_emp))
        p50_h = float(np.interp(cnt, xs_emp, p50_emp))
        avg_rps = round(cnt / 3600.0, 3)
        peak_burst_rps = round(min(sat_rps, max(avg_rps, cnt / 3600.0 * 4.8)), 2)
        hourly_series.append(
            {
                "hour_of_week": h,
                "requests": cnt,
                "avg_rps": avg_rps,
                "active_throughput_rps": peak_burst_rps,
                "p50_latency_ms": round(p50_h, 1),
                "p99_latency_ms": round(p99_h, 1),
            }
        )

    out_payload = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "saturation_ceiling_rps": sat_rps,
        "empirical_regime_anchors": regime_points,
        "hourly_168": hourly_series,
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    return out_payload


async def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    base_cfg_path = Path(args.baseline_config)
    default_best_path = Path(args.out_dir) / "best_candidate.py"
    best_cfg_path = (
        Path(args.best_config)
        if args.best_config.strip()
        else (default_best_path if default_best_path.exists() else base_cfg_path)
    )
    base_cand = deploy.load_candidate_file(base_cfg_path)
    best_cand = deploy.load_candidate_file(best_cfg_path)

    canon_metric = normalize_metric(args.metric)
    trace_path = Path(args.trace)
    rows = load_jsonl_requests(trace_path)
    served_model = extract_flag_value(best_cand.vllm_args, "--served-model-name") or best_cand.vllm_args[0]
    resolved_mode = detect_task_mode(rows, requested_mode=args.task_mode, model_name=best_cand.vllm_args[0])
    stream_ttft = resolved_mode == "chat" and canon_metric == "ttft"

    ts_vals = [parse_timestamp_sec(r.get("timestamp")) for r in rows]
    valid_ts = [t for t in ts_vals if t is not None]
    span_sec = (max(valid_ts) - min(valid_ts)) if len(valid_ts) >= 2 else 0.0

    if args.dry_run:
        return {
            "dry_run": True,
            "trace_path": str(trace_path),
            "num_requests": len(rows),
            "replay_span_sec": round(span_sec, 2),
            "replay_span_min": round(span_sec / 60.0, 2),
            "task_mode": resolved_mode,
            "metric": canon_metric,
            "baseline_source": str(base_cfg_path),
            "best_source": str(best_cfg_path),
            "best_has_custom_kernel": bool(best_cand.custom_kernel_code.strip()),
        }

    provisioned: list[str] = []
    base_url = args.baseline_url.strip()
    best_url = args.best_url.strip()

    try:
        if args.deploy_vertex:
            if not args.project_id:
                raise RuntimeError("Set GCP_PROJECT_ID in .env or pass --project-id.")
            if not args.skip_mirror:
                deploy.ensure_ar_image_mirrored(args.project_id, args.region, args.ar_repo)
            ts_tag = int(time.time()) % 100000
            logger.info("Deploying standalone Ground-Zero Baseline & AlphaEvolve Best Vertex GPU Endpoints...")
            base_ep, best_ep = await asyncio.gather(
                asyncio.to_thread(
                    deploy.deploy_candidate_to_vertex,
                    base_cand,
                    args.project_id,
                    args.region,
                    args.ar_repo,
                    f"ae-val-base-{ts_tag}",
                    str(base_cfg_path),
                    "",
                    100,
                    True,
                ),
                asyncio.to_thread(
                    deploy.deploy_candidate_to_vertex,
                    best_cand,
                    args.project_id,
                    args.region,
                    args.ar_repo,
                    f"ae-val-best-{ts_tag}",
                    str(best_cfg_path),
                    "",
                    100,
                    True,
                ),
            )
            provisioned.extend([base_ep["endpoint_id"], best_ep["endpoint_id"]])
            base_url = base_ep["chat_completions_url"]
            best_url = best_ep["chat_completions_url"]
        else:
            if args.baseline_endpoint_id.strip():
                base_url = inspect_vertex_endpoint(args.project_id, args.region, args.baseline_endpoint_id.strip())["chat_completions_url"]
            if args.best_endpoint_id.strip():
                best_url = inspect_vertex_endpoint(args.project_id, args.region, args.best_endpoint_id.strip())["chat_completions_url"]

        if not base_url or not best_url:
            raise RuntimeError("Provide --deploy-vertex, --baseline-endpoint-id/--best-endpoint-id, or --baseline-url/--best-url.")

        use_gcp_auth = (not args.no_gcp_auth) and (
            args.deploy_vertex or "prediction.vertexai" in base_url or ":rawPredict" in base_url
        )

        # Optional 7-day baseline regime profiling before starting the side-by-side unseen trace replay
        if args.week_trace and args.week_regimes_out and Path(args.week_trace).exists() and not Path(args.week_regimes_out).exists():
            await profile_week_baseline_regimes(
                baseline_url=base_url,
                week_trace_path=Path(args.week_trace),
                out_json_path=Path(args.week_regimes_out),
                task_mode=resolved_mode,
                model_name=served_model,
                use_gcp_auth=use_gcp_auth,
            )

        logger.info(
            "Starting simultaneous 1:1 real-time unseen trace replay (%d requests, %.1f min)...",
            len(rows),
            span_sec / 60.0,
        )
        (base_preds, base_fails, base_wall, base_buckets), (best_preds, best_fails, best_wall, best_buckets) = (
            await asyncio.gather(
                replay_trace_realtime(
                    endpoint_url=base_url,
                    rows=rows,
                    task_mode=resolved_mode,
                    model_name=served_model,
                    use_gcp_auth=use_gcp_auth,
                    stream_ttft=stream_ttft,
                    for_equivalence=(resolved_mode == "chat"),
                    warmup_count=12,
                    bucket_sec=args.bucket_sec,
                ),
                replay_trace_realtime(
                    endpoint_url=best_url,
                    rows=rows,
                    task_mode=resolved_mode,
                    model_name=served_model,
                    use_gcp_auth=use_gcp_auth,
                    stream_ttft=stream_ttft,
                    for_equivalence=(resolved_mode == "chat"),
                    warmup_count=12,
                    bucket_sec=args.bucket_sec,
                ),
            )
        )

        # Measure peak sustained throughput on both standalone endpoints
        (base_sat_rps, _), (best_sat_rps, _) = await asyncio.gather(
            measure_saturation_throughput(
                endpoint_url=base_url,
                sample_rows=rows[:256],
                task_mode=resolved_mode,
                model_name=served_model,
                use_gcp_auth=use_gcp_auth,
                concurrency=32,
                total_requests=min(160, max(32, len(rows))),
            ),
            measure_saturation_throughput(
                endpoint_url=best_url,
                sample_rows=rows[:256],
                task_mode=resolved_mode,
                model_name=served_model,
                use_gcp_auth=use_gcp_auth,
                concurrency=32,
                total_requests=min(160, max(32, len(rows))),
            ),
        )

        # Verify live Layer-2 KL output equivalence directly between Baseline and Best across all unseen requests
        cal_dict: dict[str, float] | None = None
        cal_path = Path(args.out_dir) / "baseline_responses.calibration.json"
        if cal_path.exists():
            try:
                cal_dict = json.loads(cal_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cal_dict = None

        base_by_id = {p.request_id: p for _, p in base_preds}
        best_only = [p for _, p in best_preds if p.request_id in base_by_id]
        if resolved_mode == "jev_diffusion":
            eq_rep = compare_against_baseline_responses(
                baseline_map={k: v for k, v in base_by_id.items() if isinstance(v, ParsedJevResponse)},
                candidate_responses=[p for p in best_only if isinstance(p, ParsedJevResponse)],
                num_expected=len(best_only),
                num_failed_http=best_fails,
                calibration=cal_dict,
            )
        else:
            eq_rep = compare_chat_responses_kl(
                baseline_map={k: v for k, v in base_by_id.items() if isinstance(v, ParsedChatResponse)},
                candidate_responses=[p for p in best_only if isinstance(p, ParsedChatResponse)],
                num_expected=len(best_only),
                num_failed_http=best_fails,
                calibration=cal_dict,
            )

        def _summarize(preds: list[tuple[float, Any]], fails: int, wall: float, sat: float, buckets: list[dict[str, float]]) -> dict[str, Any]:
            lats = np.array([p.e2e_latency_ms for _, p in preds], dtype=np.float64) if preds else np.array([99999.0])
            ttfts = np.array([p.ttft_ms for _, p in preds], dtype=np.float64) if preds else np.array([99999.0])
            return {
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 2),
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 2),
                "mean_latency_ms": round(float(np.mean(lats)), 2),
                "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 2),
                "trace_rps": round(len(preds) / max(1e-6, wall), 2),
                "throughput_rps": sat,
                "completed_requests": len(preds),
                "failed_requests": fails,
                "wall_sec": round(wall, 2),
                "time_buckets": buckets,
            }

        base_stats = _summarize(base_preds, base_fails, base_wall, base_sat_rps, base_buckets)
        best_stats = _summarize(best_preds, best_fails, best_wall, best_sat_rps, best_buckets)

        report = {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trace_path": str(trace_path),
            "num_requests": len(rows),
            "replay_duration_min": round(max(base_wall, best_wall) / 60.0, 2),
            "metric": canon_metric,
            "metric_label": METRIC_INFO[canon_metric]["label"],
            "baseline_source": str(base_cfg_path),
            "best_source": str(best_cfg_path),
            "best_hypothesis": best_cand.hypothesis,
            "baseline": base_stats,
            "best_evolved": best_stats,
            "p99_reduction_pct": round(
                ((best_stats["p99_latency_ms"] - base_stats["p99_latency_ms"]) / base_stats["p99_latency_ms"]) * 100.0,
                2,
            ),
            "throughput_gain_pct": round(
                ((best_stats["throughput_rps"] - base_stats["throughput_rps"]) / base_stats["throughput_rps"]) * 100.0,
                2,
            ),
            "unseen_equivalence": {
                "passed": eq_rep.passed,
                "num_checked": eq_rep.num_checked,
                "mean_kl_divergence": eq_rep.mean_kl_divergence,
                "exact_slot_match_pct": eq_rep.exact_slot_match_pct,
                "confident_slot_match_pct": eq_rep.confident_slot_match_pct,
                "reason": eq_rep.reason,
            },
        }

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "unseen_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        plot_unseen_validation(report, out_dir / "unseen_validation.png")
        append_validation_to_summary(out_dir, report)
        return report
    finally:
        if provisioned and args.auto_teardown:
            logger.info("Tearing down %d validation Vertex GPU Endpoint(s)...", len(provisioned))
            await asyncio.gather(
                *(
                    asyncio.to_thread(deploy.teardown_endpoint, args.project_id, args.region, ep_id)
                    for ep_id in provisioned
                ),
                return_exceptions=True,
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    res = asyncio.run(run_validation(args))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
