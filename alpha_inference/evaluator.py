"""1:1 Real-Time Production Trace Replay, KL Gate & Multi-Metric Evaluator."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import aiohttp
import numpy as np

from alpha_inference.adapters import (
    SERVED_MODEL_NAME,
    ParsedResponse,
    ResolvedTaskMode,
    detect_task_mode,
    fetch_gcp_bearer_token,
    load_jsonl_requests,
    send_one_request,
)
from alpha_inference.equivalence import EquivalenceReport, run_equivalence_gate

MetricName = Literal["p99_latency", "latency", "throughput", "ttft"]
METRIC_CHOICES: tuple[str, ...] = ("p99_latency", "latency", "throughput", "ttft")

METRIC_INFO: dict[str, dict[str, str]] = {
    "p99_latency": {"label": "P99 Latency", "unit": "ms", "direction": "minimize"},
    "latency": {"label": "Median (P50) Latency", "unit": "ms", "direction": "minimize"},
    "throughput": {"label": "Sustained Throughput", "unit": "req/s", "direction": "maximize"},
    "ttft": {"label": "Time To First Token (P99 TTFT)", "unit": "ms", "direction": "minimize"},
}


def normalize_metric(metric: str) -> MetricName:
    m = metric.strip().lower()
    aliases = {
        "p99": "p99_latency",
        "p99_latency": "p99_latency",
        "p99_e2e_latency": "p99_latency",
        "latency": "latency",
        "p50_latency": "latency",
        "mean_latency": "latency",
        "throughput": "throughput",
        "ttft": "ttft",
    }
    if m not in aliases:
        raise ValueError(f"Unsupported --metric {metric!r}; choose from {METRIC_CHOICES}")
    return aliases[m]  # type: ignore[return-value]


def is_better(candidate_val: float, best_val: float, metric: str) -> bool:
    m = normalize_metric(metric)
    if math.isnan(candidate_val) or math.isinf(candidate_val):
        return False
    if METRIC_INFO[m]["direction"] == "maximize":
        return candidate_val > best_val + 1e-6 and candidate_val > 0.0
    return 0.0 < candidate_val < best_val - 1e-6 and candidate_val < 90000.0


def format_metric(val: float, metric: str) -> str:
    m = normalize_metric(metric)
    unit = METRIC_INFO[m]["unit"]
    return f"{val:.2f} {unit}"


def to_alpha_evolve_score(val: float, metric: str) -> float:
    """Convert raw metric to a 32-bit-safe maximized fitness score for Google Cloud AlphaEvolve."""
    m = normalize_metric(metric)
    if math.isnan(val) or math.isinf(val):
        return -1_000_000.0
    raw = val if METRIC_INFO[m]["direction"] == "maximize" else -val
    return float(max(-1_000_000.0, min(1_000_000.0, raw)))


@dataclass
class TraceEvalResult:
    """Complete evaluation telemetry for one candidate configuration."""

    candidate_name: str
    equivalence: EquivalenceReport
    p99_latency_ms: float
    p50_latency_ms: float
    mean_latency_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    trace_rps: float
    throughput_rps: float
    num_requests: int
    num_failures: int
    eval_wall_sec: float
    time_buckets: list[dict[str, float]] = field(default_factory=list)
    responses_by_id: dict[str, ParsedResponse] = field(default_factory=dict, repr=False)

    def metric_value(self, metric: str) -> float:
        m = normalize_metric(metric)
        if m == "p99_latency":
            return self.p99_latency_ms
        if m == "latency":
            return self.p50_latency_ms
        if m == "throughput":
            return self.throughput_rps
        return self.ttft_p99_ms

    @classmethod
    def failed(cls, candidate_name: str, equivalence: EquivalenceReport, wall_sec: float = 0.0) -> TraceEvalResult:
        return cls(
            candidate_name=candidate_name,
            equivalence=equivalence,
            p99_latency_ms=99999.0,
            p50_latency_ms=99999.0,
            mean_latency_ms=99999.0,
            ttft_p50_ms=99999.0,
            ttft_p99_ms=99999.0,
            trace_rps=0.0,
            throughput_rps=0.0,
            num_requests=0,
            num_failures=equivalence.num_failed_http,
            eval_wall_sec=wall_sec,
        )


def parse_timestamp_sec(raw_ts: Any) -> float | None:
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)):
        return float(raw_ts)
    s = str(raw_ts).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


async def replay_trace_realtime(
    endpoint_url: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    task_mode: ResolvedTaskMode,
    model_name: str,
    use_gcp_auth: bool = True,
    stream_ttft: bool = False,
    for_equivalence: bool = False,
    warmup_count: int = 8,
    max_replay_span_sec: float = 3600.0,
    fallback_concurrency: int = 24,
    bucket_sec: float = 60.0,
) -> tuple[list[tuple[float, ParsedResponse]], int, float, list[dict[str, float]]]:
    """Replay `rows` at exact 1:1 production arrival timestamps (or concurrent fallback if untimed)."""
    parsed_ts = [parse_timestamp_sec(r.get("timestamp")) for r in rows]
    has_timestamps = (
        len(rows) >= 2
        and all(t is not None for t in parsed_ts)
        and 0.0 < (max(t for t in parsed_ts if t is not None) - min(t for t in parsed_ts if t is not None)) <= max_replay_span_sec
    )

    connector = aiohttp.TCPConnector(limit=512, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=180.0)
    timed_preds: list[tuple[float, ParsedResponse]] = []
    failures = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup requests (excluded from timed metrics) so CUDA graphs and caches start warm
        if warmup_count > 0 and rows:
            tok = fetch_gcp_bearer_token() if use_gcp_auth else None
            warm_n = min(warmup_count, len(rows))
            await asyncio.gather(
                *(
                    send_one_request(
                        session,
                        endpoint_url,
                        rows[i],
                        task_mode=task_mode,
                        model_name=model_name,
                        bearer_token=tok,
                        for_equivalence=False,
                        stream_ttft=stream_ttft,
                    )
                    for i in range(warm_n)
                ),
                return_exceptions=True,
            )

        t_wall_0 = time.perf_counter()

        if has_timestamps:
            indexed = sorted(
                ((float(t), r) for t, r in zip(parsed_ts, rows) if t is not None),
                key=lambda x: x[0],
            )
            t_first = indexed[0][0]
            in_flight: set[asyncio.Task[None]] = set()
            # Cap concurrent in-flight HTTP sockets per Vertex replica while starting the latency
            # clock at exact 1:1 arrival time (MLPerf Server open-loop queueing semantics).
            in_flight_sem = asyncio.Semaphore(48)

            async def _dispatch_one(offset_s: float, req_obj: Mapping[str, Any]) -> None:
                nonlocal failures
                t_arrival = time.perf_counter()
                async with in_flight_sem:
                    client_queue_ms = max(0.0, (time.perf_counter() - t_arrival) * 1000.0)
                    tok = fetch_gcp_bearer_token() if use_gcp_auth else None
                    try:
                        p = await send_one_request(
                            session,
                            endpoint_url,
                            req_obj,
                            task_mode=task_mode,
                            model_name=model_name,
                            bearer_token=tok,
                            for_equivalence=for_equivalence,
                            stream_ttft=stream_ttft,
                        )
                        if client_queue_ms > 0.0:
                            p.e2e_latency_ms = round(p.e2e_latency_ms + client_queue_ms, 3)
                            p.ttft_ms = round(p.ttft_ms + client_queue_ms, 3)
                        timed_preds.append((offset_s, p))
                    except Exception:  # noqa: BLE001
                        failures += 1

            for abs_t, req_obj in indexed:
                offset_s = abs_t - t_first
                wait_s = (t_wall_0 + offset_s) - time.perf_counter()
                if wait_s > 0.0005:
                    await asyncio.sleep(wait_s)
                task = asyncio.create_task(_dispatch_one(offset_s, req_obj))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)

            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
        else:
            sem = asyncio.Semaphore(min(fallback_concurrency, max(1, len(rows))))

            async def _fire_untimed(req_obj: Mapping[str, Any]) -> None:
                nonlocal failures
                async with sem:
                    off = time.perf_counter() - t_wall_0
                    tok = fetch_gcp_bearer_token() if use_gcp_auth else None
                    try:
                        p = await send_one_request(
                            session,
                            endpoint_url,
                            req_obj,
                            task_mode=task_mode,
                            model_name=model_name,
                            bearer_token=tok,
                            for_equivalence=for_equivalence,
                            stream_ttft=stream_ttft,
                        )
                        timed_preds.append((off, p))
                    except Exception:  # noqa: BLE001
                        failures += 1

            await asyncio.gather(*(asyncio.create_task(_fire_untimed(r)) for r in rows))

        wall_sec = max(1e-6, time.perf_counter() - t_wall_0)

    # Build time-series buckets (e.g. per-minute over 30-min validation or per-30s over 3-min search)
    buckets: list[dict[str, float]] = []
    if timed_preds and bucket_sec > 0:
        max_off = max(off for off, _ in timed_preds)
        n_b = max(1, int(max_off // bucket_sec) + 1)
        grouped: list[list[ParsedResponse]] = [[] for _ in range(n_b)]
        for off, pr in timed_preds:
            bi = min(n_b - 1, max(0, int(off // bucket_sec)))
            grouped[bi].append(pr)
        for bi, plist in enumerate(grouped):
            if not plist:
                continue
            lats = np.array([x.e2e_latency_ms for x in plist], dtype=np.float64)
            ttfts = np.array([x.ttft_ms for x in plist], dtype=np.float64)
            eff_span = min(bucket_sec, max(1.0, max_off - bi * bucket_sec)) if bi == n_b - 1 else bucket_sec
            buckets.append(
                {
                    "bucket_idx": float(bi),
                    "start_sec": round(bi * bucket_sec, 1),
                    "requests": float(len(plist)),
                    "rps": round(len(plist) / eff_span, 2),
                    "p50_ms": round(float(np.percentile(lats, 50)), 2),
                    "p99_ms": round(float(np.percentile(lats, 99)), 2),
                    "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 2),
                }
            )

    return timed_preds, failures, wall_sec, buckets


async def measure_saturation_throughput(
    endpoint_url: str,
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    task_mode: ResolvedTaskMode,
    model_name: str,
    use_gcp_auth: bool = True,
    concurrency: int = 32,
    total_requests: int = 128,
) -> tuple[float, int]:
    """Short saturation probe (`c=32`) to measure peak sustained GPU throughput (`req/s`)."""
    sem = asyncio.Semaphore(concurrency)
    ok_count = 0
    failures = 0
    connector = aiohttp.TCPConnector(limit=256, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=90.0)) as session:
        async def _one(i: int) -> None:
            nonlocal ok_count, failures
            req = sample_rows[i % len(sample_rows)]
            async with sem:
                tok = fetch_gcp_bearer_token() if use_gcp_auth else None
                try:
                    await send_one_request(
                        session,
                        endpoint_url,
                        req,
                        task_mode=task_mode,
                        model_name=model_name,
                        bearer_token=tok,
                        stream_ttft=False,
                    )
                    ok_count += 1
                except Exception:  # noqa: BLE001
                    failures += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(asyncio.create_task(_one(i)) for i in range(total_requests)))
        wall = max(1e-6, time.perf_counter() - t0)
    return round(ok_count / wall, 2), failures


async def evaluate_candidate_on_endpoint(
    candidate_name: str,
    endpoint_url: str,
    trace_path: Path,
    equiv_path: Path,
    baseline_responses_path: Path,
    *,
    record_baseline: bool = False,
    secondary_baseline_url: str = "",
    use_gcp_auth: bool = True,
    task_mode: str = "auto",
    model_name: str = SERVED_MODEL_NAME,
    metric: str = "p99_latency",
) -> TraceEvalResult:
    """Tier-2 Layer-2 KL Equivalence Gate + Tier-3 1:1 Real-Time Trace Replay + Saturation Throughput."""
    t0 = time.perf_counter()
    trace_rows = load_jsonl_requests(trace_path)
    equiv_rows = load_jsonl_requests(equiv_path) if equiv_path.exists() else trace_rows[: min(80, len(trace_rows))]
    resolved_mode = detect_task_mode(trace_rows, requested_mode=task_mode, model_name=model_name)
    canon_metric = normalize_metric(metric)
    stream_ttft = resolved_mode == "chat" and canon_metric == "ttft"
    tok = fetch_gcp_bearer_token() if use_gcp_auth else None

    equiv_report = await run_equivalence_gate(
        endpoint_url=endpoint_url,
        equiv_requests=equiv_rows,
        baseline_responses_path=baseline_responses_path,
        record_baseline=record_baseline,
        bearer_token=tok,
        secondary_baseline_url=secondary_baseline_url,
        task_mode=resolved_mode,
        model_name=model_name,
    )
    if not equiv_report.passed:
        return TraceEvalResult.failed(
            candidate_name=candidate_name,
            equivalence=equiv_report,
            wall_sec=round(time.perf_counter() - t0, 2),
        )

    timed_preds, trace_fails, trace_wall, buckets = await replay_trace_realtime(
        endpoint_url=endpoint_url,
        rows=trace_rows,
        task_mode=resolved_mode,
        model_name=model_name,
        use_gcp_auth=use_gcp_auth,
        stream_ttft=stream_ttft,
        warmup_count=8,
        bucket_sec=30.0,
    )

    sat_rps, sat_fails = await measure_saturation_throughput(
        endpoint_url=endpoint_url,
        sample_rows=trace_rows,
        task_mode=resolved_mode,
        model_name=model_name,
        use_gcp_auth=use_gcp_auth,
        concurrency=32,
        total_requests=min(128, max(32, len(trace_rows))),
    )

    total_fails = trace_fails + sat_fails
    max_tolerable = max(3, int(0.01 * len(trace_rows))) if record_baseline else 0
    if total_fails > max_tolerable or not timed_preds:
        bad_eq = EquivalenceReport(
            passed=False,
            num_checked=equiv_report.num_checked,
            num_failed_http=total_fails,
            exact_slot_matches=equiv_report.exact_slot_matches,
            total_slots=equiv_report.total_slots,
            exact_slot_match_pct=equiv_report.exact_slot_match_pct,
            max_prob_abs_diff=equiv_report.max_prob_abs_diff,
            reason=f"Failed {total_fails} HTTP request(s) under burst trace replay",
            mean_kl_divergence=equiv_report.mean_kl_divergence,
            mean_prob_abs_diff=equiv_report.mean_prob_abs_diff,
            confident_slot_match_pct=equiv_report.confident_slot_match_pct,
            confident_slots=equiv_report.confident_slots,
        )
        return TraceEvalResult.failed(
            candidate_name=candidate_name,
            equivalence=bad_eq,
            wall_sec=round(time.perf_counter() - t0, 2),
        )

    lats = np.array([p.e2e_latency_ms for _, p in timed_preds], dtype=np.float64)
    ttfts = np.array([p.ttft_ms for _, p in timed_preds], dtype=np.float64)

    return TraceEvalResult(
        candidate_name=candidate_name,
        equivalence=equiv_report,
        p99_latency_ms=round(float(np.percentile(lats, 99)), 2),
        p50_latency_ms=round(float(np.percentile(lats, 50)), 2),
        mean_latency_ms=round(float(np.mean(lats)), 2),
        ttft_p50_ms=round(float(np.percentile(ttfts, 50)), 2),
        ttft_p99_ms=round(float(np.percentile(ttfts, 99)), 2),
        trace_rps=round(len(timed_preds) / trace_wall, 2),
        throughput_rps=sat_rps,
        num_requests=len(timed_preds),
        num_failures=0,
        eval_wall_sec=round(time.perf_counter() - t0, 2),
        time_buckets=buckets,
    )
