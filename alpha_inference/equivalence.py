"""Layer-2 Logprob / KL-Divergence Output Equivalence Gate (`jev_diffusion` & generic OpenAI `chat`)."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp

from alpha_inference.adapters import (
    DECISION_DIMENSIONS,
    SERVED_MODEL_NAME,
    ParsedChatResponse,
    ParsedJevResponse,
    ParsedResponse,
    ResolvedTaskMode,
    detect_task_mode,
    send_one_chat_request,
    send_one_jev_request,
)


@dataclass
class EquivalenceReport:
    """Result of comparing a candidate server/kernel against Ground-Zero baseline outputs."""

    passed: bool
    num_checked: int
    num_failed_http: int
    exact_slot_matches: int
    total_slots: int
    exact_slot_match_pct: float
    max_prob_abs_diff: float
    reason: str
    mean_kl_divergence: float = 0.0
    mean_prob_abs_diff: float = 0.0
    confident_slot_match_pct: float = 100.0
    confident_slots: int = 0


def _unwrap_jev_ref(ref_entry: Mapping[str, Any] | ParsedJevResponse) -> tuple[Mapping[str, int], Mapping[str, Mapping[str, float]]]:
    if isinstance(ref_entry, ParsedJevResponse):
        return ref_entry.ratings, ref_entry.slot_probabilities
    return ref_entry["ratings"], ref_entry["slot_probabilities"]


def compare_against_baseline_responses(
    baseline_map: Mapping[str, Mapping[str, Any] | ParsedJevResponse],
    candidate_responses: Sequence[ParsedJevResponse],
    num_expected: int | None = None,
    num_failed_http: int = 0,
    max_allowed_mean_kl: float = 0.020,
    max_allowed_mean_prob_diff: float = 0.020,
    min_confident_slot_pct: float = 99.0,
    min_exact_slot_pct: float = 95.0,
    confident_margin: float = 0.20,
    calibration: Mapping[str, float] | None = None,
) -> EquivalenceReport:
    """Layer-2 Logprob / KL-Divergence + Confident-Decision Equivalence Gate for 1-step Jev."""
    expected = len(candidate_responses) if num_expected is None else num_expected
    if num_failed_http > 0 or len(candidate_responses) < expected:
        return EquivalenceReport(
            passed=False,
            num_checked=len(candidate_responses),
            num_failed_http=num_failed_http,
            exact_slot_matches=0,
            total_slots=expected * 3,
            exact_slot_match_pct=0.0,
            max_prob_abs_diff=1.0,
            reason=f"HTTP/parse failures during equivalence gate ({num_failed_http} failed)",
            mean_kl_divergence=10.0,
            mean_prob_abs_diff=1.0,
            confident_slot_match_pct=0.0,
            confident_slots=0,
        )

    eff_max_kl = max_allowed_mean_kl
    eff_max_mean_prob = max_allowed_mean_prob_diff
    eff_min_conf_pct = min_confident_slot_pct
    eff_min_exact_pct = min_exact_slot_pct
    eff_min_within_one_pct = 98.5
    eff_max_catastrophic_pct = 1.0
    if calibration:
        eff_max_kl = min(0.25, max(max_allowed_mean_kl, float(calibration.get("self_mean_kl", 0.0)) * 2.2 + 0.015))
        eff_max_mean_prob = min(
            0.08, max(max_allowed_mean_prob_diff, float(calibration.get("self_mean_prob_diff", 0.0)) * 2.2 + 0.012)
        )
        eff_min_conf_pct = max(
            90.0, min(min_confident_slot_pct, float(calibration.get("self_confident_match_pct", 100.0)) - 2.5)
        )
        eff_min_exact_pct = max(
            90.0, min(min_exact_slot_pct, float(calibration.get("self_exact_slot_match_pct", calibration.get("self_exact_match_pct", 100.0))) - 2.5)
        )
        eff_min_within_one_pct = max(
            96.0, min(99.5, float(calibration.get("self_within_one_pct", 99.5)) - 1.5)
        )
        eff_max_catastrophic_pct = min(
            2.5, max(1.0, float(calibration.get("self_catastrophic_pct", 0.0)) + 0.75)
        )

    eps = 1e-7
    exact_matches = 0
    within_one_matches = 0
    total_slots = 0
    confident_slots = 0
    confident_matches = 0
    catastrophic_count = 0
    max_diff = 0.0
    sum_abs_diff = 0.0
    sum_kl = 0.0
    first_conf_mismatch = ""
    catastrophic_jump = ""

    for pred in candidate_responses:
        ref = baseline_map.get(pred.request_id)
        if ref is None:
            continue
        ref_ratings, ref_probs = _unwrap_jev_ref(ref)
        for dim in DECISION_DIMENSIONS:
            total_slots += 1
            p_vec = [float(ref_probs[dim][k_str]) for k_str in ("1", "2", "3", "4", "5")]
            q_vec = [float(pred.slot_probabilities[dim][k_str]) for k_str in ("1", "2", "3", "4", "5")]

            p_sorted = sorted(p_vec, reverse=True)
            margin = p_sorted[0] - p_sorted[1]
            same_label = int(pred.ratings[dim]) == int(ref_ratings[dim])
            label_jump = abs(int(pred.ratings[dim]) - int(ref_ratings[dim]))

            if same_label:
                exact_matches += 1
            if label_jump <= 1:
                within_one_matches += 1
            if margin >= confident_margin:
                confident_slots += 1
                if same_label:
                    confident_matches += 1
                elif not first_conf_mismatch:
                    first_conf_mismatch = (
                        f"{pred.request_id}.{dim} (margin={margin:.2f}): "
                        f"got {pred.ratings[dim]} vs baseline {ref_ratings[dim]}"
                    )
                if margin >= 0.35 and label_jump >= 2:
                    catastrophic_count += 1
                    if not catastrophic_jump:
                        catastrophic_jump = (
                            f"{pred.request_id}.{dim} jumped {label_jump} points "
                            f"(got {pred.ratings[dim]} vs confident baseline {ref_ratings[dim]})"
                        )

            slot_kl = 0.0
            for pk, qk in zip(p_vec, q_vec):
                diff = abs(pk - qk)
                sum_abs_diff += diff
                if diff > max_diff:
                    max_diff = diff
                if pk > eps:
                    slot_kl += pk * math.log(max(pk, eps) / max(qk, eps))
            sum_kl += max(0.0, slot_kl)

    match_pct = (100.0 * exact_matches / max(1, total_slots)) if total_slots else 0.0
    within_one_pct = (100.0 * within_one_matches / max(1, total_slots)) if total_slots else 0.0
    catastrophic_pct = (100.0 * catastrophic_count / max(1, total_slots)) if total_slots else 0.0
    conf_pct = (100.0 * confident_matches / confident_slots) if confident_slots else 100.0
    mean_kl = sum_kl / max(1, total_slots)
    mean_abs_diff = sum_abs_diff / max(1, total_slots * 5)

    def _build(passed: bool, reason: str) -> EquivalenceReport:
        return EquivalenceReport(
            passed=passed,
            num_checked=len(candidate_responses),
            num_failed_http=0,
            exact_slot_matches=exact_matches,
            total_slots=total_slots,
            exact_slot_match_pct=round(match_pct, 2),
            max_prob_abs_diff=round(max_diff, 5),
            reason=reason,
            mean_kl_divergence=round(mean_kl, 5),
            mean_prob_abs_diff=round(mean_abs_diff, 5),
            confident_slot_match_pct=round(conf_pct, 2),
            confident_slots=confident_slots,
        )

    if catastrophic_pct > eff_max_catastrophic_pct:
        return _build(
            False,
            f"Catastrophic confident jumps {catastrophic_pct:.2f}% > {eff_max_catastrophic_pct:.2f}% ({catastrophic_jump})",
        )
    if within_one_pct + 1e-9 < eff_min_within_one_pct:
        return _build(
            False,
            f"Within-1-point ordinal match {within_one_pct:.2f}% < {eff_min_within_one_pct:.1f}%",
        )
    if mean_kl > eff_max_kl:
        return _build(
            False,
            f"KL divergence {mean_kl:.5f} exceeds Layer-2 ceiling {eff_max_kl:.5f}",
        )
    if mean_abs_diff > eff_max_mean_prob:
        return _build(
            False,
            f"Mean probability drift {mean_abs_diff:.4f} exceeds Layer-2 ceiling {eff_max_mean_prob:.4f}",
        )
    if confident_slots > 0 and conf_pct + 1e-9 < eff_min_conf_pct:
        return _build(
            False,
            f"Confident-slot match {conf_pct:.2f}% < {eff_min_conf_pct:.1f}% ({first_conf_mismatch})",
        )
    if match_pct + 1e-9 < eff_min_exact_pct:
        return _build(
            False,
            f"Overall slot match {match_pct:.2f}% < {eff_min_exact_pct:.1f}%",
        )
    return _build(True, "ok")


def _build_aligned_top_prob_vectors(
    ref_top: Mapping[str, float],
    cand_top: Mapping[str, float],
) -> tuple[list[float], list[float]]:
    """Construct normalized probability vectors over `union(ref_top, cand_top) + ['__other__']`."""
    vocab_union = list(dict.fromkeys([*ref_top.keys(), *cand_top.keys()]))

    def _to_probs(lp_map: Mapping[str, float]) -> list[float]:
        present = {tok: math.exp(max(-30.0, min(0.0, float(lp_map[tok])))) for tok in vocab_union if tok in lp_map}
        mass = sum(present.values())
        min_present = min(present.values()) if present else 1e-4
        missing_count = (len(vocab_union) - len(present)) + 1  # +1 for __other__ tail bucket
        rem = max(1e-6, 1.0 - mass)
        per_missing = min(min_present * 0.5, rem / max(1, missing_count))
        vec: list[float] = []
        used = 0.0
        for tok in vocab_union:
            if tok in present:
                p = present[tok]
            else:
                p = max(1e-7, per_missing)
            vec.append(p)
            used += p
        other_p = max(1e-7, 1.0 - used)
        vec.append(other_p)
        total = sum(vec) or 1.0
        return [x / total for x in vec]

    return _to_probs(ref_top), _to_probs(cand_top)


def compare_chat_responses_kl(
    baseline_map: Mapping[str, Mapping[str, Any] | ParsedChatResponse],
    candidate_responses: Sequence[ParsedChatResponse],
    num_expected: int | None = None,
    num_failed_http: int = 0,
    max_positions: int = 32,
    max_allowed_mean_kl: float = 0.030,
    max_allowed_mean_prob_diff: float = 0.030,
    min_confident_token_pct: float = 98.0,
    min_exact_token_pct: float = 90.0,
    confident_margin: float = 0.20,
    calibration: Mapping[str, float] | None = None,
) -> EquivalenceReport:
    """Layer-2 Token-Logprob KL Equivalence Gate for generic OpenAI `/v1/chat/completions`."""
    expected = len(candidate_responses) if num_expected is None else num_expected
    if num_failed_http > 0 or len(candidate_responses) < expected:
        return EquivalenceReport(
            passed=False,
            num_checked=len(candidate_responses),
            num_failed_http=num_failed_http,
            exact_slot_matches=0,
            total_slots=expected,
            exact_slot_match_pct=0.0,
            max_prob_abs_diff=1.0,
            reason=f"HTTP/parse failures during chat equivalence gate ({num_failed_http} failed)",
            mean_kl_divergence=10.0,
            mean_prob_abs_diff=1.0,
            confident_slot_match_pct=0.0,
            confident_slots=0,
        )

    eff_max_kl = max_allowed_mean_kl
    eff_max_mean_prob = max_allowed_mean_prob_diff
    eff_min_conf_pct = min_confident_token_pct
    eff_min_exact_pct = min_exact_token_pct
    if calibration:
        eff_max_kl = min(0.30, max(max_allowed_mean_kl, float(calibration.get("self_mean_kl", 0.0)) * 2.2 + 0.015))
        eff_max_mean_prob = min(
            0.10, max(max_allowed_mean_prob_diff, float(calibration.get("self_mean_prob_diff", 0.0)) * 2.2 + 0.012)
        )
        eff_min_conf_pct = max(
            90.0, min(min_confident_token_pct, float(calibration.get("self_confident_match_pct", 100.0)) - 2.5)
        )
        eff_min_exact_pct = max(
            85.0, min(min_exact_token_pct, float(calibration.get("self_exact_match_pct", 100.0)) - 3.0)
        )

    eps = 1e-7
    exact_matches = 0
    total_positions = 0
    confident_positions = 0
    confident_matches = 0
    max_diff = 0.0
    sum_abs_diff = 0.0
    total_prob_entries = 0
    sum_kl = 0.0
    first_conf_mismatch = ""

    for pred in candidate_responses:
        ref = baseline_map.get(pred.request_id)
        if ref is None:
            continue
        if isinstance(ref, ParsedChatResponse):
            ref_content = ref.content
            ref_positions = ref.token_positions
        else:
            ref_content = str(ref.get("content", ""))
            ref_positions = list(ref.get("token_positions") or [])

        # Fallback to normalized exact text match only when the server does not emit logprobs at all
        if not ref_positions:
            total_positions += 1
            confident_positions += 1
            if pred.content.strip() == ref_content.strip():
                exact_matches += 1
                confident_matches += 1
            elif not first_conf_mismatch:
                first_conf_mismatch = f"{pred.request_id}: text mismatch vs baseline"
            continue

        if not pred.token_positions:
            return EquivalenceReport(
                passed=False,
                num_checked=len(candidate_responses),
                num_failed_http=0,
                exact_slot_matches=0,
                total_slots=max(1, len(ref_positions)),
                exact_slot_match_pct=0.0,
                max_prob_abs_diff=1.0,
                reason=f"Candidate {pred.request_id} omitted token logprobs required by Layer-2 KL gate",
                mean_kl_divergence=10.0,
                mean_prob_abs_diff=1.0,
                confident_slot_match_pct=0.0,
                confident_slots=0,
            )

        n_ref = min(len(ref_positions), max_positions)
        n_cand = min(len(pred.token_positions), max_positions)
        n_shared = min(n_ref, n_cand)
        diverged = False

        for pos_idx in range(n_shared):
            r_pos = ref_positions[pos_idx]
            c_pos = pred.token_positions[pos_idx]
            r_tok = str(r_pos.get("token", ""))
            c_tok = str(c_pos.get("token", ""))
            r_top = dict(r_pos.get("top_logprobs") or {})
            c_top = dict(c_pos.get("top_logprobs") or {})

            p_vec, q_vec = _build_aligned_top_prob_vectors(r_top, c_top)
            # Compute confidence margin over actual candidate tokens (excluding __other__ bucket)
            token_ps = sorted(p_vec[:-1], reverse=True)
            margin = (token_ps[0] - token_ps[1]) if len(token_ps) >= 2 else (token_ps[0] if token_ps else 0.0)
            same_tok = r_tok == c_tok

            total_positions += 1
            if same_tok:
                exact_matches += 1
            if margin >= confident_margin:
                confident_positions += 1
                if same_tok:
                    confident_matches += 1
                elif not first_conf_mismatch:
                    first_conf_mismatch = (
                        f"{pred.request_id}@pos{pos_idx} (margin={margin:.2f}): "
                        f"got {c_tok!r} vs baseline {r_tok!r}"
                    )

            pos_kl = 0.0
            for pk, qk in zip(p_vec, q_vec):
                diff = abs(pk - qk)
                sum_abs_diff += diff
                total_prob_entries += 1
                if diff > max_diff:
                    max_diff = diff
                if pk > eps:
                    pos_kl += pk * math.log(max(pk, eps) / max(qk, eps))
            sum_kl += max(0.0, pos_kl)

            if not same_tok:
                # Subsequent autoregressive tokens condition on a different prefix; stop here for this prompt
                diverged = True
                break

        if not diverged and n_cand < n_ref:
            # Candidate truncated output early despite matching prefix
            total_positions += 1
            confident_positions += 1
            if not first_conf_mismatch:
                first_conf_mismatch = f"{pred.request_id}: truncated at {n_cand} tokens (baseline {n_ref})"

    match_pct = (100.0 * exact_matches / max(1, total_positions)) if total_positions else 0.0
    conf_pct = (100.0 * confident_matches / confident_positions) if confident_positions else 100.0
    mean_kl = sum_kl / max(1, total_positions)
    mean_abs_diff = sum_abs_diff / max(1, total_prob_entries)

    def _build(passed: bool, reason: str) -> EquivalenceReport:
        return EquivalenceReport(
            passed=passed,
            num_checked=len(candidate_responses),
            num_failed_http=0,
            exact_slot_matches=exact_matches,
            total_slots=total_positions,
            exact_slot_match_pct=round(match_pct, 2),
            max_prob_abs_diff=round(max_diff, 5),
            reason=reason,
            mean_kl_divergence=round(mean_kl, 5),
            mean_prob_abs_diff=round(mean_abs_diff, 5),
            confident_slot_match_pct=round(conf_pct, 2),
            confident_slots=confident_positions,
        )

    if mean_kl > eff_max_kl:
        return _build(False, f"KL divergence {mean_kl:.5f} exceeds Layer-2 ceiling {eff_max_kl:.5f}")
    if mean_abs_diff > eff_max_mean_prob:
        return _build(False, f"Mean probability drift {mean_abs_diff:.4f} exceeds Layer-2 ceiling {eff_max_mean_prob:.4f}")
    if confident_positions > 0 and conf_pct + 1e-9 < eff_min_conf_pct:
        return _build(False, f"Confident-token match {conf_pct:.2f}% < {eff_min_conf_pct:.1f}% ({first_conf_mismatch})")
    if match_pct + 1e-9 < eff_min_exact_pct:
        return _build(False, f"Token prefix match {match_pct:.2f}% < {eff_min_exact_pct:.1f}%")
    return _build(True, "ok")


async def _fetch_equiv_batch(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    requests_seq: Sequence[Mapping[str, Any]],
    *,
    task_mode: ResolvedTaskMode,
    model_name: str,
    bearer_token: str | None,
    concurrency: int,
) -> tuple[list[ParsedResponse], int, str]:
    """Run equivalence requests at controlled concurrency while preserving request order."""
    first_err = ""

    async def _call_one(req_obj: Mapping[str, Any]) -> ParsedResponse:
        if task_mode == "jev_diffusion":
            return await send_one_jev_request(
                session,
                endpoint_url,
                req_obj,
                bearer_token=bearer_token,
                served_model_name=model_name or SERVED_MODEL_NAME,
            )
        return await send_one_chat_request(
            session,
            endpoint_url,
            req_obj,
            model_name=model_name,
            bearer_token=bearer_token,
            for_equivalence=True,
            stream=False,
        )

    if concurrency <= 1:
        preds: list[ParsedResponse] = []
        failures = 0
        for req in requests_seq:
            try:
                preds.append(await _call_one(req))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if not first_err:
                    first_err = str(exc)[:200]
        return preds, failures, first_err

    sem = asyncio.Semaphore(concurrency)
    results: list[ParsedResponse | None] = [None] * len(requests_seq)
    failures = 0

    async def _one(idx: int, req_obj: Mapping[str, Any]) -> None:
        nonlocal failures, first_err
        async with sem:
            try:
                results[idx] = await _call_one(req_obj)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if not first_err:
                    first_err = str(exc)[:200]

    await asyncio.gather(*(asyncio.create_task(_one(i, r)) for i, r in enumerate(requests_seq)))
    return [r for r in results if r is not None], failures, first_err


async def run_equivalence_gate(
    endpoint_url: str,
    equiv_requests: Sequence[Mapping[str, Any]],
    baseline_responses_path: Path,
    record_baseline: bool = False,
    bearer_token: str | None = None,
    eval_concurrency: int = 8,
    secondary_baseline_url: str = "",
    task_mode: str = "auto",
    model_name: str = SERVED_MODEL_NAME,
) -> EquivalenceReport:
    """Execute Layer-2 logprob/KL equivalence gate (`c=8` batched check; `c=16` calibration on iter 0)."""
    resolved_mode = detect_task_mode(equiv_requests, requested_mode=task_mode, model_name=model_name)
    calibration_path = baseline_responses_path.with_suffix(".calibration.json")

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=32)) as session:
        # Warmup with brief DNS/route propagation retry
        for attempt in range(5):
            try:
                await _fetch_equiv_batch(
                    session,
                    endpoint_url,
                    equiv_requests[: min(2, len(equiv_requests))],
                    task_mode=resolved_mode,
                    model_name=model_name,
                    bearer_token=bearer_token,
                    concurrency=1,
                )
                break
            except Exception:  # noqa: BLE001
                if attempt < 4:
                    await asyncio.sleep(2.5)

        if record_baseline:
            preds, failures, first_err = await _fetch_equiv_batch(
                session,
                endpoint_url,
                equiv_requests,
                task_mode=resolved_mode,
                model_name=model_name,
                bearer_token=bearer_token,
                concurrency=1,
            )
            if failures > 0 or len(preds) < len(equiv_requests):
                return EquivalenceReport(
                    passed=False,
                    num_checked=len(preds),
                    num_failed_http=failures,
                    exact_slot_matches=0,
                    total_slots=len(equiv_requests) * (3 if resolved_mode == "jev_diffusion" else 1),
                    exact_slot_match_pct=0.0,
                    max_prob_abs_diff=1.0,
                    reason=f"Baseline recording had {failures} failed requests (first error: {first_err})",
                )

            baseline_responses_path.parent.mkdir(parents=True, exist_ok=True)
            req_lookup = {str(r["request_id"]): r for r in equiv_requests}
            baseline_map_rec: dict[str, dict[str, Any]] = {}
            with baseline_responses_path.open("w", encoding="utf-8") as f:
                for p in preds:
                    src = req_lookup[p.request_id]
                    if isinstance(p, ParsedJevResponse):
                        entry_resp: dict[str, Any] = {
                            "ratings": p.ratings,
                            "slot_probabilities": p.slot_probabilities,
                        }
                    else:
                        assert isinstance(p, ParsedChatResponse)
                        entry_resp = {
                            "content": p.content,
                            "token_positions": p.token_positions,
                        }
                    baseline_map_rec[p.request_id] = entry_resp
                    row = {
                        "request_id": p.request_id,
                        "task_mode": resolved_mode,
                        "timestamp": src.get("timestamp", ""),
                        "baseline_response": entry_resp,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            cal_target_url = secondary_baseline_url or endpoint_url
            burst_preds, burst_fails, _ = await _fetch_equiv_batch(
                session,
                cal_target_url,
                equiv_requests,
                task_mode=resolved_mode,
                model_name=model_name,
                bearer_token=bearer_token,
                concurrency=16,
            )
            if burst_fails == 0 and len(burst_preds) == len(equiv_requests):
                if resolved_mode == "jev_diffusion":
                    jev_burst = [bp for bp in burst_preds if isinstance(bp, ParsedJevResponse)]
                    self_rep = compare_against_baseline_responses(
                        baseline_map=baseline_map_rec,
                        candidate_responses=jev_burst,
                        num_expected=len(equiv_requests),
                        num_failed_http=0,
                    )
                    within_one_cnt = 0
                    cat_cnt = 0
                    tot_s = 0
                    for bp in jev_burst:
                        ref_r = baseline_map_rec[bp.request_id]["ratings"]
                        ref_p = baseline_map_rec[bp.request_id]["slot_probabilities"]
                        for dm in DECISION_DIMENSIONS:
                            tot_s += 1
                            jmp = abs(int(bp.ratings[dm]) - int(ref_r[dm]))
                            if jmp <= 1:
                                within_one_cnt += 1
                            ps = sorted((float(ref_p[dm][ks]) for ks in ("1", "2", "3", "4", "5")), reverse=True)
                            if (ps[0] - ps[1]) >= 0.35 and jmp >= 2:
                                cat_cnt += 1
                    cal_data = {
                        "self_exact_match_pct": self_rep.exact_slot_match_pct,
                        "self_within_one_pct": round(100.0 * within_one_cnt / max(1, tot_s), 2),
                        "self_catastrophic_pct": round(100.0 * cat_cnt / max(1, tot_s), 2),
                        "self_confident_match_pct": self_rep.confident_slot_match_pct,
                        "self_mean_kl": self_rep.mean_kl_divergence,
                        "self_mean_prob_diff": self_rep.mean_prob_abs_diff,
                        "self_max_prob_diff": self_rep.max_prob_abs_diff,
                    }
                else:
                    chat_burst = [bp for bp in burst_preds if isinstance(bp, ParsedChatResponse)]
                    self_rep = compare_chat_responses_kl(
                        baseline_map=baseline_map_rec,
                        candidate_responses=chat_burst,
                        num_expected=len(equiv_requests),
                        num_failed_http=0,
                    )
                    cal_data = {
                        "self_exact_match_pct": self_rep.exact_slot_match_pct,
                        "self_confident_match_pct": self_rep.confident_slot_match_pct,
                        "self_mean_kl": self_rep.mean_kl_divergence,
                        "self_mean_prob_diff": self_rep.mean_prob_abs_diff,
                        "self_max_prob_diff": self_rep.max_prob_abs_diff,
                    }
                calibration_path.write_text(json.dumps(cal_data, indent=2) + "\n", encoding="utf-8")

            total_s = (
                len(preds) * 3
                if resolved_mode == "jev_diffusion"
                else sum(max(1, len(p.token_positions)) for p in preds if isinstance(p, ParsedChatResponse))
            )
            return EquivalenceReport(
                passed=True,
                num_checked=len(preds),
                num_failed_http=0,
                exact_slot_matches=total_s,
                total_slots=total_s,
                exact_slot_match_pct=100.0,
                max_prob_abs_diff=0.0,
                reason="recorded_ground_zero",
                mean_kl_divergence=0.0,
                mean_prob_abs_diff=0.0,
                confident_slot_match_pct=100.0,
                confident_slots=total_s,
            )

        preds, failures, first_err = await _fetch_equiv_batch(
            session,
            endpoint_url,
            equiv_requests,
            task_mode=resolved_mode,
            model_name=model_name,
            bearer_token=bearer_token,
            concurrency=eval_concurrency,
        )

    if not baseline_responses_path.exists():
        raise FileNotFoundError(
            f"Missing {baseline_responses_path}; record Ground-Zero baseline first."
        )
    baseline_map: dict[str, dict[str, Any]] = {}
    for line in baseline_responses_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            baseline_map[str(obj["request_id"])] = obj["baseline_response"]

    cal_dict: dict[str, float] | None = None
    if calibration_path.exists():
        try:
            cal_dict = json.loads(calibration_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cal_dict = None

    if failures > 0 or len(preds) < len(equiv_requests):
        return EquivalenceReport(
            passed=False,
            num_checked=len(preds),
            num_failed_http=failures,
            exact_slot_matches=0,
            total_slots=len(equiv_requests) * (3 if resolved_mode == "jev_diffusion" else 1),
            exact_slot_match_pct=0.0,
            max_prob_abs_diff=1.0,
            reason=f"HTTP/parse failures during equivalence gate ({failures} failed; first: {first_err})",
            mean_kl_divergence=10.0,
            mean_prob_abs_diff=1.0,
            confident_slot_match_pct=0.0,
            confident_slots=0,
        )

    if resolved_mode == "jev_diffusion":
        return compare_against_baseline_responses(
            baseline_map=baseline_map,
            candidate_responses=[p for p in preds if isinstance(p, ParsedJevResponse)],
            num_expected=len(equiv_requests),
            num_failed_http=failures,
            calibration=cal_dict,
        )
    return compare_chat_responses_kl(
        baseline_map=baseline_map,
        candidate_responses=[p for p in preds if isinstance(p, ParsedChatResponse)],
        num_expected=len(equiv_requests),
        num_failed_http=failures,
        calibration=cal_dict,
    )
