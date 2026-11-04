"""Offline unit & integration test suite for `alpha_inference` ($0 cloud spend)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from aiohttp import web

from alpha_inference.adapters import (
    ParsedChatResponse,
    detect_task_mode,
    load_jsonl_requests,
)
from alpha_inference.equivalence import compare_chat_responses_kl
from alpha_inference.evaluator import (
    is_better,
    normalize_metric,
    parse_timestamp_sec,
    replay_trace_realtime,
    to_alpha_evolve_score,
)
from alpha_inference.reporting import (
    REPO_ROOT,
    build_seed_program_source,
    parse_candidate_code,
    validate_candidate_spec,
)


def test_seed_candidate_has_official_evolve_block() -> None:
    seed_src = (REPO_ROOT / "candidate.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in seed_src
    assert "# EVOLVE-BLOCK-END" in seed_src
    cand = parse_candidate_code(seed_src, name="seed")
    ok, msg = validate_candidate_spec(cand, cand)
    assert ok, msg
    assert cand.vllm_args[0] == "nvidia/diffusiongemma-26B-A4B-it-NVFP4"


def test_candidate_validation_rejects_cheating_and_unsafe_kernels() -> None:
    base = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="base")

    # 1. Reject model checkpoint swap
    swapped = parse_candidate_code(build_seed_program_source("Qwen/Qwen2.5-7B-Instruct"), name="swap")
    ok, reason = validate_candidate_spec(swapped, base)
    assert not ok and "Model checkpoint" in reason

    # 2. Reject lossy FP8 KV cache flag
    bad_kv = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="fp8")
    bad_kv.vllm_args.append("--kv-cache-dtype=fp8")
    ok, reason = validate_candidate_spec(bad_kv, base)
    assert not ok and "Forbidden" in reason

    # 3. Reject network imports inside custom Triton kernel code
    bad_kern = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="net")
    bad_kern.custom_kernel_code = "import socket\nx = 1\n"
    ok, reason = validate_candidate_spec(bad_kern, base)
    assert not ok and "socket" in reason


def test_metrics_and_32bit_alpha_evolve_scores() -> None:
    assert normalize_metric("p99") == "p99_latency"
    assert normalize_metric("throughput") == "throughput"
    assert normalize_metric("ttft") == "ttft"

    assert is_better(420.0, 500.0, "p99_latency")
    assert not is_better(520.0, 500.0, "p99_latency")
    assert is_better(45.0, 30.0, "throughput")

    # Discovery Engine protobuf uses 32-bit TYPE_FLOAT; ensure clamping works
    assert to_alpha_evolve_score(450.0, "p99_latency") == -450.0
    assert to_alpha_evolve_score(42.5, "throughput") == 42.5
    assert to_alpha_evolve_score(float("inf"), "p99_latency") == -1_000_000.0


def test_monday_and_tuesday_traces_are_disjoint_and_realtime_timed() -> None:
    mon = load_jsonl_requests(REPO_ROOT / "examples" / "email_triage" / "monday_peak_3m.jsonl")
    tue = load_jsonl_requests(REPO_ROOT / "examples" / "email_triage" / "tuesday_peak_30m.jsonl")
    eq = load_jsonl_requests(REPO_ROOT / "examples" / "email_triage" / "equivalence_golden.jsonl")

    mon_ids = {r["request_id"] for r in mon}
    tue_ids = {r["request_id"] for r in tue}
    eq_ids = {r["request_id"] for r in eq}
    assert len(mon_ids & tue_ids) == 0
    assert len(eq_ids & tue_ids) == 0
    assert detect_task_mode(mon) == "jev_diffusion"

    mon_ts = [parse_timestamp_sec(r["timestamp"]) for r in mon]
    tue_ts = [parse_timestamp_sec(r["timestamp"]) for r in tue]
    assert all(t is not None for t in mon_ts)
    assert all(t is not None for t in tue_ts)
    mon_span = max(t for t in mon_ts if t) - min(t for t in mon_ts if t)
    tue_span = max(t for t in tue_ts if t) - min(t for t in tue_ts if t)
    assert 150.0 <= mon_span <= 190.0
    assert 1700.0 <= tue_span <= 1820.0


def test_realtime_replay_and_kl_equivalence_against_mock_server() -> None:
    async def _run() -> None:
        app = web.Application()

        async def handle_chat(req: web.Request) -> web.StreamResponse:
            body = await req.json()
            if body.get("stream"):
                resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
                await resp.prepare(req)
                chunk = {"choices": [{"delta": {"content": "ok"}}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                await resp.write_eof()
                return resp
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "ok",
                                        "logprob": -0.05,
                                        "top_logprobs": [{"token": "ok", "logprob": -0.05}, {"token": "yes", "logprob": -3.1}],
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 1},
                }
            )

        app.router.add_post("/v1/chat/completions", handle_chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        url = f"http://127.0.0.1:{port}"
        try:
            rows = [
                {"request_id": "r1", "timestamp": 1000.00, "messages": [{"role": "user", "content": "hi"}]},
                {"request_id": "r2", "timestamp": 1000.08, "messages": [{"role": "user", "content": "hello"}]},
                {"request_id": "r3", "timestamp": 1000.16, "messages": [{"role": "user", "content": "hey"}]},
            ]
            preds, fails, wall, buckets = await replay_trace_realtime(
                endpoint_url=url,
                rows=rows,
                task_mode="chat",
                model_name="mock-model",
                use_gcp_auth=False,
                stream_ttft=False,
                for_equivalence=True,
                warmup_count=1,
                bucket_sec=1.0,
            )
            assert fails == 0
            assert len(preds) == 3
            assert wall >= 0.14
            assert len(buckets) == 1

            base_map = {p.request_id: p for _, p in preds if isinstance(p, ParsedChatResponse)}
            cand_ok = [p for _, p in preds if isinstance(p, ParsedChatResponse)]
            rep_ok = compare_chat_responses_kl(base_map, cand_ok)
            assert rep_ok.passed and rep_ok.mean_kl_divergence == 0.0

            # Corrupt logprobs -> KL gate must reject
            cand_bad = [
                ParsedChatResponse(
                    request_id=p.request_id,
                    content="wrong",
                    token_positions=[{"token": "wrong", "top_logprobs": {"wrong": -0.01, "ok": -9.5}}],
                    e2e_latency_ms=10.0,
                )
                for p in cand_ok
            ]
            rep_bad = compare_chat_responses_kl(base_map, cand_bad)
            assert not rep_bad.passed
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_cli_dry_runs_succeed() -> None:
    for cmd in (
        [sys.executable, "optimize.py", "--vertex-endpoint-id", "1234567890", "--dry-run"],
        [sys.executable, "validate.py", "--baseline-endpoint-id", "111", "--best-endpoint-id", "222", "--dry-run"],
        [sys.executable, "deploy.py", "--config", "candidate.py", "--dry-run"],
        [sys.executable, "deploy.py", "--model", "google/gemma-4-26B-A4B-it", "--hot-reload", "--dry-run"],
    ):
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        assert isinstance(payload, dict)


def test_coding_agent_onboarding_skill_present() -> None:
    skill_md = REPO_ROOT / ".agents" / "skills" / "alpha-inference" / "SKILL.md"
    agents_md = REPO_ROOT / "AGENTS.md"
    readme_md = REPO_ROOT / "README.md"
    assert skill_md.exists(), "Missing .agents/skills/alpha-inference/SKILL.md"
    assert agents_md.exists(), "Missing AGENTS.md"
    skill_txt = skill_md.read_text(encoding="utf-8")
    assert skill_txt.startswith("---\nname: alpha-inference\n")
    assert ".agents/skills/alpha-inference/SKILL.md" in readme_md.read_text(encoding="utf-8")


def test_repo_has_zero_leaked_project_ids_or_blog_clutter() -> None:
    forbidden_tokens = ("amirdemoenv", "37043574641", "adavidoff", "agentspace-1759227377962")
    assert not (REPO_ROOT / "blog").exists(), "Blog folder must stay outside the code repository"

    for path in REPO_ROOT.rglob("*"):
        rel = path.relative_to(REPO_ROOT).parts
        if any(p in (".git", ".venv", "__pycache__", ".pytest_cache", "runs", "alpha_inference.egg-info") for p in rel):
            continue
        if path.name == ".env" or not path.is_file():
            continue
        if path.suffix in (".png", ".gif", ".jpg", ".pyc"):
            continue
        # Skip test file's own forbidden_tokens tuple literal
        if path.name == "test_alpha_inference.py":
            continue
        txt = path.read_text(encoding="utf-8", errors="ignore")
        for bad in forbidden_tokens:
            assert bad not in txt, f"Found non-portable token {bad!r} inside {path.relative_to(REPO_ROOT)}"
