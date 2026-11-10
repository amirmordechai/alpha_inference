"""Offline unit & integration test suite for `alpha_inference` ($0 cloud spend)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
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
    compose_sitecustomize_source,
    parse_candidate_code,
    summarize_engine_metrics_delta,
    upgrade_candidate_source_for_evolution,
    validate_candidate_spec,
)
from alpha_inference.vertex import build_gcloud_rollout_command, teardown_endpoint


def test_seed_candidate_has_official_evolve_block() -> None:
    seed_src = (REPO_ROOT / "candidate.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in seed_src
    assert "# EVOLVE-BLOCK-END" in seed_src
    cand = parse_candidate_code(seed_src, name="seed")
    ok, msg = validate_candidate_spec(cand, cand)
    assert ok, msg
    assert cand.vllm_args[0] == "google/gemma-4-26B-A4B-it"


def test_candidate_validation_rejects_cheating_and_unsafe_kernels() -> None:
    base = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="base")

    # 1. Reject model checkpoint swap
    swapped = parse_candidate_code(build_seed_program_source("Qwen/Qwen2.5-7B-Instruct"), name="swap")
    ok, reason = validate_candidate_spec(swapped, base)
    assert not ok and "Model checkpoint" in reason

    # 2. Reject dummy weights & non-FP8 lossy quantizations
    bad_dummy = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="dummy")
    bad_dummy.vllm_args.append("--load-format=dummy")
    ok, reason = validate_candidate_spec(bad_dummy, base)
    assert not ok and "Forbidden" in reason

    bad_quant = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="bnb")
    bad_quant.vllm_args.append("--quantization=bitsandbytes")
    ok, reason = validate_candidate_spec(bad_quant, base)
    assert not ok and "--quantization" in reason

    # 3. Reject external draft model checkpoints in --speculative-config
    bad_spec = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="bad_spec")
    bad_spec.vllm_args.append('--speculative-config={"method":"eagle","model":"other/draft-1b","num_speculative_tokens":3}')
    ok, reason = validate_candidate_spec(bad_spec, base)
    assert not ok and "speculative" in reason.lower()

    # 4. Allow KL-gated hardware FP8, self-contained GPU n-gram speculation, kernel-config, and MoE Triton tile schedules
    good_hw = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="good_hw")
    good_hw.vllm_args.extend(
        [
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}',
        ]
    )
    good_hw.moe_tuned_config = {
        16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}
    }
    ok, reason = validate_candidate_spec(good_hw, base)
    assert ok, reason

    # 5. Reject network imports inside custom Triton kernel code
    bad_kern = parse_candidate_code((REPO_ROOT / "candidate.py").read_text(encoding="utf-8"), name="net")
    bad_kern.custom_kernel_code = "import socket\nx = 1\n"
    ok, reason = validate_candidate_spec(bad_kern, base)
    assert not ok and "socket" in reason


def test_sitecustomize_hooks_and_warm_start_continue_from() -> None:
    site_src = compose_sitecustomize_source(
        custom_kernel_code="register_vllm_patch('math', lambda m: setattr(m, '_ae_patched', 42))",
        moe_tuned_config={8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}},
    )
    assert "_AE_MOE_TUNED_CONFIG" in site_src
    assert "register_vllm_patch" in site_src

    # Verify reload_proxy.py runs cleanly in an isolated container-like env with alpha_inference blocked
    iso_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.modules['alpha_inference'] = None; "
            f"ns = runpy.run_path({str(REPO_ROOT / 'alpha_inference' / 'reload_proxy.py')!r}); "
            "sc = ns['_compose_sitecustomize']('x = 1', {16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}}); "
            "compile(sc, 'sitecustomize.py', 'exec')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert iso_proc.returncode == 0

    # Verify post-import hook executes cleanly in an isolated subprocess
    hook_proc = subprocess.run(
        [sys.executable, "-c", site_src + "\nimport importlib, sys; sys.modules.pop('decimal', None)\nregister_vllm_patch('decimal', lambda m: setattr(m, '_ae_tag', 'ok'))\nimport decimal\nassert decimal._ae_tag == 'ok'\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert hook_proc.returncode == 0

    delta = summarize_engine_metrics_delta(
        {"itl_sum": 1.0, "itl_count": 100.0, "ttft_sum": 2.0, "ttft_count": 10.0, "spec_draft_tokens": 0.0, "spec_accepted_tokens": 0.0},
        {"itl_sum": 2.2, "itl_count": 300.0, "ttft_sum": 5.0, "ttft_count": 30.0, "spec_draft_tokens": 100.0, "spec_accepted_tokens": 55.0},
    )
    assert delta["mean_itl_ms"] == 6.0
    assert delta["mean_ttft_ms"] == 150.0
    assert delta["spec_accept_pct"] == 55.0

    stage1_dir = REPO_ROOT / "examples" / "email_triage" / "results"
    if (stage1_dir / "best_candidate.py").exists() and (stage1_dir / "optimization_summary.json").exists():
        upgraded = upgrade_candidate_source_for_evolution((stage1_dir / "best_candidate.py").read_text(encoding="utf-8"))
        assert '"moe_tuned_config"' in upgraded
        proc = subprocess.run(
            [
                sys.executable,
                "optimize.py",
                "--continue-from",
                str(stage1_dir),
                "--trace",
                "examples/email_triage/monday_peak_3m.jsonl",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        assert payload["prior_programs_evaluated"] >= 300
        assert payload["seed_candidate"]["plain_title"]


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
    assert len(mon_ids & eq_ids) == 0
    assert detect_task_mode(mon) == "chat"

    # Anti-cache-bias guarantee: 0% recycled user emails within or across Monday, Tuesday, and Golden
    all_user_payloads = [r["messages"][1]["content"] for r in (*mon, *tue, *eq)]
    assert len(all_user_payloads) == len(set(all_user_payloads)), "Duplicate user payload found across traces!"

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


def test_cli_dry_runs_and_hardware_agnostic_gcloud_rollout() -> None:
    for cmd in (
        [
            sys.executable,
            "optimize.py",
            "--vertex-endpoint-id",
            "1234567890",
            "--model",
            "google/gemma-4-26B-A4B-it",
            "--machine-type",
            "g2-standard-8",
            "--accelerator-type",
            "NVIDIA_L4",
            "--trace",
            "examples/custom_openai_chat/sample_trace.jsonl",
            "--dry-run",
        ],
        [
            sys.executable,
            "validate.py",
            "--baseline-endpoint-id",
            "111",
            "--best-endpoint-id",
            "222",
            "--trace",
            "examples/custom_openai_chat/sample_trace.jsonl",
            "--dry-run",
        ],
    ):
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        assert isinstance(payload, dict)

    diff_proc = subprocess.run([sys.executable, "optimize.py", "--diff"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert "--- candidate.py" in diff_proc.stdout

    seed_g4 = parse_candidate_code(build_seed_program_source("google/gemma-4-26B-A4B-it"), name="g4")
    gcloud_cmd = build_gcloud_rollout_command(
        seed_g4,
        project_id="demo-proj",
        region="us-central1",
        endpoint_id="999888777",
        image_uri="us-central1-docker.pkg.dev/demo-proj/repo/vllm-openai:v0.31.0",
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
    )
    assert "gcloud ai models upload" in gcloud_cmd
    assert "gcloud ai endpoints deploy-model 999888777" in gcloud_cmd
    assert "type=nvidia-tesla-a100,count=1" in gcloud_cmd


def test_coding_agent_onboarding_skill_present() -> None:
    skill_md = REPO_ROOT / ".agents" / "skills" / "alpha-inference" / "SKILL.md"
    agents_md = REPO_ROOT / "AGENTS.md"
    readme_md = REPO_ROOT / "README.md"
    assert skill_md.exists(), "Missing .agents/skills/alpha-inference/SKILL.md"
    assert agents_md.exists(), "Missing AGENTS.md"
    skill_txt = skill_md.read_text(encoding="utf-8")
    readme_txt = readme_md.read_text(encoding="utf-8")
    assert skill_txt.startswith("---\nname: alpha-inference\n")
    assert ".agents/skills/alpha-inference/SKILL.md" in readme_txt
    assert "roles/aiplatform.user" in readme_txt
    assert "roles/discoveryengine.admin" in readme_txt


@pytest.mark.live_e2e
def test_live_e2e_skill_workflow_with_native_gcloud_endpoint() -> None:
    """Opt-in live cloud E2E test reproducing the `.agents/skills/alpha-inference/SKILL.md` workflow:
    1. Verify GCP env & skill prerequisites (`GCP_PROJECT_ID`, `GE_APP_ID`).
    2. Create a live Vertex AI GPU Endpoint using native `gcloud ai` commands.
    3. Run `optimize.py --vertex-endpoint-id <ID> --max-programs 2` and verify `summary.md` & `gcloud_rollout.sh`.
    4. Unconditionally tear down the endpoint & model so `$0/hr` active compute remains.
    """
    if os.environ.get("RUN_LIVE_VERTEX_E2E") != "1":
        pytest.skip("Set RUN_LIVE_VERTEX_E2E=1 to run the live Vertex AI + AlphaEvolve E2E cycle (incurs GPU spend).")

    project_id = os.environ.get("GCP_PROJECT_ID", "").strip()
    ge_app_id = os.environ.get("GE_APP_ID", "").strip()
    region = os.environ.get("GCP_REGION", "us-central1").strip()
    ar_repo = os.environ.get("GCP_AR_REPO", "alpha-inference-images").strip()
    model_id_hf = os.environ.get("E2E_MODEL_ID", "google/gemma-4-26B-A4B-it").strip()
    machine_type = os.environ.get("E2E_MACHINE_TYPE", "g4-standard-48").strip()
    accel_type = os.environ.get("E2E_ACCELERATOR_TYPE", "nvidia-rtx-pro-6000").strip()
    assert project_id and ge_app_id, "GCP_PROJECT_ID and GE_APP_ID must be set for RUN_LIVE_VERTEX_E2E=1"

    gcloud_env = os.environ.copy()
    if gcloud_env.get("GOOGLE_APPLICATION_CREDENTIALS"):
        gcloud_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = gcloud_env["GOOGLE_APPLICATION_CREDENTIALS"]
    if gcloud_env.get("GOOGLE_API_USE_CLIENT_CERTIFICATE", "").lower() == "false":
        gcloud_env["CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE"] = "false"

    image_uri = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/vllm-openai:v0.31.0"
    created_endpoint_id = ""
    uploaded_model_id = ""

    with tempfile.TemporaryDirectory(prefix="ae_live_e2e_") as tmp_dir:
        out_dir = Path(tmp_dir) / "results"
        try:
            # Step 1 (from SKILL.md): Zero-cost dry-run verification
            subprocess.run(
                [
                    sys.executable,
                    "optimize.py",
                    "--trace",
                    "examples/custom_openai_chat/sample_trace.jsonl",
                    "--model",
                    model_id_hf,
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            # Step 2 (from SKILL.md): Create a live Vertex AI Endpoint via native `gcloud ai`
            ep_out = subprocess.run(
                [
                    "gcloud",
                    "ai",
                    "endpoints",
                    "create",
                    f"--project={project_id}",
                    f"--region={region}",
                    "--display-name=ae-live-e2e-test",
                    "--format=value(name)",
                    "--quiet",
                ],
                env=gcloud_env,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            created_endpoint_id = ep_out.rsplit("/", 1)[-1]
            assert created_endpoint_id

            mod_out = subprocess.run(
                [
                    "gcloud",
                    "ai",
                    "models",
                    "upload",
                    f"--project={project_id}",
                    f"--region={region}",
                    "--display-name=ae-live-e2e-model",
                    f"--container-image-uri={image_uri}",
                    "--container-command=vllm,serve",
                    f"--container-args={model_id_hf},--served-model-name={model_id_hf},--host=0.0.0.0,--port=8080,--max-model-len=8192,--max-logprobs=16,--gpu-memory-utilization=0.90",
                    "--container-env-vars=HF_HOME=/dev/shm/hf_cache",
                    "--container-shared-memory-size-mb=81920",
                    "--container-ports=8080",
                    "--container-predict-route=/v1/chat/completions",
                    "--container-health-route=/health",
                    "--container-deployment-timeout-seconds=3600",
                    "--format=value(model)",
                    "--quiet",
                ],
                env=gcloud_env,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            uploaded_model_id = mod_out.rsplit("/", 1)[-1]

            subprocess.run(
                [
                    "gcloud",
                    "ai",
                    "endpoints",
                    "deploy-model",
                    created_endpoint_id,
                    f"--project={project_id}",
                    f"--region={region}",
                    f"--model={uploaded_model_id}",
                    "--display-name=ae-live-e2e-model",
                    f"--machine-type={machine_type}",
                    f"--accelerator=type={accel_type},count=1",
                    "--min-replica-count=1",
                    "--max-replica-count=1",
                    "--quiet",
                ],
                env=gcloud_env,
                check=True,
            )

            # Step 3 (from SKILL.md): Run short AlphaEvolve E2E cycle against `--vertex-endpoint-id`
            subprocess.run(
                [
                    sys.executable,
                    "optimize.py",
                    "--vertex-endpoint-id",
                    created_endpoint_id,
                    "--trace",
                    "examples/custom_openai_chat/sample_trace.jsonl",
                    "--workers",
                    "1",
                    "--max-programs",
                    "2",
                    "--budget-usd",
                    "15",
                    "--skip-mirror",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                check=True,
            )
            assert (out_dir / "summary.md").exists()
            assert (out_dir / "results.tsv").exists()
            assert (out_dir / "best_candidate.py").exists()
            assert (out_dir / "gcloud_rollout.sh").exists()
        finally:
            if created_endpoint_id:
                teardown_endpoint(project_id, region, created_endpoint_id)
            if uploaded_model_id:
                subprocess.run(
                    ["gcloud", "ai", "models", "delete", uploaded_model_id, f"--project={project_id}", f"--region={region}", "--quiet"],
                    env=gcloud_env,
                    capture_output=True,
                )


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
