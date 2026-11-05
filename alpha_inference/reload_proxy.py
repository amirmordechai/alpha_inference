#!/usr/bin/env python3
"""In-Container / Local vLLM Hot-Reload Supervisor (`alpha_inference.reload_proxy`).

Listens on `0.0.0.0:8080` inside a Google Cloud Vertex AI Dedicated Endpoint container:
  - During initial boot: `/health` returns `503` until Ground-Zero `vLLM` (`127.0.0.1:8001`) finishes
    loading weights and passes health checks.
  - During optimization iterations: `/health` stays `200 OK` while restarting the inner `vLLM`
    process group with new CLI flags, env vars, or custom Triton/CUDA kernels (`sitecustomize.py`),
    cutting reload time from ~35 minutes (full cloud VM redeploy) to ~60-90 seconds.
  - Supports real-time SSE chunk pass-through (`text/event-stream`) so Time-To-First-Token (`ttft`)
    measurements reflect true first-token latency with zero proxy buffering.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

INNER_HOST = "127.0.0.1"
INNER_PORT = 8001
INNER_BASE = f"http://{INNER_HOST}:{INNER_PORT}"
OUTER_PORT = int(os.environ.get("AIP_HTTP_PORT", "8080"))
ADMIN_MODEL_TAG = "__ae_admin_reload__"
STATUS_MODEL_TAG = "__ae_admin_status__"
PROFILE_MODEL_TAG = "__ae_admin_profile__"
KERNEL_DIR = Path("/tmp/ae_candidate_kernel")
PROFILER_DIR = Path("/tmp/ae_vllm_profiler")


class VllmSupervisor:
    """Manages the inner `vllm serve` process group on `127.0.0.1:8001`."""

    def __init__(self, vllm_cmd: list[str] | None = None) -> None:
        self.vllm_cmd = vllm_cmd or ["vllm", "serve"]
        self.proc: subprocess.Popen[str] | None = None
        self.ever_ready: bool = False
        self.inner_ready: bool = False
        self.reloading: bool = False
        self.active_config: dict[str, Any] = {}
        self.log_tail: deque[str] = deque(maxlen=160)
        self._lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None

    @staticmethod
    def _rewrite_args_for_inner_port(vllm_args: list[str]) -> list[str]:
        cleaned: list[str] = []
        skip_next = False
        for arg in vllm_args:
            if skip_next:
                skip_next = False
                continue
            if arg in ("--port", "--host"):
                skip_next = True
                continue
            if arg.startswith("--port=") or arg.startswith("--host="):
                continue
            cleaned.append(arg)
        cleaned.extend([f"--host={INNER_HOST}", f"--port={INNER_PORT}"])
        return cleaned

    async def _stream_logs(self, proc: subprocess.Popen[str]) -> None:
        loop = asyncio.get_running_loop()
        assert proc.stdout is not None
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            stripped = line.rstrip("\n")
            self.log_tail.append(stripped)
            print(f"[vllm] {stripped}", flush=True)

    async def stop_inner(self) -> None:
        self.inner_ready = False
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(25):
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.4)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.sleep(0.5)
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        await asyncio.sleep(1.0)

    async def reload(
        self,
        vllm_args: list[str],
        env_vars: dict[str, str],
        custom_kernel_code: str = "",
        startup_timeout_sec: float = 420.0,
    ) -> dict[str, Any]:
        async with self._lock:
            self.reloading = True
            t0 = time.perf_counter()
            try:
                await self.stop_inner()
                self.log_tail.clear()

                if KERNEL_DIR.exists():
                    shutil.rmtree(KERNEL_DIR)
                PROFILER_DIR.mkdir(parents=True, exist_ok=True)
                child_env = os.environ.copy()
                child_env.pop("AE_ADMIN_TOKEN", None)
                child_env.pop("AE_INITIAL_SPEC_JSON", None)
                child_env["VLLM_TORCH_PROFILER_DIR"] = str(PROFILER_DIR)
                for k, v in env_vars.items():
                    child_env[str(k)] = str(v)

                if custom_kernel_code.strip():
                    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
                    (KERNEL_DIR / "sitecustomize.py").write_text(custom_kernel_code, encoding="utf-8")
                    existing_pp = child_env.get("PYTHONPATH", "")
                    child_env["PYTHONPATH"] = f"{KERNEL_DIR}:{existing_pp}" if existing_pp else str(KERNEL_DIR)

                inner_args = self._rewrite_args_for_inner_port(vllm_args)
                full_cmd = [*self.vllm_cmd, *inner_args]
                print(f"[supervisor] Starting inner process: {full_cmd}", flush=True)

                self.proc = subprocess.Popen(
                    full_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=child_env,
                    start_new_session=True,
                )
                self._reader_task = asyncio.create_task(self._stream_logs(self.proc))

                deadline = time.perf_counter() + startup_timeout_sec
                async with aiohttp.ClientSession() as session:
                    while time.perf_counter() < deadline:
                        if self.proc.poll() is not None:
                            return {
                                "ok": False,
                                "error": f"vLLM exited early with code {self.proc.returncode}",
                                "reload_sec": round(time.perf_counter() - t0, 2),
                                "log_tail": list(self.log_tail)[-45:],
                            }
                        try:
                            async with session.get(
                                f"{INNER_BASE}/health",
                                timeout=aiohttp.ClientTimeout(total=2.0),
                            ) as resp:
                                if resp.status == 200:
                                    self.inner_ready = True
                                    self.ever_ready = True
                                    self.active_config = {
                                        "vllm_args": vllm_args,
                                        "env_vars": {
                                            k: v for k, v in env_vars.items() if "TOKEN" not in k.upper()
                                        },
                                        "has_custom_kernel": bool(custom_kernel_code.strip()),
                                    }
                                    return {
                                        "ok": True,
                                        "reload_sec": round(time.perf_counter() - t0, 2),
                                        "active_config": self.active_config,
                                        "log_tail": list(self.log_tail)[-20:],
                                    }
                        except Exception:  # noqa: BLE001
                            pass
                        await asyncio.sleep(1.0)

                await self.stop_inner()
                return {
                    "ok": False,
                    "error": f"Timed out after {startup_timeout_sec}s waiting for vLLM /health",
                    "reload_sec": round(time.perf_counter() - t0, 2),
                    "log_tail": list(self.log_tail)[-45:],
                }
            finally:
                self.reloading = False

    async def capture_gpu_kernel_profile(self, sample_payload: dict[str, Any], num_burst: int = 8) -> dict[str, Any]:
        """Trigger vLLM `/start_profile` -> burst -> `/stop_profile` and summarize top GPU CUDA kernels."""
        if not self.inner_ready:
            return {"ok": False, "error": "Inner vLLM not ready"}
        for f in PROFILER_DIR.glob("*"):
            f.unlink(missing_ok=True)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60.0)) as sess:
            async with sess.post(f"{INNER_BASE}/start_profile") as r_start:
                if r_start.status >= 400:
                    return {"ok": False, "error": f"/start_profile returned HTTP {r_start.status}"}
            await asyncio.gather(
                *(
                    sess.post(f"{INNER_BASE}/v1/chat/completions", json=sample_payload)
                    for _ in range(max(1, num_burst))
                ),
                return_exceptions=True,
            )
            async with sess.post(f"{INNER_BASE}/stop_profile") as r_stop:
                if r_stop.status >= 400:
                    return {"ok": False, "error": f"/stop_profile returned HTTP {r_stop.status}"}

        await asyncio.sleep(1.5)
        traces = sorted(PROFILER_DIR.rglob("*.json*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not traces:
            return {"ok": False, "error": "No trace file produced in VLLM_TORCH_PROFILER_DIR"}

        trace_file = traces[0]
        raw = gzip.decompress(trace_file.read_bytes()) if trace_file.suffix == ".gz" else trace_file.read_bytes()
        trace_json = json.loads(raw.decode("utf-8", errors="ignore"))
        events = trace_json.get("traceEvents") or []

        dur_us_by_name: Counter[str] = Counter()
        calls_by_name: Counter[str] = Counter()
        for ev in events:
            cat = str(ev.get("cat", "")).lower()
            if "kernel" in cat or "gpu_op" in cat:
                name = str(ev.get("name", "unknown"))[:110]
                dur = float(ev.get("dur", 0.0) or 0.0)
                if dur > 0:
                    dur_us_by_name[name] += dur
                    calls_by_name[name] += 1

        total_us = sum(dur_us_by_name.values()) or 1.0
        top_kernels = [
            {
                "kernel": k_name,
                "total_ms": round(us / 1000.0, 3),
                "pct_gpu_time": round(100.0 * us / total_us, 2),
                "calls": calls_by_name[k_name],
            }
            for k_name, us in dur_us_by_name.most_common(20)
        ]
        for f in PROFILER_DIR.glob("*"):
            f.unlink(missing_ok=True)
        return {"ok": True, "total_gpu_ms": round(total_us / 1000.0, 2), "top_gpu_kernels": top_kernels}


HTTP_CLIENT_KEY = web.AppKey("http_client", aiohttp.ClientSession)


def create_app(supervisor: VllmSupervisor, admin_token: str) -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024)

    async def on_startup(app_inst: web.Application) -> None:
        app_inst[HTTP_CLIENT_KEY] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=512, limit_per_host=0),
            timeout=aiohttp.ClientTimeout(total=180.0),
        )

    async def on_cleanup(app_inst: web.Application) -> None:
        await app_inst[HTTP_CLIENT_KEY].close()
        await supervisor.stop_inner()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def handle_health(_request: web.Request) -> web.Response:
        if supervisor.ever_ready:
            return web.json_response({"status": "ok", "inner_ready": supervisor.inner_ready})
        return web.json_response({"status": "starting"}, status=503)

    async def handle_proxy(request: web.Request) -> web.StreamResponse:
        raw_body = await request.read()
        if request.method == "POST" and raw_body:
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict) and parsed.get("model") in (
                ADMIN_MODEL_TAG,
                STATUS_MODEL_TAG,
                PROFILE_MODEL_TAG,
            ):
                supplied_tok = str(parsed.get("admin_token") or request.headers.get("X-AE-Admin-Token", ""))
                if not admin_token or supplied_tok != admin_token:
                    return web.json_response({"ok": False, "error": "Unauthorized admin token"}, status=403)
                if parsed.get("model") == STATUS_MODEL_TAG:
                    return web.json_response(
                        {
                            "ok": True,
                            "inner_ready": supervisor.inner_ready,
                            "reloading": supervisor.reloading,
                            "active_config": supervisor.active_config,
                            "log_tail": list(supervisor.log_tail)[-25:],
                        }
                    )
                if parsed.get("model") == PROFILE_MODEL_TAG:
                    prof = await supervisor.capture_gpu_kernel_profile(
                        sample_payload=dict(parsed.get("sample_payload") or {}),
                        num_burst=int(parsed.get("num_burst", 8)),
                    )
                    return web.json_response(prof, status=200 if prof.get("ok") else 422)
                res = await supervisor.reload(
                    vllm_args=list(parsed.get("vllm_args") or []),
                    env_vars=dict(parsed.get("env_vars") or {}),
                    custom_kernel_code=str(parsed.get("custom_kernel_code") or ""),
                    startup_timeout_sec=float(parsed.get("startup_timeout_sec", 420.0)),
                )
                return web.json_response(res, status=200 if res.get("ok") else 422)

        if not supervisor.inner_ready:
            return web.json_response(
                {"error": "Inner vLLM engine is reloading or not ready"},
                status=503,
            )

        target_path = request.rel_url.path_qs
        if target_path.endswith(":rawPredict") or target_path == "/":
            target_path = "/v1/chat/completions"
        target_url = f"{INNER_BASE}{target_path}"

        fwd_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "authorization", "x-ae-admin-token")
        }
        client: aiohttp.ClientSession = request.app[HTTP_CLIENT_KEY]
        async with client.request(
            method=request.method,
            url=target_url,
            data=raw_body,
            headers=fwd_headers,
        ) as upstream:
            ctype = upstream.content_type or "application/json"
            if "text/event-stream" in ctype.lower():
                stream_resp = web.StreamResponse(
                    status=upstream.status,
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                )
                await stream_resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if chunk:
                        await stream_resp.write(chunk)
                await stream_resp.write_eof()
                return stream_resp

            resp_bytes = await upstream.read()
            return web.Response(
                body=resp_bytes,
                status=upstream.status,
                content_type=ctype,
            )

    app.router.add_get("/health", handle_health)
    app.router.add_route("*", "/{tail:.*}", handle_proxy)
    return app


async def _main_async(cli_args: argparse.Namespace) -> None:
    admin_token = cli_args.admin_token or os.environ.get("AE_ADMIN_TOKEN", "")
    init_spec_raw = os.environ.get("AE_INITIAL_SPEC_JSON", "{}")
    init_spec = json.loads(init_spec_raw)

    if cli_args.model and not init_spec.get("vllm_args"):
        vllm_args = [
            cli_args.model,
            f"--served-model-name={cli_args.served_model_name or cli_args.model}",
            "--max-logprobs=16",
            *cli_args.extra_vllm_args,
        ]
        init_spec = {"vllm_args": vllm_args, "env_vars": {"PYTHONUNBUFFERED": "1"}}

    vllm_cmd_override = init_spec.get("vllm_cmd")
    supervisor = VllmSupervisor(vllm_cmd=vllm_cmd_override)

    app = create_app(supervisor, admin_token=admin_token)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cli_args.host, cli_args.port, backlog=1024)
    await site.start()
    print(f"[supervisor] Listening on {cli_args.host}:{cli_args.port}", flush=True)

    if init_spec.get("vllm_args"):
        res = await supervisor.reload(
            vllm_args=list(init_spec["vllm_args"]),
            env_vars=dict(init_spec.get("env_vars") or {}),
            custom_kernel_code=str(init_spec.get("custom_kernel_code") or ""),
            startup_timeout_sec=float(init_spec.get("startup_timeout_sec", 900.0)),
        )
        if not res.get("ok"):
            print(f"[supervisor] Initial startup failed: {res}", file=sys.stderr, flush=True)
            sys.exit(1)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    await stop_event.wait()
    await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local or in-container vLLM hot-reload supervisor.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=OUTER_PORT)
    parser.add_argument("--admin-token", default="", help="Secret token required for hot-reload API calls")
    parser.add_argument("--model", default="", help="Optional HuggingFace model ID to boot initially")
    parser.add_argument("--served-model-name", default="", help="Optional --served-model-name override")
    parser.add_argument("extra_vllm_args", nargs="*", help="Additional initial vLLM CLI flags")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
