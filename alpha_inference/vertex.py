"""Hardware- and model-agnostic Vertex AI Endpoint inspection, search-worker cloning, and `gcloud ai` helpers."""

from __future__ import annotations

import base64
import difflib
import json
import os
import re
import secrets
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_inference.reporting import (
    ACCELERATOR_COUNT,
    ACCELERATOR_TYPE,
    MACHINE_TYPE,
    RELOAD_PROXY_FILE,
    CandidateSpec,
    parse_candidate_code,
)

DEFAULT_UPSTREAM_VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "docker.io/vllm/vllm-openai:v0.31.0")


def load_candidate_file(config_path: str | Path) -> CandidateSpec:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate configuration file not found: {p}")
    return parse_candidate_code(p.read_text(encoding="utf-8"), name=p.stem)


def format_candidate_diff(baseline_path: str | Path, candidate_path: str | Path) -> str:
    b_lines = Path(baseline_path).read_text(encoding="utf-8").splitlines(keepends=True)
    e_lines = Path(candidate_path).read_text(encoding="utf-8").splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(b_lines, e_lines, fromfile=str(baseline_path), tofile=str(candidate_path))
    )
    return diff or f"--- {baseline_path}\n+++ {candidate_path}\n# (Identical to baseline)\n"


def ensure_ar_image_mirrored(
    project_id: str,
    region: str,
    ar_repo: str,
    upstream_image: str = DEFAULT_UPSTREAM_VLLM_IMAGE,
) -> str:
    """Ensure `upstream_image` is available in regional Artifact Registry (pass through if already pkg.dev/gcr.io)."""
    if ".pkg.dev/" in upstream_image or "gcr.io/" in upstream_image:
        return upstream_image

    image_leaf = upstream_image.rsplit("/", 1)[-1]
    ar_image_uri = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/{image_leaf}"
    gcloud_env = os.environ.copy()
    cred_path = gcloud_env.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path:
        gcloud_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = cred_path
    if gcloud_env.get("GOOGLE_API_USE_CLIENT_CERTIFICATE", "").lower() == "false":
        gcloud_env["CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE"] = "false"

    img_check = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "describe",
            ar_image_uri,
            f"--project={project_id}",
            "--quiet",
        ],
        env=gcloud_env,
        capture_output=True,
        text=True,
    )
    if img_check.returncode == 0 or "PERMISSION_DENIED" in (img_check.stderr or ""):
        return ar_image_uri

    subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "create",
            ar_repo,
            "--repository-format=docker",
            f"--location={region}",
            f"--project={project_id}",
            "--quiet",
        ],
        env=gcloud_env,
        capture_output=True,
        text=True,
    )

    cb_cfg = Path(f"/tmp/ae_mirror_{os.getpid()}.json")
    cb_cfg.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "gcr.io/go-containerregistry/crane:latest",
                        "args": ["copy", "--platform=linux/amd64", upstream_image, ar_image_uri],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            ["gcloud", "builds", "submit", "--no-source", f"--config={cb_cfg}", f"--project={project_id}", "--quiet"],
            env=gcloud_env,
            check=True,
        )
    finally:
        cb_cfg.unlink(missing_ok=True)
    return ar_image_uri


def inspect_vertex_endpoint(project_id: str, region: str, endpoint_id: str) -> dict[str, Any]:
    """Inspect any live Vertex AI Endpoint (created via `gcloud ai`, Model Garden, or SDK).

    Extracts its `:rawPredict` URL, machine/GPU spec, serving image, `vLLM` args, and hot-reload token (if any).
    """
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
    image_uri = ""
    machine_type = ""
    accelerator_type = ""
    accelerator_count = 0

    for dm in ep.gca_resource.deployed_models:
        try:
            dr = getattr(dm, "dedicated_resources", None)
            mspec = getattr(dr, "machine_spec", None) if dr else None
            if mspec:
                machine_type = str(getattr(mspec, "machine_type", "") or "")
                accel_raw = getattr(mspec, "accelerator_type", None)
                if accel_raw is not None:
                    accel_name = getattr(accel_raw, "name", str(accel_raw))
                    if accel_name and accel_name != "ACCELERATOR_TYPE_UNSPECIFIED":
                        accelerator_type = accel_name
                accelerator_count = int(getattr(mspec, "accelerator_count", 0) or 0)

            m = aiplatform.Model(model_name=dm.model)
            cspec = m.gca_resource.container_spec
            image_uri = str(getattr(cspec, "image_uri", "") or "")
            env_map = {e.name: e.value for e in (cspec.env or [])}
            if env_map.get("AE_ADMIN_TOKEN"):
                admin_token = env_map["AE_ADMIN_TOKEN"]
            if env_map.get("AE_INITIAL_SPEC_JSON"):
                init_obj = json.loads(env_map["AE_INITIAL_SPEC_JSON"])
                vllm_args = [str(x) for x in (init_obj.get("vllm_args") or [])]
            else:
                # Standard `gcloud ai` / Model Garden vLLM containers pass args in `args` or after `vllm serve` in `command`
                raw_cmd = [str(x) for x in (cspec.command or [])]
                raw_args = [str(x) for x in (cspec.args or [])]
                combined = [*raw_cmd, *raw_args]
                if "serve" in combined:
                    idx = combined.index("serve")
                    vllm_args = combined[idx + 1 :]
                elif raw_args:
                    vllm_args = [a for a in raw_args if a not in ("vllm", "serve", "-m", "vllm.entrypoints.openai.api_server")]
            break
        except Exception:  # noqa: BLE001
            continue

    # Normalize `--model <id>` to positional `vllm_args[0]` if the user's container used `--model`
    if vllm_args and vllm_args[0].startswith("-"):
        norm_args: list[str] = []
        model_val = ""
        skip_next = False
        for i, tok in enumerate(vllm_args):
            if skip_next:
                skip_next = False
                continue
            if tok == "--model" and i + 1 < len(vllm_args):
                model_val = vllm_args[i + 1]
                skip_next = True
            elif tok.startswith("--model="):
                model_val = tok.split("=", 1)[1]
            else:
                norm_args.append(tok)
        if model_val:
            vllm_args = [model_val, *norm_args]

    return {
        "endpoint_id": ep.name,
        "endpoint_resource_name": ep.resource_name,
        "dedicated_endpoint_dns": dns,
        "chat_completions_url": url,
        "admin_token": admin_token,
        "has_hot_reload": bool(admin_token),
        "vllm_args": vllm_args,
        "image_uri": image_uri,
        "machine_type": machine_type,
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
    }


def recommend_shared_memory_mb(machine_type: str) -> int:
    """Estimate safe `/dev/shm` reservation (~43% of host RAM) so 25B-70B+ checkpoints cache in RAM without overflowing Vertex's ~50 GiB container overlay."""
    mt = (machine_type or "").strip().lower()
    host_ram_gb = 64
    m_std = re.search(r"-standard-(\d+)$", mt)
    m_high = re.search(r"^(a2|a3)-(highgpu|ultragpu|megagpu)-(\d+)g$", mt)
    if m_std:
        host_ram_gb = int(m_std.group(1)) * 4
    elif m_high:
        fam, tier, gpus = m_high.group(1), m_high.group(2), int(m_high.group(3))
        per_gpu_gb = 234 if fam == "a3" else (170 if tier == "ultragpu" else 85)
        host_ram_gb = gpus * per_gpu_gb
    shm_mb = int((host_ram_gb * 1024 * 0.43) // 1024) * 1024
    return max(8192, min(shm_mb, 524288))


def build_container_launch_spec(
    cand: CandidateSpec,
    *,
    hot_reload: bool = False,
    admin_token: str = "",
    shared_memory_mb: int = 0,
) -> tuple[list[str], list[str], dict[str, str], str]:
    """Build Vertex container `command`, `args`, `env`, and optional hot-reload admin token."""
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hot_reload:
        tok = admin_token or secrets.token_urlsafe(24)
        proxy_b64 = base64.b64encode(RELOAD_PROXY_FILE.read_bytes()).decode("ascii")
        init_spec = json.dumps(
            {
                "vllm_args": cand.vllm_args,
                "env_vars": cand.env_vars,
                "custom_kernel_code": cand.custom_kernel_code,
                "startup_timeout_sec": 1500.0,
            }
        )
        env = {
            "AE_ADMIN_TOKEN": tok,
            "AE_PROXY_B64": proxy_b64,
            "AE_INITIAL_SPEC_JSON": init_spec,
            "PYTHONUNBUFFERED": "1",
        }
        if shared_memory_mb >= 65536:
            env["HF_HOME"] = "/dev/shm/hf_cache"
        if hf_token:
            env["HF_TOKEN"] = hf_token
        bootstrap = (
            'python3 -c "import base64, os, pathlib; '
            "pathlib.Path('/tmp/reload_proxy.py').write_bytes(base64.b64decode(os.environ['AE_PROXY_B64']))\" && "
            "exec python3 /tmp/reload_proxy.py"
        )
        return ["bash", "-lc", bootstrap], [], env, tok

    env = dict(cand.env_vars)
    if shared_memory_mb >= 65536:
        env.setdefault("HF_HOME", "/dev/shm/hf_cache")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    if not cand.custom_kernel_code.strip():
        return ["vllm", "serve"], list(cand.vllm_args), env, ""

    env["AE_KERNEL_B64"] = base64.b64encode(cand.custom_kernel_code.encode("utf-8")).decode("ascii")
    bootstrap = (
        "mkdir -p /tmp/ae_kernel && "
        'python3 -c "import base64, os, pathlib; '
        "pathlib.Path('/tmp/ae_kernel/sitecustomize.py').write_bytes(base64.b64decode(os.environ['AE_KERNEL_B64']))\" && "
        'export PYTHONPATH="/tmp/ae_kernel:${PYTHONPATH:-}" && '
        'exec vllm serve "$@"'
    )
    return ["bash", "-lc", bootstrap, "--"], list(cand.vllm_args), env, ""


def deploy_vertex_worker_or_endpoint(
    cand: CandidateSpec,
    project_id: str,
    region: str,
    ar_repo: str,
    display_name: str,
    *,
    source_path: str = "candidate.py",
    existing_endpoint_id: str = "",
    traffic_percentage: int = 100,
    skip_mirror: bool = False,
    upstream_image: str = DEFAULT_UPSTREAM_VLLM_IMAGE,
    machine_type: str = MACHINE_TYPE,
    accelerator_type: str = ACCELERATOR_TYPE,
    accelerator_count: int = ACCELERATOR_COUNT,
    hot_reload: bool = False,
) -> dict[str, Any]:
    """Provision a standalone or hot-reloadable Vertex AI Dedicated GPU Endpoint."""
    from google.cloud import aiplatform

    t0 = time.perf_counter()
    aiplatform.init(project=project_id, location=region)
    if skip_mirror and ".pkg.dev/" not in upstream_image and "gcr.io/" not in upstream_image:
        image_leaf = upstream_image.rsplit("/", 1)[-1]
        ar_image_uri = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/{image_leaf}"
    else:
        ar_image_uri = ensure_ar_image_mirrored(project_id, region, ar_repo, upstream_image=upstream_image)

    shm_mb = recommend_shared_memory_mb(machine_type)
    cmd, args, env, admin_token = build_container_launch_spec(cand, hot_reload=hot_reload, shared_memory_mb=shm_mb)
    model = aiplatform.Model.upload(
        display_name=f"{display_name}-model",
        serving_container_image_uri=ar_image_uri,
        serving_container_command=cmd,
        serving_container_args=args,
        serving_container_environment_variables=env,
        serving_container_ports=[8080],
        serving_container_predict_route="/v1/chat/completions",
        serving_container_health_route="/health",
        serving_container_deployment_timeout=3600,
        serving_container_shared_memory_size_mb=shm_mb,
    )

    last_exc: Exception | None = None
    endpoint = None
    for attempt in range(3):
        created_new = not bool(existing_endpoint_id)
        if existing_endpoint_id:
            endpoint = aiplatform.Endpoint(endpoint_name=existing_endpoint_id)
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=display_name,
                dedicated_endpoint_enabled=True,
            )

        try:
            model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=display_name,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                min_replica_count=1,
                max_replica_count=1,
                traffic_percentage=traffic_percentage,
                deploy_request_timeout=3600,
                sync=True,
            )
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if created_new and endpoint is not None:
                try:
                    endpoint.delete(sync=True)
                except Exception:  # noqa: BLE001
                    pass
            if attempt < 2:
                time.sleep(20.0 * (attempt + 1))
    if last_exc is not None or endpoint is None:
        raise last_exc or RuntimeError("Endpoint deployment failed")

    endpoint = aiplatform.Endpoint(endpoint.resource_name)
    dns = getattr(endpoint, "dedicated_endpoint_dns", "") or ""
    res: dict[str, Any] = {
        "deployed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate": str(source_path),
        "plain_title": cand.plain_title,
        "endpoint_id": endpoint.name,
        "endpoint_resource_name": endpoint.resource_name,
        "model_resource_name": model.resource_name,
        "dedicated_endpoint_dns": dns,
        "chat_completions_url": (
            f"https://{dns}/v1/projects/{project_id}/locations/{region}/endpoints/{endpoint.name}:rawPredict"
            if dns
            else f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{endpoint.name}:rawPredict"
        ),
        "machine_type": machine_type,
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
        "image_uri": ar_image_uri,
        "hot_reload_enabled": hot_reload,
        "total_deploy_sec": round(time.perf_counter() - t0, 2),
    }
    if admin_token:
        res["admin_token"] = admin_token
    return res


def teardown_endpoint(project_id: str, region: str, endpoint_id: str) -> None:
    """Undeploy all models from a Vertex Endpoint and delete it ($0/hr active compute remaining)."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    model_names: set[str] = set()
    deadline = time.time() + 900.0
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            ep = aiplatform.Endpoint(endpoint_name=endpoint_id)
        except Exception:  # noqa: BLE001
            break
        for dm in ep.gca_resource.deployed_models:
            if dm.model:
                model_names.add(dm.model)
        try:
            if ep.gca_resource.deployed_models:
                ep.undeploy_all(sync=True)
            ep.delete(sync=True)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "other operations running on the Endpoint" in msg or "FailedPrecondition" in msg:
                last_exc = exc
                time.sleep(20.0)
                continue
            if "NotFound" in msg or "404" in msg:
                last_exc = None
                break
            raise
    if last_exc is not None:
        raise last_exc
    for m_res in model_names:
        try:
            aiplatform.Model(model_name=m_res).delete(sync=True)
        except Exception:  # noqa: BLE001
            pass


def build_gcloud_rollout_command(
    cand: CandidateSpec,
    *,
    project_id: str = "${GCP_PROJECT_ID}",
    region: str = "${GCP_REGION:-us-central1}",
    endpoint_id: str = "${VERTEX_ENDPOINT_ID}",
    image_uri: str = "${VLLM_IMAGE_URI}",
    machine_type: str = MACHINE_TYPE,
    accelerator_type: str = ACCELERATOR_TYPE,
    accelerator_count: int = ACCELERATOR_COUNT,
) -> str:
    """Generate native `gcloud ai models upload` + `gcloud ai endpoints deploy-model` commands for any winner."""
    shm_mb = recommend_shared_memory_mb(machine_type)
    cmd, vllm_args, env, _ = build_container_launch_spec(cand, hot_reload=False, shared_memory_mb=shm_mb)
    cmd_csv = ",".join(cmd)
    args_csv = "^||^" + "||".join(vllm_args)
    env_pairs = [f"{k}={v}" for k, v in env.items() if k != "HF_TOKEN"]
    env_flag = f" \\\n  --container-env-vars={shlex.quote('^||^' + '||'.join(env_pairs))}" if env_pairs else ""
    accel_flag = f"type={accelerator_type.lower().replace('_', '-')},count={accelerator_count}"

    return (
        "# Roll out the AlphaEvolve winning configuration using native Vertex AI (`gcloud ai`) CLI:\n"
        "MODEL_ID=$(gcloud ai models upload \\\n"
        f"  --project={project_id} \\\n"
        f"  --region={region} \\\n"
        "  --display-name=alpha-inference-best \\\n"
        f"  --container-image-uri={image_uri} \\\n"
        f"  --container-command={shlex.quote(cmd_csv)} \\\n"
        f"  --container-args={shlex.quote(args_csv)}"
        f"{env_flag} \\\n"
        "  --container-ports=8080 \\\n"
        "  --container-predict-route=/v1/chat/completions \\\n"
        "  --container-health-route=/health \\\n"
        f"  --container-shared-memory-size-mb={shm_mb} \\\n"
        "  --container-deployment-timeout-seconds=3600 \\\n"
        "  --format='value(model)' --quiet)\n\n"
        "gcloud ai endpoints deploy-model "
        f"{endpoint_id} \\\n"
        f"  --project={project_id} \\\n"
        f"  --region={region} \\\n"
        "  --model=\"$MODEL_ID\" \\\n"
        "  --display-name=alpha-inference-best \\\n"
        f"  --machine-type={machine_type} \\\n"
        f"  --accelerator={accel_flag} \\\n"
        "  --min-replica-count=1 --max-replica-count=1 \\\n"
        "  --traffic-split=0=100 --quiet\n"
    )

