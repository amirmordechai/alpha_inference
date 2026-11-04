#!/usr/bin/env python3
"""Standalone Vertex AI Dedicated Endpoint deployer for any `candidate.py` / `best_candidate.py`.

Usage:
    # 1. Need a live Vertex AI GPU Endpoint? Deploy any HuggingFace model in 1 command:
    python deploy.py --model google/gemma-4-26B-A4B-it --hot-reload --deploy --out-json endpoint.json

    # 2. Preview container launch spec ($0 spend):
    python deploy.py --config candidate.py --dry-run

    # 3. Compare Ground-Zero Baseline vs. AlphaEvolve Best:
    python deploy.py --diff

    # 4. Roll out the winning configuration to a new or existing Vertex AI Endpoint:
    python deploy.py --config examples/email_triage/results/best_candidate.py --endpoint-id <ENDPOINT_ID> --deploy

    # 5. Tear down an endpoint ($0/hr active compute):
    python deploy.py --delete --endpoint-id <ENDPOINT_ID>
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from alpha_inference.reporting import (
    ACCELERATOR_COUNT,
    ACCELERATOR_TYPE,
    MACHINE_TYPE,
    RELOAD_PROXY_FILE,
    CandidateSpec,
    build_seed_program_source,
    parse_candidate_code,
)

UPSTREAM_VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.31.0"


def ensure_ar_image_mirrored(project_id: str, region: str, ar_repo: str) -> str:
    """Ensure upstream `vllm-openai:v0.31.0` is mirrored in Artifact Registry via Cloud Build `crane`."""
    ar_image_uri = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/vllm-openai:v0.31.0"
    gcloud_env = os.environ.copy()
    cred_path = gcloud_env.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path:
        gcloud_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = cred_path

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

    cb_cfg = Path("/tmp/ae_mirror_vllm.json")
    cb_cfg.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "gcr.io/go-containerregistry/crane:latest",
                        "args": ["copy", "--platform=linux/amd64", UPSTREAM_VLLM_IMAGE, ar_image_uri],
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


def load_candidate_file(config_path: str | Path) -> CandidateSpec:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate configuration file not found: {p}")
    return parse_candidate_code(p.read_text(encoding="utf-8"), name=p.stem)


def build_container_launch_spec(
    cand: CandidateSpec,
    *,
    hot_reload: bool = False,
    admin_token: str = "",
) -> tuple[list[str], list[str], dict[str, str], str]:
    """Build Vertex container command, args, env vars, and optional hot-reload admin token."""
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hot_reload:
        tok = admin_token or secrets.token_urlsafe(24)
        proxy_b64 = base64.b64encode(RELOAD_PROXY_FILE.read_bytes()).decode("ascii")
        init_spec = json.dumps(
            {
                "vllm_args": cand.vllm_args,
                "env_vars": cand.env_vars,
                "custom_kernel_code": cand.custom_kernel_code,
                "startup_timeout_sec": 900.0,
            }
        )
        env = {
            "AE_ADMIN_TOKEN": tok,
            "AE_PROXY_B64": proxy_b64,
            "AE_INITIAL_SPEC_JSON": init_spec,
            "PYTHONUNBUFFERED": "1",
        }
        if hf_token:
            env["HF_TOKEN"] = hf_token
        bootstrap = (
            'python3 -c "import base64, os, pathlib; '
            "pathlib.Path('/tmp/reload_proxy.py').write_bytes(base64.b64decode(os.environ['AE_PROXY_B64']))\" && "
            "exec python3 /tmp/reload_proxy.py"
        )
        return ["bash", "-lc", bootstrap], [], env, tok

    env = dict(cand.env_vars)
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


def deploy_candidate_to_vertex(
    cand: CandidateSpec,
    project_id: str,
    region: str,
    ar_repo: str,
    display_name: str,
    source_path: str = "candidate.py",
    existing_endpoint_id: str = "",
    traffic_percentage: int = 100,
    skip_mirror: bool = False,
    machine_type: str = MACHINE_TYPE,
    accelerator_type: str = ACCELERATOR_TYPE,
    accelerator_count: int = ACCELERATOR_COUNT,
    hot_reload: bool = False,
) -> dict[str, Any]:
    """Deploy a `CandidateSpec` to a new or existing Vertex AI Dedicated GPU Endpoint."""
    from google.cloud import aiplatform

    t0 = time.perf_counter()
    aiplatform.init(project=project_id, location=region)
    ar_image_uri = (
        f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/vllm-openai:v0.31.0"
        if skip_mirror
        else ensure_ar_image_mirrored(project_id, region, ar_repo)
    )
    cmd, args, env, admin_token = build_container_launch_spec(cand, hot_reload=hot_reload)

    model = aiplatform.Model.upload(
        display_name=f"{display_name}-model",
        serving_container_image_uri=ar_image_uri,
        serving_container_command=cmd,
        serving_container_args=args,
        serving_container_environment_variables=env,
        serving_container_ports=[8080],
        serving_container_predict_route="/v1/chat/completions",
        serving_container_health_route="/health",
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
    res = {
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
            else ""
        ),
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
    ep = aiplatform.Endpoint(endpoint_name=endpoint_id)
    model_names = [dm.model for dm in ep.gca_resource.deployed_models]
    if ep.gca_resource.deployed_models:
        ep.undeploy_all(sync=True)
    ep.delete(sync=True)
    for m_res in set(model_names):
        try:
            aiplatform.Model(model_name=m_res).delete(sync=True)
        except Exception:  # noqa: BLE001
            pass
    print(f"Deleted Vertex Endpoint {endpoint_id} in {region} ($0/hr active compute).")


def main() -> None:
    load_dotenv()
    default_best = Path("examples/email_triage/results/best_candidate.py")
    default_cfg = str(default_best) if default_best.exists() else "candidate.py"

    parser = argparse.ArgumentParser(
        description="Deploy, diff, or tear down a candidate configuration on a Vertex AI Dedicated GPU Endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=default_cfg, help="Path to candidate.py or best_candidate.py")
    parser.add_argument("--model", default="", help="Optional HuggingFace model ID (generates clean seed config on the fly)")
    parser.add_argument("--served-model-name", default="", help="Optional --served-model-name when using --model")
    parser.add_argument("--baseline-config", default="candidate.py", help="Baseline candidate file for --diff")
    parser.add_argument("--diff", action="store_true", help="Print unified diff between --baseline-config and --config")
    parser.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", ""))
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", os.environ.get("GCP_REGION_GPU", "us-central1")))
    parser.add_argument("--ar-repo", default=os.environ.get("GCP_AR_REPO", "geap-models"))
    parser.add_argument("--machine-type", default=MACHINE_TYPE)
    parser.add_argument("--accelerator-type", default=ACCELERATOR_TYPE)
    parser.add_argument("--accelerator-count", type=int, default=ACCELERATOR_COUNT)
    parser.add_argument("--display-name", default="ae-inference-endpoint")
    parser.add_argument("--endpoint-id", default="", help="Existing endpoint ID (for rollout, A/B split, or --delete)")
    parser.add_argument("--hot-reload", action="store_true", help="Deploy with reload_proxy so optimize.py can hot-swap in place")
    parser.add_argument("--traffic-pct", type=int, default=100)
    parser.add_argument("--skip-mirror", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if args.diff:
        b_lines = Path(args.baseline_config).read_text(encoding="utf-8").splitlines(keepends=True)
        e_lines = Path(args.config).read_text(encoding="utf-8").splitlines(keepends=True)
        sys.stdout.writelines(
            difflib.unified_diff(b_lines, e_lines, fromfile=args.baseline_config, tofile=args.config)
        )
        return

    if args.delete:
        if not args.project_id or not args.endpoint_id:
            raise SystemExit("--project-id and --endpoint-id are required with --delete.")
        teardown_endpoint(args.project_id, args.region, args.endpoint_id)
        return

    if args.model.strip():
        cand = parse_candidate_code(build_seed_program_source(args.model, args.served_model_name), name="model_seed")
        src_label = f"--model {args.model.strip()}"
    else:
        cand = load_candidate_file(args.config)
        src_label = args.config

    cmd, vllm_args, env, _tok = build_container_launch_spec(cand, hot_reload=args.hot_reload)
    safe_env = {
        k: ("***REDACTED***" if k in ("HF_TOKEN", "AE_ADMIN_TOKEN") else v)
        for k, v in env.items()
        if k not in ("AE_KERNEL_B64", "AE_PROXY_B64")
    }
    spec_preview = {
        "source_candidate": src_label,
        "plain_title": cand.plain_title,
        "hypothesis": cand.hypothesis,
        "category": cand.category,
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "hot_reload_enabled": args.hot_reload,
        "container_command": cmd,
        "vllm_args": cand.vllm_args if args.hot_reload else vllm_args,
        "env_vars": safe_env,
        "has_custom_kernel": bool(cand.custom_kernel_code.strip()),
    }
    if args.dry_run or not args.deploy:
        print(json.dumps(spec_preview, indent=2))
        return

    if not args.project_id:
        raise SystemExit("Set GCP_PROJECT_ID in .env or pass --project-id.")

    info = deploy_candidate_to_vertex(
        cand=cand,
        project_id=args.project_id,
        region=args.region,
        ar_repo=args.ar_repo,
        display_name=args.display_name,
        source_path=src_label,
        existing_endpoint_id=args.endpoint_id,
        traffic_percentage=args.traffic_pct,
        skip_mirror=args.skip_mirror,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        hot_reload=args.hot_reload,
    )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
