"""Workload trace loading and OpenAI `/v1/chat/completions` HTTP/SSE adapters."""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import threading as _threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Mapping, Sequence

import aiohttp

TaskMode = Literal["auto", "chat"]
ResolvedTaskMode = Literal["chat"]

TASK_MODE_CHOICES: tuple[str, ...] = ("auto", "chat")
SERVED_MODEL_NAME = "google/gemma-4-26B-A4B-it"


@dataclass
class ParsedChatResponse:
    """Parsed OpenAI `/v1/chat/completions` response with optional token-position logprobs & TTFT."""

    request_id: str
    content: str
    token_positions: list[dict[str, Any]] = field(default_factory=list)
    e2e_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.ttft_ms <= 0.0 and self.e2e_latency_ms > 0.0:
            self.ttft_ms = self.e2e_latency_ms


ParsedResponse = ParsedChatResponse


def load_jsonl_requests(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL (or `.jsonl.gz`) request rows, assigning deterministic `request_id` if omitted."""
    p = Path(path)
    if not p.exists() and Path(f"{p}.gz").exists():
        p = Path(f"{p}.gz")
    rows: list[dict[str, Any]] = []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {idx} of {p}")
            if not obj.get("request_id"):
                obj = {"request_id": f"req-{idx:05d}", **obj}
            rows.append(obj)
    if not rows:
        raise ValueError(f"No request rows found in {p}")
    return rows


def detect_task_mode(
    rows: Sequence[Mapping[str, Any]],
    requested_mode: str = "auto",
    model_name: str = "",
) -> ResolvedTaskMode:
    """Validate and resolve the OpenAI `/v1/chat/completions` task mode (`chat`)."""
    del rows, model_name
    mode = requested_mode.strip().lower()
    if mode not in TASK_MODE_CHOICES:
        raise ValueError(f"Unknown --task-mode {requested_mode!r}; choose from {TASK_MODE_CHOICES}")
    return "chat"


def trace_uses_guided_decoding(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return True if any request uses grammar/schema-constrained decoding (`response_format`, `tools`, etc.)."""
    for req in rows:
        rf = req.get("response_format")
        if isinstance(rf, Mapping) and str(rf.get("type") or "").strip().lower() not in ("", "text"):
            return True
        if isinstance(rf, str) and rf.strip().lower() not in ("", "text"):
            return True
        if req.get("guided_decoding") or req.get("structured_outputs") or req.get("tools"):
            return True
        xargs = req.get("vllm_xargs")
        if isinstance(xargs, Mapping) and any(str(k).startswith("guided_") for k in xargs):
            return True
    return False


def resolve_completions_url(endpoint_url: str) -> str:
    """Normalize base server URL or full Vertex `:rawPredict` URL."""
    if endpoint_url.endswith(("/chat/completions", ":rawPredict")):
        return endpoint_url
    return f"{endpoint_url.rstrip('/')}/v1/chat/completions"


_TOKEN_LOCK = _threading.Lock()
_CACHED_CREDS: Any = None
_CACHED_TOKEN_TS: float = 0.0


def fetch_gcp_bearer_token(max_age_sec: float = 600.0) -> str:
    """Fetch/refresh OAuth2 bearer token using ADC / GOOGLE_APPLICATION_CREDENTIALS (cached 10m)."""
    global _CACHED_CREDS, _CACHED_TOKEN_TS
    import google.auth
    import google.auth.transport.requests

    with _TOKEN_LOCK:
        now = time.monotonic()
        if _CACHED_CREDS is None:
            _CACHED_CREDS, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if (
            (not getattr(_CACHED_CREDS, "token", None))
            or (not _CACHED_CREDS.valid)
            or (now - _CACHED_TOKEN_TS > max_age_sec)
        ):
            _CACHED_CREDS.refresh(google.auth.transport.requests.Request())
            _CACHED_TOKEN_TS = now
        if not _CACHED_CREDS.token:
            raise RuntimeError("Failed to refresh GCP bearer token.")
        return str(_CACHED_CREDS.token)


def build_chat_payload(
    req: Mapping[str, Any],
    model_name: str,
    *,
    for_equivalence: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    """Build an OpenAI `/v1/chat/completions` payload from `messages`, `prompt`, or `input_text`."""
    if req.get("messages"):
        messages = list(req["messages"])
    elif req.get("prompt") is not None:
        messages = []
        if req.get("system"):
            messages.append({"role": "system", "content": str(req["system"])})
        messages.append({"role": "user", "content": str(req["prompt"])})
    elif req.get("input_text") is not None:
        messages = [{"role": "user", "content": str(req["input_text"])}]
    else:
        raise ValueError(f"Request {req.get('request_id')} must contain 'messages', 'prompt', or 'input_text'")

    resolved_model = str(req.get("model") or model_name or "default")
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": int(req.get("max_tokens", 64)),
        "temperature": 0.0 if for_equivalence else float(req.get("temperature", 0.0)),
    }
    for opt_key in (
        "chat_template_kwargs",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "guided_decoding",
        "structured_outputs",
        "tools",
        "tool_choice",
    ):
        if opt_key in req:
            payload[opt_key] = req[opt_key]
    if for_equivalence:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(req.get("top_logprobs", 5))
        payload["stream"] = False
    elif stream:
        payload["stream"] = True
    else:
        if "logprobs" in req:
            payload["logprobs"] = bool(req["logprobs"])
        if "top_logprobs" in req:
            payload["top_logprobs"] = int(req["top_logprobs"])
    return payload


def _extract_token_positions(lp_content: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for entry in lp_content:
        chosen_tok = str(entry.get("token", ""))
        chosen_lp = float(entry.get("logprob", -20.0))
        top_map: dict[str, float] = {}
        for item in entry.get("top_logprobs") or []:
            tok = str(item.get("token", ""))
            if tok and tok not in top_map:
                top_map[tok] = round(float(item.get("logprob", -20.0)), 6)
        if chosen_tok and chosen_tok not in top_map:
            top_map[chosen_tok] = round(chosen_lp, 6)
        positions.append({"token": chosen_tok, "top_logprobs": top_map})
    return positions


def parse_chat_response(
    request_id: str,
    body: Mapping[str, Any],
    e2e_latency_ms: float,
    ttft_ms: float | None = None,
) -> ParsedChatResponse:
    """Parse a non-streaming `/v1/chat/completions` JSON response."""
    usage = body.get("usage") or {}
    prompt_toks = int(usage.get("prompt_tokens") or 0)
    comp_toks = int(usage.get("completion_tokens") or 0)
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = str(msg.get("content") or choice.get("text") or "")
    lp_content = ((choice.get("logprobs") or {}).get("content")) or []
    positions = _extract_token_positions(lp_content)

    rounded_e2e = round(e2e_latency_ms, 3)
    resolved_ttft = round(ttft_ms, 3) if ttft_ms is not None and ttft_ms > 0.0 else rounded_e2e
    return ParsedChatResponse(
        request_id=request_id,
        content=content,
        token_positions=positions,
        e2e_latency_ms=rounded_e2e,
        ttft_ms=resolved_ttft,
        prompt_tokens=prompt_toks,
        completion_tokens=comp_toks or len(positions),
    )


async def consume_sse_chat_stream(
    raw_lines: AsyncIterator[bytes | str],
    t0: float,
) -> tuple[str, list[dict[str, Any]], float, float]:
    """Consume an OpenAI SSE stream (`data: {...}`) and measure exact first-token TTFT and E2E latency."""
    ttft_ms: float | None = None
    text_parts: list[str] = []
    streamed_lp_entries: list[Mapping[str, Any]] = []

    async for raw_line in raw_lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        stripped = line.strip()
        if not stripped or not stripped.startswith("data:"):
            continue
        data_str = stripped[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        c0 = choices[0]
        delta = c0.get("delta") or {}
        piece = delta.get("content")
        reasoning_piece = delta.get("reasoning_content")
        tool_calls = delta.get("tool_calls")
        alt_text = c0.get("text")

        has_first_token = bool(piece) or bool(reasoning_piece) or bool(tool_calls) or bool(alt_text)
        if has_first_token and ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        if piece:
            text_parts.append(str(piece))
        elif alt_text:
            text_parts.append(str(alt_text))

        lp_chunk = ((c0.get("logprobs") or {}).get("content")) or []
        if lp_chunk:
            streamed_lp_entries.extend(lp_chunk)

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    final_ttft = ttft_ms if ttft_ms is not None else e2e_ms
    return "".join(text_parts), _extract_token_positions(streamed_lp_entries), round(final_ttft, 3), round(e2e_ms, 3)


async def send_one_chat_request(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    req: Mapping[str, Any],
    model_name: str,
    *,
    bearer_token: str | None = None,
    timeout_sec: float = 120.0,
    for_equivalence: bool = False,
    stream: bool = False,
    max_retries: int = 7,
) -> ParsedChatResponse:
    """Send one OpenAI `/v1/chat/completions` request (streaming SSE TTFT or non-streaming JSON)."""
    url = resolve_completions_url(endpoint_url)
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    payload = build_chat_payload(req, model_name, for_equivalence=for_equivalence, stream=stream)
    req_id = str(req["request_id"])
    t0 = time.perf_counter()
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as resp:
                if resp.status in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                    err_body = (await resp.text())[:180]
                    last_err = RuntimeError(f"HTTP {resp.status}: {err_body}")
                    await asyncio.sleep(0.22 * (1.8**attempt) + random.uniform(0.0, 0.08))
                    continue
                if resp.status >= 400:
                    err_body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {err_body[:240]}")

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if stream and "application/json" not in ctype:
                    content, positions, ttft_ms, e2e_ms = await consume_sse_chat_stream(resp.content, t0)
                    return ParsedChatResponse(
                        request_id=req_id,
                        content=content,
                        token_positions=positions,
                        e2e_latency_ms=e2e_ms,
                        ttft_ms=ttft_ms,
                        completion_tokens=len(positions) or max(1, len(content) // 4),
                    )

                data = await resp.json(content_type=None)
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                return parse_chat_response(req_id, data, e2e_ms)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                raise
            await asyncio.sleep(0.25 * (1.8**attempt) + random.uniform(0.0, 0.08))

    raise last_err or RuntimeError("Chat request failed after retries")


async def send_one_request(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    req: Mapping[str, Any],
    *,
    task_mode: ResolvedTaskMode = "chat",
    model_name: str = SERVED_MODEL_NAME,
    bearer_token: str | None = None,
    timeout_sec: float = 120.0,
    for_equivalence: bool = False,
    stream_ttft: bool = False,
) -> ParsedResponse:
    """Send a single OpenAI `/v1/chat/completions` request."""
    del task_mode
    return await send_one_chat_request(
        session=session,
        endpoint_url=endpoint_url,
        req=req,
        model_name=model_name,
        bearer_token=bearer_token,
        timeout_sec=timeout_sec,
        for_equivalence=for_equivalence,
        stream=stream_ttft and not for_equivalence,
    )
