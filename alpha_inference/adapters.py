"""Workload loading, task-mode auto-detection, and HTTP request adapters.

Supports two built-in request modes:
  1. `chat`: Standard OpenAI `/v1/chat/completions` (`{"messages": [...]}` or `{"prompt": "..."}`),
     supporting `logprobs=True, top_logprobs=5` for Layer-2 token-logprob KL equivalence and
     streaming SSE (`"stream": true`) for real Time-To-First-Token (`ttft`) measurement.
  2. `jev_diffusion`: Frozen 1-step DiffusionGemma block-diffusion protocol (`diffusion_read_only=True`,
     32-token masked JSON canvas) with automatic `ttft_ms = e2e_latency_ms` fallback.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Mapping, Sequence

import aiohttp

TaskMode = Literal["auto", "chat", "jev_diffusion"]
ResolvedTaskMode = Literal["chat", "jev_diffusion"]

TASK_MODE_CHOICES: tuple[str, ...] = ("auto", "chat", "jev_diffusion")

# ==============================================================================
# FROZEN JEV REQUEST PROTOCOL (NEVER MODIFIED BY CANDIDATES)
# ==============================================================================
SERVED_MODEL_NAME = "diffusiongemma-26b"
DECISION_DIMENSIONS: tuple[str, str, str] = ("safe", "spam", "regular")
RATING_VALUES: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
DIGIT_TOKEN_IDS: dict[int, int] = {
    1: 236770,
    2: 236778,
    3: 236800,
    4: 236812,
    5: 236810,
}
ID_TO_DIGIT: dict[int, int] = {tid: digit for digit, tid in DIGIT_TOKEN_IDS.items()}
LABEL_TOKEN_ID_LIST: list[int] = [236770, 236778, 236800, 236812, 236810]

# Exact 32-token Ground-Zero DiffusionGemma JSON canvas:
# `{"safe": <69>, "spam": <69>, "regular": <69>}` padded with 0s to 32 tokens (slots at 4, 10, 16)
MASK_TOKEN_ID = 69
FROZEN_SEED_CANVAS_32: list[int] = [
    14937, 12977, 1083, 236743, MASK_TOKEN_ID,
    236764, 623, 173732, 1083, 236743, MASK_TOKEN_ID,
    236764, 623, 16749, 1083, 236743, MASK_TOKEN_ID,
    236783,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
FROZEN_SLOT_POSITIONS: dict[str, int] = {"safe": 4, "spam": 10, "regular": 16}

FROZEN_SYSTEM_PROMPT = (
    "Analyze the provided input and answer all requested questions. "
    "Respond only with a valid JSON object matching the schema."
)

FROZEN_DECISION_RUBRIC = """Evaluate the incoming organization email across three independent 1-to-5 integer scales.
Return ONLY the structured JSON ratings using single-digit integers 1, 2, 3, 4, or 5:

1. safe (1-5): Sender authenticity, link/attachment safety, and absence of social-engineering risk.
   - 1 = Confirmed or high-confidence phishing, credential harvesting, BEC wire fraud, or malware risk
   - 2 = Suspicious sender/domain mismatch, urgent deceptive lure, or unverified external action request
   - 3 = Neutral or ambiguous external sender with no overt malicious indicators
   - 4 = Known commercial/SaaS sender or routine external business correspondence
   - 5 = Verified internal colleague, trusted enterprise partner, or authenticated system notification

2. spam (1-5): Unsolicited bulk, promotional, graymail, or abusive commercial nature.
   - 1 = Direct person-to-person work email or critical operational transaction (not spam)
   - 2 = Opted-in operational notification or low-noise transactional update
   - 3 = Subscribed industry newsletter, digest, or recurring vendor bulletin
   - 4 = Unsolicited cold sales outreach, webinar blast, or high-frequency marketing graymail
   - 5 = Aggressive bulk spam, deceptive affiliate blast, or mass unsolicited campaign

3. regular (1-5): Routine day-to-day business workflow relevance requiring normal inbox delivery.
   - 1 = Zero business value; quarantine, block, or junk immediately
   - 2 = Low operational priority; background marketing or bulk digest suited for archival/low-priority tab
   - 3 = Standard automated system notification, receipt, or informational broadcast
   - 4 = Legitimate external customer, vendor, or partner operational correspondence
   - 5 = Core internal engineering, finance, product, or leadership work communication"""

FROZEN_KEY_HINT = (
    'Fill in every field of the JSON object {"safe": <1-5>, "spam": <1-5>, "regular": <1-5>} accurately.'
)


@dataclass
class ParsedJevResponse:
    """Parsed 1-step Jev DiffusionGemma structured decision response."""

    request_id: str
    ratings: dict[str, int]
    slot_probabilities: dict[str, dict[str, float]]
    e2e_latency_ms: float
    prompt_tokens: int
    ttft_ms: float = 0.0

    def __post_init__(self) -> None:
        # 1-step non-streaming diffusion produces all canvas positions in one forward pass
        if self.ttft_ms <= 0.0 and self.e2e_latency_ms > 0.0:
            self.ttft_ms = self.e2e_latency_ms


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


ParsedResponse = ParsedJevResponse | ParsedChatResponse


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
    """Auto-detect whether a dataset uses standard OpenAI `chat` or `jev_diffusion`."""
    mode = requested_mode.strip().lower()
    if mode in ("jev", "jev_diffusion"):
        return "jev_diffusion"
    if mode == "chat":
        return "chat"
    if mode != "auto":
        raise ValueError(f"Unknown --task-mode {requested_mode!r}; choose from {TASK_MODE_CHOICES}")

    if not rows:
        return "jev_diffusion" if "diffusiongemma" in model_name.lower() else "chat"

    sample = rows[0]
    if "vllm_xargs" in sample:
        return "jev_diffusion"
    if "messages" in sample or "prompt" in sample:
        return "chat"
    if any(k in sample for k in ("sender", "subject", "body", "recipient_dept")):
        return "jev_diffusion"
    if "input_text" in sample and "diffusiongemma" in model_name.lower():
        return "jev_diffusion"
    return "chat"


def resolve_completions_url(endpoint_url: str) -> str:
    """Normalize base server URL or full Vertex `:rawPredict` URL."""
    if endpoint_url.endswith(("/chat/completions", ":rawPredict")):
        return endpoint_url
    return f"{endpoint_url.rstrip('/')}/v1/chat/completions"


import threading as _threading

_TOKEN_LOCK = _threading.Lock()
_CACHED_CREDS: Any = None
_CACHED_TOKEN_TS: float = 0.0


def fetch_gcp_bearer_token(max_age_sec: float = 600.0) -> str:
    """Fetch/refresh OAuth2 bearer token using GOOGLE_APPLICATION_CREDENTIALS (cached 10m, thread-safe)."""
    global _CACHED_CREDS, _CACHED_TOKEN_TS
    import google.auth
    import google.auth.transport.requests

    with _TOKEN_LOCK:
        now = time.monotonic()
        if _CACHED_CREDS is None:
            _CACHED_CREDS, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if (not getattr(_CACHED_CREDS, "token", None)) or (not _CACHED_CREDS.valid) or (now - _CACHED_TOKEN_TS > max_age_sec):
            _CACHED_CREDS.refresh(google.auth.transport.requests.Request())
            _CACHED_TOKEN_TS = now
        if not _CACHED_CREDS.token:
            raise RuntimeError("Failed to refresh GCP bearer token.")
        return str(_CACHED_CREDS.token)


# ==============================================================================
# ADAPTER 1: FROZEN JEV 1-STEP DIFFUSIONGEMMA
# ==============================================================================
def render_email_text(req: Mapping[str, Any]) -> str:
    if req.get("input_text"):
        return str(req["input_text"]).strip()
    lines: list[str] = []
    if req.get("sender"):
        lines.append(f"From: {req['sender']}")
    if req.get("recipient_dept"):
        lines.append(f"Department: {req['recipient_dept']}")
    if req.get("subject"):
        lines.append(f"Subject: {req['subject']}")
    lines.append("")
    lines.append(str(req.get("body", "")).strip())
    return "\n".join(lines).strip()


def build_frozen_jev_payload(
    req: Mapping[str, Any],
    served_model_name: str = SERVED_MODEL_NAME,
) -> dict[str, Any]:
    """Build the 100% immutable Ground-Zero Jev `/v1/chat/completions` JSON request body."""
    email_state = render_email_text(req)
    user_content = f"### State\n{email_state}\n\n### Questions\n{FROZEN_DECISION_RUBRIC}\n{FROZEN_KEY_HINT}"
    return {
        "model": served_model_name,
        "messages": [
            {"role": "system", "content": FROZEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 32,
        "logprobs": True,
        "top_logprobs": 5,
        "logprob_token_ids": LABEL_TOKEN_ID_LIST,
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {
            "diffusion_seed_canvas": FROZEN_SEED_CANVAS_32,
            "diffusion_max_steps": 1,
            "diffusion_read_only": True,
            "diffusion_constrained": True,
            "diffusion_canvas_length": 32,
        },
    }


def _parse_token_digit(token_str: str) -> int | None:
    if token_str.startswith("token_id:"):
        try:
            tid = int(token_str.split(":", 1)[1])
            return ID_TO_DIGIT.get(tid)
        except ValueError:
            return None
    stripped = token_str.strip()
    if stripped in ("1", "2", "3", "4", "5"):
        return int(stripped)
    return None


def parse_jev_response(
    request_id: str,
    body: Mapping[str, Any],
    e2e_latency_ms: float,
) -> ParsedJevResponse:
    """Extract discrete 1-5 ratings and normalized 5-class probabilities at slots 4, 10, 16."""
    usage = body.get("usage") or {}
    prompt_toks = int(usage.get("prompt_tokens") or 0)
    choice = (body.get("choices") or [{}])[0]
    lp_content = ((choice.get("logprobs") or {}).get("content")) or []

    ratings: dict[str, int] = {}
    slot_probs: dict[str, dict[str, float]] = {}

    for dim in DECISION_DIMENSIONS:
        pos = FROZEN_SLOT_POSITIONS[dim]
        if pos >= len(lp_content):
            raise ValueError(f"Response logprobs length ({len(lp_content)}) shorter than slot {dim}@{pos}")
        entry = lp_content[pos]
        digit_lps: dict[int, float] = {}
        for cand in entry.get("top_logprobs") or []:
            d = _parse_token_digit(str(cand.get("token", "")))
            if d is not None:
                digit_lps[d] = float(cand["logprob"])
        if not digit_lps:
            d_top = _parse_token_digit(str(entry.get("token", "")))
            if d_top is not None:
                digit_lps[d_top] = float(entry.get("logprob", 0.0))
        if not digit_lps:
            raise ValueError(f"No valid 1..5 digit token IDs found at slot {dim}@{pos}")

        raw = {k: float(digit_lps.get(k, -30.0)) for k in RATING_VALUES}
        max_lp = max(raw.values())
        exps = {k: math.exp(lp - max_lp) for k, lp in raw.items()}
        denom = sum(exps.values()) or 1.0
        probs = {k: exps[k] / denom for k in RATING_VALUES}
        best_k = max(RATING_VALUES, key=lambda k: (probs[k], -abs(k - 3)))

        ratings[dim] = int(best_k)
        slot_probs[dim] = {str(k): round(probs[k], 6) for k in RATING_VALUES}

    rounded_e2e = round(e2e_latency_ms, 3)
    return ParsedJevResponse(
        request_id=request_id,
        ratings=ratings,
        slot_probabilities=slot_probs,
        e2e_latency_ms=rounded_e2e,
        prompt_tokens=prompt_toks,
        ttft_ms=rounded_e2e,
    )


async def send_one_jev_request(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    req: Mapping[str, Any],
    bearer_token: str | None = None,
    timeout_sec: float = 120.0,
    served_model_name: str = SERVED_MODEL_NAME,
    max_retries: int = 7,
) -> ParsedJevResponse:
    """Send one frozen 1-step Jev request (`diffusion_read_only=True`; `ttft_ms == e2e_latency_ms`).

    Anchors `t0` before retries so any cloud gateway `429/503` backpressure under burst load is
    honestly counted inside `e2e_latency_ms` rather than hidden or dropped.
    """
    import asyncio
    import random

    url = resolve_completions_url(endpoint_url)
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    payload = build_frozen_jev_payload(req, served_model_name=served_model_name)
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
                if resp.status in (429, 502, 503, 504) and attempt + 1 < max_retries:
                    err_txt = (await resp.text())[:180]
                    last_err = RuntimeError(f"HTTP {resp.status}: {err_txt}")
                    await asyncio.sleep(0.22 * (1.8 ** attempt) + random.uniform(0.0, 0.08))
                    continue
                data = await resp.json(content_type=None)
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {str(data)[:240]}")
                return parse_jev_response(str(req["request_id"]), data, e2e_ms)
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                raise
            await asyncio.sleep(0.25 * (1.8 ** attempt) + random.uniform(0.0, 0.08))

    raise last_err or RuntimeError("Jev request failed after retries")


# ==============================================================================
# ADAPTER 2: GENERIC OPENAI `/v1/chat/completions` + STREAMING SSE TTFT
# ==============================================================================
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
    max_tokens = int(req.get("max_tokens", 64))
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0 if for_equivalence else float(req.get("temperature", 0.0)),
    }
    for opt_key in ("chat_template_kwargs", "top_p", "presence_penalty", "frequency_penalty", "stop", "response_format"):
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
    """Send one OpenAI `/v1/chat/completions` request (supporting streaming SSE TTFT or non-streaming JSON)."""
    import asyncio
    import random

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
                    await asyncio.sleep(0.22 * (1.8 ** attempt) + random.uniform(0.0, 0.08))
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
            await asyncio.sleep(0.25 * (1.8 ** attempt) + random.uniform(0.0, 0.08))

    raise last_err or RuntimeError("Chat request failed after retries")


async def send_one_request(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    req: Mapping[str, Any],
    *,
    task_mode: ResolvedTaskMode = "jev_diffusion",
    model_name: str = SERVED_MODEL_NAME,
    bearer_token: str | None = None,
    timeout_sec: float = 120.0,
    for_equivalence: bool = False,
    stream_ttft: bool = False,
) -> ParsedResponse:
    """Unified request dispatcher supporting both `jev_diffusion` and generic OpenAI `chat`."""
    if task_mode == "jev_diffusion":
        # 1-step Jev diffusion_read_only=True is non-streaming; ttft_ms automatically equals e2e_latency_ms
        return await send_one_jev_request(
            session=session,
            endpoint_url=endpoint_url,
            req=req,
            bearer_token=bearer_token,
            timeout_sec=timeout_sec,
            served_model_name=model_name or SERVED_MODEL_NAME,
        )
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
