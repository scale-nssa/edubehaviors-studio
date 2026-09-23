"""The model layer: a thin shim over the official provider SDKs.

Replaces the `../inference` sibling repo, which a user cloning this one repo
could never install. The app only ever used a sliver of it —

    model.ask(prompt) -> .text .input_tokens .output_tokens .thinking_tokens

— so that is all this module provides, over `anthropic`, `openai` and
`google-genai`, plus one OpenAI-compatible base-URL path that covers
OpenRouter, Together, vLLM, Ollama and LM Studio at once.

**Token accounting has one invariant: `output_tokens` is everything billed as
output, reasoning included.** `thinking_tokens` is the part of it that was
reasoning, where the provider says, and is informational only — never add it
to `output_tokens` again. The providers disagree about this, which is the
whole reason for stating it:

  - Anthropic folds thinking into `usage.output_tokens` and does not break it
    out, so `thinking_tokens` is None.
  - OpenAI (and OpenRouter) count reasoning *inside* `completion_tokens` and
    report the subset under `completion_tokens_details.reasoning_tokens`.
    Adding the two double-counts reasoning.
  - Gemini reports thoughts *outside* `candidates_token_count`, under
    `thoughts_token_count`. Dropping it understated Gemini ~2.6x on the hosted
    branch until it was caught.

Reasoning control is per provider and not all of them can switch it off. The
`REASONING` table below is what the Models page offers; a setting a provider
cannot honour is not offered, rather than sent and silently ignored.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Prompt:
    system_prompt: str | None
    model_prompt: str


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int
    output_tokens: int              # ALL billed output, reasoning included
    thinking_tokens: int | None = None   # the reasoning share, where reported


@dataclass(frozen=True)
class Provider:
    id: str
    title: str
    key_env: str | None          # None: no key (Vertex uses GCP credentials)
    key_optional: bool = False   # local servers usually want no key at all
    needs_base_url: bool = False
    needs_project: bool = False
    reasoning: tuple[str, ...] = ("default",)
    reasoning_note: str = ""
    key_url: str = ""


PROVIDERS: dict[str, Provider] = {
    p.id: p
    for p in (
        Provider(
            "anthropic", "Anthropic", "ANTHROPIC_API_KEY",
            reasoning=("default", "off", "low", "medium", "high"),
            reasoning_note=(
                "Claude Haiku 4.5 and older do not reason unless asked; Sonnet 5 "
                "and newer reason by default. 'off' is refused by models whose "
                "reasoning cannot be disabled (Fable, Opus 5.5) — Test connection "
                "will show the error. Temperature cannot be pinned: the SDK no "
                "longer accepts it."
            ),
            key_url="https://console.anthropic.com/settings/keys",
        ),
        Provider(
            "openai", "OpenAI", "OPENAI_API_KEY",
            reasoning=("default", "low", "medium", "high"),
            reasoning_note=(
                "OpenAI reasoning models cannot switch reasoning off, only lower "
                "its effort. Non-reasoning models reject an effort setting; leave "
                "them on 'default'."
            ),
            key_url="https://platform.openai.com/api-keys",
        ),
        Provider(
            "google", "Google Gemini", "GEMINI_API_KEY",
            reasoning=("default", "minimal", "low", "medium", "high"),
            reasoning_note=(
                "Gemini 3 cannot turn thinking off entirely; 'minimal' is the "
                "lowest it goes. Thinking is billed as output."
            ),
            key_url="https://aistudio.google.com/apikey",
        ),
        Provider(
            "openai_compatible", "OpenAI-compatible endpoint", "OPENAI_COMPATIBLE_API_KEY",
            key_optional=True, needs_base_url=True,
            reasoning=("default",),
            reasoning_note=(
                "Covers OpenRouter, Together, vLLM, Ollama and LM Studio. There "
                "is no portable way to control reasoning here, so the server's "
                "default applies. Local servers usually need no key."
            ),
        ),
        Provider(
            "vertex", "Google Vertex AI", None, needs_project=True,
            reasoning=("default", "off", "low", "medium", "high"),
            reasoning_note=(
                "Uses your GCP application-default credentials "
                "(gcloud auth application-default login), not a key. Claude "
                "models go through Anthropic-on-Vertex, anything else through "
                "Gemini. 'off' means minimal for Gemini, which cannot fully "
                "disable thinking."
            ),
        ),
    )
}

# Starting points shown on the Models page, with list prices (USD per 1M
# tokens, input/output) where known. The user can type any model id and any
# price; these only save a lookup. Prices drift — the page says so.
SUGGESTED: dict[str, list[tuple[str, str, float | None, float | None]]] = {
    "anthropic": [
        ("claude-haiku-4-5", "Claude Haiku 4.5", 1.00, 5.00),
        ("claude-sonnet-5", "Claude Sonnet 5", 2.00, 10.00),
        ("claude-opus-5", "Claude Opus 5", 5.00, 25.00),
    ],
    "openai": [],
    "google": [],
    "openai_compatible": [],
    "vertex": [
        ("gemini-3.8-flash", "Gemini 3.8 Flash", 0.75, 3.75),
        ("claude-haiku-4-5", "Claude Haiku 4.5", 1.00, 5.00),
        ("claude-sonnet-5", "Claude Sonnet 5", 3.00, 15.00),
    ],
}


def known_price(provider: str, model: str) -> tuple[float, float] | None:
    for mid, _title, pin, pout in SUGGESTED.get(provider, []):
        if mid == model and pin is not None and pout is not None:
            return (pin, pout)
    return None


# --------------------------------------------------------------------------- #
# Clients. One per distinct (provider, credentials, endpoint), shared across
# threads: constructing an SDK client per call under the annotation pool leaks
# sockets ("Too many open files") and skips connection reuse.
# --------------------------------------------------------------------------- #

_clients: dict[tuple, Any] = {}
_clients_lock = threading.Lock()

TIMEOUT_S = 600.0


def _client(key: tuple, make: Callable[[], Any]) -> Any:
    with _clients_lock:
        if key not in _clients:
            _clients[key] = make()
        return _clients[key]


def reset_clients() -> None:
    """Drop pooled clients — after a key changes, so the old one is not reused."""
    with _clients_lock:
        _clients.clear()


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

# The anthropic SDK refuses a NON-streaming request whose max_tokens implies a
# possible ten-minute call (~21k). 16k is the documented safe default; the
# hosted branch hit this with 32768 on Sonnet 5.
ANTHROPIC_MAX_TOKENS = 16000

# Models on the legacy pinned-budget thinking API. Everything newer takes
# adaptive thinking and rejects `budget_tokens` with a 400.
_LEGACY_THINKING = re.compile(
    r"claude-(3|haiku-4-5|sonnet-4-5|opus-4-5|opus-4-1|(opus|sonnet)-4(-0|-2025|@|$))"
)
_BUDGET = {"low": 1024, "medium": 4096, "high": 8192}


def _anthropic_kwargs(model: str, reasoning: str) -> dict:
    kw: dict = {"max_tokens": ANTHROPIC_MAX_TOKENS}
    legacy = bool(_LEGACY_THINKING.search(model))
    if reasoning == "default":
        return kw  # the model's own default, nothing sent
    if reasoning == "off":
        # Legacy models do not think unless asked, so omitting it IS off.
        # Adaptive models think by default and need an explicit disable.
        if not legacy:
            kw["thinking"] = {"type": "disabled"}
        return kw
    if legacy:
        kw["thinking"] = {"type": "enabled", "budget_tokens": _BUDGET[reasoning]}
    else:
        kw["thinking"] = {"type": "adaptive"}
        kw["output_config"] = {"effort": reasoning}
    return kw


def _anthropic_response(msg) -> Response:
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return Response(
        text=text,
        input_tokens=msg.usage.input_tokens or 0,
        # Thinking is already inside output_tokens and not broken out.
        output_tokens=msg.usage.output_tokens or 0,
        thinking_tokens=None,
    )


def _ask_anthropic(spec: dict, key: str | None, prompt: Prompt) -> Response:
    import anthropic

    client = _client(
        ("anthropic", key),
        lambda: anthropic.Anthropic(api_key=key, timeout=TIMEOUT_S),
    )
    # No `temperature`: anthropic>=1.3 removed it from Messages.create, and
    # Sonnet 5 rejects sampling parameters server-side anyway.
    msg = client.messages.create(
        model=spec["model"],
        system=prompt.system_prompt or "",
        messages=[{"role": "user", "content": prompt.model_prompt}],
        **_anthropic_kwargs(spec["model"], spec.get("reasoning", "default")),
    )
    return _anthropic_response(msg)


# --------------------------------------------------------------------------- #
# OpenAI and OpenAI-compatible
# --------------------------------------------------------------------------- #

# Models discovered at runtime to reject `temperature` (OpenAI reasoning models
# accept only the default). Remembered so the first failure is the only one.
_no_temperature: set[tuple[str, str]] = set()
_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)


def _openai_response(resp) -> Response:
    msg = resp.choices[0].message
    text = _THINK_TAG.sub("", msg.content or "").strip()
    usage = resp.usage
    if usage is None:
        raise ValueError("the endpoint returned no token usage, so cost cannot be counted")
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    return Response(
        text=text,
        input_tokens=usage.prompt_tokens or 0,
        # completion_tokens already INCLUDES reasoning_tokens.
        output_tokens=usage.completion_tokens or 0,
        thinking_tokens=reasoning,
    )


def _ask_openai(spec: dict, key: str | None, prompt: Prompt, *, compatible: bool) -> Response:
    import openai

    base_url = (spec.get("base_url") or "").strip() or None if compatible else None
    if compatible and not base_url:
        raise ValueError("an OpenAI-compatible endpoint needs a base URL")
    # Local servers accept any key but the SDK insists on one.
    api_key = key or ("not-needed" if compatible else None)
    client = _client(
        ("openai", api_key, base_url),
        lambda: openai.OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT_S),
    )
    messages = []
    if prompt.system_prompt:
        messages.append({"role": "system", "content": prompt.system_prompt})
    messages.append({"role": "user", "content": prompt.model_prompt})
    kw: dict = {"model": spec["model"], "messages": messages}
    reasoning = spec.get("reasoning", "default")
    if not compatible and reasoning != "default":
        kw["reasoning_effort"] = reasoning
    slot = (base_url or "openai", spec["model"])
    if slot not in _no_temperature:
        kw["temperature"] = 0.0
    try:
        resp = client.chat.completions.create(**kw)
    except openai.BadRequestError as exc:
        if "temperature" not in str(exc).lower() or "temperature" not in kw:
            raise
        _no_temperature.add(slot)
        kw.pop("temperature")
        resp = client.chat.completions.create(**kw)
    return _openai_response(resp)


# --------------------------------------------------------------------------- #
# Gemini (API key, or Vertex)
# --------------------------------------------------------------------------- #

_LEVEL = {"off": "MINIMAL", "minimal": "MINIMAL", "low": "LOW",
          "medium": "MEDIUM", "high": "HIGH"}


def _gemini(client, spec: dict, prompt: Prompt) -> Response:
    from google.genai import types

    cfg: dict = {"temperature": 0.0}
    if prompt.system_prompt:
        cfg["system_instruction"] = prompt.system_prompt
    reasoning = spec.get("reasoning", "default")
    if reasoning != "default":
        # Sent explicitly. Omitting the config is NOT "off" — the server
        # default (MEDIUM) applies, which is how the hosted branch paid for
        # 206k thinking tokens it believed it had disabled.
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=_LEVEL[reasoning])
    resp = client.models.generate_content(
        model=spec["model"],
        contents=prompt.model_prompt,
        config=types.GenerateContentConfig(**cfg),
    )
    usage = resp.usage_metadata
    visible = (usage.candidates_token_count or 0) if usage else 0
    thoughts = (usage.thoughts_token_count or 0) if usage else 0
    return Response(
        text=resp.text or "",
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        # Gemini reports thoughts OUTSIDE candidates; both are billed output.
        output_tokens=visible + thoughts,
        thinking_tokens=thoughts,
    )


def _ask_google(spec: dict, key: str | None, prompt: Prompt) -> Response:
    from google import genai

    client = _client(("google", key), lambda: genai.Client(api_key=key))
    return _gemini(client, spec, prompt)


def _ask_vertex(spec: dict, _key: str | None, prompt: Prompt) -> Response:
    project = (spec.get("project") or "").strip()
    region = (spec.get("region") or "global").strip()
    if not project:
        raise ValueError("Vertex needs a GCP project id")
    if spec["model"].startswith("claude"):
        import anthropic

        client = _client(
            ("vertex-anthropic", project, region),
            lambda: anthropic.AnthropicVertex(
                project_id=project, region=region, timeout=TIMEOUT_S
            ),
        )
        msg = client.messages.create(
            model=spec["model"],
            system=prompt.system_prompt or "",
            messages=[{"role": "user", "content": prompt.model_prompt}],
            **_anthropic_kwargs(spec["model"], spec.get("reasoning", "default")),
        )
        return _anthropic_response(msg)
    from google import genai

    client = _client(
        ("vertex-gemini", project, region),
        lambda: genai.Client(vertexai=True, project=project, location=region),
    )
    return _gemini(client, spec, prompt)


_DISPATCH = {
    "anthropic": _ask_anthropic,
    "openai": lambda s, k, p: _ask_openai(s, k, p, compatible=False),
    "openai_compatible": lambda s, k, p: _ask_openai(s, k, p, compatible=True),
    "google": _ask_google,
    "vertex": _ask_vertex,
}


class Model:
    """What `llm.model()` hands back: one bound (spec, key) pair."""

    def __init__(self, spec: dict, key: str | None):
        if spec.get("provider") not in _DISPATCH:
            raise ValueError(f"unknown provider {spec.get('provider')!r}")
        reasoning = spec.get("reasoning", "default")
        if reasoning not in PROVIDERS[spec["provider"]].reasoning:
            raise ValueError(
                f"{PROVIDERS[spec['provider']].title} does not support reasoning "
                f"setting {reasoning!r}"
            )
        self.spec = spec
        self._key = key

    def ask(self, prompt: Prompt) -> Response:
        return _DISPATCH[self.spec["provider"]](self.spec, self._key, prompt)
