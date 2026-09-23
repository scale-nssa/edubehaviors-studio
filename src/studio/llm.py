"""LLM plumbing: JSON extraction with repair, retries, and token logging.

Free-text JSON with a repair pass rather than structured output (Q131).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

from . import model_settings, spend, telemetry
from .config import ANNOTATION_RETRIES, MODEL_CONCURRENCY, is_permitted
from .providers import Model, Prompt, Response

log = logging.getLogger("studio.llm")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_models: dict[str, Any] = {}
_models_lock = threading.Lock()

_usage_lock = threading.Lock()
_usage: dict[str, dict[str, int]] = {}

# One gate per model, covering EVERY caller. It lives here rather than at the
# three reasoning call sites because the failure it prevents — twenty people
# clicking "Generate criteria" in the first five minutes of a workshop, i.e.
# twenty simultaneous Sonnet calls — happens at the worst possible moment.
#
# `annotate._semaphores` used to gate the annotators separately; it is gone, so
# nothing is gated twice (which would silently halve throughput).
_gates_lock = threading.Lock()
_gates: dict[str, threading.Semaphore] = {}


def gate(nick: str) -> threading.Semaphore:
    # Keyed on the underlying model: two annotators bound to the same model
    # (`x` and `x#2`) share one rate limit on the provider's side, so they
    # share one gate here.
    base = model_settings.strip_sample(nick)
    with _gates_lock:
        if base not in _gates:
            _gates[base] = threading.Semaphore(MODEL_CONCURRENCY(nick))
        return _gates[base]


class ModelNotReady(RuntimeError):
    """The nickname is not held by a verified role — see model_settings."""


def model(nick: str, *, profile: str = "default") -> Model:
    """Resolve a nickname to a callable model.

    Only a nickname a *verified* role currently holds resolves. That is the
    validation layer that replaced the hosted allowlist: nothing is called
    that Test connection has not confirmed works, with this key and these
    settings. `profile` is kept for the call sites and the spend ledger; the
    reasoning setting is part of the binding, and so of the nickname.
    """
    if not is_permitted(nick):
        raise ModelNotReady(
            f"{model_settings.title(nick)} is not a verified model. Open the "
            "Models page and press Test connection for each role."
        )
    spec = model_settings.spec_for(nick)
    if spec is None:
        raise ModelNotReady(f"no role is bound to {nick!r}")
    key = f"{nick}::{model_settings.fingerprint(spec)}"
    with _models_lock:
        if key not in _models:
            _models[key] = Model(spec, model_settings.key_for(spec["provider"]))
        return _models[key]


def _call(nick: str, prompt: Prompt, profile: str) -> Response:
    """One gated, recorded, cap-checked call."""
    spend.check()
    with gate(nick):
        t0 = telemetry.call_started(nick)
        try:
            resp = model(nick, profile=profile).ask(prompt)
        except BaseException:
            telemetry.call_finished(nick, t0, ok=False)
            raise
        telemetry.call_finished(nick, t0, ok=True)
    _record(nick, resp.input_tokens, resp.output_tokens, resp.thinking_tokens, profile)
    return resp


def test_role(role: str) -> tuple[bool, str]:
    """Make one cheap real call for a role and report what happened.

    The error text is returned verbatim (keys redacted): auth, region and
    model-not-found errors are the whole support burden of local mode, and a
    paraphrase of them is useless. Success marks the binding verified.
    """
    spec = model_settings.role_spec(role)
    nick = model_settings.role_nick(role)
    try:
        spend.check()
        m = Model(spec, model_settings.key_for(spec["provider"]))
        t0 = time.time()
        resp = m.ask(Prompt(
            system_prompt="Reply with exactly the word OK.",
            model_prompt="Say OK.",
        ))
        _record(nick, resp.input_tokens, resp.output_tokens, resp.thinking_tokens, "test")
    except Exception as exc:  # noqa: BLE001 — surface whatever the provider said
        model_settings.mark_verified(role, False)
        return False, model_settings.redact(f"{type(exc).__name__}: {exc}")
    model_settings.mark_verified(role, True)
    with _models_lock:
        _models.clear()
    return True, (
        f"Answered in {time.time() - t0:.1f}s: {resp.text.strip()[:40]!r} "
        f"({resp.input_tokens} in / {resp.output_tokens} out tokens)"
    )


def usage_report() -> dict[str, dict[str, int]]:
    with _usage_lock:
        return {k: dict(v) for k, v in _usage.items()}


def _record(nick: str, inp: int, out: int, thinking: int | None = None,
            profile: str = "default") -> None:
    """`out` is ALL billed output, reasoning included — the invariant
    `providers.Response` guarantees, so it is never added to again here.
    (Gemini reports thoughts outside candidates and they were once dropped;
    OpenAI reports reasoning inside completion_tokens and adding it again
    would double-count. The shim normalises both.) `thinking_tokens` is kept
    alongside so the panel can show how much of the bill is reasoning."""
    with _usage_lock:
        u = _usage.setdefault(
            nick,
            {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "calls": 0},
        )
        u["input_tokens"] += inp or 0
        u["output_tokens"] += out or 0
        u["thinking_tokens"] += thinking or 0
        u["calls"] += 1
    try:
        spend.record(nick, profile, inp or 0, out or 0, thinking)
    except Exception:  # noqa: BLE001 — a ledger failure must not lose the answer
        log.warning("could not write the spend ledger", exc_info=True)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response, repairing common wrappers."""
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _OBJECT.search(cleaned)
        if not m:
            raise ValueError(f"no JSON object in response: {cleaned[:200]!r}")
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def ask_json(
    nick: str,
    prompt: Prompt,
    *,
    retries: int = ANNOTATION_RETRIES,
    profile: str = "default",
) -> tuple[dict, int, int]:
    """Call a model and parse JSON, retrying transient failures and bad JSON.

    A session-level call is expensive enough that one malformed response is worth
    a retry rather than silently yielding an empty answer (upstream policy, Q304).
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _call(nick, prompt, profile)
            return extract_json(resp.text), resp.input_tokens, resp.output_tokens
        except (ModelNotReady, spend.SpendCapReached):
            raise   # not transient: retrying cannot help and only delays the message
        except Exception as exc:  # noqa: BLE001 — provider errors are heterogeneous
            last = exc
            if attempt < retries - 1:
                sleep = min(2.0 * (2**attempt), 20.0)
                log.warning("%s attempt %d failed (%s); retrying in %.0fs",
                            nick, attempt + 1, model_settings.redact(str(exc)), sleep)
                time.sleep(sleep)
    raise RuntimeError(
        f"{model_settings.title(nick)} failed after {retries} attempts: "
        f"{model_settings.redact(str(last))}"
    ) from last


def ask_text(
    nick: str,
    prompt: Prompt,
    *,
    retries: int = ANNOTATION_RETRIES,
    profile: str = "default",
) -> tuple[str, int, int]:
    """Call a model and return raw text, retrying only transport failures."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _call(nick, prompt, profile)
            return resp.text, resp.input_tokens, resp.output_tokens
        except (ModelNotReady, spend.SpendCapReached):
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(min(2.0 * (2**attempt), 20.0))
    raise RuntimeError(
        f"{model_settings.title(nick)} failed after {retries} attempts: "
        f"{model_settings.redact(str(last))}"
    ) from last
