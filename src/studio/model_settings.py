"""Which model plays which role, and the keys to call them — set on /models.

This is what replaced the hosted branch's three-name allowlist. The app still
needs exactly three *roles* — two annotators and a reasoning model — and the
rest of the code only ever asks for those. What changed is who binds them: the
researcher, on the Models page, with their own key.

The allowlist's property survives as **verification**. A role's model may only
be called once Test connection has made a real call to that exact model, with
that exact key and those exact settings, and it answered. Change any of them
and the role is unverified again until re-tested. "Any model" therefore means
"a model the app has confirmed it can call", never "any string typed into a
box" (docs/local-mode-plan.md §2.1).

Storage, both beside the database in the data directory:

  models.json  role bindings, prices, spend cap. Not secret.
  .env         API keys, chmod 600. **Never the SQLite file**: that is the
               thing people copy around and send to collaborators.

Nicknames. Everything downstream — the annotation cache, the export, the
panel — keys on a nickname string, as it always did. A nickname is now derived
from the binding: `provider/model`, plus `@reasoning` when reasoning is not the
provider default, plus the endpoint host for an OpenAI-compatible server. That
puts the settings that change an answer *into the cache key*, so a reasoning
change cannot silently reuse answers produced under the old setting (§4.5,
§8), and two servers both calling their model "llama3" cannot collide.

When both annotators are bound to the same model, the second gets a `#2`
suffix. Without it the second annotator would hit the first one's cache entry
and "agree" perfectly by construction; with it, each is an independent sample
and α measures sampling noise, which is what the Models page warns it means.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

from .providers import PROVIDERS, known_price

log = logging.getLogger("studio.settings")

ROLES = ("annotator_a", "annotator_b", "reasoning")
ROLE_TITLES = {
    "annotator_a": "Annotator A",
    "annotator_b": "Annotator B",
    "reasoning": "Reasoning",
}

# What a fresh install is bound to. One provider on purpose: a user with only
# an Anthropic key can complete a round. Unverified until tested, so nothing is
# ever called on these defaults without the user having pressed Test.
DEFAULT_ROLES: dict[str, dict] = {
    "annotator_a": {"provider": "anthropic", "model": "claude-haiku-4-5", "reasoning": "low"},
    "annotator_b": {"provider": "anthropic", "model": "claude-haiku-4-5", "reasoning": "low"},
    "reasoning": {"provider": "anthropic", "model": "claude-sonnet-5", "reasoning": "medium"},
}

# Refuse to start anything once lifetime spend on this data directory would
# pass this. Deliberately low: it is the user's own card, and raising it is
# one field on the Models page.
DEFAULT_SPEND_CAP_USD = 10.0

SPEC_FIELDS = ("provider", "model", "reasoning", "base_url", "project", "region",
               "price_in", "price_out")

_lock = threading.RLock()
_state: dict = {}
_dir: Path | None = None


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #

def _defaults() -> dict:
    return {
        "roles": {r: dict(s) for r, s in DEFAULT_ROLES.items()},
        "verified": {},
        "spend_cap_usd": DEFAULT_SPEND_CAP_USD,
    }


def init(data_dir: Path) -> None:
    """Bind to a data directory and load what is there. Called by create_app."""
    global _dir, _state
    with _lock:
        _dir = Path(data_dir)
        _dir.mkdir(parents=True, exist_ok=True)
        load_env(_dir / ".env")
        path = _dir / "models.json"
        state = _defaults()
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                state["roles"].update(
                    {r: _clean(s) for r, s in (raw.get("roles") or {}).items() if r in ROLES}
                )
                state["verified"] = dict(raw.get("verified") or {})
                if raw.get("spend_cap_usd") is not None:
                    state["spend_cap_usd"] = float(raw["spend_cap_usd"])
            except (ValueError, TypeError, OSError) as exc:
                log.warning("could not read %s (%s); using defaults", path, exc)
        _state = state


def _ensure() -> dict:
    # Import-time callers (and tests that never build an app) get defaults.
    if not _state:
        _state.update(_defaults())
    return _state


def _save() -> None:
    if _dir is None:
        return
    path = _dir / "models.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_ensure(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def data_dir() -> Path | None:
    return _dir


def _clean(spec: dict) -> dict:
    out = {k: spec.get(k) for k in SPEC_FIELDS if spec.get(k) not in (None, "")}
    out.setdefault("reasoning", "default")
    for k in ("price_in", "price_out"):
        if k in out:
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                out.pop(k)
    return out


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def load_env(path: Path) -> None:
    """Load KEY=value lines into os.environ without overriding what is set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        m = _ENV_LINE.match(line)
        if not m or line.strip().startswith("#"):
            continue
        os.environ.setdefault(m.group(1), m.group(2).strip().strip("'\""))


def key_for(provider: str) -> str | None:
    env = PROVIDERS[provider].key_env
    return os.environ.get(env) if env else None


def set_key(provider: str, value: str) -> None:
    """Write one provider's key to `.env` (chmod 600) and the live environment."""
    env = PROVIDERS[provider].key_env
    if env is None:
        raise ValueError(f"{PROVIDERS[provider].title} does not use a key")
    value = value.strip()
    with _lock:
        if _dir is not None:
            path = _dir / ".env"
            lines = []
            if path.exists():
                lines = [
                    l for l in path.read_text().splitlines()
                    if (m := _ENV_LINE.match(l)) is None or m.group(1) != env
                ]
            if value:
                lines.append(f"{env}={value}")
            # Created 600 rather than chmod-ed after, so the key is never
            # briefly world-readable.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            path.chmod(0o600)
        if value:
            os.environ[env] = value
        else:
            os.environ.pop(env, None)
    from .providers import reset_clients

    reset_clients()


def redact_key(value: str | None) -> str:
    if not value:
        return ""
    return value[:4] + "…" + value[-4:] if len(value) > 12 else "…"


def secrets() -> list[str]:
    """Every configured key, for log redaction."""
    out = []
    for p in PROVIDERS.values():
        if p.key_env and (v := os.environ.get(p.key_env)) and len(v) >= 8:
            out.append(v)
    return out


def redact(text: str) -> str:
    for s in secrets():
        text = text.replace(s, redact_key(s))
    return text


class RedactingFilter(logging.Filter):
    """Scrub API keys out of every log record, whatever logged it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not secrets():
            return True
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        return True


def install_log_redaction() -> None:
    root = logging.getLogger()
    for h in root.handlers or []:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())
    if not any(isinstance(f, RedactingFilter) for f in root.filters):
        root.addFilter(RedactingFilter())


# --------------------------------------------------------------------------- #
# Roles and nicknames
# --------------------------------------------------------------------------- #

def role_spec(role: str) -> dict:
    with _lock:
        return dict(_ensure()["roles"][role])


def base_nick(spec: dict) -> str:
    nick = f"{spec['provider']}/{spec['model']}"
    if spec["provider"] == "openai_compatible":
        host = urlparse(spec.get("base_url") or "").netloc or "endpoint"
        nick = f"{spec['provider']}[{host}]/{spec['model']}"
    if spec["provider"] == "vertex":
        nick = f"vertex[{spec.get('project', '')}]/{spec['model']}"
    if spec.get("reasoning", "default") != "default":
        nick += f"@{spec['reasoning']}"
    return nick


def role_nick(role: str) -> str:
    nick = base_nick(role_spec(role))
    if role == "annotator_b" and nick == base_nick(role_spec("annotator_a")):
        nick += "#2"
    return nick


def annotators() -> tuple[str, str]:
    return (role_nick("annotator_a"), role_nick("annotator_b"))


def reasoning_model() -> str:
    return role_nick("reasoning")


def same_annotator_model() -> bool:
    return base_nick(role_spec("annotator_a")) == base_nick(role_spec("annotator_b"))


def spec_for(nick: str) -> dict | None:
    """The binding behind a nickname, if a role currently holds it."""
    for role in ROLES:
        if role_nick(role) == nick:
            return role_spec(role)
    return None


def strip_sample(nick: str) -> str:
    """`x#2` -> `x`: the same underlying model, for gating and pricing."""
    return nick.split("#", 1)[0]


def title(nick: str) -> str:
    """Human-readable, for anything a reviewer reads."""
    base = strip_sample(nick)
    provider, _, rest = base.partition("/")
    model, _, reasoning = rest.partition("@")
    ptitle = PROVIDERS.get(provider.split("[", 1)[0])
    out = model or nick
    if ptitle:
        out += f" ({ptitle.title})"
    if nick.endswith("#2"):
        out += ", 2nd sample"
    return out


def short(nick: str) -> str:
    """Compact form for progress lines."""
    base = strip_sample(nick)
    s = base.rsplit("/", 1)[-1]
    return s + (" #2" if nick.endswith("#2") else "")


# --------------------------------------------------------------------------- #
# Verification — the validation layer where the allowlist was
# --------------------------------------------------------------------------- #

def fingerprint(spec: dict) -> str:
    """Everything that decides whether a call works: binding plus the key."""
    key = key_for(spec["provider"]) or ""
    parts = [str(spec.get(k, "")) for k in
             ("provider", "model", "reasoning", "base_url", "project", "region")]
    parts.append(hashlib.sha256(key.encode()).hexdigest())
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def is_verified(role: str) -> bool:
    with _lock:
        spec = role_spec(role)
        return _ensure()["verified"].get(role) == fingerprint(spec)


def mark_verified(role: str, ok: bool) -> None:
    with _lock:
        st = _ensure()
        if ok:
            st["verified"][role] = fingerprint(role_spec(role))
        else:
            st["verified"].pop(role, None)
        _save()


def ready() -> bool:
    return all(is_verified(r) for r in ROLES)


def unverified_roles() -> list[str]:
    return [r for r in ROLES if not is_verified(r)]


def is_permitted(nick: str) -> bool:
    """A nickname may be called only while a *verified* role holds it."""
    for role in ROLES:
        if role_nick(role) == nick:
            return is_verified(role)
    return False


def set_role(role: str, spec: dict) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    spec = _clean(spec)
    if spec.get("provider") not in PROVIDERS:
        raise ValueError(f"unknown provider {spec.get('provider')!r}")
    if not spec.get("model"):
        raise ValueError("a model id is required")
    if spec["reasoning"] not in PROVIDERS[spec["provider"]].reasoning:
        spec["reasoning"] = "default"
    with _lock:
        _ensure()["roles"][role] = spec
        _save()


# --------------------------------------------------------------------------- #
# Prices and the cap
# --------------------------------------------------------------------------- #

def price(nick: str) -> tuple[float, float] | None:
    """USD per 1M tokens (input, output), or None if nobody has said."""
    base = strip_sample(nick)
    for role in ROLES:
        spec = role_spec(role)
        if base_nick(spec) == base:
            if spec.get("price_in") is not None and spec.get("price_out") is not None:
                return (spec["price_in"], spec["price_out"])
            return known_price(spec["provider"], spec["model"])
    # A nickname no role holds any more (history in the cache): best effort
    # from the suggestion table.
    provider, _, rest = base.partition("/")
    return known_price(provider.split("[", 1)[0], rest.split("@", 1)[0])


def spend_cap() -> float:
    with _lock:
        return float(_ensure().get("spend_cap_usd") or 0.0)


def set_spend_cap(value: float) -> None:
    if value < 0:
        raise ValueError("the cap cannot be negative")
    with _lock:
        _ensure()["spend_cap_usd"] = float(value)
        _save()
