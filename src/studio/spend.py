"""What this data directory has spent, and the cap that stops it spending more.

Hosted, cost was the operator's concern and lived in process memory for the
panel. Locally it is the user's own card, so every call is written to the
`spend_log` table and the figure survives a restart.

Every number here is an **estimate**: token counts come from the provider's
usage report, prices from the Models page (or a built-in list price), and a
call that fails before reporting usage is not counted at all. The UI says so
wherever it shows one.

The cap is enforced in two places. Routes refuse to *start* a paid action whose
estimate would cross it (`would_exceed`), and `llm` refuses every individual
call once the ledger is already past it (`check`) — the second catches spend
nobody clicked for, such as chained sessions and orphan recovery after a
restart.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from . import model_settings

_db_path: Path | None = None
_local = threading.local()
_started = time.time()
_session_lock = threading.Lock()
_session_usd = 0.0


class SpendCapReached(RuntimeError):
    """Raised instead of making a call that the spend cap forbids."""


def init(db_path: Path) -> None:
    global _db_path
    _db_path = Path(db_path)
    _local.__dict__.clear()


def _conn() -> sqlite3.Connection | None:
    if _db_path is None:
        return None
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != _db_path:
        from .db import connect

        conn = _local.conn = connect(_db_path)
        _local.path = _db_path
    return conn


def cost(nick: str, input_tokens: int, output_tokens: int) -> float | None:
    """None when the price is unknown — not zero, which would read as free."""
    p = model_settings.price(nick)
    if p is None:
        return None
    return (input_tokens / 1e6) * p[0] + (output_tokens / 1e6) * p[1]


def record(nick: str, profile: str, input_tokens: int, output_tokens: int,
           thinking_tokens: int | None) -> None:
    global _session_usd
    usd = cost(nick, input_tokens, output_tokens)
    with _session_lock:
        _session_usd += usd or 0.0
    conn = _conn()
    if conn is None:
        return
    conn.execute(
        "INSERT INTO spend_log (model, profile, input_tokens, output_tokens, "
        "thinking_tokens, cost_usd) VALUES (?,?,?,?,?,?)",
        (nick, profile, input_tokens, output_tokens, thinking_tokens, usd),
    )
    conn.commit()


def lifetime_usd() -> float:
    conn = _conn()
    if conn is None:
        return 0.0
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) s FROM spend_log").fetchone()
    return float(row["s"] or 0.0)


def session_usd() -> float:
    """Since this server process started."""
    with _session_lock:
        return _session_usd


def unpriced_calls() -> int:
    conn = _conn()
    if conn is None:
        return 0
    return conn.execute(
        "SELECT COUNT(*) n FROM spend_log WHERE cost_usd IS NULL"
    ).fetchone()["n"]


def averages(nick: str, profile: str) -> tuple[float, float] | None:
    """Mean (input, output) tokens per call for one model and profile, from
    what this data directory has actually seen. None until there is history."""
    conn = _conn()
    if conn is None:
        return None
    row = conn.execute(
        "SELECT COUNT(*) n, AVG(input_tokens) i, AVG(output_tokens) o FROM spend_log "
        "WHERE model=? AND profile=?", (nick, profile),
    ).fetchone()
    if not row or not row["n"]:
        return None
    return float(row["i"] or 0), float(row["o"] or 0)


def remaining_usd() -> float:
    return model_settings.spend_cap() - lifetime_usd()


def would_exceed(estimate_usd: float) -> bool:
    return lifetime_usd() + estimate_usd > model_settings.spend_cap()


def check() -> None:
    cap = model_settings.spend_cap()
    spent = lifetime_usd()
    if spent >= cap:
        raise SpendCapReached(
            f"Spend cap reached: an estimated ${spent:.2f} spent against a cap of "
            f"${cap:.2f}. Raise it on the Models page to continue."
        )
