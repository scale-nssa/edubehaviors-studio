"""In-process telemetry for the control panel.

One lock, a few counters and a ring buffer. Everything here is process-local and
dies with the process — that is fine, because the panel is a live view and the
durable record is the database.

Hooks are deliberately one line each at the call site (`llm.py` around the gated
call, `service.py` at round lifecycle points) so this module stays easy to merge
around. See `docs/multi-user/02-concurrency.md §7`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

MAX_EVENTS = 200


@dataclass
class ModelStats:
    in_flight: int = 0
    completed: int = 0
    failed: int = 0
    last_latency_s: float | None = None
    # Latency matters more than errors for spotting a rate limit: `ask_json`
    # backs off 2/4/8/20s, so throttling shows up as a climbing number here
    # long before it shows up as a failure.
    total_latency_s: float = 0.0

    @property
    def mean_latency_s(self) -> float | None:
        return self.total_latency_s / self.completed if self.completed else None


_lock = threading.Lock()
_models: dict[str, ModelStats] = {}
_events: deque = deque(maxlen=MAX_EVENTS)
_touched: dict[int, float] = {}
_started_at = time.time()


# --------------------------------------------------------------------------- #
# Model call accounting
# --------------------------------------------------------------------------- #

def call_started(nick: str) -> float:
    with _lock:
        _models.setdefault(nick, ModelStats()).in_flight += 1
    return time.time()


def call_finished(nick: str, t0: float, ok: bool = True) -> None:
    dt = time.time() - t0
    with _lock:
        st = _models.setdefault(nick, ModelStats())
        st.in_flight = max(0, st.in_flight - 1)
        if ok:
            st.completed += 1
            st.last_latency_s = dt
            st.total_latency_s += dt
        else:
            st.failed += 1


def model_stats() -> dict[str, dict]:
    with _lock:
        return {
            nick: {
                "in_flight": s.in_flight,
                "completed": s.completed,
                "failed": s.failed,
                "last_latency_s": s.last_latency_s,
                "mean_latency_s": s.mean_latency_s,
            }
            for nick, s in sorted(_models.items())
        }


def total_in_flight() -> int:
    with _lock:
        return sum(s.in_flight for s in _models.values())


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

def push_event(kind: str, **fields) -> None:
    """Append to the ring buffer. Never raises — telemetry must not break work."""
    try:
        with _lock:
            _events.appendleft({"t": time.time(), "kind": kind, **fields})
    except Exception:  # noqa: BLE001
        pass


def events(limit: int = 50) -> list[dict]:
    with _lock:
        return list(_events)[:limit]


def uptime_s() -> float:
    return time.time() - _started_at


# --------------------------------------------------------------------------- #
# Presence throttle
# --------------------------------------------------------------------------- #

def should_touch(reviewer_id: int, every: float = 30.0) -> bool:
    """True at most once per `every` seconds per reviewer.

    Presence is written from `before_request`. Twenty people with HTMX polling
    every 1.2s would otherwise mean ~17 writes/second to one table for a number
    nobody reads more than once every two seconds.
    """
    now = time.time()
    with _lock:
        last = _touched.get(reviewer_id, 0.0)
        if now - last < every:
            return False
        _touched[reviewer_id] = now
        return True


def reset() -> None:
    """Test helper."""
    with _lock:
        _models.clear()
        _events.clear()
        _touched.clear()
