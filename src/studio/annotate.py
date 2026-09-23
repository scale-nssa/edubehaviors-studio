"""Session-level annotation: one call per (session × criterion × model).

Spec §3. The model returns only the indices where the criterion fires, so there
is no per-utterance alignment to get wrong. Failure policy is upstream's (Q304):
retry an unparseable response, drop individually bad indices with a warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from . import prompts, telemetry
from .config import ANNOTATION_CONCURRENCY, MAX_CONCURRENT_ROUNDS, annotators
from .model_settings import short
from .corpus import Session
from .llm import ask_json, ask_text

log = logging.getLogger("studio.annotate")

# ONE executor for the whole process, not one per round. Twenty concurrent
# rounds used to mean ~320 worker threads, virtually all of them parked waiting
# for a model slot. Sized to the work that can actually be in flight.
#
# Rate limiting is NOT done here — `llm.gate` gates every call, including the
# reasoning tier. Gating in both places would silently halve throughput.
_POOL = ThreadPoolExecutor(
    max_workers=ANNOTATION_CONCURRENCY * 2 + 8,   # two annotators, always
    thread_name_prefix="annot",
)

# Round admission. Beyond this, rounds wait with a visible position rather than
# everybody crawling at once: a queue that is shown reads as a working system.
_round_gate = threading.Semaphore(MAX_CONCURRENT_ROUNDS)
_waiting_lock = threading.Lock()
_waiting: list[int] = []   # round ids waiting for admission, in arrival order

# The annotation cache dedupes SEQUENTIALLY: two reviewers who start the same
# (session, criterion text, model) in the same minute both miss and both pay.
# With a pinned 20-session corpus and criteria from the same seeding prompt,
# exact text matches across people are plausible. First caller does the work,
# the rest wait on its result.
_inflight_lock = threading.Lock()
_inflight: dict[tuple, "Future"] = {}


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    failed: int = 0
    status: str = "running"   # queued | running | complete | failed
    error: str = ""
    position: int = 0         # rounds ahead of this one, while queued
    started_at: float = field(default_factory=time.time)
    recent: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def tick(self, label: str, ok: bool = True) -> None:
        with self._lock:
            self.done += 1
            if not ok:
                self.failed += 1
            self.recent.append(label)
            del self.recent[:-6]

    def snapshot(self) -> dict:
        with self._lock:
            pct = int(100 * self.done / self.total) if self.total else 0
            return {
                "total": self.total,
                "done": self.done,
                "failed": self.failed,
                "pct": pct,
                "status": self.status,
                "error": self.error,
                "position": self.position,
                "elapsed_s": time.time() - self.started_at,
                "recent": list(reversed(self.recent)),
            }


# round_id -> Progress
JOBS: dict[int, Progress] = {}
_jobs_lock = threading.Lock()


def busy() -> bool:
    """Is any round annotating or queued right now?"""
    with _jobs_lock:
        return any(p.status in ("queued", "running") for p in JOBS.values())


def progress_for(round_id: int) -> Progress | None:
    with _jobs_lock:
        return JOBS.get(round_id)


_INDEX_ARRAY = re.compile(r'"true_indices"\s*:\s*\[(.*?)(?:\]|$)', re.DOTALL)


def salvage_true_indices(text: str) -> list[int] | None:
    """Recover the index list from a response whose JSON is malformed.

    Observed live at roughly 1 call in 20: a model emits a well-formed prefix and
    then breaks the array syntax mid-way. The payload shape here is so
    constrained — one key, one array of ints — that pulling the integers out is
    unambiguous, and it saves a full retry of a large session-level call.

    Returns None when there is nothing recognisable, so the caller still retries.
    """
    m = _INDEX_ARRAY.search(text or "")
    if not m:
        return None
    nums = re.findall(r"-?\d+", m.group(1))
    return [int(n) for n in nums] if nums else []


def parse_true_indices(payload: dict) -> list[int]:
    """Strict parse. Raises so the caller retries rather than accepting silence."""
    if "true_indices" not in payload:
        raise ValueError("response JSON missing 'true_indices'")
    raw = payload["true_indices"]
    if not isinstance(raw, list):
        raise ValueError("'true_indices' must be an array")
    out: list[int] = []
    for entry in raw:
        # Accept both the bare-int form and upstream's {"idx": n} form.
        if isinstance(entry, bool):
            raise ValueError("index must be an integer, got bool")
        if isinstance(entry, int):
            out.append(entry)
        elif isinstance(entry, dict) and "idx" in entry:
            idx = entry["idx"]
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError("'idx' must be an integer")
            out.append(idx)
        else:
            raise ValueError(f"unrecognised true_indices entry: {entry!r}")
    return out


def validate_indices(
    raw: list[int], candidates: set[int], all_indices: set[int]
) -> tuple[list[int], dict[str, int]]:
    """Drop bad indices with counted warnings; never raise.

    An out-of-range or out-of-scope index is a model mistake, not signal.
    """
    kept: list[int] = []
    seen: set[int] = set()
    warn = {"out_of_range": 0, "out_of_scope": 0, "duplicate": 0}
    for i in raw:
        if i in seen:
            warn["duplicate"] += 1
            continue
        seen.add(i)
        if i not in all_indices:
            warn["out_of_range"] += 1
            continue
        if i not in candidates:
            warn["out_of_scope"] += 1
            continue
        kept.append(i)
    return sorted(kept), warn


def _call_id(session_id: str, criterion_text: str, nick: str) -> str:
    h = hashlib.sha256(f"{session_id}\x00{criterion_text}\x00{nick}".encode()).hexdigest()
    return h[:16]


def annotate_one(
    conn: sqlite3.Connection,
    conn_lock: threading.Lock,
    session: Session,
    criterion_text: str,
    nick: str,
    scope: str,
    parent_text: str | None = None,
    parent_hits: list[int] | None = None,
) -> list[int]:
    """Annotate one (session, criterion, model), using the cache when present.

    With `parent_text`, this is a sub-criterion: it is judged only over the
    utterances the parent already selected, and cached under a key that carries
    the parent so rewording the parent cannot leave a stale child behind.
    """
    from .service import annotation_key

    key = annotation_key(criterion_text, parent_text)
    with conn_lock:
        row = conn.execute(
            "SELECT true_indices FROM annotation WHERE session_id=? AND criterion_text=? AND model=?",
            (session.session_id, key, nick),
        ).fetchone()
    if row:
        return json.loads(row["true_indices"])

    # Cache miss. Claim this (session, criterion, model) so a concurrent caller
    # waits for our answer instead of paying for the same call.
    slot = (session.session_id, key, nick)
    with _inflight_lock:
        pending = _inflight.get(slot)
        if pending is None:
            mine = Future()
            _inflight[slot] = mine
        else:
            mine = None
    if mine is None:
        return pending.result()   # someone else is already asking
    try:
        result = _annotate_uncached(
            conn, conn_lock, session, criterion_text, nick, scope, key,
            parent_text, parent_hits,
        )
    except BaseException as exc:
        with _inflight_lock:
            _inflight.pop(slot, None)
        mine.set_exception(exc)
        raise
    with _inflight_lock:
        _inflight.pop(slot, None)
    mine.set_result(result)
    return result


def _annotate_uncached(
    conn, conn_lock, session, criterion_text, nick, scope, key,
    parent_text=None, parent_hits=None,
) -> list[int]:
    """The call itself. Split out so `annotate_one` can own the in-flight map."""

    if parent_text is not None:
        if not parent_hits:
            return []  # gate never fired: nothing to ask about, and no call to make
        prompt = prompts.annotate_child(
            session, criterion_text, parent_text, sorted(parent_hits), scope
        )
    else:
        prompt = prompts.annotate(session, criterion_text, scope)
    # No semaphore here: `llm.gate` admits calls per model for every caller.
    try:
        payload, tin, tout = ask_json(nick, prompt, profile="annotate")
        raw = parse_true_indices(payload)
    except ValueError:
        # JSON was malformed. Before paying for a full retry, try to salvage
        # the index list directly out of the text.
        text, tin, tout = ask_text(nick, prompt, profile="annotate", retries=1)
        salvaged = salvage_true_indices(text)
        if salvaged is None:
            raise
        log.info("salvaged %d indices from malformed %s response", len(salvaged), nick)
        raw = salvaged
    all_idx = {u.index for u in session.utterances}
    # A child may only fire where its parent did. Anything else the model
    # returns is dropped as out-of-scope, same as a student index under a tutor
    # scope — the gate is not advisory.
    cand = (
        set(parent_hits)
        if parent_text is not None
        else {u.index for u in session.in_scope(scope)}
    )
    kept, warn = validate_indices(raw, cand, all_idx)
    if any(warn.values()):
        log.info("validation warnings for %s/%s: %s", session.session_id, nick, warn)

    with conn_lock:
        conn.execute(
            "INSERT OR REPLACE INTO annotation "
            "(session_id, criterion_text, model, true_indices, call_id, warnings, "
            " input_tokens, output_tokens) VALUES (?,?,?,?,?,?,?,?)",
            (
                session.session_id,
                key,
                nick,
                json.dumps(kept),
                _call_id(session.session_id, key, nick),
                json.dumps(warn),
                tin or 0,
                tout or 0,
            ),
        )
        conn.commit()
    return kept


def run_round(
    cfg,
    round_id: int,
    session: Session,
    criteria: list,
    scope: str,
    on_done=None,
) -> Progress:
    """Annotate every (criterion × model) for one session, concurrently.

    `criteria` are `(text, parent_text_or_None)` pairs. Parents run first and in
    parallel; each child then runs against its own parent's firing set, which is
    per-model — the two annotators may gate on different utterances, and that is
    genuine signal rather than something to average away.

    Sessions are never partially annotated (Q302/Q220): the whole grid runs
    before anything downstream reads it.

    Takes `cfg` rather than a connection: every worker resolves its **own**
    thread-local connection via `service.bg_conn`. Handing one connection to a
    pool of threads would put concurrent statements on a single sqlite3 object,
    which is the thing the per-thread connections exist to avoid. Workers are
    reused, so the number of connections is bounded by the pool, not by calls.
    """
    from .service import bg_conn
    tops = [t for t, parent in criteria if parent is None]
    kids = [(t, parent) for t, parent in criteria if parent is not None]

    prog = Progress(total=len(criteria) * len(annotators()))
    with _jobs_lock:
        JOBS[round_id] = prog

    def work_top(text: str, nick: str) -> None:
        conn, conn_lock = bg_conn(cfg)
        try:
            hits = annotate_one(conn, conn_lock, session, text, nick, scope)
            prog.tick(f"{text[:44]}… → {len(hits)} hits ({short(nick)})")
        except Exception as exc:  # noqa: BLE001
            log.exception("annotation failed: %s / %s", text[:60], nick)
            prog.tick(f"FAILED: {text[:44]}… ({exc})", ok=False)

    def work_child(text: str, parent_text: str, nick: str) -> None:
        conn, conn_lock = bg_conn(cfg)
        try:
            with conn_lock:
                row = conn.execute(
                    "SELECT true_indices FROM annotation "
                    "WHERE session_id=? AND criterion_text=? AND model=?",
                    (session.session_id, parent_text, nick),
                ).fetchone()
            gate = json.loads(row["true_indices"]) if row else []
            hits = annotate_one(
                conn, conn_lock, session, text, nick, scope,
                parent_text=parent_text, parent_hits=gate,
            )
            prog.tick(
                f"↳ {text[:40]}… → {len(hits)}/{len(gate)} ({short(nick)})"
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("child annotation failed: %s / %s", text[:60], nick)
            prog.tick(f"FAILED: ↳ {text[:40]}… ({exc})", ok=False)

    def driver() -> None:
        # Wait for admission first, advertising our place in the queue. The
        # driver thread is a plain Thread, never a _POOL worker: a pool worker
        # blocking on futures from the same pool can deadlock once every worker
        # is a driver.
        with _waiting_lock:
            _waiting.append(round_id)
        try:
            while not _round_gate.acquire(timeout=0.5):
                with _waiting_lock:
                    prog.position = max(0, _waiting.index(round_id))
                prog.status = "queued"
        finally:
            with _waiting_lock:
                if round_id in _waiting:
                    _waiting.remove(round_id)
        prog.status = "running"
        prog.position = 0

        try:
            # Parents first, all of them, then children — a child gates on its
            # parent's firing set. The barrier is per round and must stay per
            # round; it must not become a global barrier across users.
            for f in [_POOL.submit(work_top, t, n) for t in tops for n in annotators()]:
                f.result()
            if kids:
                for f in [
                    _POOL.submit(work_child, t, p, n) for t, p in kids for n in annotators()
                ]:
                    f.result()
            prog.status = "complete"
            telemetry.push_event("round_finished", round_id=round_id,
                                 done=prog.done, failed=prog.failed)
            if on_done:
                on_done(prog)
        except Exception as exc:  # noqa: BLE001
            prog.status = "failed"
            prog.error = str(exc)
            telemetry.push_event("round_failed", round_id=round_id, error=str(exc))
            log.exception("round %s failed", round_id)
        finally:
            _round_gate.release()

    telemetry.push_event("round_started", round_id=round_id,
                         session=session.session_id, calls=prog.total)
    threading.Thread(target=driver, daemon=True, name=f"round-{round_id}").start()
    return prog


def firing_matrix(
    conn: sqlite3.Connection,
    session_id: str,
    criteria: list[tuple],
    nick: str,
) -> dict[int, dict[int, bool]]:
    """utterance_index -> {criterion_id: bool} for one model.

    Accepts `(id, text)` or `(id, text, parent_text)`; children are looked up
    under the composite key that carries their parent.
    """
    from .service import annotation_key

    out: dict[int, dict[int, bool]] = {}
    for entry in criteria:
        cid, text = entry[0], entry[1]
        parent_text = entry[2] if len(entry) > 2 else None
        row = conn.execute(
            "SELECT true_indices FROM annotation WHERE session_id=? AND criterion_text=? AND model=?",
            (session_id, annotation_key(text, parent_text), nick),
        ).fetchone()
        hits = set(json.loads(row["true_indices"])) if row else set()
        for idx in hits:
            out.setdefault(idx, {})[cid] = True
    return out
