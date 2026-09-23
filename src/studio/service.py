"""Domain operations shared by the routes. Keeps app.py about HTTP."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading

from . import annotate, corpus, prompts, selection, telemetry
from .config import (
    MAX_CHAINED_SESSIONS,
    MIN_POSITIVES_PER_LABEL,
    Config,
    annotators,
    reasoning_model,
)
from .db import jload, snapshot_schema
from .llm import ask_json
from .schema_map import Criterion, Edge, apply_map, validate_edges

log = logging.getLogger("studio.service")

# Background threads need their own long-lived connection, because Flask's
# request-scoped `g.db` dies with the request (spec §8).
#
# ONE CONNECTION PER THREAD, not one per process. The process-wide connection
# and lock this replaces serialised every background write in the app behind
# one mutex — and two callers held it across seconds of real work
# (`_materialize_selection` building firing matrices, `finish` finalising), so
# with fifteen rounds finishing in the same minute nobody else could write.
#
# SQLite in WAL mode with busy_timeout=30000 (see `db.connect`) already handles
# concurrent writers: readers never block and writers serialise per statement,
# not per function. The per-thread lock below is therefore uncontended — it
# exists only so the `(conn, lock)` shape and every `with lock:` call site stay
# exactly as written.
_local = threading.local()


def bg_conn(cfg: Config) -> tuple[sqlite3.Connection, threading.RLock]:
    """This thread's background connection. **Call it on the thread that uses
    it** — never capture the tuple on the request thread for a callback that
    runs elsewhere, or you hand one thread another thread's connection."""
    if getattr(_local, "conn", None) is None:
        from .db import connect

        _local.conn = connect(cfg.db_path)
        _local.lock = threading.RLock()
    return _local.conn, _local.lock


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #

def load_criteria(conn, construct_id: int, *, status: str | None = "active") -> list[Criterion]:
    q = "SELECT id, text, edges, parent_id FROM criterion WHERE construct_id=?"
    args: list = [construct_id]
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY ord, id"
    return [
        Criterion(
            id=r["id"],
            text=r["text"],
            # Stored edges are `{"label": ...}`. A legacy row may still carry a
            # "kind" key from the typed-edge era; it is ignored, not migrated.
            edges=tuple(Edge(e["label"]) for e in jload(r["edges"], [])),
            parent_id=r["parent_id"],
        )
        for r in conn.execute(q, args).fetchall()
    ]


def criteria_keys(criteria) -> list[tuple]:
    """(id, text, parent_text) for each criterion, for firing_matrix lookups."""
    by_id = {c.id: c for c in criteria}
    out = []
    for c in criteria:
        parent = by_id.get(c.parent_id) if c.parent_id else None
        out.append((c.id, c.text, parent.text if parent else None))
    return out


def load_rules(conn, construct_id: int):
    """The construct's boundary rules, or an empty set if no catch-all is named."""
    from .schema_map import LabelRules

    row = conn.execute(
        "SELECT other_label FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    other = row["other_label"] if row and "other_label" in row.keys() else None
    pairs = tuple(
        (r["winner"], r["loser"])
        for r in conn.execute(
            "SELECT winner, loser FROM label_rule WHERE construct_id=? ORDER BY id",
            (construct_id,),
        )
    )
    if not other and not pairs:
        return None  # no rules at all: behave exactly as before
    return LabelRules(other_label=other, priorities=pairs)


def annotation_key(text: str, parent_text: str | None) -> str:
    """Cache key for one criterion's annotation.

    A child's answer depends on which utterances the parent selected, so the key
    has to carry the parent. Without this, rewording a parent would leave every
    child's cached firing intact and silently wrong, since the plain key is
    unchanged.
    """
    return text if not parent_text else f"{text}\n\u2282 {parent_text}"


# construct_id -> {"status": running|complete|failed, "error": str, "n": int}
SEED_JOBS: dict[int, dict] = {}
_seed_lock = threading.Lock()


def seed_status(construct_id: int) -> dict:
    with _seed_lock:
        return dict(SEED_JOBS.get(construct_id, {"status": "complete", "n": 0, "error": ""}))


def seed_criteria_async(cfg: Config, construct_id: int) -> None:
    """Seed in the background.

    P1 is a reasoning call and can take minutes. Run synchronously inside the
    form POST it produces a dead browser tab with no indication anything is
    happening — observed at ~5 minutes in a live session.
    """
    with _seed_lock:
        SEED_JOBS[construct_id] = {"status": "running", "n": 0, "error": ""}

    def work() -> None:
        conn, lock = bg_conn(cfg)
        try:
            with lock:
                n = seed_criteria(conn, cfg, construct_id)
            with _seed_lock:
                SEED_JOBS[construct_id] = {"status": "complete", "n": n, "error": ""}
        except Exception as exc:  # noqa: BLE001
            with _seed_lock:
                SEED_JOBS[construct_id] = {"status": "failed", "n": 0, "error": str(exc)}

    threading.Thread(target=work, daemon=True, name=f"seed-{construct_id}").start()


def seed_criteria(conn, cfg: Config, construct_id: int) -> int:
    row = conn.execute(
        "SELECT description, label_space, scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    labels = jload(row["label_space"], [])
    payload, _, _ = ask_json(
        reasoning_model(),
        prompts.seed_criteria(row["description"], labels, row["scope"]),
    )
    def _edges(raw):
        seen: list[str] = []
        for e in (raw or []):
            lbl = e.get("label")
            if lbl in labels and lbl not in seen:
                seen.append(lbl)
        out = [{"label": lbl} for lbl in seen]
        problems = validate_edges([Edge(e["label"]) for e in out], labels)
        if problems and out:
            # Still broken somehow; keep the first edge rather than discarding a
            # plausible criterion outright.
            out = out[:1]
        return out

    def _insert(text, edges, ord_, parent_id=None):
        cur = conn.execute(
            "INSERT INTO criterion (construct_id, text, edges, status, origin, ord, parent_id) "
            "VALUES (?,?,?,'pending','seed',?,?)",
            (construct_id, text, json.dumps(edges), ord_, parent_id),
        )
        return cur.lastrowid

    n = 0
    for i, c in enumerate(payload.get("criteria", [])):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        kids = c.get("children") or []
        edges = _edges(c.get("edges"))
        if kids:
            # A gate carries no edges by construction; drop any the model added.
            pid = _insert(text, [], i, None)
            n += 1
            for j, k in enumerate(kids):
                ktext = (k.get("text") or "").strip()
                kedges = _edges(k.get("edges"))
                if not ktext or not kedges:
                    continue
                _insert(ktext, kedges, i * 100 + j, pid)
                n += 1
        elif edges:
            _insert(text, edges, i)
            n += 1
    conn.commit()
    return n


# --------------------------------------------------------------------------- #
# Rounds
# --------------------------------------------------------------------------- #

def criterion_texts_at(conn, construct_id: int, version: int) -> dict[int, str]:
    """criterion id -> its wording at a given schema version, from the snapshot.

    Used to decide whether a human verdict still applies. A verdict is about a
    criterion's WORDING, not its id: reword a criterion and the fire rate and
    alpha reset on their own, because both are keyed on criterion text and the
    new text has no annotations. Anything keyed on id has to check this
    explicitly or it silently compares old judgements against new wording.
    """
    row = conn.execute(
        "SELECT snapshot FROM schema_version WHERE construct_id=? AND version=?",
        (construct_id, version),
    ).fetchone()
    snap = jload(row["snapshot"], {}) if row else {}
    return {c["id"]: c["text"] for c in snap.get("criteria", [])}


def live_verdicts(
    conn, construct_id: int, verdicts: dict, version: int, criteria: list[Criterion]
) -> dict[int, bool]:
    """The verdicts in `verdicts` that still describe the current wording.

    `verdicts` is the raw `review.criterion_verdicts` map (string keys),
    `version` the schema version of the round it was recorded under. Anything
    unconfirmable is dropped: under-reporting matches what alpha does when it
    cannot be computed, and over-reporting here produced a model-human alpha
    of 0.55 where every comparable criterion agreed perfectly, dragged down by
    two reworded criteria whose new text had no annotations to agree with.
    """
    then = criterion_texts_at(conn, construct_id, version)
    now = {c.id: c.text for c in criteria}
    out: dict[int, bool] = {}
    for k, v in (verdicts or {}).items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        if cid in now and then.get(cid) == now[cid]:
            out[cid] = bool(v)
    return out


def criterion_stats(conn, cfg: Config, construct_id: int) -> dict[int, dict]:
    """Per-criterion fire rate, inter-model alpha, and human agreement.

    Computed over every session this construct has annotated. Feeds the schema
    table (response plan §1.1) and, via `SchemaView.summary_lines`, P4.
    """
    from .metrics import _alpha

    criteria = load_criteria(conn, construct_id, status="active")
    scope = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["scope"]
    sessions = sorted(used_sessions(conn, construct_id))

    by_id = {c.id: c for c in criteria}
    # Read the binding once. Asked per utterance, a rebind on the Models page
    # mid-computation could pair one model's hits with another's.
    models = annotators()
    out: dict[int, dict] = {}
    for c in criteria:
        # A child is only defined where its parent fired. Scoring it over the
        # whole session would show a near-zero fire rate and near-perfect
        # agreement for every child, which says nothing.
        parent = by_id.get(c.parent_id) if c.parent_id else None
        a_vec, b_vec = [], []
        for sid in sessions:
            hits = {}
            for nick in models:
                row = conn.execute(
                    "SELECT true_indices FROM annotation "
                    "WHERE session_id=? AND criterion_text=? AND model=?",
                    (sid, annotation_key(c.text, parent.text if parent else None), nick),
                ).fetchone()
                if row is None:
                    break
                hits[nick] = set(jload(row["true_indices"], []))
            if len(hits) != len(models):
                continue
            applicable = None
            if parent is not None:
                applicable = set()
                for nick in models:
                    prow = conn.execute(
                        "SELECT true_indices FROM annotation "
                        "WHERE session_id=? AND criterion_text=? AND model=?",
                        (sid, parent.text, nick),
                    ).fetchone()
                    applicable |= set(jload(prow["true_indices"], [])) if prow else set()
            for u in corpus.get(cfg, sid).in_scope(scope):
                if applicable is not None and u.index not in applicable:
                    continue
                a_vec.append(int(u.index in hits[models[0]]))
                b_vec.append(int(u.index in hits[models[1]]))

        fires = (
            sum(1 for a, b in zip(a_vec, b_vec) if a or b) / len(a_vec)
            if a_vec else None
        )
        out[c.id] = {
            "fires": fires,
            "alpha": _alpha(a_vec, b_vec) if a_vec else None,
            "human_flips": 0,
            "human_total": 0,
        }

    # A verdict is about a criterion's WORDING, not its id. Reword a criterion
    # and the fire rate and alpha reset on their own, because both are keyed on
    # criterion text and the new text has no annotations yet. The human column
    # did not: verdicts are keyed by id, so "3 / 20" carried over from wording
    # nobody had judged any more. Count a verdict only where the snapshot for
    # the round it was recorded under still shows the current text.
    for rv in conn.execute(
        "SELECT rv.criterion_verdicts, rv.utterance_index, r.session_id, "
        "r.schema_version FROM review rv JOIN round r ON r.id = rv.round_id "
        "WHERE r.construct_id=? AND rv.complete=1",
        (construct_id,),
    ):
        verdicts = live_verdicts(
            conn, construct_id, jload(rv["criterion_verdicts"], {}),
            rv["schema_version"], criteria,
        )
        for c in criteria:
            if c.id not in verdicts:
                continue
            fired = False
            for nick in models:
                # annotation_key, not c.text: a sub-criterion is stored under
                # its parent-qualified key, so a bare-text lookup missed every
                # one of them and scored them all as "never fired".
                parent = by_id.get(c.parent_id) if c.parent_id else None
                row = conn.execute(
                    "SELECT true_indices FROM annotation "
                    "WHERE session_id=? AND criterion_text=? AND model=?",
                    (
                        rv["session_id"],
                        annotation_key(c.text, parent.text if parent else None),
                        nick,
                    ),
                ).fetchone()
                if row and rv["utterance_index"] in set(jload(row["true_indices"], [])):
                    fired = True
            out[c.id]["human_total"] += 1
            out[c.id]["human_flips"] += int(fired != verdicts[c.id])
    return out


def build_schema_view(conn, cfg: Config, construct_id: int, *, with_stats: bool = True):
    from . import schema_view as sv

    c = conn.execute("SELECT label_space FROM construct WHERE id=?", (construct_id,)).fetchone()
    criteria = load_criteria(conn, construct_id, status="active")
    stats = criterion_stats(conn, cfg, construct_id) if with_stats else {}
    return sv.build(criteria, jload(c["label_space"], []), stats=stats)


# --------------------------------------------------------------------------- #
# Session-scoped gold (notes-3)
#
# Reviews are stored per (round, utterance), but a human judgement about an
# utterance is a fact about the SESSION, not about the round that happened to
# surface it. Keyed by round, re-running a session orphaned everything you had
# already judged, which made "perfect this session" impossible.
# --------------------------------------------------------------------------- #

def criteria_from_snapshot(snapshot: dict) -> list[Criterion]:
    """Rebuild the criterion set as it stood at some schema version.

    Annotations are cached on criterion *text*, so an old schema can be replayed
    over a session with no new model calls.
    """
    out = []
    for c in snapshot.get("criteria", []):
        if c.get("status") == "excluded":
            continue
        edges = c["edges"] if not isinstance(c["edges"], str) else json.loads(c["edges"])
        out.append(
            Criterion(
                id=c["id"],
                text=c["text"],
                edges=tuple(Edge(e["label"]) for e in edges),
            )
        )
    return out


def session_gold(conn, construct_id: int, session_id: str) -> dict[int, dict]:
    """Every human judgement recorded on this session, latest per utterance.

    Per-criterion verdicts are marked stale when the criterion's text has
    changed since the verdict was given — the label survives a reword, the
    verdict about specific wording does not.
    """
    current = {
        c.id: c.text for c in load_criteria(conn, construct_id, status=None)
    }
    snaps: dict[int, dict] = {}

    out: dict[int, dict] = {}
    for rv in conn.execute(
        "SELECT rv.*, r.schema_version FROM review rv "
        "JOIN round r ON r.id = rv.round_id "
        "WHERE r.construct_id=? AND r.session_id=? AND rv.complete=1 "
        "ORDER BY rv.id",
        (construct_id, session_id),
    ):
        v = rv["schema_version"]
        if v not in snaps:
            row = conn.execute(
                "SELECT snapshot FROM schema_version WHERE construct_id=? AND version=?",
                (construct_id, v),
            ).fetchone()
            snaps[v] = jload(row["snapshot"], {}) if row else {}
        then = {c["id"]: c["text"] for c in snaps[v].get("criteria", [])}

        verdicts, stale = {}, {}
        for cid_s, val in jload(rv["criterion_verdicts"], {}).items():
            cid = int(cid_s)
            was, now = then.get(cid), current.get(cid)
            if now is None or (was is not None and was != now):
                stale[cid] = bool(val)
            else:
                verdicts[cid] = bool(val)

        out[rv["utterance_index"]] = {
            "gold_label": rv["gold_label"],
            "verdicts": verdicts,
            "stale_verdicts": stale,
            "note": rv["note"],
            "schema_version": v,
            "round_id": rv["round_id"],
        }
    return out


def _consensus_over(conn, cfg, session_id, scope, criteria):
    firings = {
        nick: annotate.firing_matrix(
            conn, session_id, criteria_keys(criteria), nick
        )
        for nick in annotators()
    }
    return firings, {
        u.index: consensus_firing(firings, u.index, criteria)
        for u in corpus.get(cfg, session_id).in_scope(scope)
    }


def evaluate_gold(conn, cfg, construct_id, session_id, criteria, gold, rules="auto") -> dict:
    """How the given schema routes the utterances you have judged.

    Rules count here: a boundary rule that routes correctly is a real
    improvement, and it is part of the schema you would export.
    """
    from collections import Counter

    if rules == "auto":
        rules = load_rules(conn, construct_id)
    scope = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["scope"]
    _, consensus = _consensus_over(conn, cfg, session_id, scope, criteria)

    per: dict[int, str] = {}
    counts: Counter = Counter()
    for idx, g in gold.items():
        if not g["gold_label"]:
            continue
        res = apply_map(criteria, consensus.get(idx, {}), rules)
        cands = set(res.candidates)
        if cands == {g["gold_label"]}:
            outcome = "exact"
        elif not cands:
            outcome = "under"
        elif g["gold_label"] in cands:
            outcome = "over_contains"
        else:
            outcome = "wrong"
        per[idx] = outcome
        counts[outcome] += 1

    n = sum(counts.values())
    return {
        "per_utterance": per,
        "counts": dict(counts),
        "n": n,
        "exact": counts.get("exact", 0),
        "pct": (counts.get("exact", 0) / n) if n else None,
    }


def compare_schema_versions(
    conn, cfg, construct_id: int, session_id: str, old_version: int
) -> dict | None:
    """Replay this session's gold under an older schema and under the current one.

    Free: schema versions are full snapshots and annotations are cached by
    criterion text, so nothing needs re-annotating.
    """
    row = conn.execute(
        "SELECT snapshot FROM schema_version WHERE construct_id=? AND version=?",
        (construct_id, old_version),
    ).fetchone()
    if row is None:
        return None
    gold = session_gold(conn, construct_id, session_id)
    if not gold:
        return None

    old_criteria = criteria_from_snapshot(jload(row["snapshot"], {}))
    new_criteria = load_criteria(conn, construct_id, status="active")
    before = evaluate_gold(conn, cfg, construct_id, session_id, old_criteria, gold)
    after = evaluate_gold(conn, cfg, construct_id, session_id, new_criteria, gold)

    # Gold is ~10 utterances; the session is ~300. Measuring only the gold
    # subset let a round that degraded whole-session consistency from 193 to 179
    # report "no change" — because none of the ten judged utterances happened to
    # cross into or out of exact.
    from .metrics import schema_logic

    scope = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["scope"]
    universe = [u.index for u in corpus.get(cfg, session_id).in_scope(scope)]
    _, cons_old = _consensus_over(conn, cfg, session_id, scope, old_criteria)
    _, cons_new = _consensus_over(conn, cfg, session_id, scope, new_criteria)
    rules = load_rules(conn, construct_id)
    logic_before = schema_logic(old_criteria, cons_old, universe, rules)
    logic_after = schema_logic(new_criteria, cons_new, universe, rules)

    fixed, regressed = [], []
    for idx, out_after in after["per_utterance"].items():
        out_before = before["per_utterance"].get(idx)
        if out_before == out_after:
            continue
        moved = {"index": idx, "before": out_before, "after": out_after}
        if out_after == "exact":
            fixed.append(moved)
        elif out_before == "exact":
            regressed.append(moved)

    wd_before = logic_before["counts"]["determined"]
    wd_after = logic_after["counts"]["determined"]
    return {
        "old_version": old_version,
        "before": before,
        "after": after,
        "fixed": sorted(fixed, key=lambda d: d["index"]),
        "regressed": sorted(regressed, key=lambda d: d["index"]),
        "n_utterances": len(universe),
        "welldef_before": wd_before,
        "welldef_after": wd_after,
        "welldef_delta": wd_after - wd_before,
        # The headline judgement. Gold movement alone is too small a sample to
        # carry it, so the whole-session numbers get a veto.
        "verdict": (
            "worse" if wd_after < wd_before
            else "better" if (wd_after > wd_before or after["exact"] > before["exact"])
            else "no change"
        ),
    }


def criterion_blame(conn, cfg, construct_id, session_id, criteria, gold, per_utterance):
    """For misrouted utterances, which criteria disagree with your corrections.

    The "up to assertion level variability" drill-down: turns "6 wrong" into
    "criterion #7 accounts for 4 of them".
    """
    from collections import Counter

    scope = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["scope"]
    _, consensus = _consensus_over(conn, cfg, session_id, scope, criteria)

    blame: Counter = Counter()
    for idx, outcome in per_utterance.items():
        if outcome == "exact":
            continue
        for cid, human in (gold.get(idx, {}).get("verdicts") or {}).items():
            if bool(consensus.get(idx, {}).get(cid, False)) != bool(human):
                blame[cid] += 1
    by_id = {c.id: c for c in criteria}
    return [
        {"criterion": by_id[cid], "n": n}
        for cid, n in blame.most_common()
        if cid in by_id
    ]


def gold_by_label(conn, cfg: Config, construct_id: int, session_id: str) -> list[dict]:
    """Accuracy on judged utterances, split per label (notes-4 item 15).

    One row per gold label: how many you judged that way, and how often the
    schema routes them to exactly that label. A single aggregate hides which
    label the schema is actually failing on.
    """
    criteria = load_criteria(conn, construct_id, status="active")
    gold = session_gold(conn, construct_id, session_id)
    ev = evaluate_gold(conn, cfg, construct_id, session_id, criteria, gold)
    rows: dict[str, dict] = {}
    for idx, g in gold.items():
        lbl = g["gold_label"]
        if not lbl:
            continue
        r = rows.setdefault(lbl, {"label": lbl, "n": 0, "exact": 0, "outcomes": {}})
        r["n"] += 1
        outcome = ev["per_utterance"].get(idx, "?")
        r["outcomes"][outcome] = r["outcomes"].get(outcome, 0) + 1
        if outcome == "exact":
            r["exact"] += 1
    for r in rows.values():
        r["pct"] = r["exact"] / r["n"] if r["n"] else None
    return sorted(rows.values(), key=lambda r: (-r["n"], r["label"]))


def round_queue(conn, round_id: int) -> list[dict]:
    """The round's ten utterances, in the order selection picked them.

    A round reviews a fixed batch materialised at annotation time
    (`round_utterance`), not a live slice of the session. The batch is the unit:
    you work it front to back and then end the round.
    """
    return [
        {
            "index": r["utterance_index"],
            "tier": r["tier"],
            "ord": r["ord"],
            "judged": bool(r["complete"]),
            "gold_label": r["gold_label"],
        }
        for r in conn.execute(
            "SELECT ru.*, rv.complete, rv.gold_label FROM round_utterance ru "
            "LEFT JOIN review rv ON rv.round_id=ru.round_id "
            "AND rv.utterance_index=ru.utterance_index "
            "WHERE ru.round_id=? ORDER BY ru.ord",
            (round_id,),
        ).fetchall()
    ]


def session_snapshot(conn, cfg: Config, round_id: int, construct_id: int) -> dict:
    """What the current schema says about this whole session (notes 22-28).

    Shown when a round finishes annotating and again when review ends, so the
    same numbers visibly move as the schema is fixed.
    """
    from collections import Counter

    from .metrics import schema_logic
    from .schema_map import Outcome, apply_map

    rnd = conn.execute("SELECT * FROM round WHERE id=?", (round_id,)).fetchone()
    scope = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["scope"]
    criteria = load_criteria(conn, construct_id, status="active")
    session = corpus.get(cfg, rnd["session_id"])
    in_scope = session.in_scope(scope)

    firings = {
        nick: annotate.firing_matrix(
            conn, rnd["session_id"], criteria_keys(criteria), nick
        )
        for nick in annotators()
    }
    consensus = {
        u.index: consensus_firing(firings, u.index, criteria) for u in in_scope
    }
    universe = [u.index for u in in_scope]
    logic = schema_logic(criteria, consensus, universe, load_rules(conn, construct_id))

    # Count with the SAME rules the real map uses. Without them, every
    # nothing-fired utterance landed in a "(none)" bucket sitting in the table
    # beside a real label usually called None — two different things, one of
    # them fictional, impossible to tell apart. With the rules passed, silence
    # resolves to the catch-all exactly as it does everywhere else, and the
    # unresolved bucket only survives for a construct that designated none.
    rules = load_rules(conn, construct_id)
    per_label: Counter = Counter()
    for idx in universe:
        res = apply_map(criteria, consensus[idx], rules)
        if not res.candidates:
            per_label["— no label —"] += 1
        for lbl in res.candidates:
            per_label[lbl] += 1

    # Gold is session-scoped, so it accumulates across every round run on this
    # transcript rather than resetting each time.
    gold = session_gold(conn, construct_id, rnd["session_id"])
    correctness = evaluate_gold(
        conn, cfg, construct_id, rnd["session_id"], criteria, gold
    )
    blame = criterion_blame(
        conn, cfg, construct_id, rnd["session_id"], criteria, gold,
        correctness["per_utterance"],
    )
    reviewed = conn.execute(
        "SELECT COUNT(*) n FROM review WHERE round_id=? AND complete=1", (round_id,)
    ).fetchone()["n"]
    queued = conn.execute(
        "SELECT COUNT(*) n FROM round_utterance WHERE round_id=?", (round_id,)
    ).fetchone()["n"]

    # How many schema versions have been cut while working this session. Staying
    # on one transcript means the counts below improve partly because you are
    # tuning to it; this makes that legible rather than invisible.
    first_seen = conn.execute(
        "SELECT MIN(schema_version) v FROM round WHERE construct_id=? AND session_id=?",
        (construct_id, rnd["session_id"]),
    ).fetchone()["v"]
    current_v = conn.execute(
        "SELECT version FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["version"]

    return {
        "round": rnd,
        "session_id": rnd["session_id"],
        "n_utterances": len(session.utterances),
        "n_in_scope": len(in_scope),
        "scope": scope,
        "per_label": dict(sorted(per_label.items(), key=lambda kv: -kv[1])),
        "logic": logic,
        "reviewed": reviewed,
        "queued": queued,
        "gold": gold,
        "correctness": correctness,
        "blame": blame,
        # Consistency needs no gold and covers the whole transcript, so unlike
        # correctness it can legitimately reach zero defects. That is the part
        # that makes "complete this session" a real finish line.
        "consistent": logic["counts"]["determined"],
        # Requires at least one in-scope utterance. Without that guard a
        # session the scope filter empties has zero over- and zero
        # under-labelled utterances and reported itself "clean" — a green
        # finish line earned by having nothing to label.
        "is_clean": (
            len(in_scope) > 0
            and logic["counts"]["over"] == 0
            and logic["counts"]["under"] == 0
        ),
        "edits_on_this_session": max(0, current_v - (first_seen or current_v)),
        "n_criteria": len(criteria),
        # Binary constructs report positives rather than a candidate breakdown.
        "positive_label": positive_label(conn, construct_id),
    }


def group_criteria_by_label(
    conn, construct_id: int, criteria, firing=None, predicted=None
):
    """Criteria grouped by the label they bear on, for the Review screen.

    A criterion appears under every label it touches — deliberately. Reading
    down a label shows everything that could have raised it, which is what the
    reviewer needs when deciding whether the judgement is right.

    Ordering puts the work in front of you: labels the schema actually predicted
    come first, then labels with the most firing criteria, then the rest. Within
    a group, firing criteria lead. Reading order used to be the label space's,
    which buried the handful of TRUE judgements that matter among a dozen FALSE
    ones.
    """
    from . import schema_view as sv

    firing = firing or {}
    predicted = set(predicted or ())
    row = conn.execute(
        "SELECT label_space FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    view = sv.build(criteria, jload(row["label_space"], []))
    by_id = {c.id: c for c in criteria}

    groups = []
    for g in view.labels:
        rows = [by_id[r.criterion_id] for r in g.rows if r.criterion_id in by_id]
        if not rows:
            continue
        rows.sort(key=lambda cr: (not firing.get(cr.id, False), cr.id))
        n_true = sum(1 for cr in rows if firing.get(cr.id, False))
        groups.append(
            {
                "label": g.label,
                "rows": rows,
                "n_true": n_true,
                "predicted": g.label in predicted,
            }
        )
    groups.sort(key=lambda g: (not g["predicted"], -g["n_true"], g["label"]))
    return groups


def positive_label(conn, construct_id: int) -> str | None:
    """The non-catch-all label, when the construct is binary.

    "Binary" means two labels with one designated the catch-all — "does this
    behaviour occur?" rather than "which of these does it belong to". In that
    shape the interesting quantity is the positive class: nothing-fired is a
    negative *prediction*, not a gap, and a candidate-set breakdown over two
    labels tells you less than a straight count of positives. Returns None for
    every other shape, and callers fall back to the general display.
    """
    row = conn.execute(
        "SELECT label_space, other_label FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    if row is None:
        return None
    labels = jload(row["label_space"], [])
    other = row["other_label"] if "other_label" in row.keys() else None
    if len(labels) != 2 or not other or other not in labels:
        return None
    return next(l for l in labels if l != other)


def label_positives(conn, cfg: Config, construct_id: int) -> dict[str, int]:
    """Model firings routed to each label, over every session annotated so far.

    The same computation the session board does under "Predicted labels", run
    across sessions and summed. Deliberately not gold: this decides whether
    there is enough of a label to *review*, and gold does not exist yet.

    A session whose annotations are missing (still running, or the corpus file
    moved) contributes nothing rather than raising — the caller is deciding
    whether to spend money, and the safe answer to "I don't know" is "don't".
    """
    from .schema_map import apply_map

    criteria = load_criteria(conn, construct_id, status="active")
    labels = jload(
        conn.execute(
            "SELECT label_space FROM construct WHERE id=?", (construct_id,)
        ).fetchone()["label_space"],
        [],
    )
    scope = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["scope"]
    rules = load_rules(conn, construct_id)
    keys = criteria_keys(criteria)

    counts = {lbl: 0 for lbl in labels}
    for sid in used_sessions(conn, construct_id):
        try:
            session = corpus.get(cfg, sid)
        except (FileNotFoundError, KeyError):
            continue
        firings = {
            nick: annotate.firing_matrix(conn, sid, keys, nick) for nick in annotators()
        }
        for u in session.in_scope(scope):
            res = apply_map(criteria, consensus_firing(firings, u.index, criteria), rules)
            for lbl in res.candidates:
                if lbl in counts:
                    counts[lbl] += 1
    return counts


def labels_short_of_positives(conn, cfg: Config, construct_id: int) -> list[str]:
    """Labels with fewer than MIN_POSITIVES_PER_LABEL firings so far."""
    return [
        lbl
        for lbl, n in label_positives(conn, cfg, construct_id).items()
        if n < MIN_POSITIVES_PER_LABEL
    ]


def should_chain(conn, cfg: Config, construct_id: int) -> list[str]:
    """Labels that justify annotating another session right now.

    Empty means stop, for any of three reasons: every label has enough
    positives, the dataset is used up, or the construct has already had
    MAX_CHAINED_SESSIONS sessions annotated. The cap counts sessions, so it
    also stops a construct that chained, was revised, and chained again from
    walking the dataset one session per revision.
    """
    if len(used_sessions(conn, construct_id)) >= MAX_CHAINED_SESSIONS:
        return []
    remaining, _ = sessions_remaining(conn, cfg, construct_id)
    if not remaining:
        return []
    return labels_short_of_positives(conn, cfg, construct_id)


def sessions_remaining(conn, cfg: Config, construct_id: int) -> tuple[int, int]:
    """(unused eligible sessions, total in the dataset) for this construct.

    A round always draws a session the construct has not used, so a dataset can
    run out — `upchieve_human` has five. Before this was checked, the sixth
    "Add a session" raised out of `pick_session` before any round row was
    written and surfaced as an unhandled 500.
    """
    row = conn.execute(
        "SELECT scope, dataset FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    if row is None:
        return (0, 0)
    dataset = row["dataset"]
    used = used_sessions(conn, construct_id)
    try:
        remaining = len(
            corpus.eligible_session_ids(cfg, row["scope"], used, dataset)
        )
        total = len(corpus.all_sessions(cfg, dataset))
    except (FileNotFoundError, KeyError):
        return (0, 0)
    return (remaining, total)


def used_sessions(conn, construct_id: int) -> set[str]:
    return {
        r["session_id"]
        for r in conn.execute(
            "SELECT DISTINCT session_id FROM round WHERE construct_id=?", (construct_id,)
        )
    }


def start_round(
    conn,
    cfg: Config,
    construct_id: int,
    kind: str,
    *,
    revisit_session: str | None = None,
    chain: bool = True,
    chained: bool = False,
) -> int:
    """Pick a session, create the round, and kick off annotation in the background."""
    row = conn.execute(
        "SELECT scope, version, dataset FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    scope = row["scope"]

    # For Approve (round 0) every seeded criterion is annotated, since sessions
    # are never partially annotated (Q302).
    status = None if kind == "approve" else "active"
    criteria = load_criteria(conn, construct_id, status=status)
    if not criteria:
        raise RuntimeError("no criteria to annotate")

    if revisit_session is not None:
        # Re-run a session already seen, under the current schema (note 16).
        # The cache is keyed on criterion text, so unchanged criteria cost
        # nothing — only reworded or added ones are paid for. That makes this
        # the cheapest possible way to see what a schema change actually did.
        session_id = revisit_session
    else:
        used = used_sessions(conn, construct_id)
        session_id = selection.pick_session(cfg, scope, used, dataset=row["dataset"])

    cur = conn.execute(
        "INSERT INTO round (construct_id, kind, session_id, schema_version, "
        "status, chained) VALUES (?,?,?,?,'annotating',?)",
        (construct_id, kind, session_id, row["version"], int(chained)),
    )
    round_id = cur.lastrowid
    conn.commit()

    session = corpus.get(cfg, session_id)

    def finish(prog):
        # This thread's connection, not the request thread's: `finish` runs on
        # the round driver, and bg_conn is thread-local.
        bconn, block = bg_conn(cfg)
        with block:
            if prog.failed and prog.failed == prog.total:
                bconn.execute(
                    "UPDATE round SET status='failed', error=? WHERE id=?",
                    ("every annotation call failed", round_id),
                )
                bconn.commit()
                return
        try:
            # An approve round produces examples, not a review queue — it must
            # not land in 'reviewing', or the construct page offers a review
            # link that leads to an empty round.
            if kind == "review":
                _materialize_selection(cfg, round_id, construct_id, session, scope)
                status, done = "reviewing", None
            else:
                status, done = "complete", "datetime('now')"
            with block:
                bconn.execute(
                    f"UPDATE round SET status=?, completed_at={done or 'completed_at'} "
                    "WHERE id=?",
                    (status, round_id),
                )
                bconn.commit()
            if kind == "review" and chain:
                _maybe_chain(cfg, bconn, block, construct_id)
        except Exception as exc:  # noqa: BLE001
            with block:
                bconn.execute(
                    "UPDATE round SET status='failed', error=? WHERE id=?",
                    (str(exc), round_id),
                )
                bconn.commit()

    by_id = {c.id: c for c in criteria}
    pairs = [
        (c.text, by_id[c.parent_id].text if c.parent_id in by_id else None)
        for c in criteria
    ]
    annotate.run_round(cfg, round_id, session, pairs, scope, on_done=finish)
    return round_id


def _maybe_chain(cfg: Config, bconn, block, construct_id: int) -> int | None:
    """Annotate one more session if some label is still short of positives.

    Runs on the round driver thread once a review round lands in `reviewing`,
    so the reviewer sitting on the progress page is carried straight to the
    next session rather than being asked. Chaining is one session at a time:
    each new round re-enters this on completion, which re-reads the counts the
    session just produced, so the chain stops as soon as it has enough instead
    of committing up front to a number of sessions.

    A failure here must not take the round down with it — the round is already
    `reviewing` and usable, and an unchained round is a smaller problem than a
    round that reports itself failed after succeeding.
    """
    try:
        with block:
            short = should_chain(bconn, cfg, construct_id)
        if not short:
            return None
        log.info(
            "construct %s: chaining another session, short on %s",
            construct_id, ", ".join(short),
        )
        with block:
            return start_round(bconn, cfg, construct_id, "review", chained=True)
    except Exception:  # noqa: BLE001
        log.exception("construct %s: chained round failed to start", construct_id)
        return None


def _materialize_selection(cfg, round_id, construct_id, session, scope):
    bconn, block = bg_conn(cfg)   # this thread's connection
    with block:
        criteria = load_criteria(bconn, construct_id, status="active")
        firings = {
            nick: annotate.firing_matrix(
                bconn, session.session_id, criteria_keys(criteria), nick
            )
            for nick in annotators()
        }
    # Regressions lead: utterances this session's gold says were routed right
    # before and are now wrong are the most direct evidence the last edit
    # backfired, and nothing else in the ladder would surface them.
    with block:
        gold = session_gold(bconn, construct_id, session.session_id)
        prev = bconn.execute(
            "SELECT MAX(schema_version) v FROM round "
            "WHERE construct_id=? AND session_id=? AND schema_version < "
            "(SELECT schema_version FROM round WHERE id=?)",
            (construct_id, session.session_id, round_id),
        ).fetchone()["v"]
        cmp_ = (
            compare_schema_versions(bconn, cfg, construct_id, session.session_id, prev)
            if prev and gold else None
        )
    regressed = [d["index"] for d in (cmp_ or {}).get("regressed", [])]

    with block:
        rules = load_rules(bconn, construct_id)
        pos = positive_label(bconn, construct_id)
    picks = selection.select_round(
        bconn, session, scope, criteria, firings,
        prioritise=regressed, rules=rules, positive_label=pos,
    )
    with block:
        for i, cand in enumerate(picks):
            bconn.execute(
                "INSERT OR REPLACE INTO round_utterance (round_id, utterance_index, tier, ord) "
                "VALUES (?,?,?,?)",
                # cand.why, not TIER_LABEL[outcome]: the binary ladder does not
                # tier on Outcome at all, and the badge must say what the
                # utterance was actually picked for.
                (round_id, cand.index, cand.why, i),
            )
        bconn.commit()


def round_firings(conn, cfg: Config, round_id: int, construct_id: int) -> dict:
    r = conn.execute("SELECT session_id FROM round WHERE id=?", (round_id,)).fetchone()
    criteria = load_criteria(conn, construct_id, status="active")
    return {
        nick: annotate.firing_matrix(
            conn, r["session_id"], criteria_keys(criteria), nick
        )
        for nick in annotators()
    }


def consensus_firing(firings: dict, index: int, criteria: list[Criterion]) -> dict[int, bool]:
    """What we show the reviewer: a criterion fires if either model says so.

    Matches the worst-case-across-models rule selection uses (docs/selection.md
    §3.1), so the candidate set shown is the one the round was chosen on.
    """
    out: dict[int, bool] = {}
    for c in criteria:
        out[c.id] = any(f.get(index, {}).get(c.id, False) for f in firings.values())
    return out


def models_disagree(
    firings: dict, index: int, criteria: list[Criterion], rules=None
) -> bool:
    """Do the models land on different labels *after* the boundary rules?

    With the rules, because that is what the reader is shown. Without them a
    binary construct reports a split on every utterance where one model fired
    a catch-all criterion — a difference the catch-all rule then erases.
    """
    sets = {
        apply_map(criteria, f.get(index, {}), rules).candidates
        for f in firings.values()
    }
    return len(sets) > 1


def per_model_labels(
    firings: dict, index: int, criteria: list[Criterion], rules=None,
    positive_label: str | None = None,
) -> list[tuple[str, str]]:
    """(model title, what it said) for one utterance, for the disagreement tip.

    Phrased in the construct's own terms: a binary construct reads as
    "Metacognition" / "Not Metacognition" rather than as a candidate set,
    matching what Review shows above it.
    """
    from .config import model_title

    out = []
    for nick in annotators():
        cands = apply_map(criteria, firings.get(nick, {}).get(index, {}), rules).candidates
        if positive_label:
            said = positive_label if positive_label in cands else f"Not {positive_label}"
        else:
            said = ", ".join(sorted(cands)) or "no label"
        out.append((model_title(nick), said))
    return out


# --------------------------------------------------------------------------- #
# Revise
# --------------------------------------------------------------------------- #

def revise_context(conn, cfg: Config, construct_id: int) -> str:
    c = conn.execute("SELECT * FROM construct WHERE id=?", (construct_id,)).fetchone()
    criteria = load_criteria(conn, construct_id, status="active")
    rows = conn.execute(
        "SELECT rv.*, r.session_id FROM review rv JOIN round r ON r.id=rv.round_id "
        "WHERE r.construct_id=? AND rv.complete=1 ORDER BY rv.id",
        (construct_id,),
    ).fetchall()

    # Gold verdicts are keyed by criterion id and are never rewritten, so the
    # history keeps referring to criteria that have since been excluded. Listing
    # only the active ones left those ids unresolvable, and the model — quite
    # reasonably — spent a clarifying question asking what #65 and #71 were
    # instead of asking something only the researcher could answer.
    active_ids = {cr.id for cr in criteria}
    seen_ids: set[int] = set()
    for rv in rows:
        for k in jload(rv["criterion_verdicts"], {}):
            try:
                seen_ids.add(int(k))
            except (TypeError, ValueError):
                continue
    by_id = {cr.id: cr for cr in load_criteria(conn, construct_id, status=None)}
    removed = [by_id[i] for i in sorted(seen_ids - active_ids) if i in by_id]

    lines = [
        f"Construct: {c['name']}",
        f"Description:\n{c['description']}",
        f"Label space: {jload(c['label_space'], [])}",
        "",
        "Current criteria:",
    ]
    for cr in criteria:
        edges = ", ".join(f"->{e.label}" for e in cr.edges)
        lines.append(f'  [{cr.id}] "{cr.text}"  ({edges or "no edges — a gate"})')

    if removed:
        lines += [
            "",
            "Criteria REMOVED from the schema. They are not part of it any more "
            "and must not be edited, rewired or reasoned about as if they were. "
            "They are listed only so that every criterion id in the review "
            "history below resolves: those judgements were recorded while these "
            "criteria were still active.",
        ]
        for cr in removed:
            edges = ", ".join(f"->{e.label}" for e in cr.edges)
            lines.append(
                f'  [{cr.id}] "{cr.text}"  ({edges or "no edges"})  — REMOVED'
            )

    lines += ["", "Review evidence (whole history):"]
    if not rows:
        lines.append("  (none yet)")
    for rv in rows:
        session = corpus.get(cfg, rv["session_id"])
        utt = next(
            (u for u in session.utterances if u.index == rv["utterance_index"]), None
        )
        text = utt.text if utt else "?"
        verdicts = jload(rv["criterion_verdicts"], {})
        lines.append(
            f'  - "{text[:160]}"\n'
            f"      human label: {rv['gold_label']}   schema outcome: {rv['outcome']}"
        )
        if verdicts:
            lines.append(f"      human criterion verdicts: {verdicts}")
        if rv["note"]:
            lines.append(f"      note: {rv['note']}")

    pad = c["scratchpad"] if "scratchpad" in c.keys() else ""
    if pad:
        lines += ["", "Your running note on outstanding schema gaps:", pad]

    bnotes = conn.execute(
        "SELECT label_a, label_b, note FROM boundary_note WHERE construct_id=? "
        "ORDER BY label_a, label_b, id",
        (construct_id,),
    ).fetchall()
    if bnotes:
        lines += ["", "What the reviewer says separates specific label pairs:"]
        seen_pair = None
        for b in bnotes:
            pair = f"{b['label_a']} vs {b['label_b']}"
            if pair != seen_pair:
                lines.append(f"  {pair}:")
                seen_pair = pair
            lines.append(f"    - {b['note']}")
        lines.append(
            "  A boundary with several notes is a candidate either for a "
            "priority rule or for a criterion that separates the two."
        )

    answered = conn.execute(
        "SELECT text, answer FROM question WHERE construct_id=? AND answer<>'' "
        "AND round_id = (SELECT MAX(id) FROM round WHERE construct_id=?)",
        (construct_id, construct_id),
    ).fetchall()
    if answered:
        lines += ["", "Answers you previously received:"]
        lines += [f"  Q: {q['text']}\n  A: {q['answer']}" for q in answered]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Schema history: rollback, and the branching that falls out of it
# --------------------------------------------------------------------------- #

def restore_version(conn, construct_id: int, version: int) -> int:
    """Roll the live schema back to `version`, append-only.

    Writes a *new* version whose content equals the restored one and whose
    parent is the restored one — so nothing is destroyed and the history becomes
    a tree rather than a line (Q120, Q222).

    Criteria are upserted **by id**, and criteria absent from the snapshot are
    marked excluded rather than deleted. `review.criterion_verdicts` is keyed by
    criterion id, so deleting would orphan the review history and break every
    model-vs-human metric that references it.
    """
    row = conn.execute(
        "SELECT snapshot FROM schema_version WHERE construct_id=? AND version=?",
        (construct_id, version),
    ).fetchone()
    if row is None:
        raise KeyError(f"construct {construct_id} has no version {version}")
    snap = jload(row["snapshot"], {})

    keep: list[int] = []
    for c in snap.get("criteria", []):
        keep.append(c["id"])
        edges = c["edges"] if isinstance(c["edges"], str) else json.dumps(c["edges"])
        updated = conn.execute(
            "UPDATE criterion SET text=?, edges=?, status=?, ord=? "
            "WHERE id=? AND construct_id=?",
            (c["text"], edges, c["status"], c.get("ord", 0), c["id"], construct_id),
        ).rowcount
        if not updated:
            # The criterion was hard-deleted at some point; reinstate its id so
            # historical verdicts still resolve.
            conn.execute(
                "INSERT INTO criterion (id, construct_id, text, edges, status, origin, ord) "
                "VALUES (?,?,?,?,?,?,?)",
                (c["id"], construct_id, c["text"], edges, c["status"],
                 c.get("origin", "seed"), c.get("ord", 0)),
            )

    placeholders = ",".join("?" * len(keep)) if keep else "NULL"
    conn.execute(
        f"UPDATE criterion SET status='excluded' "
        f"WHERE construct_id=? AND id NOT IN ({placeholders})",
        (construct_id, *keep),
    )
    conn.execute(
        "UPDATE construct SET description=?, label_space=? WHERE id=?",
        (snap.get("description", ""), json.dumps(snap.get("label_space", [])), construct_id),
    )
    conn.commit()
    return snapshot_schema(conn, construct_id, f"restored v{version}", parent=version)


def version_tree(conn, construct_id: int) -> list[dict]:
    """Versions in tree order, with depth and a criteria diff against the parent."""
    rows = conn.execute(
        "SELECT version, parent_version, note, created_at, snapshot "
        "FROM schema_version WHERE construct_id=? ORDER BY version",
        (construct_id,),
    ).fetchall()
    if not rows:
        return []

    snaps = {r["version"]: jload(r["snapshot"], {}) for r in rows}

    def active(v: int) -> dict[int, str]:
        return {
            c["id"]: c["text"]
            for c in snaps.get(v, {}).get("criteria", [])
            if c.get("status") != "excluded"
        }

    children: dict[int | None, list] = {}
    for r in rows:
        children.setdefault(r["parent_version"], []).append(r)

    out: list[dict] = []

    def walk(parent: int | None, depth: int) -> None:
        for r in children.get(parent, []):
            cur, prev = active(r["version"]), active(r["parent_version"])
            changed = sum(
                1 for i, t in cur.items() if i in prev and prev[i] != t
            )
            out.append(
                {
                    "version": r["version"],
                    "parent": r["parent_version"],
                    "note": r["note"],
                    "created_at": r["created_at"],
                    "depth": depth,
                    "n_criteria": len(cur),
                    "added": len(set(cur) - set(prev)),
                    "removed": len(set(prev) - set(cur)),
                    "changed": changed,
                }
            )
            walk(r["version"], depth + 1)

    walk(None, 0)
    # Defensive: a row whose parent is missing would otherwise vanish from the
    # view entirely. List it at the root with real diff numbers.
    seen = {o["version"] for o in out}
    for r in rows:
        if r["version"] in seen:
            continue
        cur, prev = active(r["version"]), active(r["parent_version"])
        out.append({
            "version": r["version"], "parent": r["parent_version"], "note": r["note"],
            "created_at": r["created_at"], "depth": 0, "n_criteria": len(cur),
            "added": len(set(cur) - set(prev)), "removed": len(set(prev) - set(cur)),
            "changed": sum(1 for i, t in cur.items() if i in prev and prev[i] != t),
        })
    out.sort(key=lambda o: o["version"])
    return out


# --------------------------------------------------------------------------- #
# Revise: async P3/P4 (they are LLM calls and were blocking the POST)
# --------------------------------------------------------------------------- #

REVISE_JOBS: dict[int, dict] = {}
_revise_lock = threading.Lock()


def revise_status(construct_id: int) -> dict:
    with _revise_lock:
        return dict(REVISE_JOBS.get(construct_id, {"status": "complete", "error": ""}))


def revise_async(cfg: Config, construct_id: int, what: str) -> None:
    """`what` is 'questions' (P3) or 'deltas' (P4)."""
    with _revise_lock:
        REVISE_JOBS[construct_id] = {"status": "running", "error": "", "what": what}

    def work() -> None:
        conn, lock = bg_conn(cfg)
        try:
            with lock:
                ctx = revise_context(conn, cfg, construct_id)
                view = build_schema_view(conn, cfg, construct_id)
            # Structured facts P4 could not previously see: unreachable labels
            # and criteria that never fire.
            ctx += "\n\nStructural summary of the current schema:\n" + "\n".join(
                view.summary_lines()
            )
            if what == "questions":
                from .config import MAX_CLARIFYING_QUESTIONS

                with lock:
                    latest = conn.execute(
                        "SELECT id FROM round WHERE construct_id=? ORDER BY id DESC LIMIT 1",
                        (construct_id,),
                    ).fetchone()
                round_id = latest["id"] if latest else None
                payload, _, _ = ask_json(
                    reasoning_model(),
                    prompts.clarifying_questions(ctx, MAX_CLARIFYING_QUESTIONS),
                )
                rows = [
                    (construct_id, round_id, str(q))
                    for q in (payload.get("questions") or [])[:MAX_CLARIFYING_QUESTIONS]
                ]
                with lock:
                    conn.executemany(
                        "INSERT INTO question (construct_id, round_id, text) VALUES (?,?,?)",
                        rows,
                    )
                    conn.commit()
            else:
                payload, _, _ = ask_json(reasoning_model(), prompts.propose_deltas(ctx))
                # Refresh the running note while the evidence is in hand.
                try:
                    with lock:
                        cur_pad = conn.execute(
                            "SELECT scratchpad FROM construct WHERE id=?", (construct_id,)
                        ).fetchone()["scratchpad"]
                    pad, _, _ = ask_json(
                        reasoning_model(), prompts.scratchpad(ctx, cur_pad)
                    )
                    with lock:
                        conn.execute(
                            "UPDATE construct SET scratchpad=? WHERE id=?",
                            (str(pad.get("scratchpad", ""))[:4000], construct_id),
                        )
                        conn.commit()
                except Exception:  # noqa: BLE001 — the note is a nicety, not the job
                    log.warning("scratchpad refresh failed", exc_info=True)
                valid = {
                    "add_criterion", "remove_criterion", "reword",
                    "rewire", "edit_description", "add_rule",
                }
                with lock:
                    # A proposal replaces the last one. The page says "whatever
                    # the model proposed when you last asked", and without this
                    # it was really "everything it has ever proposed and you
                    # never acted on" — stale suggestions against wording that
                    # has since changed, piling up round after round.
                    #
                    # Deleted, not marked: the delta table carries
                    # CHECK (status IN ('staged','applied','rejected')) and
                    # SQLite cannot alter a CHECK without rebuilding the
                    # table, which this project avoids. Only untouched
                    # suggestions go — applied and rejected rows are the
                    # record of what you decided and are left alone.
                    n = conn.execute(
                        "DELETE FROM delta WHERE construct_id=? AND status='staged'",
                        (construct_id,),
                    ).rowcount
                    if n:
                        log.info("discarded %d stale staged delta(s)", n)
                    for d in payload.get("deltas", []):
                        if d.get("kind") not in valid:
                            continue
                        if is_noop_delta(conn, construct_id, d["kind"], d):
                            log.info("dropped no-op %s delta from P4", d["kind"])
                            continue
                        conn.execute(
                            "INSERT INTO delta (construct_id, kind, payload, rationale) "
                            "VALUES (?,?,?,?)",
                            (construct_id, d["kind"], json.dumps(d),
                             str(d.get("rationale", ""))),
                        )
                    conn.commit()
            with _revise_lock:
                REVISE_JOBS[construct_id] = {"status": "complete", "error": ""}
        except Exception as exc:  # noqa: BLE001
            with _revise_lock:
                REVISE_JOBS[construct_id] = {"status": "failed", "error": str(exc)}

    threading.Thread(target=work, daemon=True, name=f"revise-{construct_id}").start()


# An edge-set change. "rewire" is the only kind written now; the two older names
# are still read so deltas staged before this branch still render and apply.
REWIRE_KINDS = ("rewire", "reassign", "change_kind")


def is_noop_delta(conn, construct_id: int, kind: str, payload: dict) -> bool:
    """Would applying this change nothing?

    P4 has a habit of proposing the edge set a criterion already has, with a
    rationale claiming it is changing something. Rendered faithfully, that reads
    as a bug in the app.
    """
    cid = payload.get("criterion_id")
    if cid is None:
        return False
    row = conn.execute(
        "SELECT text, edges FROM criterion WHERE id=? AND construct_id=?",
        (cid, construct_id),
    ).fetchone()
    if row is None:
        return False
    if kind == "reword":
        return (payload.get("text") or "").strip() == (row["text"] or "").strip()
    if kind in REWIRE_KINDS:
        now = {e["label"] for e in jload(row["edges"], [])}
        new = {e.get("label") for e in (payload.get("edges") or [])}
        return now == new
    return False


def delta_warnings(kind: str, payload: dict) -> list[str]:
    """Atomicity problems in a *proposed* change, surfaced before you apply it.

    The failure this exists for: P4 answering "this criterion overlaps another
    label" by appending exclusions until one criterion encodes the whole label
    space (note 13).
    """
    from .schema_map import lint_criterion_text

    out: list[str] = []
    new = payload.get("text")
    if new:
        out += lint_criterion_text(new)
        old = payload.get("_old_text")
        if old and len(new.split()) > 1.4 * len(old.split()) + 5:
            out.insert(
                0,
                f"The rewording is much longer than the original "
                f"({len(old.split())} → {len(new.split())} words). Usually this "
                "means label boundaries are being written into the criterion "
                "instead of expressed as edges.",
            )
    return out


def describe_delta(kind: str, payload: dict, names: dict[int, str]) -> dict:
    """Turn a raw delta into something readable (response plan §7).

    The Revise page used to dump the JSON, which made the single easiest change
    to evaluate — a reword — impossible to read.
    """
    cid = payload.get("criterion_id")
    who = names.get(cid, f"#{cid}") if cid is not None else None

    def edges(es):
        return [f"→ {e.get('label', '?')}" for e in (es or [])]

    if kind == "reword":
        return {"title": f"Reword {who}", "before": payload.get("_old_text"),
                "after": payload.get("text"), "kind_label": "reword"}
    if kind == "add_criterion":
        return {"title": "Add criterion", "after": payload.get("text"),
                "after_edges": edges(payload.get("edges")), "kind_label": "add"}
    if kind == "remove_criterion":
        return {"title": f"Remove {who}", "kind_label": "remove",
                "metrics": payload.get("_metrics")}
    if kind == "add_rule":
        return {
            "title": f"Rule: {payload.get('winner')} wins over {payload.get('loser')}",
            "kind_label": "rule",
            "metrics": payload.get("_metrics"),
        }
    if kind in REWIRE_KINDS:
        return {"title": f"Rewire {who}", "before_edges": payload.get("_old_edges"),
                "after_edges": edges(payload.get("edges")), "kind_label": "rewire",
                "metrics": payload.get("_metrics")}
    if kind == "edit_description":
        return {"title": "Edit description", "after": payload.get("description"),
                "kind_label": "description"}
    return {"title": kind, "kind_label": kind}


def enrich_delta(conn, construct_id: int, kind: str, payload: dict) -> dict:
    """Attach the current values a delta would replace, so the UI can diff."""
    cid = payload.get("criterion_id")
    if cid is None:
        return payload
    row = conn.execute(
        "SELECT text, edges FROM criterion WHERE id=? AND construct_id=?",
        (cid, construct_id),
    ).fetchone()
    if row is None:
        return payload
    out = dict(payload)
    out["_old_text"] = row["text"]
    out["_old_edges"] = [f"→ {e['label']}" for e in jload(row["edges"], [])]
    return out


def apply_delta(conn, construct_id: int, delta_id: int) -> None:
    d = conn.execute("SELECT * FROM delta WHERE id=?", (delta_id,)).fetchone()
    p = jload(d["payload"], {})
    kind = d["kind"]

    if kind == "add_criterion":
        # Active immediately. It used to land as 'pending' to honour the Q212
        # approval gate, but that gate was never given a screen, so accepted
        # criteria disappeared: absent from the criteria list, the hand-edit
        # dropdown, the graph, and never annotated. Accepting the delta IS the
        # decision; the criterion is judged afterwards by its fire rate and flags.
        conn.execute(
            "INSERT INTO criterion (construct_id, text, edges, status, origin, ord) "
            "VALUES (?,?,?,'active','llm-revision',999)",
            (construct_id, p["text"], json.dumps(p.get("edges", []))),
        )
    elif kind == "remove_criterion":
        conn.execute(
            "UPDATE criterion SET status='excluded' WHERE id=? AND construct_id=?",
            (p["criterion_id"], construct_id),
        )
    elif kind == "reword":
        conn.execute(
            "UPDATE criterion SET text=? WHERE id=? AND construct_id=?",
            (p["text"], p["criterion_id"], construct_id),
        )
    elif kind in REWIRE_KINDS:
        # Validate here too. P4's output never passed through the Approve screen,
        # so without this it can install an edge set the UI would have rejected —
        # a label outside the space, or the same label twice.
        labels = jload(
            conn.execute(
                "SELECT label_space FROM construct WHERE id=?", (construct_id,)
            ).fetchone()["label_space"],
            [],
        )
        edges = [{"label": e.get("label")} for e in p.get("edges", [])]
        problems = validate_edges([Edge(e["label"]) for e in edges], labels)
        if problems:
            raise ValueError("; ".join(problems))
        # Also check the change against the rest of the schema, not just in
        # isolation — a legal edge set can still break the nesting rules.
        from .schema_map import validate_nesting

        trial = [
            Criterion(c.id, c.text,
                      tuple(Edge(e["label"]) for e in edges) if c.id == p["criterion_id"] else c.edges,
                      c.parent_id)
            for c in load_criteria(conn, construct_id, status=None)
        ]
        nest = validate_nesting(trial)
        if nest:
            raise ValueError("; ".join(nest))
        conn.execute(
            "UPDATE criterion SET edges=? WHERE id=? AND construct_id=?",
            (json.dumps(edges), p["criterion_id"], construct_id),
        )
    elif kind == "add_rule":
        from .schema_map import LabelRules

        existing = load_rules(conn, construct_id)
        pairs = (existing.priorities if existing else ()) + ((p["winner"], p["loser"]),)
        if LabelRules(priorities=pairs).cycles():
            raise ValueError(
                f"{p['winner']} > {p['loser']} would create a precedence cycle."
            )
        conn.execute(
            "INSERT OR IGNORE INTO label_rule (construct_id, winner, loser, note) "
            "VALUES (?,?,?,?)",
            (construct_id, p["winner"], p["loser"], str(p.get("rationale", ""))),
        )
    elif kind == "edit_description":
        conn.execute(
            "UPDATE construct SET description=? WHERE id=?", (p["description"], construct_id)
        )

    conn.execute("UPDATE delta SET status='applied' WHERE id=?", (delta_id,))
    conn.commit()
    snapshot_schema(conn, construct_id, f"{kind}: {d['rationale'][:120]}")


# --------------------------------------------------------------------------- #
# Restart recovery and resume (docs/multi-user/02 §6, 01 §3)
# --------------------------------------------------------------------------- #

MAX_REQUEUE = 20


def resume_round(cfg: Config, round_id: int) -> None:
    """Re-run the annotation pass for a round left mid-flight by a restart.

    Nearly free: every call already written is in the cache, so `annotate_one`
    returns immediately for completed work and only the unfinished calls cost
    anything. A round that was 90% done finishes in seconds.
    """
    conn, lock = bg_conn(cfg)
    with lock:
        rnd = conn.execute("SELECT * FROM round WHERE id=?", (round_id,)).fetchone()
    if rnd is None or rnd["status"] != "annotating":
        return
    cid = rnd["construct_id"]
    status = None if rnd["kind"] == "approve" else "active"
    with lock:
        criteria = load_criteria(conn, cid, status=status)
        scope = conn.execute(
            "SELECT scope FROM construct WHERE id=?", (cid,)
        ).fetchone()["scope"]
    if not criteria:
        with lock:
            conn.execute(
                "UPDATE round SET status='failed', error=? WHERE id=?",
                ("no criteria to annotate after restart", round_id),
            )
            conn.commit()
        return

    session = corpus.get(cfg, rnd["session_id"])
    by_id = {c.id: c for c in criteria}
    pairs = [
        (c.text, by_id[c.parent_id].text if c.parent_id in by_id else None)
        for c in criteria
    ]

    def finish(prog):
        bconn, block = bg_conn(cfg)
        try:
            if rnd["kind"] == "review":
                _materialize_selection(cfg, round_id, cid, session, scope)
                new_status, done = "reviewing", None
            else:
                new_status, done = "complete", "datetime('now')"
            with block:
                bconn.execute(
                    f"UPDATE round SET status=?, completed_at={done or 'completed_at'} "
                    "WHERE id=?", (new_status, round_id),
                )
                bconn.commit()
        except Exception as exc:  # noqa: BLE001
            with block:
                bconn.execute(
                    "UPDATE round SET status='failed', error=? WHERE id=?",
                    (str(exc), round_id),
                )
                bconn.commit()

    annotate.run_round(cfg, round_id, session, pairs, scope, on_done=finish)


def recover_orphans(cfg: Config) -> dict:
    """Requeue rounds stranded at `annotating` by a restart.

    Called once from `create_app`, in a background thread so startup does not
    block. Without it the reviewer sits on a progress page pinned at 100% that
    polls forever with no exit — `JOBS` is process-local, so after a restart no
    `Progress` exists for their round.
    """
    from .db import connect

    conn = connect(cfg.db_path)
    orphans = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM round WHERE status='annotating' ORDER BY id"
        )
    ]
    if not orphans:
        conn.close()
        return {"requeued": 0, "failed": 0}

    # Cap the stampede. Restarting twice in a row should not fire every
    # orphaned round at the providers simultaneously.
    requeue, excess = orphans[:MAX_REQUEUE], orphans[MAX_REQUEUE:]
    for rid in excess:
        conn.execute(
            "UPDATE round SET status='failed', error=? WHERE id=?",
            ("server restarted with too many rounds in flight", rid),
        )
    conn.commit()
    conn.close()

    telemetry.push_event(
        "server_restarted", requeued=len(requeue), failed=len(excess)
    )

    def work():
        for rid in requeue:
            try:
                resume_round(cfg, rid)
                telemetry.push_event("round_requeued", round_id=rid)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not requeue round %s: %s", rid, exc)

    threading.Thread(target=work, daemon=True, name="recover-orphans").start()
    return {"requeued": len(requeue), "failed": len(excess)}


def resume_url(conn, construct_id: int) -> tuple[str, str]:
    """(endpoint-ish url, human label) for 'where this construct is'.

    Derived from `construct.phase` plus the latest round's status rather than
    stored on the reviewer. A stored frame goes stale the moment a background
    job changes state behind the user's back, which in this app is most of the
    interesting transitions; deriving is correct after a browser crash, a
    server restart, or a job finishing while the tab was closed.

    Returns a url built with `flask.url_for`, so call it inside a request.
    """
    from flask import url_for

    c = conn.execute(
        "SELECT id, phase FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    if c is None:
        return "#", "Missing"
    home = url_for("construct_home", cid=construct_id)

    n_criteria = conn.execute(
        "SELECT COUNT(*) n FROM criterion WHERE construct_id=? AND status<>'excluded'",
        (construct_id,),
    ).fetchone()["n"]
    if not n_criteria:
        if seed_status(construct_id)["status"] == "running":
            return url_for("seeding", cid=construct_id), "Seeding criteria…"
        return home, "Seed criteria"

    latest = conn.execute(
        "SELECT * FROM round WHERE construct_id=? ORDER BY id DESC LIMIT 1",
        (construct_id,),
    ).fetchone()

    if c["phase"] == "approve":
        if latest is not None and latest["status"] == "annotating":
            return (url_for("annotating", cid=construct_id, rid=latest["id"]),
                    "Annotating…")
        pending = conn.execute(
            "SELECT COUNT(*) n FROM criterion WHERE construct_id=? AND status='pending'",
            (construct_id,),
        ).fetchone()["n"]
        if pending:
            return url_for("approve", cid=construct_id), f"Approve {pending} criteria"

    if latest is not None and latest["kind"] == "review":
        rid = latest["id"]
        if latest["status"] == "annotating":
            return url_for("annotating", cid=construct_id, rid=rid), "Annotating…"
        if latest["status"] == "failed":
            return home, "Last round failed — start another"
        if latest["status"] == "reviewing":
            queued = conn.execute(
                "SELECT COUNT(*) n FROM round_utterance WHERE round_id=?", (rid,)
            ).fetchone()["n"]
            done = conn.execute(
                "SELECT COUNT(*) n FROM review WHERE round_id=? AND complete=1", (rid,)
            ).fetchone()["n"]
            if queued and done >= queued:
                return url_for("round_summary", cid=construct_id, rid=rid), \
                    f"Finish round {rid}"
            return url_for("review", cid=construct_id, rid=rid), \
                f"Review — {done} of {queued} done"

    if c["phase"] == "revise":
        return url_for("revise", cid=construct_id), "Revise the schema"
    return home, "Open"
