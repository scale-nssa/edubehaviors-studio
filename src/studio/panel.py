"""The facilitator's control panel. Read-only, no buttons, no auth.

`GET /panel` renders a shell; `GET /panel/snapshot` returns JSON that the page
polls every 2s. It answers four questions, in order of urgency: is anyone
stuck, is the model layer healthy, where is everyone in the flow, and how much
has this cost.

Deliberately its own module and its own Blueprint — `create_app` gains one
line. It also deliberately does **not** go through `app._construct()`, whose
404-unless-you-own-it check is exactly what the panel needs to bypass.

The snapshot runs every two seconds while twenty people annotate, so it is
restricted to COUNT(*) and single-row lookups. It must never call
`_metrics_summary`, `session_snapshot`, `compare_schema_versions` or anything
that builds a firing matrix: those are seconds of work each and would make the
panel the heaviest user of the database.
"""

from __future__ import annotations

import time

from flask import Blueprint, current_app, jsonify, render_template, request

from . import annotate, corpus, llm, service, telemetry
from .config import annotators, estimate_cost, reasoning_model
from .db import get_db

bp = Blueprint("panel", __name__)

STUCK_AFTER_S = 4 * 60      # a round is budgeted at about two minutes
IDLE_AFTER_S = 5 * 60


@bp.route("/panel")
def panel():
    return render_template("panel.html")


@bp.route("/dev/transcripts")
def transcripts():
    """Read every corpus side by side, as the annotator sees it.

    A build-time conversion decides role mapping and which rows survive; those
    decisions are invisible in the CSVs and only show up here. One scrollable
    window per dataset, `?<dataset_id>=<session_id>` to change which session.
    """
    cfg = current_app.config["CFG"]
    cards = []
    for d in corpus.datasets(cfg):
        try:
            sessions = corpus.all_sessions(cfg, d["id"])
        except (FileNotFoundError, KeyError) as exc:
            cards.append({**d, "error": str(exc), "sessions": []})
            continue
        ids = [s.session_id for s in sessions]
        chosen = request.args.get(d["id"]) or (ids[0] if ids else None)
        if chosen not in ids:
            chosen = ids[0] if ids else None
        session = next((s for s in sessions if s.session_id == chosen), None)
        roles: dict[str, int] = {}
        for u in (session.utterances if session else ()):
            roles[u.role] = roles.get(u.role, 0) + 1
        cards.append({
            **d,
            "error": None,
            "session_ids": ids,
            "chosen": chosen,
            "utterances": session.utterances if session else (),
            "roles": roles,
            "n_sessions": len(ids),
        })
    return render_template("dev_transcripts.html", cards=cards)


def _age_s(conn, ts: str | None) -> float | None:
    """Seconds since a SQLite `datetime('now')` timestamp (UTC), or None."""
    if not ts:
        return None
    row = conn.execute(
        "SELECT CAST((julianday('now') - julianday(?)) * 86400.0 AS REAL) AS s", (ts,)
    ).fetchone()
    return row["s"] if row else None


def _participants(conn) -> list[dict]:
    out = []
    for r in conn.execute(
        "SELECT id, display_name, last_seen_at, last_construct_id FROM reviewer "
        "ORDER BY COALESCE(last_seen_at, '') DESC, id"
    ):
        cid = r["last_construct_id"]
        con = (
            conn.execute("SELECT id, name, phase FROM construct WHERE id=?", (cid,)).fetchone()
            if cid else None
        )
        idle_s = _age_s(conn, r["last_seen_at"])

        n_rounds = conn.execute(
            "SELECT COUNT(*) n FROM round r JOIN construct c ON c.id=r.construct_id "
            "WHERE c.owner_id=?", (r["id"],)
        ).fetchone()["n"]
        n_reviewed = conn.execute(
            "SELECT COUNT(*) n FROM review rv JOIN round r ON r.id=rv.round_id "
            "JOIN construct c ON c.id=r.construct_id WHERE c.owner_id=? AND rv.complete=1",
            (r["id"],),
        ).fetchone()["n"]

        latest = conn.execute(
            "SELECT r.id, r.status, r.started_at, r.error FROM round r "
            "JOIN construct c ON c.id=r.construct_id "
            "WHERE c.owner_id=? ORDER BY r.id DESC LIMIT 1", (r["id"],)
        ).fetchone()

        flags = []
        if latest is not None:
            if latest["status"] == "annotating":
                age = _age_s(conn, latest["started_at"]) or 0
                if age > STUCK_AFTER_S:
                    flags.append("stuck")
            if latest["status"] == "failed":
                flags.append("failed")
        # A failed background job counts too — the reviewer sees an error and
        # you want to know before they raise a hand.
        if cid and service.seed_status(cid).get("status") == "failed":
            flags.append("seed failed")
        if cid and service.revise_status(cid).get("status") == "failed":
            flags.append("revise failed")
        if idle_s is not None and idle_s > IDLE_AFTER_S and latest is not None \
                and latest["status"] in ("annotating", "reviewing"):
            flags.append("idle")

        where = ""
        if con is not None:
            try:
                _, where = service.resume_url(conn, con["id"])
            except Exception:  # noqa: BLE001 — the panel must never 500
                where = ""

        out.append({
            "id": r["id"],
            "name": r["display_name"],
            "idle_s": idle_s,
            "construct": con["name"] if con else None,
            "construct_id": con["id"] if con else None,
            "phase": con["phase"] if con else None,
            "where": where,
            "rounds": n_rounds,
            "reviewed": n_reviewed,
            "flags": flags,
            "error": latest["error"] if latest and latest["error"] else "",
        })
    return out


def _live_rounds(conn) -> list[dict]:
    with annotate._jobs_lock:
        jobs = dict(annotate.JOBS)
    out = []
    for rid, prog in jobs.items():
        snap = prog.snapshot()
        if snap["status"] in ("complete", "failed") and snap["elapsed_s"] > 300:
            continue   # finished a while ago; not "live" any more
        row = conn.execute(
            "SELECT r.session_id, r.kind, c.name AS cname, rev.display_name AS owner "
            "FROM round r JOIN construct c ON c.id=r.construct_id "
            "JOIN reviewer rev ON rev.id=c.owner_id WHERE r.id=?", (rid,)
        ).fetchone()
        out.append({
            "round_id": rid,
            "owner": row["owner"] if row else "?",
            "construct": row["cname"] if row else "?",
            "session": row["session_id"] if row else "?",
            "kind": row["kind"] if row else "?",
            "status": snap["status"],
            "position": snap["position"],
            "done": snap["done"],
            "total": snap["total"],
            "failed": snap["failed"],
            "pct": snap["pct"],
            "elapsed_s": snap["elapsed_s"],
            "last": snap["recent"][0] if snap["recent"] else "",
        })
    out.sort(key=lambda r: (r["status"] != "running", -r["elapsed_s"]))
    return out


def _models(conn) -> list[dict]:
    live = telemetry.model_stats()
    usage = llm.usage_report()
    cached = {
        r["model"]: r
        for r in conn.execute(
            "SELECT model, COUNT(*) n, SUM(input_tokens) tin, SUM(output_tokens) tout "
            "FROM annotation GROUP BY model"
        )
    }
    nicks = sorted(set(live) | set(usage) | set(cached) | set(annotators()) | {reasoning_model()})
    out = []
    for nick in nicks:
        l, u = live.get(nick, {}), usage.get(nick, {})
        cch = cached.get(nick)
        tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        out.append({
            "model": nick,
            "in_flight": l.get("in_flight", 0),
            "completed": l.get("completed", 0),
            "failed": l.get("failed", 0),
            "last_latency_s": l.get("last_latency_s"),
            "mean_latency_s": l.get("mean_latency_s"),
            "calls": u.get("calls", 0),
            "in_tokens": tin,
            "out_tokens": tout,
            "cost_usd": estimate_cost(nick, tin, tout),
            "cached_rows": cch["n"] if cch else 0,
            "cached_cost_usd": estimate_cost(nick, cch["tin"] or 0, cch["tout"] or 0) if cch else 0.0,
        })
    return out


def _corpus(conn) -> list[dict]:
    return [
        {"session": r["session_id"], "constructs": r["n"]}
        for r in conn.execute(
            "SELECT session_id, COUNT(DISTINCT construct_id) n FROM round "
            "GROUP BY session_id ORDER BY n DESC, session_id"
        )
    ]


@bp.route("/panel/snapshot")
def snapshot():
    conn = get_db()
    models = _models(conn)
    people = _participants(conn)
    return jsonify({
        "now": time.time(),
        "uptime_s": telemetry.uptime_s(),
        "participants": people,
        "rounds": _live_rounds(conn),
        "models": models,
        "corpus": _corpus(conn),
        "events": telemetry.events(50),
        "totals": {
            "people": len(people),
            "flagged": sum(1 for p in people if p["flags"]),
            "in_flight": telemetry.total_in_flight(),
            "session_cost_usd": sum(m["cost_usd"] for m in models),
            "lifetime_cache_cost_usd": sum(m["cached_cost_usd"] for m in models),
            "constructs": conn.execute("SELECT COUNT(*) n FROM construct").fetchone()["n"],
            "reviews": conn.execute(
                "SELECT COUNT(*) n FROM review WHERE complete=1").fetchone()["n"],
        },
    })
