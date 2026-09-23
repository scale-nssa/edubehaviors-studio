"""Flask + HTMX app. Spec §8."""

from __future__ import annotations

import json
import logging

from flask import (
    Flask,
    abort,
    g,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
)

from . import (
    annotate,
    corpus,
    estimate,
    export,
    local_setup,
    metrics,
    model_settings,
    selection,
    service,
    spend,
    telemetry,
)
from .config import (
    EXAMPLE_WINDOW,
    MIN_POSITIVES_PER_LABEL,
    TRANSCRIPT_WINDOW,
    Config,
    annotators,
    model_title,
)
from .db import close_db, get_db, init_db, jload, snapshot_schema
from .llm import usage_report
from .schema_map import (
    Criterion,
    Edge,
    apply_map,
    validate_edges,
    validate_nesting,
)

log = logging.getLogger("studio")

LOCAL_REVIEWER = "Me"


def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or Config.from_env()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(cfg.db_path)
    # Keys and role bindings live beside the database, never inside it.
    model_settings.init(cfg.home)
    model_settings.install_log_redaction()
    spend.init(cfg.db_path)

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=cfg.secret_key, DB_PATH=cfg.db_path, CFG=cfg, TEMPLATES_AUTO_RELOAD=True,
        MAX_CONTENT_LENGTH=110 * 1024 * 1024,   # dataset uploads; see datasets_page
    )
    app.teardown_appcontext(close_db)

    from .panel import bp as panel_bp
    app.register_blueprint(panel_bp)
    local_setup.register(app)

    # A round left at 'annotating' by a restart has no in-process Progress, so
    # its owner would sit on a bar pinned at 100% polling forever. Requeue in
    # the background — almost everything is already cached, so a round that was
    # 90% done finishes in seconds.
    service.recover_orphans(cfg)

    # ---------------------------------------------------------------- identity
    @app.before_request
    def _load_reviewer():
        name = request.cookies.get("reviewer")
        g.reviewer = None
        g.single_user = cfg.single_user
        if cfg.single_user and not name:
            # One researcher on their own machine: nobody to tell apart, so
            # no sign-in. The reviewer row still exists — everything keys on
            # it — it is just created for them.
            name = LOCAL_REVIEWER
            db = get_db()
            db.execute(
                "INSERT OR IGNORE INTO reviewer (display_name) VALUES (?)", (name,)
            )
            db.commit()
        if name:
            db = get_db()
            row = db.execute(
                "SELECT * FROM reviewer WHERE display_name=?", (name,)
            ).fetchone()
            g.reviewer = row
            # Presence, for the panel only. Throttled: twenty people polling
            # HTMX every 1.2s would otherwise be ~17 writes/second to one row.
            if row is not None and telemetry.should_touch(row["id"]):
                db.execute(
                    "UPDATE reviewer SET last_seen_at=datetime('now') WHERE id=?",
                    (row["id"],),
                )
                db.commit()

    def require_reviewer():
        if not g.reviewer:
            abort(redirect(url_for("start")))
        return g.reviewer

    @app.route("/", methods=["GET"])
    def start():
        if g.reviewer:
            if cfg.single_user and not model_settings.ready():
                # First run: nothing works until a model is set up, so that
                # is where a new user should land.
                return redirect(url_for("setup.models"))
            return redirect(url_for("home"))
        return render_template("start.html")

    @app.route("/identify", methods=["POST"])
    def identify():
        # Collapse internal whitespace too, so " Sam" and "Sam  " are one person.
        name = " ".join((request.form.get("name") or "").split())
        if not name:
            return redirect(url_for("start"))
        db = get_db()
        existing = db.execute(
            "SELECT id, last_seen_at FROM reviewer WHERE display_name=?", (name,)
        ).fetchone()
        # There is no auth, so this is advisory — but without it the second
        # "Sam" silently lands in the first Sam's account and sees their work.
        if existing and not request.form.get("confirm_existing"):
            n = db.execute(
                "SELECT COUNT(*) n FROM construct WHERE owner_id=?", (existing["id"],)
            ).fetchone()["n"]
            seen = db.execute(
                "SELECT CAST((julianday('now') - julianday(?)) * 86400 AS INT) s",
                (existing["last_seen_at"],),
            ).fetchone()["s"] if existing["last_seen_at"] else None
            return render_template(
                "identify_confirm.html", name=name, n_constructs=n, seen_s=seen,
                suggestion=f"{name} (2)",
            )
        db.execute("INSERT OR IGNORE INTO reviewer (display_name) VALUES (?)", (name,))
        db.commit()
        resp = make_response(redirect(url_for("home")))
        resp.set_cookie("reviewer", name, max_age=60 * 60 * 24 * 90, samesite="Lax")
        return resp

    @app.route("/logout")
    def logout():
        resp = make_response(redirect(url_for("start")))
        resp.delete_cookie("reviewer")
        return resp

    # -------------------------------------------------------------------- home
    @app.route("/home")
    def home():
        r = require_reviewer()
        db = get_db()
        constructs = db.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM round WHERE construct_id=c.id) AS rounds "
            "FROM construct c WHERE owner_id=? ORDER BY c.id DESC",
            (r["id"],),
        ).fetchall()
        # Derived, not stored: correct after a browser crash, a server restart,
        # or a background job finishing while the tab was closed.
        resume = {c["id"]: service.resume_url(db, c["id"]) for c in constructs}
        return render_template(
            "home.html", constructs=constructs, resume=resume,
            last_cid=r["last_construct_id"],
        )

    # ---------------------------------------------------------------- describe
    @app.route("/construct/new", methods=["GET", "POST"])
    def new_construct():
        r = require_reviewer()
        if request.method == "GET":
            return render_template(
                "describe.html", construct=None, warn_other=False,
                datasets=corpus.datasets(cfg),
            )

        name = (request.form.get("name") or "Untitled").strip()
        description = (request.form.get("description") or "").strip()
        scope = request.form.get("scope") or "tutor"
        dataset = request.form.get("dataset") or corpus.DEFAULT_DATASET
        labels = [
            s.strip() for s in (request.form.get("labels") or "").split(",") if s.strip()
        ]

        # Boundary rules are set here and only here (see describe.html): they are
        # part of defining the label space, and fixing them at creation keeps the
        # schema's tie-breaking from drifting mid-project.
        from .schema_map import LabelRules

        raw_other = (request.form.get("other_label") or "").strip()
        rules_text = (request.form.get("rules") or "").strip()
        form = {
            "name": name,
            "description": description,
            "scope": scope,
            "dataset": dataset,
            "labels": ", ".join(labels),
            "other_label": raw_other,
            "rules": rules_text,
        }

        problems: list[str] = []
        if dataset not in corpus.dataset_ids(cfg):
            problems.append(f"Unknown dataset {dataset!r}.")
        # Designating the catch-all switches on the two default boundary rules:
        # silence routes here, and it never wins against a real label.
        other = raw_other or next(
            (l for l in labels if l.lower() in ("other", "none", "n/a")), None
        )
        if raw_other and raw_other not in labels:
            problems.append(f"Catch-all {raw_other!r} is not one of the labels.")

        pairs: list[tuple[str, str]] = []
        for line in rules_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ">" not in line:
                problems.append(f"Rule {line!r} is not of the form 'Winner > Loser'.")
                continue
            w, _, l = line.partition(">")
            w, l = w.strip(), l.strip()
            for side in (w, l):
                if side not in labels:
                    problems.append(f"Rule {line!r} names {side!r}, which is not a label.")
            if w and l and w == l:
                problems.append(f"Rule {line!r} puts a label above itself.")
            pairs.append((w, l))
        if not problems and LabelRules(priorities=tuple(pairs)).cycles():
            problems.append(
                "Those rules form a precedence cycle, which would resolve by list "
                "order rather than by anything meaningful."
            )

        if problems:
            return render_template(
                "describe.html", construct=None, warn_other=False,
                problems=problems, form=form, datasets=corpus.datasets(cfg),
            )

        # Q209: warn at save if there's no catch-all, with a one-click add.
        if labels and not other and not request.form.get("ack_other"):
            return render_template(
                "describe.html", construct=None, warn_other=True, form=form,
                datasets=corpus.datasets(cfg),
            )

        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        if (resp := local_setup.preflight(
            estimate.seeding(description, labels, scope),
            title="Generate criteria", back=url_for("new_construct"),
        )) is not None:
            return resp

        db = get_db()
        cur = db.execute(
            "INSERT INTO construct (owner_id, name, description, label_space, scope, "
            "phase, other_label, dataset) VALUES (?,?,?,?,?,'approve',?,?)",
            (r["id"], name, description, json.dumps(labels), scope, other, dataset),
        )
        cid = cur.lastrowid
        for w, l in pairs:
            db.execute(
                "INSERT OR IGNORE INTO label_rule (construct_id, winner, loser) "
                "VALUES (?,?,?)", (cid, w, l),
            )
        db.commit()
        snapshot_schema(db, cid, "initial")
        service.seed_criteria_async(cfg, cid)
        return redirect(url_for("seeding", cid=cid))

    @app.route("/construct/<int:cid>/seeding")
    def seeding(cid: int):
        require_reviewer()
        return render_template("seeding.html", c=_construct(get_db(), cid))

    @app.route("/construct/<int:cid>/seed-progress")
    def seed_progress(cid: int):
        require_reviewer()
        st = service.seed_status(cid)
        st["next_url"] = url_for("approve", cid=cid)
        st["retry_url"] = url_for("seed_retry", cid=cid)
        return render_template("_seed_progress.html", p=st)

    @app.route("/construct/<int:cid>/seed-retry", methods=["POST"])
    def seed_retry(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        if (resp := local_setup.preflight(
            estimate.seeding(c["description"], jload(c["label_space"], []), c["scope"]),
            title="Generate criteria again", back=url_for("seeding", cid=cid),
        )) is not None:
            return resp
        db.execute("DELETE FROM criterion WHERE construct_id=? AND status='pending'", (cid,))
        db.commit()
        service.seed_criteria_async(cfg, cid)
        return redirect(url_for("seeding", cid=cid))

    @app.route("/construct/<int:cid>")
    def construct_home(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        # One row per SESSION, not per round. A session accumulates rounds —
        # re-running it, chaining onto it — and listing each separately made
        # the same transcript appear three times with three identical-looking
        # actions, inviting you to start work on an old one. Gold is
        # session-scoped anyway, so the session is the real unit: what you
        # judged on it carries across every round run against it.
        rounds = [
            dict(r)
            for r in db.execute(
                """
                SELECT session_id,
                       MAX(id)                                        AS last_round,
                       COUNT(*)                                       AS n_rounds,
                       MAX(CASE WHEN status='annotating' THEN id END) AS annotating_id,
                       MAX(CASE WHEN status='reviewing'  THEN id END) AS reviewing_id,
                       MAX(CASE WHEN status='failed' THEN error END)  AS error
                FROM round
                WHERE construct_id=? AND kind='review'
                GROUP BY session_id
                ORDER BY last_round DESC
                """,
                (cid,),
            )
        ]
        for r in rounds:
            r["judged"] = db.execute(
                "SELECT COUNT(DISTINCT rv.utterance_index) n FROM review rv "
                "JOIN round rd ON rd.id = rv.round_id "
                "WHERE rd.construct_id=? AND rd.session_id=? AND rv.complete=1",
                (cid, r["session_id"]),
            ).fetchone()["n"]
        versions = db.execute(
            "SELECT version, note, created_at FROM schema_version WHERE construct_id=? "
            "ORDER BY version DESC",
            (cid,),
        ).fetchall()
        criteria = service.load_criteria(db, cid, status="active")
        summary = _metrics_summary(db, cid)
        view = service.build_schema_view(db, cfg, cid)
        export_models = export.available_models(db, cid)
        remaining, total_sessions = service.sessions_remaining(db, cfg, cid)
        rules_rows = db.execute(
            "SELECT * FROM label_rule WHERE construct_id=? ORDER BY id", (cid,)
        ).fetchall()
        return render_template(
            "construct.html",
            rules=rules_rows,
            rule_load=_rule_load(db, cid),
            other_label=c["other_label"] if "other_label" in c.keys() else None,
            c=c,
            rounds=rounds,
            versions=versions,
            criteria=criteria,
            labels=jload(c["label_space"], []),
            summary=summary,
            view=view,
            export_models=export_models,
            sessions_remaining=remaining,
            sessions_total=total_sessions,
            dataset_title=next(
                (d["title"] for d in corpus.datasets(cfg) if d["id"] == c["dataset"]),
                c["dataset"],
            ),
        )

    # ------------------------------------------------- manual edits (note 15)
    def _schema_fragment(db, cid):
        """The schema table on its own, for an HTMX swap."""
        return render_template(
            "_schema.html",
            view=service.build_schema_view(db, cfg, cid),
            actions=True,
            cid=cid,
        )

    @app.route("/construct/<int:cid>/schema-table")
    def schema_table(cid: int):
        """Re-render the table — used by Cancel, and after a save."""
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        return _schema_fragment(db, cid)

    @app.route("/construct/<int:cid>/criterion/<int:crit_id>/edit-form")
    def criterion_edit_form(cid: int, crit_id: int):
        """Swap a display row for its inline editor."""
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        row = db.execute(
            "SELECT * FROM criterion WHERE id=? AND construct_id=?", (crit_id, cid)
        ).fetchone()
        if row is None:
            abort(404)
        return render_template(
            "_criterion_edit_row.html",
            cid=cid,
            crit_id=crit_id,
            text=row["text"],
            labels=jload(c["label_space"], []),
            wired=[e["label"] for e in jload(row["edges"], [])],
            problems=[],
        )

    @app.route("/construct/<int:cid>/criterion/<int:crit_id>/edit", methods=["POST"])
    def criterion_edit(cid: int, crit_id: int):
        """Direct edit of a criterion's text and edges, outside Approve."""
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        labels = jload(c["label_space"], [])
        action = request.form.get("action")

        if action == "exclude":
            db.execute(
                "UPDATE criterion SET status='excluded' WHERE id=? AND construct_id=?",
                (crit_id, cid),
            )
            db.commit()
            snapshot_schema(db, cid, f"excluded criterion {crit_id}")
            if request.headers.get("HX-Request"):
                return _schema_fragment(db, cid)
            return redirect(request.referrer or url_for("construct_home", cid=cid))

        text = (request.form.get("text") or "").strip()
        edges = _edges_from_form(request.form, labels)
        problems = validate_edges([Edge(e["label"]) for e in edges], labels)
        if not problems:
            trial = [
                Criterion(
                    cr.id, cr.text,
                    tuple(Edge(e["label"]) for e in edges)
                    if cr.id == crit_id else cr.edges,
                    cr.parent_id,
                )
                for cr in service.load_criteria(db, cid, status=None)
            ]
            problems = validate_nesting(trial)
        if problems or not text:
            # Inline: hand the editor back with the complaint attached, rather
            # than replacing the page with an error and losing what was typed.
            # 200, not 400: htmx does not swap a response with an error
            # status, so a 400 here would silently do nothing and the save
            # button would look broken. The complaint rides in the fragment.
            if request.headers.get("HX-Request"):
                return render_template(
                    "_criterion_edit_row.html",
                    cid=cid,
                    crit_id=crit_id,
                    text=text,
                    labels=labels,
                    wired=[e["label"] for e in edges],
                    problems=problems or ["A criterion needs some text."],
                )
            return render_template(
                "error.html",
                message="; ".join(problems) or "A criterion needs some text.",
                cid=cid,
            ), 400
        db.execute(
            "UPDATE criterion SET text=?, edges=? WHERE id=? AND construct_id=?",
            (text, json.dumps(edges), crit_id, cid),
        )
        db.commit()
        snapshot_schema(db, cid, f"edited criterion {crit_id}")
        if request.headers.get("HX-Request"):
            return _schema_fragment(db, cid)
        return redirect(url_for("revise", cid=cid) + "#editors")

    @app.route("/construct/<int:cid>/criterion/new", methods=["POST"])
    def criterion_new(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        labels = jload(c["label_space"], [])
        text = (request.form.get("text") or "").strip()
        edges = _edges_from_form(request.form, labels)
        problems = validate_edges([Edge(e["label"]) for e in edges], labels)
        if problems or not text:
            return render_template(
                "error.html",
                message="; ".join(problems) or "A criterion needs some text.",
                cid=cid,
            ), 400
        db.execute(
            "INSERT INTO criterion (construct_id, text, edges, status, origin, ord) "
            "VALUES (?,?,?,'active','user',999)",
            (cid, text, json.dumps(edges)),
        )
        db.commit()
        snapshot_schema(db, cid, "added a criterion by hand")
        return redirect(url_for("revise", cid=cid) + "#editors")

    @app.route("/construct/<int:cid>/description", methods=["POST"])
    def edit_description(cid: int):
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        db.execute(
            "UPDATE construct SET description=? WHERE id=?",
            ((request.form.get("description") or "").strip(), cid),
        )
        db.commit()
        snapshot_schema(db, cid, "edited description")
        return redirect(url_for("revise", cid=cid) + "#editors")

    @app.route("/construct/<int:cid>/scratchpad", methods=["POST"])
    def edit_scratchpad(cid: int):
        """The agent's note is its theory of what is wrong; you can correct it."""
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        db.execute(
            "UPDATE construct SET scratchpad=? WHERE id=?",
            ((request.form.get("scratchpad") or "").strip(), cid),
        )
        db.commit()
        return redirect(url_for("revise", cid=cid) + "#scratchpad")

    # ----------------------------------------------------------- schema history
    @app.route("/construct/<int:cid>/history")
    def history(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        return render_template(
            "history.html", c=c, versions=service.version_tree(db, cid)
        )

    @app.route("/construct/<int:cid>/restore/<int:version>", methods=["POST"])
    def restore(cid: int, version: int):
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        service.restore_version(db, cid, version)
        return redirect(url_for("history", cid=cid))

    # ----------------------------------------------------------------- approve
    @app.route("/construct/<int:cid>/approve")
    def approve(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        if service.seed_status(cid)["status"] == "running":
            return redirect(url_for("seeding", cid=cid))
        rnd = db.execute(
            "SELECT * FROM round WHERE construct_id=? AND kind='approve' ORDER BY id DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if rnd is None:
            # Round 0 costs money like any other: confirm it first.
            return redirect(url_for("approve_start", cid=cid))
        if rnd["status"] == "annotating":
            return redirect(url_for("annotating", cid=cid, rid=rnd["id"]))

        pending = db.execute(
            "SELECT * FROM criterion WHERE construct_id=? AND status='pending' ORDER BY ord, id",
            (cid,),
        ).fetchall()
        if not pending:
            # Snapshot the approved set. Without this the only version older than
            # the first edit is the empty "initial" one taken before seeding, so
            # there is nothing meaningful to roll back to — restoring would wipe
            # every criterion rather than undo one change.
            already = db.execute(
                "SELECT 1 FROM schema_version WHERE construct_id=? AND note=?",
                (cid, "approved criteria"),
            ).fetchone()
            if not already:
                snapshot_schema(db, cid, "approved criteria")
            db.execute("UPDATE construct SET phase='annotate' WHERE id=?", (cid,))
            db.commit()
            return redirect(url_for("construct_home", cid=cid))

        crit = pending[0]
        crit_children = db.execute(
            "SELECT id, text, edges FROM criterion WHERE parent_id=? ORDER BY ord, id",
            (crit["id"],),
        ).fetchall()
        session = corpus.get(cfg, rnd["session_id"])
        examples = _criterion_examples(db, session, crit["text"], c["scope"])
        # Sidebar: what has been approved so far, so you can see the label space
        # filling in rather than working blind through 10+ screens. No stats yet
        # (nothing is annotated beyond round 0), so skip the expensive pass.
        side = service.build_schema_view(db, cfg, cid, with_stats=False)
        return render_template(
            "approve.html",
            c=c,
            crit=crit,
            edges=jload(crit["edges"], []),
            labels=jload(c["label_space"], []),
            examples=examples,
            children=[(k, jload(k["edges"], [])) for k in crit_children],
            side=side,
            remaining=len(pending),
            total=db.execute(
                "SELECT COUNT(*) n FROM criterion WHERE construct_id=?", (cid,)
            ).fetchone()["n"],
        )

    @app.route("/construct/<int:cid>/approve/start", methods=["GET", "POST"])
    def approve_start(cid: int):
        """Annotate one session so Approve has real examples to show."""
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        if db.execute(
            "SELECT 1 FROM round WHERE construct_id=? AND kind='approve'", (cid,)
        ).fetchone():
            return redirect(url_for("approve", cid=cid))
        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        sid = selection.pick_session(
            cfg, c["scope"], service.used_sessions(db, cid), dataset=c["dataset"]
        )
        if (resp := local_setup.preflight(
            estimate.annotation(db, cfg, cid, "approve", sid),
            title="Annotate a first session", back=url_for("construct_home", cid=cid),
        )) is not None:
            return resp
        rid = service.start_round(db, cfg, cid, "approve")
        return redirect(url_for("annotating", cid=cid, rid=rid))

    @app.route("/construct/<int:cid>/approve/<int:crit_id>", methods=["POST"])
    def approve_decide(cid: int, crit_id: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        labels = jload(c["label_space"], [])
        action = request.form.get("action")

        if action == "exclude":
            db.execute("UPDATE criterion SET status='excluded' WHERE id=?", (crit_id,))
            db.commit()
            return redirect(url_for("approve", cid=cid))

        text = (request.form.get("text") or "").strip()
        edges = _edges_from_form(request.form, labels)

        # A gate carries no edges by design — it only decides whether its
        # sub-criteria get asked. Requiring an edge here made gates impossible
        # to approve at all.
        is_gate = bool(
            db.execute(
                "SELECT 1 FROM criterion WHERE parent_id=? LIMIT 1", (crit_id,)
            ).fetchone()
        )
        if is_gate:
            problems = (
                ["A criterion with sub-criteria must carry no edges of its own."]
                if edges else []
            )
        else:
            problems = validate_edges([Edge(e["label"]) for e in edges], labels)

        if problems:
            crit = db.execute("SELECT * FROM criterion WHERE id=?", (crit_id,)).fetchone()
            rnd = db.execute(
                "SELECT * FROM round WHERE construct_id=? AND kind='approve' "
                "ORDER BY id DESC LIMIT 1",
                (cid,),
            ).fetchone()
            session = corpus.get(cfg, rnd["session_id"])
            side = service.build_schema_view(db, cfg, cid, with_stats=False)
            return render_template(
                "approve.html",
                c=c,
                crit=crit,
                edges=edges,
                labels=labels,
                examples=_criterion_examples(db, session, crit["text"], c["scope"]),
                side=side,
                    problems=problems,
                remaining=1,
                total=1,
            )

        db.execute(
            "UPDATE criterion SET text=?, edges=?, status='active' WHERE id=?",
            (text, json.dumps(edges), crit_id),
        )
        db.commit()
        return redirect(url_for("approve", cid=cid))

    # ---------------------------------------------------------------- annotate
    @app.route("/construct/<int:cid>/round/<int:rid>/annotating")
    def annotating(cid: int, rid: int):
        require_reviewer()
        db = get_db()
        rnd = db.execute("SELECT * FROM round WHERE id=?", (rid,)).fetchone()
        # Say why, when the app chose this session rather than the reviewer.
        # Unprompted spending is alarming without a reason attached, and the
        # reason is specific: these labels do not yet have enough firings to
        # revise against. Recomputed live, so it names the labels that are
        # still short rather than the ones that were short when it started.
        short = (
            service.labels_short_of_positives(db, cfg, cid)
            if rnd is not None and rnd["chained"] else []
        )
        return render_template(
            "annotating.html", c=_construct(db, cid), rnd=rnd,
            short_labels=short, min_positives=MIN_POSITIVES_PER_LABEL,
        )

    @app.route("/round/<int:rid>/progress")
    def progress(rid: int):
        db = get_db()
        rnd = db.execute("SELECT * FROM round WHERE id=?", (rid,)).fetchone()
        prog = annotate.progress_for(rid)
        snap = prog.snapshot() if prog else {
            "total": 0, "done": 0, "failed": 0, "pct": 100,
            "status": rnd["status"], "error": rnd["error"], "recent": [],
            "position": 0, "elapsed_s": 0.0,
        }
        snap["round_status"] = rnd["status"]
        cid = rnd["construct_id"]
        if rnd["kind"] == "approve":
            snap["next_url"] = url_for("approve", cid=cid)
        else:
            # Follow a chain. A round that finishes annotating may have started
            # another on the next session (service._maybe_chain), and the
            # reviewer should stay on the progress page for it rather than be
            # dropped into review of the first while the second is still
            # running — the chain exists because one session was not enough.
            nxt = db.execute(
                # Round rows are only ever 'annotating' while waiting — the
                # queue lives in annotate's in-memory progress, not the DB.
                "SELECT id FROM round WHERE construct_id=? AND id>? "
                "AND status='annotating' ORDER BY id LIMIT 1",
                (cid, rid),
            ).fetchone()
            if nxt:
                snap["next_url"] = url_for("annotating", cid=cid, rid=nxt["id"])
                snap["chained"] = True
            elif rnd["chained"]:
                # A round the app started on its own. Work the chain in the
                # order it was annotated, oldest unfinished first — otherwise
                # the earlier sessions in the chain are never surfaced and sit
                # in `reviewing` forever.
                first = db.execute(
                    "SELECT id FROM round WHERE construct_id=? AND status='reviewing' "
                    "ORDER BY id LIMIT 1", (cid,)
                ).fetchone()
                snap["next_url"] = url_for(
                    "review", cid=cid, rid=first["id"] if first else rid
                )
            else:
                # A round the reviewer asked for. Go to THAT session. Falling
                # through to "oldest unfinished" here meant that pressing "Add
                # a new session" while part-way through one dropped you back
                # into the session you were already in, which looked like the
                # new utterances had not loaded.
                snap["next_url"] = url_for("review", cid=cid, rid=rid)
        return render_template("_progress.html", p=snap)

    @app.route("/construct/<int:cid>/round/new", methods=["POST"])
    def new_round(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        # The UI disables this once the dataset is used up, but a stale page or
        # a direct POST would otherwise raise out of pick_session as a 500.
        remaining, total = service.sessions_remaining(db, cfg, cid)
        if not remaining:
            return render_template(
                "error.html", cid=cid,
                message=f"You have annotated all {total} sessions in this "
                        f"dataset ({c['dataset']}). There is no new session to "
                        "draw. You can re-run a session you have already seen "
                        "from the rounds table.",
            ), 409
        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        sid = selection.pick_session(
            cfg, c["scope"], service.used_sessions(db, cid), dataset=c["dataset"]
        )
        if (resp := local_setup.preflight(
            estimate.annotation(db, cfg, cid, "review", sid),
            title="Annotate a new session", back=url_for("construct_home", cid=cid),
        )) is not None:
            return resp
        rid = service.start_round(db, cfg, cid, "review")
        db.execute("UPDATE construct SET phase='annotate' WHERE id=?", (cid,))
        db.commit()
        return redirect(url_for("annotating", cid=cid, rid=rid))

    @app.route("/construct/<int:cid>/round/<int:rid>/revisit", methods=["POST"])
    def revisit_round(cid: int, rid: int):
        """Re-run an earlier round's session under the current schema (note 16)."""
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        old = db.execute(
            "SELECT session_id FROM round WHERE id=? AND construct_id=?", (rid, cid)
        ).fetchone()
        if old is None:
            abort(404)
        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        if (resp := local_setup.preflight(
            estimate.annotation(db, cfg, cid, "review", old["session_id"], chain=False),
            title="Re-annotate this session", back=url_for("construct_home", cid=cid),
        )) is not None:
            return resp
        new_rid = service.start_round(
            # Re-running a session you have already seen is a deliberate,
            # targeted act; chaining off it would draw sessions you did not ask
            # for on top of the one you did.
            db, cfg, cid, "review", revisit_session=old["session_id"], chain=False
        )
        db.execute("UPDATE construct SET phase='annotate' WHERE id=?", (cid,))
        db.commit()
        return redirect(url_for("annotating", cid=cid, rid=new_rid))

    @app.route("/construct/<int:cid>/round/<int:rid>/summary")
    def round_summary(cid: int, rid: int):
        """The session scoreboard, shown when the round's ten are done and
        reachable from Review — the same numbers, so progress is visible."""
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        snap = service.session_snapshot(db, cfg, rid, cid)
        snap["by_label"] = service.gold_by_label(db, cfg, cid, snap["session_id"])
        snap["remaining"], snap["dataset_total"] = service.sessions_remaining(db, cfg, cid)
        snap["criterion_stats"] = service.criterion_stats(db, cfg, cid)
        snap["criteria"] = service.load_criteria(db, cid, status="active")
        rules = service.load_rules(db, cid)
        snap["catch_all"] = rules.other_label
        raw_tiers = db.execute(
            "SELECT tier, COUNT(*) n FROM round_utterance WHERE round_id=? "
            "GROUP BY tier ORDER BY n DESC",
            (rid,),
        ).fetchall()
        # Which ladder produced THIS round, read off the stored tiers rather
        # than the construct's current shape. A binary construct can hold
        # rounds drawn before the binary ladder existed, and describing them
        # with the wrong ladder contradicts the table directly above the text.
        snap["binary_round"] = any(
            r["tier"] in selection.BIN_TIER_LABEL.values() for r in raw_tiers
        )
        tiers = [
            {
                "tier": selection.name_tier(
                    r["tier"], snap["positive_label"], rules.other_label
                ),
                "n": r["n"],
            }
            for r in raw_tiers
        ]
        return render_template("round_summary.html", c=c, s=snap, tiers=tiers)

    # ------------------------------------------------------------------ review
    @app.route("/construct/<int:cid>/round/<int:rid>/review")
    def review(cid: int, rid: int):
        """Serve one of the round's ten utterances.

        The batch is fixed: `selection` picked ten when annotation finished and
        they are worked front to back. `?i=` is an utterance index within that
        batch, so the queue strip can jump anywhere in it.
        """
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        rnd = db.execute("SELECT * FROM round WHERE id=?", (rid,)).fetchone()
        if rnd["status"] == "annotating":
            return redirect(url_for("annotating", cid=cid, rid=rid))

        items = service.round_queue(db, rid)
        # `tier` is the badge text as it stood when the round was selected. A
        # round selected before the edge kinds were removed can carry a reason
        # this schema language cannot produce ("a criterion collision"), so drop
        # anything the current ladder would not say rather than show a lie.
        current_reasons = set(selection.TIER_LABEL.values())
        if not items:
            return render_template("review_empty.html", c=c, rnd=rnd)

        queue = [it["index"] for it in items]
        judged = {it["index"] for it in items if it["judged"]}
        target = request.args.get("i", type=int)
        if target is None:
            target = next((i for i in queue if i not in judged), queue[0])
        if target not in queue:
            target = queue[0]

        session = corpus.get(cfg, rnd["session_id"])
        criteria = service.load_criteria(db, cid, status="active")
        firings = service.round_firings(db, cfg, rid, cid)
        firing = service.consensus_firing(firings, target, criteria)
        # WITH the boundary rules. Without them Review called every
        # nothing-fired utterance "no label" while the session board, which
        # does pass them, showed that same utterance resolving to the
        # catch-all. Two pages disagreeing about what silence means is worse
        # than either answer on its own.
        rules = service.load_rules(db, cid)
        res = apply_map(criteria, firing, rules)
        pos_label = service.positive_label(db, cid)

        existing = db.execute(
            "SELECT * FROM review WHERE round_id=? AND utterance_index=?",
            (rid, target),
        ).fetchone()
        pos = queue.index(target)
        nxt = next((i for i in queue[pos + 1:] if i not in judged), None)

        return render_template(
            "review.html",
            c=c,
            rnd=rnd,
            pos=pos,
            n=len(queue),
            n_judged=len(judged),
            why=next(
                (it["tier"] for it in items
                 if it["index"] == target and it["tier"] in current_reasons),
                None,
            ),
            next_index=nxt,
            criteria=criteria,
            firing=firing,
            res=res,
            labels=jload(c["label_space"], []),
            names=_criterion_names(criteria),
            window=session.window(target, TRANSCRIPT_WINDOW),
            groups=service.group_criteria_by_label(
                db, cid, criteria, firing=firing, predicted=res.candidates
            ),
            gates=[c for c in criteria if not c.edges],
            queue=queue,
            judged=judged,
            target=target,
            existing=existing,
            verdicts=jload(existing["criterion_verdicts"], {}) if existing else {},
            disagree=service.models_disagree(firings, target, criteria, rules),
            reveal=bool(existing and existing["complete"]),
            positive_label=pos_label,
            catch_all=rules.other_label,
            model_labels=service.per_model_labels(
                firings, target, criteria, rules, pos_label
            ),
            board=service.session_snapshot(db, cfg, rid, cid),
        )

    @app.route("/construct/<int:cid>/round/<int:rid>/review/<int:idx>", methods=["POST"])
    def review_submit(cid: int, rid: int, idx: int):
        r = require_reviewer()
        db = get_db()
        criteria = service.load_criteria(db, cid, status="active")
        firings = service.round_firings(db, cfg, rid, cid)
        firing = service.consensus_firing(firings, idx, criteria)
        res = apply_map(criteria, firing, service.load_rules(db, cid))

        gold = request.form.get("gold_label") or None
        # Each appearance of a criterion posts under its own field name (see
        # review.html); they are kept in sync client-side, so any one is
        # authoritative. Fall back to the model's own firing if none came back.
        verdicts = {}
        for cr in criteria:
            vals = [
                v for k, v in request.form.items()
                if k == f"crit_{cr.id}" or k.startswith(f"crit_{cr.id}__")
            ]
            # No posted value means the reviewer never saw or answered this one.
            # Defaulting to the model's own firing would record a fabricated
            # human judgement that agrees with the model by construction, which
            # silently inflates every agreement statistic downstream.
            if vals:
                verdicts[str(cr.id)] = vals[0] == "true"
        note = (request.form.get("note") or "").strip()
        outcome = _outcome_for(res, gold)

        db.execute(
            "INSERT INTO review (round_id, utterance_index, reviewer_id, gold_label, "
            "outcome, criterion_verdicts, note, complete) VALUES (?,?,?,?,?,?,?,1) "
            "ON CONFLICT(round_id, utterance_index) DO UPDATE SET "
            "gold_label=excluded.gold_label, outcome=excluded.outcome, "
            "criterion_verdicts=excluded.criterion_verdicts, note=excluded.note, complete=1",
            (rid, idx, r["id"], gold, outcome, json.dumps(verdicts), note),
        )
        # A note written on an over-determined utterance is about a specific
        # LABEL BOUNDARY, so store it against the pair. Scattered per-utterance
        # notes never accumulate into an argument; four notes about the same
        # boundary do.
        bnote = (request.form.get("boundary_note") or "").strip()
        pair = sorted(res.candidates)
        if bnote and len(pair) == 2:
            db.execute(
                "INSERT INTO boundary_note (construct_id, label_a, label_b, "
                "session_id, utterance_index, note) VALUES (?,?,?,?,?,?)",
                (cid, pair[0], pair[1],
                 db.execute("SELECT session_id FROM round WHERE id=?", (rid,))
                   .fetchone()["session_id"], idx, bnote),
            )
        db.commit()

        nxt = request.form.get("next_index", type=int)
        if nxt is not None:
            return redirect(url_for("review", cid=cid, rid=rid, i=nxt))
        # The batch is done — on to the session board, which offers "end round".
        return redirect(url_for("round_summary", cid=cid, rid=rid))

    @app.route("/construct/<int:cid>/round/<int:rid>/end", methods=["POST"])
    def end_round(cid: int, rid: int):
        require_reviewer()
        db = get_db()
        db.execute(
            "UPDATE round SET status='complete', completed_at=datetime('now') WHERE id=?",
            (rid,),
        )
        db.commit()

        # Q117: nothing to correct means no Revise; go straight to a new round.
        rows = db.execute(
            "SELECT outcome, criterion_verdicts FROM review WHERE round_id=? AND complete=1",
            (rid,),
        ).fetchall()
        clean = rows and all(r["outcome"] == "exact" for r in rows)
        if clean:
            return redirect(url_for("construct_home", cid=cid))
        db.execute("UPDATE construct SET phase='revise' WHERE id=?", (cid,))
        db.commit()
        return redirect(url_for("revise", cid=cid, rid=rid))

    # ------------------------------------------------------------------ revise
    @app.route("/construct/<int:cid>/revise")
    def revise(cid: int):
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        # Questions belong to the round that raised them; carrying them forward
        # made the page accumulate stale asks (note 11).
        questions = db.execute(
            "SELECT * FROM question WHERE construct_id=? "
            "AND round_id = (SELECT MAX(id) FROM round WHERE construct_id=?) "
            "ORDER BY id",
            (cid, cid),
        ).fetchall()
        deltas = db.execute(
            "SELECT * FROM delta WHERE construct_id=? AND status='staged' ORDER BY id",
            (cid,),
        ).fetchall()
        view = service.build_schema_view(db, cfg, cid)
        criteria = service.load_criteria(db, cid, status="active")
        return render_template(
            "revise.html",
            c=c,
            questions=questions,
            deltas=_render_deltas(db, cid, deltas),
            job=service.revise_status(cid),
            criteria=criteria,
            names=_criterion_names(criteria),
            view=view,
            labels=jload(c["label_space"], []),
            edgemap={cr.id: [e.label for e in cr.edges] for cr in criteria},
        )

    @app.route("/construct/<int:cid>/revise/ask", methods=["POST"])
    def revise_ask(cid: int):
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        if (resp := local_setup.preflight(
            estimate.revise(db, cfg, cid, "questions"),
            title="Ask clarifying questions", back=url_for("revise", cid=cid),
        )) is not None:
            return resp
        service.revise_async(cfg, cid, "questions")
        return redirect(url_for("revise", cid=cid) + "#assist")

    @app.route("/construct/<int:cid>/revise/progress")
    def revise_progress(cid: int):
        require_reviewer()
        st = service.revise_status(cid)
        # Land on what the job produced, not the top of the page. The job
        # records which of the two it was running.
        anchor = "#questions" if st.get("what") == "questions" else "#deltas"
        st["next_url"] = url_for("revise", cid=cid) + anchor
        return render_template("_seed_progress.html", p=st)

    @app.route("/construct/<int:cid>/revise/answer/<int:qid>", methods=["POST"])
    def revise_answer(cid: int, qid: int):
        require_reviewer()
        db = get_db()
        db.execute(
            "UPDATE question SET answer=? WHERE id=? AND construct_id=?",
            ((request.form.get("answer") or "").strip(), qid, cid),
        )
        db.commit()
        return redirect(url_for("revise", cid=cid) + "#questions")

    @app.route("/construct/<int:cid>/revise/propose", methods=["POST"])
    def revise_propose(cid: int):
        require_reviewer()
        db = get_db()
        _construct(db, cid)
        if (resp := local_setup.models_ready_or_redirect()) is not None:
            return resp
        if (resp := local_setup.preflight(
            estimate.revise(db, cfg, cid, "deltas"),
            title="Propose schema changes", back=url_for("revise", cid=cid),
        )) is not None:
            return resp
        service.revise_async(cfg, cid, "deltas")
        # #assist, not the top of the page and not #deltas. The proposal runs
        # in the background, so what you want to see on landing is the
        # progress bar that replaces the button you just pressed — which sits
        # under this heading. #deltas would scroll past it to a section that
        # is still empty. Every other action on this page anchors too; this
        # one did not, so it threw you back to the top of a long page.
        return redirect(url_for("revise", cid=cid) + "#assist")

    @app.route("/construct/<int:cid>/delta/<int:did>/<action>", methods=["POST"])
    def delta_action(cid: int, did: int, action: str):
        require_reviewer()
        db = get_db()
        if action == "apply":
            try:
                service.apply_delta(db, cid, did)
            except ValueError as exc:
                return render_template(
                    "error.html",
                    message=f"That change would produce an invalid schema: {exc}",
                    cid=cid,
                ), 400
        else:
            db.execute("UPDATE delta SET status='rejected' WHERE id=?", (did,))
            db.commit()
        return redirect(url_for("revise", cid=cid) + "#deltas")

    # ------------------------------------------------------------------ export
    @app.route("/construct/<int:cid>/export.zip")
    def export_bundle(cid: int):
        """The firing matrix plus the assertion key, as two CSVs in a zip.

        `?models=both` (default), or one annotator by nickname or short name.
        Reads only cached annotations — nothing here costs a model call.
        """
        require_reviewer()
        db = get_db()
        c = _construct(db, cid)
        try:
            models = export.resolve_models(
                request.args.get("models"), export.available_models(db, cid)
            )
        except ValueError as exc:
            return render_template("error.html", message=str(exc), cid=cid), 400
        payload = export.bundle(db, cfg, cid, models)
        resp = make_response(payload)
        stem = f"{export.slug(c['name']) or 'construct'}_v{c['version']}"
        resp.headers["Content-Type"] = "application/zip"
        resp.headers["Content-Disposition"] = f'attachment; filename="{stem}.zip"'
        return resp

    # ------------------------------------------------------------------- admin
    @app.route("/admin")
    def admin():
        require_reviewer()
        db = get_db()
        rows = db.execute(
            "SELECT model, COUNT(*) n, SUM(input_tokens) tin, SUM(output_tokens) tout "
            "FROM annotation GROUP BY model"
        ).fetchall()
        return render_template(
            "admin.html", rows=rows, usage=usage_report(), models=annotators()
        )

    # ----------------------------------------------------------------- helpers
    def _construct(db, cid):
        c = db.execute("SELECT * FROM construct WHERE id=?", (cid,)).fetchone()
        if not c or c["owner_id"] != g.reviewer["id"]:
            abort(404)
        # Where this reviewer last was, for the panel and for Resume. Set here
        # because this is already the chokepoint every construct route passes
        # through. Throttled on the same clock as presence.
        if g.reviewer["last_construct_id"] != cid or telemetry.should_touch(
            -g.reviewer["id"], every=30.0
        ):
            db.execute(
                "UPDATE reviewer SET last_construct_id=? WHERE id=?",
                (cid, g.reviewer["id"]),
            )
            db.commit()
        return c

    def _render_deltas(db, cid, rows):
        names = _criterion_names(service.load_criteria(db, cid, status="active"))
        out = []
        for d in rows:
            payload = service.enrich_delta(db, cid, d["kind"], jload(d["payload"], {}))
            out.append((
                d,
                service.describe_delta(d["kind"], payload, names),
                service.delta_warnings(d["kind"], payload),
            ))
        return out

    def _rule_load(db, cid):
        """How much work each rule is doing on the latest session."""
        from . import rewire

        rnd = db.execute(
            "SELECT session_id FROM round WHERE construct_id=? AND kind='review' "
            "ORDER BY id DESC LIMIT 1",
            (cid,),
        ).fetchone()
        rules = service.load_rules(db, cid)
        if rnd is None or rules is None:
            return []
        criteria = service.load_criteria(db, cid, status="active")
        scope = db.execute(
            "SELECT scope FROM construct WHERE id=?", (cid,)
        ).fetchone()["scope"]
        _, cons = service._consensus_over(db, cfg, rnd["session_id"], scope, criteria)
        universe = [
            u.index for u in corpus.get(cfg, rnd["session_id"]).in_scope(scope)
        ]
        return rewire.rule_load(criteria, cons, universe, rules)

    def _criterion_names(criteria, width=None):
        """Labels for referring to criteria in prose, e.g. staged rewires.

        Not truncated: a staged rewire reading "Rewire #54 The tutor asks a
        student to explain the reas…" gives you no way to tell which criterion
        it means. CSS handles overflow where space is tight.
        """
        return {
            cr.id: f"#{cr.id} {cr.text[:width] + '…' if width and len(cr.text) > width else cr.text}"
            for cr in criteria
        }

    def _criterion_examples(db, session, text, scope, k=3, span=EXAMPLE_WINDOW):
        """True/false firings for the Approve screen, from round 0's annotation.

        Each example carries its neighbours, because a criterion is judged "in
        the context of its session" and a bare line cannot be judged at all:
        "Okay" is a plausible firing or an obvious miss depending entirely on
        what preceded it. The template renders each as a small scrollable
        transcript centred on the target, the way Review does.
        """
        row = db.execute(
            "SELECT true_indices FROM annotation WHERE session_id=? AND criterion_text=? "
            "AND model=? ",
            (session.session_id, text, annotators()[0]),
        ).fetchone()
        hits = set(jload(row["true_indices"], [])) if row else set()
        in_scope = session.in_scope(scope)

        def with_context(utts):
            return [
                {"target": u.index, "window": session.window(u.index, span)}
                for u in utts
            ]

        fired = [u for u in in_scope if u.index in hits]
        missed = [u for u in in_scope if u.index not in hits]
        return {
            "true": with_context(fired[:k]),
            "false": with_context(missed[:k]),
            # Counted over in-scope utterances, so the two headings sum to the
            # annotatable part of the session and you can read the fire rate
            # off them directly.
            "n_true": len(fired),
            "n_false": len(missed),
        }

    def _outcome_for(res, gold):
        if gold is None:
            return None
        cands = set(res.candidates)
        if cands == {gold}:
            return "exact"
        if len(cands) == 0:
            return "under"
        return "over_contains" if gold in cands else "over_missing"

    def _metrics_summary(db, cid):
        c = db.execute("SELECT * FROM construct WHERE id=?", (cid,)).fetchone()
        labels = jload(c["label_space"], [])
        criteria = service.load_criteria(db, cid, status="active")
        rules = service.load_rules(db, cid)
        rounds = db.execute(
            # Newest first. The panel grows a block per round, and the one
            # you just finished is the one you came to read — it should not be
            # at the bottom of a list that gets longer every round.
            "SELECT * FROM round WHERE construct_id=? AND kind='review' "
            "ORDER BY id DESC", (cid,)
        ).fetchall()
        out = []
        for rnd in rounds:
            reviews = db.execute(
                "SELECT * FROM review WHERE round_id=? AND complete=1", (rnd["id"],)
            ).fetchall()
            if not reviews:
                continue
            session = corpus.get(cfg, rnd["session_id"])
            firings = {
                nick: annotate.firing_matrix(
                    db, rnd["session_id"], service.criteria_keys(criteria), nick
                )
                for nick in annotators()
            }
            consensus = {
                rv["utterance_index"]: service.consensus_firing(
                    firings, rv["utterance_index"], criteria
                )
                for rv in reviews
            }
            # Inter-model agreement is measured over the WHOLE session, not the
            # handful you reviewed. The reviewed set is chosen by the selection
            # ladder, which picks contentious utterances on purpose — so an
            # alpha over it is enriched for disagreement by construction and
            # reads far worse than the session. Measured on one round: -0.14
            # over the 8 reviewed against 0.29 over all 21 in-scope, same
            # criteria and same models. It also dropped criteria none of the
            # eight happened to trigger, losing a 1.00 from the average.
            #
            # This now matches the schema table and the board's agreement pane,
            # which were already whole-session. Needs no model calls: the
            # firing matrix is already loaded for the entire transcript.
            agreement_universe = [
                u.index for u in session.in_scope(c["scope"])
            ]
            # Gold-dependent figures stay on what you actually judged.
            universe = sorted(consensus)
            pairs = []
            for rv in reviews:
                if not rv["gold_label"]:
                    continue
                pairs.append(
                    (
                        apply_map(
                            criteria, consensus[rv["utterance_index"]], rules
                        ).candidates,
                        rv["gold_label"],
                    )
                )
            # Same flush the schema table's "disagreements" column applies:
            # drop verdicts recorded against wording that has since changed.
            # Without it, a reworded criterion is compared against the new
            # text's firings — of which there are none, since annotations are
            # keyed by text — so old "true" verdicts read as total
            # disagreement. That alone pulled a round where every comparable
            # criterion agreed perfectly down to alpha 0.55.
            human = {
                rv["utterance_index"]: service.live_verdicts(
                    db, cid, jload(rv["criterion_verdicts"], {}),
                    rnd["schema_version"], criteria,
                )
                for rv in reviews
            }
            out.append(
                {
                    "round": rnd["id"],
                    "session": rnd["session_id"],
                    "reviewed": len(reviews),
                    # Counts first, rates second: a rate is unreadable without
                    # knowing how many things it is over, and these are small
                    # samples. Named to match the session board's vocabulary.
                    "n_criteria": len(criteria),
                    # Assertion-level, not utterance-level: one reviewed
                    # utterance can disagree on several criteria at once, so
                    # this is not bounded by the number judged. Only verdicts
                    # that still describe the current wording count, so it is
                    # the schema table's per-criterion column summed down.
                    "n_disagreements": sum(
                        1
                        for idx, hv in human.items()
                        for cid, v in hv.items()
                        if bool(consensus[idx].get(cid)) != v
                    ),
                    "n_verdicts": sum(len(hv) for hv in human.values()),
                    "n_utterances": len(agreement_universe),
                    "n_judged": len(pairs),
                    "n_exact": sum(1 for cands, g in pairs if cands == {g}),
                    # Sample sizes for the two alphas, which are measured over
                    # different populations: cross-model over the whole
                    # session, model-human only where you recorded a verdict.
                    "n_model_alpha": len(agreement_universe),
                    "n_human_alpha": len(human),
                    "logic": metrics.schema_logic(criteria, consensus, universe),
                    "model_alpha": metrics.model_agreement(
                        criteria, firings, agreement_universe
                    ),
                    "human_alpha": metrics.human_agreement(criteria, consensus, human),
                    "predictive": metrics.predictive(pairs, labels) if pairs else None,
                }
            )
        return out

    def _edges_from_form(form, labels):
        """Edges are untyped, so the control is a checkbox per label."""
        return [
            {"label": lbl} for lbl in labels if form.get(f"edge_{lbl}") == "on"
        ]

    # Scoreboard chip wording for the non-exact gold outcomes of
    # `service.evaluate_gold`. On a binary construct only `wrong` can occur:
    # the catch-all always assigns a label, so `under` is unreachable, and the
    # catch-all rule collapses two candidates, so `over_contains` is too.
    MISMATCH_LABEL = {
        "wrong": "mismatch",
        "under": "got no label",
        "over_contains": "got several labels, yours among them",
    }

    app.jinja_env.globals.update(MISMATCH_LABEL=MISMATCH_LABEL)

    @app.context_processor
    def _models_in_templates():
        # Per request, not once at startup: the Models page rebinds roles
        # while the server runs, and a tooltip naming last week's annotators
        # is worse than none.
        return {
            "MODELS": annotators(),
            "MODEL_TITLES": [model_title(m) for m in annotators()],
        }
    app.jinja_env.filters["from_json"] = lambda s: jload(s, [])
    return app
