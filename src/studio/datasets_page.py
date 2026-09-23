"""The Datasets page: what is installed, and bringing your own.

Most of the intended audience has their own transcripts and no interest in
TalkMoves, which ships so the app does something on first run. Upload is four
steps on two pages:

  1. choose a CSV                                   (index -> upload)
  2. map columns to session id, speaker, text,      (map, GET)
     optionally an order column and a gold label
  3. map every speaker value to tutor / student / other / drop
  4. name it, see the counts, convert and pin       (map, POST; then create)

The uploaded file is staged in the data directory between steps rather than
held in the session cookie: transcripts are large, and the cookie is signed,
not encrypted.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from . import corpus, dataset_upload as up
from .config import MAX_SESSION_UTTERANCES
from .db import get_db

bp = Blueprint("datasets", __name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _cfg():
    return current_app.config["CFG"]


def _require_user():
    if not g.get("reviewer"):
        abort(redirect(url_for("start")))


STAGING_TTL_S = 24 * 3600


def _staging(cfg) -> Path:
    d = corpus.user_dir(cfg) / "_staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sweep_staging(cfg) -> None:
    """Drop uploads abandoned mid-mapping. They are copies of someone's
    transcripts, so they should not linger in the data directory forever."""
    cutoff = time.time() - STAGING_TTL_S
    for p in _staging(cfg).iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _staged(cfg, token: str) -> tuple[Path, Path]:
    if not token.isalnum():
        abort(404)
    base = _staging(cfg)
    return base / f"{token}.csv", base / f"{token}.name"


def _in_use(dataset_id: str) -> list:
    return get_db().execute(
        "SELECT c.id, c.name, r.display_name AS owner FROM construct c "
        "JOIN reviewer r ON r.id = c.owner_id WHERE c.dataset=? ORDER BY c.id",
        (dataset_id,),
    ).fetchall()


@bp.route("/datasets")
def index():
    _require_user()
    cfg = _cfg()
    rows = []
    for d in corpus.datasets(cfg):
        err = None
        try:
            n = len(corpus.all_sessions(cfg, d["id"]))
        except Exception as exc:  # noqa: BLE001 — a broken dataset must not hide the rest
            n, err = 0, str(exc)
        rows.append({**d, "loaded_sessions": n, "error": err,
                     "constructs": len(_in_use(d["id"]))})
    return render_template("datasets.html", datasets=rows)


@bp.route("/datasets/upload", methods=["POST"])
def upload():
    _require_user()
    cfg = _cfg()
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Choose a CSV file first.")
        return redirect(url_for("datasets.index"))
    raw = f.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        flash("That file is over 100 MB.")
        return redirect(url_for("datasets.index"))
    try:
        text, _enc = up.decode(raw)
        up.read_rows(text)   # fail now, on the page they are looking at
    except up.UploadError as exc:
        flash(f"Could not read {f.filename}: {exc}")
        return redirect(url_for("datasets.index"))
    _sweep_staging(cfg)
    token = secrets.token_hex(8)
    csv_path, name_path = _staged(cfg, token)
    csv_path.write_bytes(raw)
    name_path.write_text(f.filename)
    return redirect(url_for("datasets.map_columns", token=token))


def _mapping_from_form(form, header) -> up.Mapping:
    def col(name):
        v = form.get(name) or ""
        return v if v in header else None

    roles = {}
    for key, value in form.items():
        if key.startswith("role__"):
            spk = form.get("spk__" + key[len("role__"):], "")
            if value in up.ROLES:
                roles[spk] = value
    return up.Mapping(
        session_id=col("col_session_id") or "", speaker=col("col_speaker") or "",
        text=col("col_text") or "", order=col("col_order"), gold=col("col_gold"),
        roles=roles,
    )


@bp.route("/datasets/map/<token>", methods=["GET", "POST"])
def map_columns(token: str):
    _require_user()
    cfg = _cfg()
    csv_path, name_path = _staged(cfg, token)
    if not csv_path.exists():
        flash("That upload has expired; choose the file again.")
        return redirect(url_for("datasets.index"))
    text, encoding = up.decode(csv_path.read_bytes())
    header, rows = up.read_rows(text)
    source = name_path.read_text() if name_path.exists() else csv_path.name

    if request.method == "GET":
        cols = up.guess_columns(header)
        form = {f"col_{k}": v for k, v in cols.items()}
        stem = Path(source).stem
        form["title"] = stem
        form["dataset_id"] = up.slugify(stem) or "my_dataset"
    else:
        form = request.form.to_dict()

    speaker_col = form.get("col_speaker")
    speakers = up.speaker_values(rows, speaker_col) if speaker_col in header else []
    # Carry roles the user already chose across a re-render; guess the rest.
    chosen = {}
    for i, (value, _n) in enumerate(speakers):
        same = form.get(f"spk__{i}") == value   # not a different column's values
        chosen[value] = (form.get(f"role__{i}") if same else None) or up.guess_role(value)

    built = None
    problems: list[str] = []
    if request.method == "POST":
        mapping = _mapping_from_form(request.form, header)
        ds_id = (form.get("dataset_id") or "").strip()
        if not up.ID_RE.match(ds_id):
            problems.append("The short name must be 2–41 characters of lowercase "
                            "letters, digits and underscores, starting with a letter or digit.")
        elif ds_id in corpus.dataset_ids(cfg):
            problems.append(f"A dataset called {ds_id!r} already exists.")
        try:
            built = up.build(rows, mapping, ds_id or "x")
        except up.UploadError as exc:
            problems.append(str(exc))
        if built and not built.role_counts.get("tutor") and not built.role_counts.get("student"):
            problems.append("No speaker is mapped to tutor or student, so nothing "
                            "could ever be annotated.")
        if request.form.get("create") == "1" and built and not problems:
            entry = up.write(
                built, corpus.user_dir(cfg), ds_id,
                (form.get("title") or ds_id).strip(),
                (form.get("description") or "").strip(), source,
            )
            csv_path.unlink(missing_ok=True)
            name_path.unlink(missing_ok=True)
            flash(f"Added {entry['title']}: {entry['sessions']} sessions, "
                  f"{entry['utterances']} utterances. Pick it on New construct.")
            return redirect(url_for("datasets.detail", dataset_id=ds_id))

    too_long = 0
    if built:
        too_long = sum(1 for s in built.sessions
                       if len(s["utterances"]) > MAX_SESSION_UTTERANCES)
    return render_template(
        "dataset_map.html",
        token=token, header=header, preview=rows[:up.PREVIEW_ROWS],
        n_rows=len(rows), source=source, encoding=encoding, form=form,
        speakers=speakers, chosen=chosen, roles=up.ROLES,
        speakers_truncated=speaker_col in header and len(
            {(r.get(speaker_col) or "").strip() for r in rows}) > len(speakers),
        built=built, problems=problems, too_long=too_long,
        max_len=MAX_SESSION_UTTERANCES,
    )


@bp.route("/datasets/<dataset_id>")
def detail(dataset_id: str):
    _require_user()
    cfg = _cfg()
    try:
        d = corpus.entry(cfg, dataset_id)
        sessions = corpus.all_sessions(cfg, dataset_id)
    except KeyError:
        abort(404)
    chosen = request.args.get("s") or (sessions[0].session_id if sessions else None)
    session = next((s for s in sessions if s.session_id == chosen), None)
    summary = [
        {
            "id": s.session_id,
            "n": len(s.utterances),
            "tutor": len(s.in_scope("tutor")),
            "student": len(s.in_scope("student")),
            "too_long": len(s.utterances) > MAX_SESSION_UTTERANCES,
        }
        for s in sessions
    ]
    return render_template(
        "dataset_detail.html", d=d, sessions=summary, session=session,
        in_use=_in_use(dataset_id), max_len=MAX_SESSION_UTTERANCES,
    )


@bp.route("/datasets/<dataset_id>/delete", methods=["POST"])
def delete(dataset_id: str):
    _require_user()
    cfg = _cfg()
    try:
        d = corpus.entry(cfg, dataset_id)
    except KeyError:
        abort(404)
    if d.get("bundled"):
        flash(f"{d['title']} ships with the app and cannot be deleted.")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))
    users = _in_use(dataset_id)
    if users:
        # A construct's dataset is a hard reference, and its cached
        # annotations and gold are keyed to session ids that would vanish
        # underneath it. Refuse, and say exactly what is in the way.
        names = ", ".join(f"“{u['name']}”" for u in users)
        flash(f"Not deleted: {len(users)} construct(s) use {d['title']} — {names}. "
              "Their rounds, annotations and judgements all point at its sessions.")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))
    up.remove(corpus.user_dir(cfg), dataset_id)
    flash(f"Deleted {d['title']}.")
    return redirect(url_for("datasets.index"))
