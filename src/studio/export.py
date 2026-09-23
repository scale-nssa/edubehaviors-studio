"""Bundle export: the firing matrix and the assertion key, as a zip of two CSVs.

Reads only what is already cached. Nothing here annotates, so exporting is free
and can be done at any point.

The assertion id scheme is lifted from `../atoms/assertions-experiments`
(`export_annotations_wide.py`, `alpha_by_assertion.py`) so the two toolchains
join: `<category_slug>__<i>`, and `<assertion_id>.<short_model>` when more than
one model is in play. Assertion ids are `[a-z0-9_]` only, so splitting a column
name on its LAST dot recovers the pair unambiguously.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile

from . import annotate, corpus
from .config import Config, annotators
from .db import jload

MISSING = ""   # never annotated — distinct from FALSE. See `cell` below.
TRUE = "TRUE"
FALSE = "FALSE"


def slug(s: str) -> str:
    """Verbatim from atoms/assertions-experiments."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def short_model(nick: str) -> str:
    """A column-safe model name.

    The whole nickname, slugged: provider, model, reasoning setting and the
    `#2` of a second same-model sample all distinguish real columns, so none
    of it can be dropped without two annotators colliding in one header. The
    one exception is `openrouter_`, a prefix on nicknames from before local
    mode that carries no information, still found in carried-over databases.
    """
    return slug(nick.removeprefix("openrouter_"))


def assertion_ids(criteria, label_space: list[str]) -> dict[int, str]:
    """criterion id -> `<category_slug>__<i>`.

    The upstream scheme indexes assertions within a category. Our criteria are
    not grouped that way — a criterion carries zero or more edges — so the
    category is its **first label in label-space order**, and a criterion with
    no edges (a gate) goes under `gate`. Each criterion gets exactly one id, so
    the ids stay 1:1 with the columns of the firing CSV even when a criterion
    bears on several labels.
    """
    order = {lbl: i for i, lbl in enumerate(label_space)}
    counters: dict[str, int] = {}
    out: dict[int, str] = {}
    for c in criteria:
        labels = sorted(c.labels, key=lambda l: order.get(l, len(order)))
        cat = slug(labels[0]) if labels else "gate"
        i = counters.get(cat, 0)
        counters[cat] = i + 1
        out[c.id] = f"{cat}__{i}"
    return out


def _annotated(conn, session_id: str, key: str, model: str) -> set[int] | None:
    """Hits for one (session, criterion, model), or None if never annotated.

    The distinction is the whole point of the blank cell: no row means the
    question was never put to the model, which is not the same as FALSE.
    """
    row = conn.execute(
        "SELECT true_indices FROM annotation "
        "WHERE session_id=? AND criterion_text=? AND model=?",
        (session_id, key, model),
    ).fetchone()
    return set(jload(row["true_indices"], [])) if row else None


def firing_csv(conn, cfg: Config, construct_id: int, models: list[str]) -> str:
    """One row per in-scope utterance of every session this construct annotated."""
    from .service import annotation_key, load_criteria, used_sessions

    row = conn.execute(
        "SELECT label_space, scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    labels, scope = jload(row["label_space"], []), row["scope"]
    criteria = load_criteria(conn, construct_id, status="active")
    ids = assertion_ids(criteria, labels)
    by_id = {c.id: c for c in criteria}

    multi = len(models) > 1
    cols = [
        f"{ids[c.id]}.{short_model(m)}" if multi else ids[c.id]
        for c in criteria
        for m in models
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["session_id", "utterance_index", "role", "utterance"] + cols)

    for sid in sorted(used_sessions(conn, construct_id)):
        try:
            session = corpus.get(cfg, sid)
        except KeyError:
            # A session that is no longer in the pinned corpus. Skipping keeps
            # the export honest rather than emitting rows with no text.
            continue

        # (criterion, model) -> hits, or None for never-annotated.
        hits: dict[tuple[int, str], set[int] | None] = {}
        gates: dict[tuple[int, str], set[int] | None] = {}
        for c in criteria:
            parent = by_id.get(c.parent_id) if c.parent_id else None
            key = annotation_key(c.text, parent.text if parent else None)
            for m in models:
                hits[(c.id, m)] = _annotated(conn, sid, key, m)
                # A sub-criterion is only defined where its own model's gate
                # fired; elsewhere it was never asked.
                gates[(c.id, m)] = (
                    _annotated(conn, sid, parent.text, m) if parent else None
                )

        for u in session.in_scope(scope):
            cells = []
            for c in criteria:
                for m in models:
                    h = hits[(c.id, m)]
                    if h is None:
                        cells.append(MISSING)
                        continue
                    if c.parent_id:
                        g = gates[(c.id, m)]
                        if g is None or u.index not in g:
                            cells.append(MISSING)
                            continue
                    cells.append(TRUE if u.index in h else FALSE)
            w.writerow([sid, u.index, u.role, u.text] + cells)

    return buf.getvalue()


def assertions_csv(conn, construct_id: int) -> str:
    """The key: one row per assertion, so it joins 1:1 with the columns above."""
    from .service import load_criteria

    row = conn.execute(
        "SELECT label_space FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    labels = jload(row["label_space"], [])
    criteria = load_criteria(conn, construct_id, status="active")
    ids = assertion_ids(criteria, labels)
    order = {lbl: i for i, lbl in enumerate(labels)}

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name", "shortname", "labels"])
    for c in criteria:
        ordered = sorted(c.labels, key=lambda l: order.get(l, len(order)))
        w.writerow([c.text, ids[c.id], "|".join(ordered)])
    return buf.getvalue()


def available_models(conn, construct_id: int) -> list[str]:
    """Models that actually annotated this construct's sessions.

    Not the configured pair. Annotations are keyed by model nickname and kept
    forever, so a construct worked before a model change holds rows under the
    old names; defaulting to `config.annotators()` then exports a matrix that is
    entirely blank and looks like a bug. Configured models sort first, and are
    listed even with no rows yet so a fresh construct still offers a choice.
    """
    from .service import used_sessions

    sessions = used_sessions(conn, construct_id)
    found: set[str] = set()
    if sessions:
        marks = ",".join("?" * len(sessions))
        found = {
            r["model"]
            for r in conn.execute(
                f"SELECT DISTINCT model FROM annotation WHERE session_id IN ({marks})",
                tuple(sorted(sessions)),
            )
        }
    configured = [m for m in annotators()]
    return configured + sorted(found - set(configured))


def resolve_models(requested: str | None, available: list[str] | None = None) -> list[str]:
    """`?models=` -> nicknames. Accepts 'both', a nickname, or a short name."""
    pool = available if available is not None else list(annotators())
    if not requested or requested == "both":
        return list(pool)
    for m in pool:
        if requested in (m, short_model(m)):
            return [m]
    raise ValueError(
        f"unknown model {requested!r}; expected 'both' or one of "
        f"{[short_model(m) for m in pool]}"
    )


def bundle(conn, cfg: Config, construct_id: int, models: list[str]) -> bytes:
    """Both CSVs in one zip."""
    row = conn.execute(
        "SELECT name, version FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    stem = f"{slug(row['name']) or 'construct'}_v{row['version']}"

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}/annotations.csv", firing_csv(conn, cfg, construct_id, models))
        z.writestr(f"{stem}/assertions.csv", assertions_csv(conn, construct_id))
        z.writestr(
            f"{stem}/README.txt",
            "annotations.csv  one row per in-scope utterance of every session this\n"
            "                 construct has annotated. Cells are TRUE/FALSE, or\n"
            "                 EMPTY where the model was never asked — a criterion\n"
            "                 added after that session was annotated, a failed\n"
            "                 call, or a sub-criterion on an utterance where its\n"
            "                 gate did not fire. EMPTY is not FALSE.\n"
            "assertions.csv   name, shortname, labels — one row per assertion,\n"
            "                 joining 1:1 with the columns above.\n\n"
            f"models: {', '.join(models)}\n"
            + (
                "column names are <shortname>.<model>; split on the LAST dot.\n"
                if len(models) > 1
                else "one model, so columns are the bare shortname.\n"
            ),
        )
    return out.getvalue()
