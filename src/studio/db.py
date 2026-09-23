"""SQLite persistence. WAL, short write transactions, no migrations (Q158)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviewer (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name  TEXT NOT NULL UNIQUE,
    -- Presence, for the control panel only. Nothing in the reviewer-facing
    -- flow depends on either column; both are written throttled.
    last_seen_at      TEXT,
    last_construct_id INTEGER,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS construct (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id      INTEGER NOT NULL REFERENCES reviewer(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    label_space   TEXT NOT NULL DEFAULT '[]',      -- JSON array
    scope         TEXT NOT NULL DEFAULT 'tutor'
                    CHECK (scope IN ('tutor','student','all')),
    phase         TEXT NOT NULL DEFAULT 'describe'
                    CHECK (phase IN ('describe','approve','annotate','review','revise')),
    -- Starts at 0 so the first snapshot_schema() call writes v1 with a NULL
    -- parent. Starting at 1 leaves the history rooted at a version row that
    -- never exists, and the tree walk finds nothing to hang off.
    version       INTEGER NOT NULL DEFAULT 0,
    -- The revise agent's evolving theory of what is still wrong with the schema.
    -- Rewritten each Revise round, fed back into P4, and editable by the user —
    -- you should be able to correct the agent's theory directly (note 21).
    scratchpad    TEXT NOT NULL DEFAULT '',
    -- Which corpus this construct's rounds draw from. Chosen once, on the
    -- New Construct page, and fixed thereafter: every round, every cached
    -- annotation and every gold label is tied to transcripts from this one.
    dataset       TEXT NOT NULL DEFAULT 'talkmoves',
    -- Which label is the catch-all. Designating one switches on the two default
    -- boundary rules; leaving it NULL keeps the old behaviour exactly.
    other_label   TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Full snapshots, not deltas (Q221): rollback and branching stay trivial.
CREATE TABLE IF NOT EXISTS schema_version (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    version       INTEGER NOT NULL,
    parent_version INTEGER,                        -- NULL for v1; branching falls out of this
    snapshot      TEXT NOT NULL,                   -- JSON {description,label_space,criteria}
    note          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (construct_id, version)
);

-- A reword trial: a candidate rewording held while we measure what it does to
-- inter-model and human agreement, before it touches the live criterion.
CREATE TABLE IF NOT EXISTS reword_trial (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    criterion_id  INTEGER NOT NULL REFERENCES criterion(id) ON DELETE CASCADE,
    old_text      TEXT NOT NULL,
    new_text      TEXT NOT NULL,
    rationale     TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('proposed','measuring','measured','accepted','rejected')),
    before_json   TEXT NOT NULL DEFAULT '{}',
    after_json    TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Boundary rules: pairwise precedence between labels, consulted only when the
-- evidence has already failed to settle an utterance.
CREATE TABLE IF NOT EXISTS label_rule (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    winner        TEXT NOT NULL,
    loser         TEXT NOT NULL,
    note          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (construct_id, winner, loser)
);

-- Free-text about a specific label boundary, gathered while reviewing. Keyed by
-- the PAIR rather than the utterance so Revise can hand the model a coherent
-- pile of evidence about one boundary instead of scattered notes.
CREATE TABLE IF NOT EXISTS boundary_note (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    label_a       TEXT NOT NULL,
    label_b       TEXT NOT NULL,
    session_id    TEXT NOT NULL DEFAULT '',
    utterance_index INTEGER,
    note          TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS criterion (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    edges         TEXT NOT NULL,                   -- JSON [{label,kind}]
    status        TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','active','excluded')),
    origin        TEXT NOT NULL DEFAULT 'seed'
                    CHECK (origin IN ('seed','user','llm-revision')),
    -- Two-level nesting. A parent is a pure gate: it carries no edges and its
    -- children are only annotated on utterances where it fired. Depth is capped
    -- at 2 — a child may not itself be a parent.
    parent_id     INTEGER REFERENCES criterion(id) ON DELETE CASCADE,
    ord           INTEGER NOT NULL DEFAULT 0
);

-- Cache key is (session, criterion TEXT, model) so the bank is shared across
-- constructs (Q223) and reword invalidates for free (Q41).
CREATE TABLE IF NOT EXISTS annotation (
    session_id     TEXT NOT NULL,
    criterion_text TEXT NOT NULL,
    model          TEXT NOT NULL,
    true_indices   TEXT NOT NULL,                  -- JSON array of ints
    call_id        TEXT NOT NULL DEFAULT '',
    warnings       TEXT NOT NULL DEFAULT '{}',
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, criterion_text, model)
);

CREATE TABLE IF NOT EXISTS round (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id   INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL DEFAULT 'review' CHECK (kind IN ('approve','review')),
    session_id     TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    status         TEXT NOT NULL DEFAULT 'annotating'
                     CHECK (status IN ('annotating','reviewing','complete','failed')),
    error          TEXT NOT NULL DEFAULT '',
    started_at     TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at   TEXT
);

CREATE TABLE IF NOT EXISTS round_utterance (
    round_id        INTEGER NOT NULL REFERENCES round(id) ON DELETE CASCADE,
    utterance_index INTEGER NOT NULL,
    tier            TEXT NOT NULL,
    ord             INTEGER NOT NULL,
    PRIMARY KEY (round_id, utterance_index)
);

CREATE TABLE IF NOT EXISTS review (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id           INTEGER NOT NULL REFERENCES round(id) ON DELETE CASCADE,
    utterance_index    INTEGER NOT NULL,
    reviewer_id        INTEGER NOT NULL REFERENCES reviewer(id) ON DELETE CASCADE,
    gold_label         TEXT,
    outcome            TEXT,                       -- exact|over_contains|over_missing|under
    criterion_verdicts TEXT NOT NULL DEFAULT '{}', -- JSON {criterion_id: bool}  (human-corrected firing)
    note               TEXT NOT NULL DEFAULT '',
    complete           INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (round_id, utterance_index)
);

CREATE TABLE IF NOT EXISTS delta (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    round_id      INTEGER REFERENCES round(id) ON DELETE SET NULL,
    kind          TEXT NOT NULL,                   -- add_criterion|remove_criterion|reword|reassign|change_kind|edit_description
    payload       TEXT NOT NULL,                   -- JSON
    rationale     TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'staged'
                    CHECK (status IN ('staged','applied','rejected')),
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS question (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id  INTEGER NOT NULL REFERENCES construct(id) ON DELETE CASCADE,
    round_id      INTEGER REFERENCES round(id) ON DELETE SET NULL,
    text          TEXT NOT NULL,
    answer        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per model call that reported usage. Local mode spends the user's
-- own money, so the ledger outlives the process (see spend.py). cost_usd is
-- NULL when the model's price is unknown, rather than 0, which would read as
-- free.
CREATE TABLE IF NOT EXISTS spend_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model           TEXT NOT NULL,
    profile         TEXT NOT NULL DEFAULT 'default',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,   -- reasoning included
    thinking_tokens INTEGER,
    cost_usd        REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_criterion_construct ON criterion(construct_id, status);
CREATE INDEX IF NOT EXISTS idx_round_construct ON round(construct_id, status);
CREATE INDEX IF NOT EXISTS idx_review_round ON review(round_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect(current_app.config["DB_PATH"])
    return g.db


def close_db(_exc=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(SCHEMA)
        # Idempotent add for databases created before branching existed.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(schema_version)")}
        if "parent_version" not in cols:
            conn.execute("ALTER TABLE schema_version ADD COLUMN parent_version INTEGER")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(construct)")}
        if "other_label" not in cols:
            conn.execute("ALTER TABLE construct ADD COLUMN other_label TEXT")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(criterion)")}
        if "parent_id" not in cols:
            conn.execute("ALTER TABLE criterion ADD COLUMN parent_id INTEGER")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(construct)")}
        if "dataset" not in cols:
            conn.execute(
                "ALTER TABLE construct ADD COLUMN dataset TEXT NOT NULL "
                "DEFAULT 'talkmoves'"
            )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(round)")}
        if "chained" not in cols:
            # 1 when the app started this round itself (service._maybe_chain),
            # 0 when the reviewer asked for it. They need different landings
            # when annotation finishes — see the progress route.
            conn.execute(
                "ALTER TABLE round ADD COLUMN chained INTEGER NOT NULL DEFAULT 0"
            )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(reviewer)")}
        if "last_seen_at" not in cols:
            conn.execute("ALTER TABLE reviewer ADD COLUMN last_seen_at TEXT")
        if "last_construct_id" not in cols:
            conn.execute("ALTER TABLE reviewer ADD COLUMN last_construct_id INTEGER")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(construct)")}
        if "scratchpad" not in cols:
            conn.execute(
                "ALTER TABLE construct ADD COLUMN scratchpad TEXT NOT NULL DEFAULT ''"
            )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Small helpers used across routes
# --------------------------------------------------------------------------- #

def jload(value, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def snapshot_schema(
    conn: sqlite3.Connection,
    construct_id: int,
    note: str,
    parent: int | None = None,
) -> int:
    """Write a full snapshot and bump the construct version (Q119, Q221).

    `parent` defaults to the version being superseded, so ordinary edits form a
    chain. A rollback passes the version it restored, which makes the history a
    tree — that is the whole of branching (Q120, Q222).
    """
    row = conn.execute(
        "SELECT description, label_space, version FROM construct WHERE id=?",
        (construct_id,),
    ).fetchone()
    criteria = conn.execute(
        "SELECT id, text, edges, status, origin, ord FROM criterion "
        "WHERE construct_id=? ORDER BY ord, id",
        (construct_id,),
    ).fetchall()
    new_version = row["version"] + 1
    if parent is not None:
        parent_version = parent
    else:
        parent_version = row["version"] or None  # v1 has no parent
    rules = conn.execute(
        "SELECT winner, loser, note FROM label_rule WHERE construct_id=? ORDER BY id",
        (construct_id,),
    ).fetchall()
    other = conn.execute(
        "SELECT other_label FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["other_label"]
    snapshot = {
        "description": row["description"],
        "label_space": jload(row["label_space"], []),
        "criteria": [dict(c) for c in criteria],
        "other_label": other,
        "rules": [dict(r) for r in rules],
    }
    conn.execute(
        "INSERT INTO schema_version (construct_id, version, parent_version, snapshot, note) "
        "VALUES (?,?,?,?,?)",
        (construct_id, new_version, parent_version, json.dumps(snapshot), note),
    )
    conn.execute("UPDATE construct SET version=? WHERE id=?", (new_version, construct_id))
    conn.commit()
    return new_version
