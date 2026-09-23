"""Rewording a criterion flushes its disagreement count.

The fire rate and alpha reset on their own after a reword: both are computed
from annotations keyed by criterion *text*, and the new text has none yet. The
human column did not, because verdicts are keyed by criterion *id* — so a
count like "3 / 20" survived a rewrite and described wording nobody had judged.
"""

import json

import pytest

from studio.config import Config
from studio.db import connect, init_db, snapshot_schema
from studio.service import criterion_stats


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "t.sqlite3")
    conn = connect(tmp_path / "t.sqlite3")
    conn.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    conn.execute(
        "INSERT INTO construct (owner_id,name,description,label_space,scope) "
        "VALUES (1,'c','d','[\"A\",\"B\"]','tutor')"
    )
    conn.execute(
        "INSERT INTO criterion (construct_id, text, edges, status, origin) "
        "VALUES (1,'original wording',?,'active','seed')",
        (json.dumps([{"label": "A"}]),),
    )
    conn.commit()
    snapshot_schema(conn, 1, "initial")
    return conn


def version(conn):
    return conn.execute("SELECT version FROM construct WHERE id=1").fetchone()["version"]


def add_judged_round(conn, *, verdict, n=3):
    """A round at the current schema version, with `n` judged utterances."""
    v = version(conn)
    conn.execute(
        "INSERT INTO round (construct_id,kind,session_id,schema_version,status) "
        "VALUES (1,'review','S',?, 'reviewing')", (v,),
    )
    rid = conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
    for i in range(n):
        conn.execute(
            "INSERT INTO review (round_id, reviewer_id, utterance_index, gold_label, "
            "criterion_verdicts, complete) VALUES (?,1,?, 'A', ?, 1)",
            (rid, i, json.dumps({"1": verdict})),
        )
    conn.commit()
    return rid


def stats(conn, cfg):
    return criterion_stats(conn, cfg, 1)[1]


@pytest.fixture()
def cfg(tmp_path):
    base = Config.from_env()
    return Config(host="127.0.0.1", port=0, db_path=tmp_path / "t.sqlite3",
                  dataset_path=base.dataset_path, secret_key="k", admin_token="t")


class TestFlushOnReword:
    def test_verdicts_count_against_the_wording_they_were_made_on(self, db, cfg):
        add_judged_round(db, verdict=True)
        s = stats(db, cfg)
        assert s["human_total"] == 3

    def test_rewording_flushes_the_count_to_nothing(self, db, cfg):
        add_judged_round(db, verdict=True)
        db.execute("UPDATE criterion SET text='rewritten wording' WHERE id=1")
        db.commit()
        snapshot_schema(db, 1, "edited criterion 1")
        s = stats(db, cfg)
        assert s["human_total"] == 0
        assert s["human_flips"] == 0

    def test_judgements_after_the_reword_count_again(self, db, cfg):
        add_judged_round(db, verdict=True)
        db.execute("UPDATE criterion SET text='rewritten wording' WHERE id=1")
        db.commit()
        snapshot_schema(db, 1, "edited criterion 1")
        add_judged_round(db, verdict=True, n=2)
        s = stats(db, cfg)
        assert s["human_total"] == 2   # only the new ones

    def test_an_unrelated_schema_version_bump_does_not_flush(self, db, cfg):
        """Only a change to THIS criterion's wording should clear it — adding
        another criterion bumps the version but leaves the text alone."""
        add_judged_round(db, verdict=True)
        db.execute(
            "INSERT INTO criterion (construct_id, text, edges, status, origin) "
            "VALUES (1,'a second criterion',?,'active','user')",
            (json.dumps([{"label": "B"}]),),
        )
        db.commit()
        snapshot_schema(db, 1, "added a criterion by hand")
        assert stats(db, cfg)["human_total"] == 3
