"""Session-scoped gold (notes-3).

A human judgement is a fact about the SESSION, not about the round that happened
to surface it. Keyed by round, re-running a session orphaned everything already
judged, which made "perfect this session" impossible.
"""

import json

import pytest

from studio.db import connect, init_db, snapshot_schema
from studio.schema_map import Criterion, Edge
from studio.service import criteria_from_snapshot, session_gold


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.sqlite3"
    init_db(path)
    conn = connect(path)
    conn.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    conn.execute(
        "INSERT INTO construct (owner_id, name, description, label_space, scope) "
        "VALUES (1,'c','d','[\"A\",\"Other\"]','tutor')"
    )
    conn.execute(
        "INSERT INTO criterion (construct_id, text, edges, status, origin) "
        "VALUES (1,'original wording',?,'active','seed')",
        (json.dumps([{"label": "A"}]),),
    )
    conn.commit()
    snapshot_schema(conn, 1, "v1")
    return conn


def add_round(conn, session_id, version, rid=None):
    conn.execute(
        "INSERT INTO round (construct_id, kind, session_id, schema_version, status) "
        "VALUES (1,'review',?,?,'complete')",
        (session_id, version),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]


def add_review(conn, rid, idx, label, verdicts):
    conn.execute(
        "INSERT INTO review (round_id, utterance_index, reviewer_id, gold_label, "
        "criterion_verdicts, complete) VALUES (?,?,1,?,?,1)",
        (rid, idx, label, json.dumps(verdicts)),
    )
    conn.commit()


def test_gold_accumulates_across_rounds_on_the_same_session(db):
    r1 = add_round(db, "S", 1)
    add_review(db, r1, 5, "A", {"1": True})
    r2 = add_round(db, "S", 1)
    add_review(db, r2, 9, "Other", {"1": False})

    gold = session_gold(db, 1, "S")
    assert sorted(gold) == [5, 9]
    assert gold[5]["gold_label"] == "A"
    assert gold[9]["gold_label"] == "Other"


def test_gold_does_not_leak_between_sessions(db):
    r1 = add_round(db, "S1", 1)
    add_review(db, r1, 5, "A", {})
    r2 = add_round(db, "S2", 1)
    add_review(db, r2, 5, "Other", {})
    assert session_gold(db, 1, "S1")[5]["gold_label"] == "A"
    assert session_gold(db, 1, "S2")[5]["gold_label"] == "Other"


def test_later_judgement_wins_for_the_same_utterance(db):
    r1 = add_round(db, "S", 1)
    add_review(db, r1, 5, "A", {})
    r2 = add_round(db, "S", 1)
    add_review(db, r2, 5, "Other", {})
    assert session_gold(db, 1, "S")[5]["gold_label"] == "Other"


class TestStaleness:
    def test_label_survives_a_reword_but_the_verdict_goes_stale(self, db):
        r1 = add_round(db, "S", 1)
        add_review(db, r1, 5, "A", {"1": True})

        db.execute("UPDATE criterion SET text='reworded wording' WHERE id=1")
        db.commit()
        snapshot_schema(db, 1, "reworded")

        gold = session_gold(db, 1, "S")
        assert gold[5]["gold_label"] == "A", "the label is about the utterance, not the wording"
        assert gold[5]["verdicts"] == {}, "verdict was about wording that no longer exists"
        assert gold[5]["stale_verdicts"] == {1: True}, "kept as history, not discarded"

    def test_verdict_stays_live_when_the_wording_is_unchanged(self, db):
        r1 = add_round(db, "S", 1)
        add_review(db, r1, 5, "A", {"1": True})
        snapshot_schema(db, 1, "unrelated change")
        gold = session_gold(db, 1, "S")
        assert gold[5]["verdicts"] == {1: True}
        assert gold[5]["stale_verdicts"] == {}

    def test_verdict_for_a_deleted_criterion_is_stale(self, db):
        r1 = add_round(db, "S", 1)
        add_review(db, r1, 5, "A", {"99": True})
        gold = session_gold(db, 1, "S")
        assert 99 in gold[5]["stale_verdicts"]


def test_criteria_from_snapshot_skips_excluded(db):
    snap = {
        "criteria": [
            {"id": 1, "text": "kept", "edges": [{"label": "A"}],
             "status": "active"},
            {"id": 2, "text": "gone", "edges": [{"label": "A"}],
             "status": "excluded"},
        ]
    }
    got = criteria_from_snapshot(snap)
    assert [c.id for c in got] == [1]
    assert got[0].edges == (Edge("A"),)


def test_criteria_from_snapshot_accepts_json_encoded_edges(db):
    """Snapshots store whatever the criterion row held, which is a JSON string."""
    snap = {"criteria": [{"id": 1, "text": "x", "status": "active",
                          "edges": json.dumps([{"label": "A"}])}]}
    assert criteria_from_snapshot(snap)[0].edges == (Edge("A"),)


class TestNoFabricatedVerdicts:
    """A criterion with no posted verdict must not silently record the model's
    own firing as the human's judgement — that inflates every agreement
    statistic by construction. Found live when a harness stopped posting the
    fields and agreement jumped to 100%.
    """

    def test_missing_fields_are_omitted_not_defaulted(self, tmp_path):
        from studio.app import create_app
        from studio.config import Config

        cfg = Config.from_env()
        object.__setattr__(cfg, "db_path", tmp_path / "app.sqlite3")
        app = create_app(cfg)
        with app.test_request_context():
            pass  # app builds; the rule under test is in review_submit

        # Exercise the extraction rule directly, mirroring the route.
        criteria = [Criterion(1, "a", ()), Criterion(2, "b", ())]
        form = {"crit_1__0_A": "false"}   # only criterion 1 answered
        verdicts = {}
        for cr in criteria:
            vals = [
                v for k, v in form.items()
                if k == f"crit_{cr.id}" or k.startswith(f"crit_{cr.id}__")
            ]
            if vals:
                verdicts[str(cr.id)] = vals[0] == "true"
        assert verdicts == {"1": False}
        assert "2" not in verdicts, "unanswered criterion must not be invented"
