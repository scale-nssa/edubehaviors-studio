"""Rollback and the branching that falls out of it (spec Q120/Q222)."""

import json

import pytest

from studio.db import connect, init_db, jload, snapshot_schema
from studio.service import restore_version, version_tree


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.sqlite3"
    init_db(path)
    conn = connect(path)
    conn.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    conn.execute(
        "INSERT INTO construct (owner_id, name, description, label_space, scope) "
        "VALUES (1,'c','desc','[\"A\",\"Other\"]','tutor')"
    )
    conn.commit()
    snapshot_schema(conn, 1, "initial")   # v1, as create_app does
    return conn


def add(conn, text, status="active"):
    cur = conn.execute(
        "INSERT INTO criterion (construct_id, text, edges, status, origin) "
        "VALUES (1,?,?,?,'seed')",
        (text, json.dumps([{"label": "A"}]), status),
    )
    conn.commit()
    return cur.lastrowid


def active_texts(conn):
    return {
        r["id"]: r["text"]
        for r in conn.execute(
            "SELECT id, text FROM criterion WHERE construct_id=1 AND status='active'"
        )
    }


def test_restore_brings_back_old_text_and_appends_a_version(db):
    a = add(db, "original")
    snapshot_schema(db, 1, "v2 baseline")           # v2 holds "original"
    db.execute("UPDATE criterion SET text='edited' WHERE id=?", (a,))
    db.commit()
    snapshot_schema(db, 1, "v3 edit")               # v3 holds "edited"

    new_v = restore_version(db, 1, 2)               # -> v4, content of v2
    assert new_v == 4
    assert active_texts(db)[a] == "original"
    assert db.execute("SELECT version FROM construct WHERE id=1").fetchone()[0] == 4


def test_restore_is_append_only_nothing_is_deleted(db):
    add(db, "one")
    snapshot_schema(db, 1, "v2")
    snapshot_schema(db, 1, "v3")
    restore_version(db, 1, 2)
    versions = [r[0] for r in db.execute(
        "SELECT version FROM schema_version WHERE construct_id=1 ORDER BY version"
    )]
    assert versions == [1, 2, 3, 4]


def test_restore_parents_the_new_version_to_the_restored_one(db):
    add(db, "one")
    snapshot_schema(db, 1, "v2")
    snapshot_schema(db, 1, "v3")
    restore_version(db, 1, 2)
    parent = db.execute(
        "SELECT parent_version FROM schema_version WHERE construct_id=1 AND version=4"
    ).fetchone()[0]
    assert parent == 2, "v4 should hang off v2, making v3 a sibling branch"


def test_criteria_added_after_the_restore_point_are_excluded_not_deleted(db):
    """Deleting would orphan review.criterion_verdicts, which is keyed by id."""
    a = add(db, "kept")
    snapshot_schema(db, 1, "v2")
    b = add(db, "added later")
    snapshot_schema(db, 1, "v3")

    restore_version(db, 1, 2)
    assert set(active_texts(db)) == {a}
    row = db.execute("SELECT status FROM criterion WHERE id=?", (b,)).fetchone()
    assert row is not None, "criterion row was deleted"
    assert row["status"] == "excluded"


def test_restoring_forward_again_reinstates_the_criterion(db):
    a = add(db, "kept")
    snapshot_schema(db, 1, "v2")
    b = add(db, "added later")
    snapshot_schema(db, 1, "v3")
    restore_version(db, 1, 2)          # v4: b excluded
    restore_version(db, 1, 3)          # v5: b back
    assert set(active_texts(db)) == {a, b}


def test_description_and_label_space_are_restored(db):
    add(db, "one")
    snapshot_schema(db, 1, "v2")
    db.execute("UPDATE construct SET description='changed', label_space='[\"Z\"]' WHERE id=1")
    db.commit()
    snapshot_schema(db, 1, "v3")
    restore_version(db, 1, 2)
    row = db.execute("SELECT description, label_space FROM construct WHERE id=1").fetchone()
    assert row["description"] == "desc"
    assert jload(row["label_space"], []) == ["A", "Other"]


def test_unknown_version_raises(db):
    with pytest.raises(KeyError):
        restore_version(db, 1, 99)


class TestVersionTree:
    def test_linear_history_has_increasing_depth(self, db):
        add(db, "one")
        snapshot_schema(db, 1, "v2")
        snapshot_schema(db, 1, "v3")
        tree = version_tree(db, 1)
        assert [v["version"] for v in tree] == [1, 2, 3]
        assert [v["depth"] for v in tree] == [0, 1, 2]

    def test_a_restore_creates_a_visible_branch(self, db):
        add(db, "one")
        snapshot_schema(db, 1, "v2")
        snapshot_schema(db, 1, "v3")
        restore_version(db, 1, 2)                      # v4 parented to v2
        tree = {v["version"]: v for v in version_tree(db, 1)}
        assert tree[3]["parent"] == 2
        assert tree[4]["parent"] == 2
        assert tree[3]["depth"] == tree[4]["depth"], "siblings should share a depth"

    def test_diff_counts_against_parent(self, db):
        a = add(db, "one")
        snapshot_schema(db, 1, "v2")
        add(db, "two")
        db.execute("UPDATE criterion SET text='one edited' WHERE id=?", (a,))
        db.commit()
        snapshot_schema(db, 1, "v3")
        v3 = {v["version"]: v for v in version_tree(db, 1)}[3]
        assert v3["added"] == 1
        assert v3["changed"] == 1
        assert v3["removed"] == 0
