"""Binary constructs report the positive class.

Two labels with one of them the catch-all is not "which of these?" but "does
this occur?". A candidate-set breakdown over such a pair says less than a count
of positives, and nothing-fired is the negative answer rather than a hole in
the schema — so the display switches. Every other shape is left alone.
"""

import json

import pytest

from studio.db import connect, init_db
from studio.service import positive_label


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "t.sqlite3"
    init_db(path)
    c = connect(path)
    c.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    return c


def make(conn, labels, other):
    conn.execute(
        "INSERT INTO construct (owner_id, name, description, label_space, scope, "
        "other_label) VALUES (1,'c','d',?,'tutor',?)",
        (json.dumps(labels), other),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]


class TestPositiveLabel:
    def test_binary_with_catch_all(self, conn):
        cid = make(conn, ["Revoicing", "Other"], "Other")
        assert positive_label(conn, cid) == "Revoicing"

    def test_order_does_not_matter(self, conn):
        cid = make(conn, ["Other", "Revoicing"], "Other")
        assert positive_label(conn, cid) == "Revoicing"

    def test_three_labels_is_not_binary(self, conn):
        cid = make(conn, ["A", "B", "Other"], "Other")
        assert positive_label(conn, cid) is None

    def test_two_labels_without_a_catch_all_is_a_real_choice(self, conn):
        """A vs B with no residual is a genuine two-way classification; neither
        side is the 'positive' one."""
        cid = make(conn, ["A", "B"], None)
        assert positive_label(conn, cid) is None

    def test_catch_all_outside_the_label_space_is_ignored(self, conn):
        """The label space can be edited out from under other_label."""
        cid = make(conn, ["A", "B"], "None")
        assert positive_label(conn, cid) is None

    def test_missing_construct(self, conn):
        assert positive_label(conn, 999) is None
