"""The construct page's action buttons and session table.

Both have broken silently before. The buttons are driven by a Jinja filter
over the round rows, and `selectattr` on a key that does not exist yields
nothing rather than raising — so when the rows changed shape the page quietly
offered "Start first round" on a construct with fourteen rounds behind it. The
same class of failure took out `service.gold_by_label` earlier: plausible
wrong UI, no stack trace, nothing red in the suite.
"""

import json

import pytest

from studio import telemetry
from studio.app import create_app
from studio.config import Config
from studio.db import connect, init_db, snapshot_schema

SESSIONS = ["Boats and Fish 1_Grade 4.xlsx", "Boats and Fish 2_Grade 4.xlsx", "Boats and Fish 3_Grade 4.xlsx"]


@pytest.fixture()
def env(tmp_path):
    telemetry.reset()
    base = Config.from_env()
    cfg = Config(
        host="127.0.0.1", port=0, db_path=tmp_path / "t.sqlite3",
        dataset_path=base.dataset_path, secret_key="k", admin_token="t",
    )
    init_db(cfg.db_path)
    conn = connect(cfg.db_path)
    conn.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    conn.execute(
        "INSERT INTO construct (owner_id,name,description,label_space,scope,"
        "other_label,dataset,phase) VALUES "
        "(1,'C','d','[\"A\",\"None\"]','student','None','talkmoves','revise')"
    )
    conn.execute(
        "INSERT INTO criterion (construct_id,text,edges,status,origin) "
        "VALUES (1,'a criterion',?,'active','seed')",
        (json.dumps([{"label": "A"}]),),
    )
    conn.commit()
    snapshot_schema(conn, 1, "initial")
    app = create_app(cfg)
    client = app.test_client()
    # Identity is a plain cookie, not a Flask session — there is no auth in
    # this app at all (see docs/multi-user).
    client.set_cookie("reviewer", "t")
    return client, conn


def add_round(conn, session_id, status="complete", kind="review"):
    conn.execute(
        "INSERT INTO round (construct_id,kind,session_id,schema_version,status) "
        "VALUES (1,?,?,1,?)", (kind, session_id, status),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]


def page(client):
    r = client.get("/construct/1")
    assert r.status_code == 200
    return r.get_data(as_text=True)


class TestActionButtons:
    def test_no_rounds_offers_a_first_round(self, env):
        client, _ = env
        body = page(client)
        assert "Start first round" in body
        assert "Continue in this session" not in body

    def test_any_review_round_offers_continue_and_add(self, env):
        """The regression: rows changed shape, the filter matched nothing, and
        a construct with rounds behind it advertised "Start first round"."""
        client, conn = env
        add_round(conn, "Boats and Fish 1_Grade 4.xlsx")
        body = page(client)
        assert "Continue in this session" in body
        assert "Add a session" in body
        assert "Start first round" not in body

    def test_continue_posts_the_newest_round_of_the_newest_session(self, env):
        client, conn = env
        add_round(conn, "Boats and Fish 1_Grade 4.xlsx")
        add_round(conn, "Boats and Fish 2_Grade 4.xlsx")
        newest = add_round(conn, "Boats and Fish 2_Grade 4.xlsx")
        assert f"/construct/1/round/{newest}/revisit" in page(client)

    def test_an_approve_round_alone_is_not_a_started_construct(self, env):
        """Approve draws a session too, but it is not a review round and
        cannot be continued."""
        client, conn = env
        add_round(conn, "Boats and Fish 1_Grade 4.xlsx", kind="approve")
        assert "Start first round" in page(client)


class TestSessionTable:
    def test_one_row_per_session_not_per_round(self, env):
        client, conn = env
        for _ in range(3):
            add_round(conn, "Boats and Fish 1_Grade 4.xlsx")
        add_round(conn, "Boats and Fish 2_Grade 4.xlsx")
        body = page(client)
        assert body.count(">Boats and Fish 1_Grade 4.xlsx<") == 1
        assert body.count(">Boats and Fish 2_Grade 4.xlsx<") == 1

    def test_shows_how_many_rounds_a_session_accumulated(self, env):
        client, conn = env
        for _ in range(3):
            add_round(conn, "Boats and Fish 1_Grade 4.xlsx")
        body = page(client)
        i = body.find(">Boats and Fish 1_Grade 4.xlsx<")
        assert ">3<" in body[i:i + 260]

    def test_an_unfinished_session_offers_review_not_continue(self, env):
        client, conn = env
        rid = add_round(conn, "Boats and Fish 1_Grade 4.xlsx", status="reviewing")
        body = page(client)
        assert f"/construct/1/round/{rid}/review" in body

    def test_review_points_at_the_newest_round_of_that_session(self, env):
        """Older batches of the same session are superseded; offering them
        invites starting work that the newest round has already replaced."""
        client, conn = env
        stale = add_round(conn, "Boats and Fish 1_Grade 4.xlsx", status="reviewing")
        newest = add_round(conn, "Boats and Fish 1_Grade 4.xlsx", status="reviewing")
        body = page(client)
        assert f"/construct/1/round/{newest}/review" in body
        assert f"/construct/1/round/{stale}/review" not in body

    def test_an_annotating_session_links_to_progress(self, env):
        client, conn = env
        rid = add_round(conn, "Boats and Fish 1_Grade 4.xlsx", status="annotating")
        assert f"/construct/1/round/{rid}/annotating" in page(client)

    def test_approve_rounds_are_not_listed(self, env):
        client, conn = env
        add_round(conn, "Boats and Fish 3_Grade 4.xlsx", kind="approve")
        assert ">Boats and Fish 3_Grade 4.xlsx<" not in page(client)
