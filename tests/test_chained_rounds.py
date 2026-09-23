"""Chained rounds (docs/next-round.md item 3).

One session is a thin basis for revising a schema, and it is the rare labels
that need the evidence. So a review round that leaves some label under
MIN_POSITIVES_PER_LABEL model firings pulls the next session in behind it,
until the counts are met — or the cap, or the dataset, runs out.
"""

import json

import pytest

from studio import config, service
from studio.config import Config
from studio.db import connect, init_db


@pytest.fixture()
def conn(tmp_path):
    init_db(tmp_path / "t.sqlite3")
    c = connect(tmp_path / "t.sqlite3")
    c.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    c.execute(
        "INSERT INTO construct (owner_id, name, description, label_space, scope, "
        "other_label) VALUES (1,'c','d','[\"A\",\"Other\"]','tutor','Other')"
    )
    c.commit()
    return c


def add_sessions(conn, n):
    for i in range(n):
        conn.execute(
            "INSERT INTO round (construct_id, kind, session_id, schema_version, "
            "status) VALUES (1,'review',?,1,'complete')", (f"S{i}",),
        )
    conn.commit()


def stub(monkeypatch, counts, remaining=9):
    monkeypatch.setattr(service, "label_positives", lambda *a, **k: counts)
    monkeypatch.setattr(service, "sessions_remaining", lambda *a, **k: (remaining, 20))


class TestShouldChain:
    def test_chains_while_a_label_is_short(self, conn, monkeypatch):
        add_sessions(conn, 1)
        stub(monkeypatch, {"A": 1, "Other": 40})
        assert service.should_chain(conn, None, 1) == ["A"]

    def test_stops_once_every_label_has_enough(self, conn, monkeypatch):
        add_sessions(conn, 1)
        stub(monkeypatch, {"A": config.MIN_POSITIVES_PER_LABEL, "Other": 40})
        assert service.should_chain(conn, None, 1) == []

    def test_cap_stops_it_even_with_a_label_at_zero(self, conn, monkeypatch):
        """The cap is the cost ceiling — a label the corpus simply does not
        contain would otherwise chain until the dataset ran out."""
        add_sessions(conn, config.MAX_CHAINED_SESSIONS)
        stub(monkeypatch, {"A": 0, "Other": 40})
        assert service.should_chain(conn, None, 1) == []

    def test_exhausted_dataset_stops_it(self, conn, monkeypatch):
        add_sessions(conn, 1)
        stub(monkeypatch, {"A": 0, "Other": 40}, remaining=0)
        assert service.should_chain(conn, None, 1) == []

    def test_cap_counts_sessions_not_chain_length(self, conn, monkeypatch):
        """Revise, chain, revise, chain would otherwise walk the dataset a
        session per revision without ever hitting a per-chain limit."""
        add_sessions(conn, config.MAX_CHAINED_SESSIONS + 3)
        stub(monkeypatch, {"A": 0, "Other": 0})
        assert service.should_chain(conn, None, 1) == []


class TestMaybeChain:
    """The driver-thread hook. It must start exactly one round, and it must not
    be able to take down a round that has already succeeded."""

    def test_starts_one_round_when_short(self, conn, monkeypatch):
        add_sessions(conn, 1)
        stub(monkeypatch, {"A": 0, "Other": 40})
        calls = []
        monkeypatch.setattr(service, "start_round",
                            lambda *a, **k: calls.append(k) or 99)
        assert service._maybe_chain(None, conn, _nolock(), 1) == 99
        assert len(calls) == 1

    def test_does_nothing_when_satisfied(self, conn, monkeypatch):
        add_sessions(conn, 1)
        stub(monkeypatch, {"A": 9, "Other": 9})
        monkeypatch.setattr(service, "start_round", _boom)
        assert service._maybe_chain(None, conn, _nolock(), 1) is None

    def test_swallows_a_failure_to_start(self, conn, monkeypatch):
        """The round that triggered this is already `reviewing` and usable.
        Reporting it failed because its successor could not start would be a
        worse lie than simply not chaining."""
        add_sessions(conn, 1)
        stub(monkeypatch, {"A": 0, "Other": 40})
        monkeypatch.setattr(service, "start_round", _boom)
        assert service._maybe_chain(None, conn, _nolock(), 1) is None


def _boom(*a, **k):
    raise RuntimeError("no criteria to annotate")


def _nolock():
    import threading
    return threading.RLock()


class TestProgressFollowsTheChain:
    """The progress page is where the reviewer waits. If a chained round has
    started behind this one, sending them into review of the first defeats the
    point of chaining; if none has, they must not be stranded."""

    @pytest.fixture
    def client(self, tmp_path):
        from studio.app import create_app
        from studio.config import Config
        from studio import telemetry

        telemetry.reset()
        cfg = Config(
            host="127.0.0.1", port=0, db_path=tmp_path / "t.sqlite3",
            dataset_path=Config.from_env().dataset_path,
            secret_key="k", admin_token="t",
        )
        init_db(cfg.db_path)
        c = connect(cfg.db_path)
        c.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
        c.execute(
            "INSERT INTO construct (owner_id,name,description,label_space,scope) "
            "VALUES (1,'c','d','[\"A\",\"B\"]','tutor')"
        )
        c.commit()
        return create_app(cfg).test_client(), c

    def _round(self, conn, status, chained=0):
        cur = conn.execute(
            "INSERT INTO round (construct_id,kind,session_id,schema_version,"
            "status,chained) VALUES (1,'review','S',1,?,?)", (status, chained),
        )
        conn.commit()
        return cur.lastrowid

    def test_points_at_the_next_annotating_round(self, client):
        cl, conn = client
        first = self._round(conn, "reviewing")
        second = self._round(conn, "annotating")
        body = cl.get(f"/round/{first}/progress").get_data(as_text=True)
        assert f"/construct/1/round/{second}/annotating" in body
        # normalised: the copy wraps across source lines
        assert "the next session is being annotated too" in " ".join(body.split())

    def test_a_chained_round_falls_through_to_the_oldest(self, client):
        """A chain is worked in the order it was annotated, so finishing the
        last one lands on the first — otherwise the earlier sessions in the
        chain are never surfaced and sit in `reviewing` forever."""
        cl, conn = client
        first = self._round(conn, "reviewing")
        last = self._round(conn, "reviewing", chained=1)
        body = cl.get(f"/round/{last}/progress").get_data(as_text=True)
        assert f"/construct/1/round/{first}/review" in body

    def test_a_round_you_asked_for_goes_to_that_session(self, client):
        """Pressing "Add a new session" part-way through a round used to drop
        you back into the round you were already in, which read as the new
        session's utterances having failed to load."""
        cl, conn = client
        half_done = self._round(conn, "reviewing")
        asked_for = self._round(conn, "reviewing", chained=0)
        body = cl.get(f"/round/{asked_for}/progress").get_data(as_text=True)
        assert f"/construct/1/round/{asked_for}/review" in body
        assert f"/construct/1/round/{half_done}/review" not in body

    def test_ignores_an_annotating_round_that_came_first(self, client):
        """Only *later* rounds are the chain. An older one still running is
        someone else's business — following it would go backwards."""
        cl, conn = client
        stale = self._round(conn, "annotating")
        mine = self._round(conn, "reviewing")
        body = cl.get(f"/round/{mine}/progress").get_data(as_text=True)
        assert f"/construct/1/round/{stale}/annotating" not in body
        assert f"/construct/1/round/{mine}/review" in body


class TestProposalsSupersede:
    """A new proposal replaces the last one.

    The Revise page says "whatever the model proposed when you last asked".
    Without this it was really "everything it has ever proposed and you never
    acted on" — suggestions against wording that had since been reworded,
    accumulating round after round.
    """

    @pytest.fixture()
    def conn(self, tmp_path):
        init_db(tmp_path / "t.sqlite3")
        c = connect(tmp_path / "t.sqlite3")
        c.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
        c.execute(
            "INSERT INTO construct (owner_id,name,description,label_space,scope) "
            "VALUES (1,'c','d','[\"A\",\"B\"]','tutor')"
        )
        for status in ("staged", "staged", "applied", "rejected"):
            c.execute(
                "INSERT INTO delta (construct_id,kind,payload,rationale,status) "
                "VALUES (1,'reword','{}','r',?)", (status,),
            )
        c.commit()
        return c

    def _wash(self, conn):
        return conn.execute(
            "DELETE FROM delta WHERE construct_id=? AND status='staged'", (1,),
        ).rowcount

    def test_untouched_suggestions_are_discarded(self, conn):
        assert self._wash(conn) == 2
        left = conn.execute(
            "SELECT COUNT(*) n FROM delta WHERE construct_id=1 AND status='staged'"
        ).fetchone()["n"]
        assert left == 0

    def test_decisions_you_already_made_survive(self, conn):
        """Applied and rejected are the record of what you decided; only
        suggestions you never acted on are washed."""
        self._wash(conn)
        kept = sorted(
            r["status"]
            for r in conn.execute("SELECT status FROM delta WHERE construct_id=1")
        )
        assert kept == ["applied", "rejected"]


class TestChainedRoundExplainsItself:
    """A round nobody asked for spends money unprompted. The page has to say
    why, and be specific: which labels are short, and of what."""

    @pytest.fixture()
    def env(self, tmp_path):
        from studio import telemetry
        from studio.app import create_app
        telemetry.reset()
        base = Config.from_env()
        cfg = Config(host="127.0.0.1", port=0, db_path=tmp_path / "t.sqlite3",
                     dataset_path=base.dataset_path, secret_key="k", admin_token="t")
        init_db(cfg.db_path)
        c = connect(cfg.db_path)
        c.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
        c.execute(
            "INSERT INTO construct (owner_id,name,description,label_space,scope,"
            "other_label,dataset) VALUES "
            "(1,'C','d','[\"A\",\"None\"]','student','None','talkmoves')"
        )
        c.execute(
            "INSERT INTO criterion (construct_id,text,edges,status,origin) "
            "VALUES (1,'a criterion',?,'active','seed')",
            (json.dumps([{"label": "A"}]),),
        )
        c.commit()
        cl = create_app(cfg).test_client()
        cl.set_cookie("reviewer", "t")
        return cl, c

    def _round(self, conn, chained):
        conn.execute(
            "INSERT INTO round (construct_id,kind,session_id,schema_version,"
            "status,chained) VALUES (1,'review','Boats and Fish 1_Grade 4.xlsx',1,'annotating',?)",
            (chained,),
        )
        conn.commit()
        return conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]

    def test_a_chained_round_says_why_it_is_running(self, env):
        cl, conn = env
        rid = self._round(conn, chained=1)
        body = cl.get(f"/construct/1/round/{rid}/annotating").get_data(as_text=True)
        assert "Hunting for more positive cases" in body
        assert "added automatically" in body

    def test_it_names_the_labels_that_are_short(self, env):
        """And only those. The catch-all is never short: every utterance
        nothing fires on routes to it, so it banks positives immediately.
        Chaining is therefore driven by the real labels, which is the point."""
        cl, conn = env
        rid = self._round(conn, chained=1)
        body = cl.get(f"/construct/1/round/{rid}/annotating").get_data(as_text=True)
        assert "<strong>A</strong>" in body
        assert "<strong>None</strong>" not in body

    def test_a_round_you_asked_for_says_nothing(self, env):
        cl, conn = env
        rid = self._round(conn, chained=0)
        body = cl.get(f"/construct/1/round/{rid}/annotating").get_data(as_text=True)
        assert "Hunting for more positive cases" not in body
