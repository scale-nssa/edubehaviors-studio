"""Multi-annotator behaviour: isolation, presence, resume, recovery, gating.

The concurrency claims are the ones worth pinning: they are invisible with one
user and only bite in a room of twenty, which is exactly when nobody can debug
them.
"""

import json
import threading
import time

import pytest

from studio import annotate, service, telemetry
from studio.app import create_app
from studio.config import Config
from studio.db import connect, init_db


@pytest.fixture
def cfg(tmp_path):
    return Config(
        host="127.0.0.1", port=0, db_path=tmp_path / "t.sqlite3",
        dataset_path=Config.from_env().dataset_path,
        secret_key="k", admin_token="t",
    )


@pytest.fixture
def app(cfg):
    telemetry.reset()
    return create_app(cfg)


def _reviewer(conn, name):
    conn.execute("INSERT OR IGNORE INTO reviewer (display_name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT id FROM reviewer WHERE display_name=?", (name,)).fetchone()["id"]


def _construct(conn, owner, name="C", phase="annotate"):
    cur = conn.execute(
        "INSERT INTO construct (owner_id,name,description,label_space,scope,phase) "
        "VALUES (?,?,'d','[\"A\",\"B\"]','tutor',?)", (owner, name, phase),
    )
    conn.commit()
    return cur.lastrowid


class TestIsolation:
    def test_one_reviewer_cannot_open_anothers_construct(self, app, cfg):
        conn = connect(cfg.db_path)
        a, b = _reviewer(conn, "Ann"), _reviewer(conn, "Bo")
        cid = _construct(conn, a)
        c = app.test_client()
        c.post("/identify", data={"name": "Bo", "confirm_existing": "1"})
        assert c.get(f"/construct/{cid}").status_code == 404
        assert b != a

    def test_home_lists_only_your_own(self, app, cfg):
        conn = connect(cfg.db_path)
        a, b = _reviewer(conn, "Ann"), _reviewer(conn, "Bo")
        _construct(conn, a, "AnnConstruct")
        _construct(conn, b, "BoConstruct")
        c = app.test_client(); c.post("/identify", data={"name": "Bo", "confirm_existing": "1"})
        body = c.get("/home").text
        assert "BoConstruct" in body and "AnnConstruct" not in body


class TestNameCollision:
    def test_a_taken_name_asks_before_adopting_it(self, app, cfg):
        conn = connect(cfg.db_path)
        _construct(conn, _reviewer(conn, "Sam"))
        c = app.test_client()
        r = c.post("/identify", data={"name": "Sam"})
        assert r.status_code == 200 and "That name is taken" in r.text
        assert "1 construct" in r.text, "show what you are about to walk into"
        assert "reviewer" not in (r.headers.get("Set-Cookie") or "")

    def test_confirming_signs_you_in(self, app, cfg):
        conn = connect(cfg.db_path)
        _reviewer(conn, "Sam")
        c = app.test_client()
        r = c.post("/identify", data={"name": "Sam", "confirm_existing": "1"})
        assert r.status_code == 302 and "reviewer" in r.headers.get("Set-Cookie", "")

    def test_a_fresh_name_is_not_interrupted(self, app):
        c = app.test_client()
        assert c.post("/identify", data={"name": "Brand New"}).status_code == 302

    def test_whitespace_is_collapsed_so_one_person_is_one_row(self, app, cfg):
        c = app.test_client()
        c.post("/identify", data={"name": "  Ann   Lee "})
        conn = connect(cfg.db_path)
        names = [r["display_name"] for r in conn.execute("SELECT display_name FROM reviewer")]
        assert names == ["Ann Lee"]


class TestPresence:
    def test_last_seen_and_last_construct_are_recorded(self, app, cfg):
        conn = connect(cfg.db_path)
        a = _reviewer(conn, "Ann")
        cid = _construct(conn, a)
        c = app.test_client(); c.post("/identify", data={"name": "Ann", "confirm_existing": "1"})
        c.get(f"/construct/{cid}")
        row = conn.execute("SELECT last_seen_at,last_construct_id FROM reviewer WHERE id=?", (a,)).fetchone()
        assert row["last_seen_at"] and row["last_construct_id"] == cid

    def test_the_presence_write_is_throttled(self):
        telemetry.reset()
        assert telemetry.should_touch(1, every=30.0) is True
        assert telemetry.should_touch(1, every=30.0) is False, \
            "20 people polling every 1.2s must not be 17 writes/second"
        assert telemetry.should_touch(2, every=30.0) is True


class TestResume:
    """Derived from DB state, never stored — so it survives a restart."""

    def test_no_criteria_yet_means_seed(self, app, cfg):
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"), phase="approve")
        with app.test_request_context():
            url, label = service.resume_url(conn, cid)
        assert label == "Seed criteria"

    def test_pending_criteria_means_approve(self, app, cfg):
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"), phase="approve")
        for i in range(3):
            conn.execute("INSERT INTO criterion (construct_id,text,edges,status) "
                         "VALUES (?,?,'[]','pending')", (cid, f"c{i}"))
        conn.commit()
        with app.test_request_context():
            _, label = service.resume_url(conn, cid)
        assert label == "Approve 3 criteria"

    def test_mid_review_counts_the_batch(self, app, cfg):
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"))
        conn.execute("INSERT INTO criterion (construct_id,text,edges,status) "
                     "VALUES (?,'c','[]','active')", (cid,))
        conn.execute("INSERT INTO round (id,construct_id,kind,session_id,schema_version,status) "
                     "VALUES (7,?,'review','s',1,'reviewing')", (cid,))
        for i in range(10):
            conn.execute("INSERT INTO round_utterance (round_id,utterance_index,tier,ord) "
                         "VALUES (7,?,'t',?)", (i, i))
        for i in range(4):
            conn.execute("INSERT INTO review (round_id,utterance_index,reviewer_id,complete) "
                         "VALUES (7,?,1,1)", (i,))
        conn.commit()
        with app.test_request_context():
            url, label = service.resume_url(conn, cid)
        assert label == "Review — 4 of 10 done" and "/review" in url

    def test_a_finished_batch_points_at_the_board(self, app, cfg):
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"))
        conn.execute("INSERT INTO criterion (construct_id,text,edges,status) "
                     "VALUES (?,'c','[]','active')", (cid,))
        conn.execute("INSERT INTO round (id,construct_id,kind,session_id,schema_version,status) "
                     "VALUES (8,?,'review','s',1,'reviewing')", (cid,))
        conn.execute("INSERT INTO round_utterance (round_id,utterance_index,tier,ord) VALUES (8,0,'t',0)")
        conn.execute("INSERT INTO review (round_id,utterance_index,reviewer_id,complete) VALUES (8,0,1,1)")
        conn.commit()
        with app.test_request_context():
            url, label = service.resume_url(conn, cid)
        assert label == "Finish round 8" and "/summary" in url

    def test_a_failed_round_offers_another(self, app, cfg):
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"))
        conn.execute("INSERT INTO criterion (construct_id,text,edges,status) "
                     "VALUES (?,'c','[]','active')", (cid,))
        conn.execute("INSERT INTO round (construct_id,kind,session_id,schema_version,status) "
                     "VALUES (?,'review','s',1,'failed')", (cid,))
        conn.commit()
        with app.test_request_context():
            _, label = service.resume_url(conn, cid)
        assert "failed" in label

    def test_home_shows_the_resume_label(self, app, cfg):
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"), phase="revise")
        conn.execute("INSERT INTO criterion (construct_id,text,edges,status) "
                     "VALUES (?,'c','[]','active')", (cid,))
        conn.commit()
        c = app.test_client(); c.post("/identify", data={"name": "Ann", "confirm_existing": "1"})
        assert "Revise the schema" in c.get("/home").text


class TestRestartRecovery:
    def test_orphaned_rounds_are_requeued_not_left_spinning(self, cfg, monkeypatch):
        conn = connect(cfg.db_path) if cfg.db_path.exists() else None
        init_db(cfg.db_path)
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"))
        for i in range(2):
            conn.execute("INSERT INTO round (construct_id,kind,session_id,schema_version,status) "
                         "VALUES (?,'review','s',1,'annotating')", (cid,))
        conn.commit()
        seen = []
        monkeypatch.setattr(service, "resume_round", lambda c, rid: seen.append(rid))
        out = service.recover_orphans(cfg)
        assert out == {"requeued": 2, "failed": 0}
        for _ in range(50):
            if len(seen) == 2:
                break
            time.sleep(0.02)
        assert len(seen) == 2

    def test_a_stampede_is_capped_and_the_excess_marked_failed(self, cfg, monkeypatch):
        init_db(cfg.db_path)
        conn = connect(cfg.db_path)
        cid = _construct(conn, _reviewer(conn, "Ann"))
        for _ in range(service.MAX_REQUEUE + 3):
            conn.execute("INSERT INTO round (construct_id,kind,session_id,schema_version,status) "
                         "VALUES (?,'review','s',1,'annotating')", (cid,))
        conn.commit()
        monkeypatch.setattr(service, "resume_round", lambda c, rid: None)
        out = service.recover_orphans(cfg)
        assert out["requeued"] == service.MAX_REQUEUE and out["failed"] == 3
        n_failed = conn.execute(
            "SELECT COUNT(*) n FROM round WHERE status='failed'").fetchone()["n"]
        assert n_failed == 3

    def test_recovery_is_a_no_op_with_nothing_in_flight(self, cfg):
        init_db(cfg.db_path)
        assert service.recover_orphans(cfg) == {"requeued": 0, "failed": 0}


class TestBackgroundConnections:
    def test_each_thread_gets_its_own_connection(self, cfg):
        init_db(cfg.db_path)
        got = {}

        def grab(name):
            conn, lock = service.bg_conn(cfg)
            got[name] = id(conn)

        ts = [threading.Thread(target=grab, args=(i,)) for i in range(3)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert len(set(got.values())) == 3, \
            "a shared connection serialises every background write in the process"

    def test_the_same_thread_reuses_its_connection(self, cfg):
        init_db(cfg.db_path)
        a, _ = service.bg_conn(cfg)
        b, _ = service.bg_conn(cfg)
        assert a is b


class TestGating:
    def test_the_reasoning_model_is_gated_tighter_than_the_annotators(self):
        from studio.config import MODEL_CONCURRENCY, annotators, reasoning_model
        assert MODEL_CONCURRENCY(reasoning_model()) < MODEL_CONCURRENCY(annotators()[0]), \
            "everyone clicks Generate criteria in the first five minutes"

    def test_every_model_gets_a_gate(self):
        from studio import llm
        from studio.config import annotators, reasoning_model
        for nick in (*annotators(), reasoning_model()):
            assert isinstance(llm.gate(nick), threading.Semaphore)

    def test_annotation_is_not_gated_twice(self):
        assert not hasattr(annotate, "_semaphores"), \
            "gating in both llm and annotate silently halves throughput"


class TestPanel:
    def test_the_panel_sees_across_owners(self, app, cfg):
        conn = connect(cfg.db_path)
        a, b = _reviewer(conn, "Ann"), _reviewer(conn, "Bo")
        _construct(conn, a, "Ann's"); _construct(conn, b, "Bo's")
        c = app.test_client()
        c.post("/identify", data={"name": "Ann", "confirm_existing": "1"})
        snap = c.get("/panel/snapshot").get_json()
        names = {p["name"] for p in snap["participants"]}
        assert {"Ann", "Bo"} <= names, "the panel must bypass the ownership check"

    def test_the_panel_renders_and_has_no_buttons(self, app):
        body = app.test_client().get("/panel").text
        assert "Control panel" in body
        assert "<form" not in body, "the panel is watch-only by decision"

    def test_snapshot_shape(self, app):
        snap = app.test_client().get("/panel/snapshot").get_json()
        assert {"participants", "rounds", "models", "events", "totals", "corpus"} <= set(snap)

    def test_spend_is_estimated_per_model(self, app, cfg):
        """Priced from the binding's list price. Haiku 4.5 is $1/M input."""
        from studio.config import annotators
        nick = annotators()[0]
        conn = connect(cfg.db_path)
        conn.execute(
            "INSERT INTO annotation (session_id,criterion_text,model,true_indices,"
            "input_tokens,output_tokens) VALUES ('s','c',?,'[]',1000000,0)", (nick,))
        conn.commit()
        snap = app.test_client().get("/panel/snapshot").get_json()
        m = next(m for m in snap["models"] if m["model"] == nick)
        assert m["cached_cost_usd"] == pytest.approx(1.00, abs=0.01)


class TestTelemetry:
    def test_in_flight_rises_and_falls(self):
        telemetry.reset()
        t0 = telemetry.call_started("m")
        assert telemetry.total_in_flight() == 1
        telemetry.call_finished("m", t0, ok=True)
        assert telemetry.total_in_flight() == 0
        assert telemetry.model_stats()["m"]["completed"] == 1

    def test_failures_are_counted_separately(self):
        telemetry.reset()
        telemetry.call_finished("m", telemetry.call_started("m"), ok=False)
        st = telemetry.model_stats()["m"]
        assert st["failed"] == 1 and st["completed"] == 0

    def test_the_event_ring_is_bounded_and_newest_first(self):
        telemetry.reset()
        for i in range(telemetry.MAX_EVENTS + 10):
            telemetry.push_event("x", i=i)
        evs = telemetry.events(5)
        assert evs[0]["i"] > evs[-1]["i"]
        assert len(telemetry.events(10_000)) == telemetry.MAX_EVENTS
