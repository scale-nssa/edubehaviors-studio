"""Local mode: own models, own key, own data (docs/local-mode-plan.md).

Each test pins a property that fails silently when it breaks — a doubled
token count, a stale verification, a digest nobody checks — because a silent
failure is the one a researcher alone on their laptop cannot debug.
"""

import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio import corpus, dataset_upload as up, llm, model_settings, providers, spend
from studio.app import create_app
from studio.config import Config, annotators, reasoning_model
from studio.db import connect

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for p in providers.PROVIDERS.values():
        if p.key_env:
            monkeypatch.delenv(p.key_env, raising=False)
    return Config(
        host="127.0.0.1", port=0, db_path=tmp_path / "t.sqlite3",
        dataset_path=Config.from_env().dataset_path, secret_key="k",
        admin_token="t", single_user=True,
    )


@pytest.fixture
def app(cfg):
    a = create_app(cfg)
    yield a
    model_settings._state.clear()


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Answer every Anthropic call locally, recording what was asked."""
    calls = []

    def ask(spec, key, prompt):
        calls.append((spec, key, prompt))
        return providers.Response(text='{"true_indices": []}', input_tokens=1000,
                                  output_tokens=200, thinking_tokens=None)

    monkeypatch.setitem(providers._DISPATCH, "anthropic", ask)
    return calls


def _verify_all(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000000000000")
    for role in model_settings.ROLES:
        ok, msg = llm.test_role(role)
        assert ok, msg


# --------------------------------------------------------------------------- #
# Token accounting: output_tokens is ALL billed output, never added to again
# --------------------------------------------------------------------------- #

class TestTokenAccounting:
    def test_openai_reasoning_is_inside_completion_tokens_not_added(self):
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=500,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=450),
        )
        resp = SimpleNamespace(
            usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        )
        r = providers._openai_response(resp)
        assert r.output_tokens == 500, "reasoning is a subset of completion_tokens"
        assert r.thinking_tokens == 450

    def test_think_tags_are_stripped_from_local_models(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                  completion_tokens_details=None),
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='<think>hmm</think>{"true_indices": [1]}'))],
        )
        assert providers._openai_response(resp).text == '{"true_indices": [1]}'

    def test_gemini_thoughts_are_outside_candidates_and_are_added(self):
        class FakeModels:
            def generate_content(self, **kw):
                return SimpleNamespace(text="ok", usage_metadata=SimpleNamespace(
                    prompt_token_count=100, candidates_token_count=15,
                    thoughts_token_count=2000))
        r = providers._gemini(SimpleNamespace(models=FakeModels()),
                              {"model": "gemini-x"}, providers.Prompt(None, "p"))
        assert r.output_tokens == 2015, "dropping thoughts understated Gemini ~2.6x"
        assert r.thinking_tokens == 2000

    def test_record_does_not_add_thinking_again(self):
        llm._usage.clear()
        llm._record("m", 10, 500, thinking=450)
        assert llm.usage_report()["m"]["output_tokens"] == 500


class TestAnthropicQuirks:
    def test_no_temperature_is_ever_sent(self):
        for model in ("claude-haiku-4-5", "claude-sonnet-5"):
            for r in ("default", "off", "low", "high"):
                assert "temperature" not in providers._anthropic_kwargs(model, r)

    def test_max_tokens_stays_under_the_non_streaming_limit(self):
        assert providers._anthropic_kwargs("claude-sonnet-5", "high")["max_tokens"] <= 21000

    def test_legacy_models_take_a_budget_and_are_off_by_omission(self):
        kw = providers._anthropic_kwargs("claude-haiku-4-5", "low")
        assert kw["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert "thinking" not in providers._anthropic_kwargs("claude-haiku-4-5", "off")

    def test_adaptive_models_take_effort_and_need_an_explicit_off(self):
        kw = providers._anthropic_kwargs("claude-sonnet-5", "medium")
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["output_config"] == {"effort": "medium"}
        assert providers._anthropic_kwargs("claude-sonnet-5", "off")["thinking"] == {
            "type": "disabled"}

    def test_opus_4_8_is_not_mistaken_for_a_legacy_model(self):
        assert providers._anthropic_kwargs("claude-opus-4-8", "low")["thinking"] == {
            "type": "adaptive"}


# --------------------------------------------------------------------------- #
# Roles, nicknames and verification — what replaced the allowlist
# --------------------------------------------------------------------------- #

class TestRoles:
    def test_reasoning_setting_is_in_the_nickname_and_so_the_cache_key(self, app):
        model_settings.set_role("annotator_a", {"provider": "anthropic",
                                                "model": "claude-haiku-4-5",
                                                "reasoning": "low"})
        low = annotators()[0]
        model_settings.set_role("annotator_a", {"provider": "anthropic",
                                                "model": "claude-haiku-4-5",
                                                "reasoning": "off"})
        assert annotators()[0] != low

    def test_one_model_twice_gives_two_independent_samples(self, app):
        a, b = annotators()
        assert model_settings.same_annotator_model()
        assert a != b and b == a + "#2", "otherwise B reads A's cache and agrees by construction"

    def test_nothing_is_callable_until_tested(self, app):
        assert not model_settings.ready()
        with pytest.raises(llm.ModelNotReady):
            llm.model(annotators()[0])

    def test_a_passing_test_verifies_and_a_new_key_unverifies(self, app, fake_anthropic,
                                                             monkeypatch):
        _verify_all(monkeypatch)
        assert model_settings.ready()
        model_settings.set_key("anthropic", "sk-ant-a-different-key-1111")
        assert not model_settings.ready(), "verification is tied to the key"

    def test_a_failing_test_reports_the_providers_words_without_the_key(
            self, app, monkeypatch):
        key = "sk-ant-secret-9999999999999"
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)

        def boom(spec, k, prompt):
            raise RuntimeError(f"401 invalid x-api-key {k}")
        monkeypatch.setitem(providers._DISPATCH, "anthropic", boom)
        ok, msg = llm.test_role("annotator_a")
        assert not ok and "401 invalid x-api-key" in msg
        assert key not in msg

    def test_keys_go_to_a_private_env_file_never_the_database(self, app, cfg):
        model_settings.set_key("anthropic", "sk-ant-private-12345678901")
        env = cfg.home / ".env"
        assert "sk-ant-private-12345678901" in env.read_text()
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
        assert b"sk-ant-private" not in cfg.db_path.read_bytes()
        assert "sk-ant-private" not in (cfg.home / "models.json").read_text() \
            if (cfg.home / "models.json").exists() else True

    def test_log_lines_are_redacted(self, app, caplog, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leaky-000000000000")
        f = model_settings.RedactingFilter()
        import logging
        rec = logging.LogRecord("x", logging.INFO, "", 0, "key=%s", ("sk-ant-leaky-000000000000",), None)
        f.filter(rec)
        assert "sk-ant-leaky-000000000000" not in rec.getMessage()


# --------------------------------------------------------------------------- #
# Spend
# --------------------------------------------------------------------------- #

class TestSpend:
    def test_every_call_is_written_to_the_ledger(self, app, fake_anthropic, monkeypatch, cfg):
        _verify_all(monkeypatch)
        before = connect(cfg.db_path).execute("SELECT COUNT(*) n FROM spend_log").fetchone()["n"]
        llm.ask_json(annotators()[0], providers.Prompt(None, "x"), profile="annotate")
        rows = connect(cfg.db_path).execute("SELECT * FROM spend_log").fetchall()
        assert len(rows) == before + 1
        # Haiku 4.5 list price: 1000 in at $1/M + 200 out at $5/M.
        assert rows[-1]["cost_usd"] == pytest.approx(0.002)

    def test_the_cap_refuses_the_call_itself(self, app, fake_anthropic, monkeypatch):
        _verify_all(monkeypatch)
        model_settings.set_spend_cap(0.0)
        n = len(fake_anthropic)
        with pytest.raises(spend.SpendCapReached):
            llm.ask_json(annotators()[0], providers.Prompt(None, "x"))
        assert len(fake_anthropic) == n, "nothing may reach the provider past the cap"

    def test_unknown_price_is_unknown_not_free(self, app):
        model_settings.set_role("annotator_a", {"provider": "openai_compatible",
                                                "model": "llama3",
                                                "base_url": "http://localhost:11434/v1"})
        assert spend.cost(annotators()[0], 10**6, 10**6) is None


# --------------------------------------------------------------------------- #
# Routes: no paid action without verified models, an estimate, and a yes
# --------------------------------------------------------------------------- #

def _construct(cfg, dataset="talkmoves"):
    c = connect(cfg.db_path)
    c.execute("INSERT OR IGNORE INTO reviewer (display_name) VALUES ('Me')")
    c.execute(
        "INSERT INTO construct (owner_id,name,description,label_space,scope,phase,"
        "other_label,dataset) VALUES ((SELECT id FROM reviewer WHERE display_name='Me'),"
        "'C','d','[\"A\",\"Other\"]','tutor','annotate','Other',?)", (dataset,))
    cid = c.execute("SELECT MAX(id) i FROM construct").fetchone()["i"]
    c.execute("INSERT INTO criterion (construct_id,text,edges,status,origin) "
              "VALUES (?,'asks a question','[{\"label\": \"A\"}]','active','seed')", (cid,))
    c.commit()
    return cid


class TestPaidRoutes:
    def test_first_run_lands_on_the_models_page_without_signing_in(self, app):
        r = app.test_client().get("/")
        assert r.status_code == 302 and "/models" in r.location

    def test_a_round_without_verified_models_goes_to_setup(self, app, cfg, monkeypatch):
        cid = _construct(cfg)
        from studio import service
        monkeypatch.setattr(service, "start_round", lambda *a, **k: pytest.fail("spent"))
        r = app.test_client().post(f"/construct/{cid}/round/new")
        assert r.status_code == 302 and "/models" in r.location

    def test_a_round_shows_an_estimate_first_then_runs_on_yes(
            self, app, cfg, fake_anthropic, monkeypatch):
        _verify_all(monkeypatch)
        cid = _construct(cfg)
        from studio import service
        started = []
        monkeypatch.setattr(service, "start_round",
                            lambda *a, **k: started.append(a) or 1)
        cl = app.test_client()
        body = cl.post(f"/construct/{cid}/round/new").get_data(as_text=True)
        assert "model call" in body and "Go ahead" in body
        assert not started, "the estimate page must not start the round"
        cl.post(f"/construct/{cid}/round/new", data={"confirmed": "1"})
        assert started

    def test_the_cap_blocks_a_round_whose_estimate_crosses_it(
            self, app, cfg, fake_anthropic, monkeypatch):
        _verify_all(monkeypatch)
        model_settings.set_spend_cap(0.000001)
        cid = _construct(cfg)
        from studio import service
        monkeypatch.setattr(service, "start_round", lambda *a, **k: pytest.fail("spent"))
        r = app.test_client().post(f"/construct/{cid}/round/new", data={"confirmed": "1"})
        assert r.status_code == 402
        assert "spend cap" in r.get_data(as_text=True)

    def test_the_models_page_renders(self, app):
        body = app.test_client().get("/models").get_data(as_text=True)
        assert "Test connection" in body and "Annotator B" in body


# --------------------------------------------------------------------------- #
# Bring your own dataset
# --------------------------------------------------------------------------- #

CSV = """conv,who,utt,seq
s1,Teacher,Good morning,2
s1,T,What is 3 x 4?,3
s1,Student,Twelve,4
s1,mouse,clicked,5
s1,Student,,6
s2,Teacher,Hi,1
s2,S,Hello,2
"""


def _mapping():
    return up.Mapping(session_id="conv", speaker="who", text="utt", order="seq",
                      roles={"Teacher": "tutor", "T": "tutor", "Student": "student",
                             "S": "student", "mouse": "drop"})


class TestUploadConversion:
    def test_index_is_zero_based_list_position_after_drops(self):
        _, rows = up.read_rows(CSV)
        built = up.build(rows, _mapping(), "mine")
        s1 = built.sessions[0]
        assert [u["index"] for u in s1["utterances"]] == [0, 1, 2], \
            "Session.window slices by index, so index must equal position"
        assert built.dropped_speaker == 1 and built.dropped_blank == 1

    def test_session_ids_are_namespaced(self):
        _, rows = up.read_rows(CSV)
        built = up.build(rows, _mapping(), "mine")
        assert [s["session_id"] for s in built.sessions] == ["mine:s1", "mine:s2"]

    def test_an_unlisted_speaker_is_context_not_a_candidate(self):
        _, rows = up.read_rows(CSV + "s2,Narrator,Meanwhile,3\n")
        built = up.build(rows, _mapping(), "mine")
        assert "Narrator" in built.unmapped_speakers
        last = built.sessions[1]["utterances"][-1]
        assert last["speaker"]["role"] == "other"


class TestUploadedDatasets:
    def _write(self, cfg, ds="mine"):
        _, rows = up.read_rows(CSV)
        return up.write(up.build(rows, _mapping(), ds), corpus.user_dir(cfg), ds,
                        "Mine", "", "x.csv")

    def test_it_loads_and_resolves_alongside_talkmoves(self, app, cfg):
        self._write(cfg)
        assert corpus.dataset_ids(cfg) == ["talkmoves", "mine"]
        s = corpus.get(cfg, "mine:s1")
        assert [u.role for u in s.utterances] == ["tutor", "tutor", "student"]
        assert len(corpus.all_sessions(cfg, "talkmoves")) == 20, "TalkMoves untouched"

    def test_it_is_pinned_like_the_bundled_corpus(self, app, cfg):
        e = self._write(cfg)
        path = corpus.user_dir(cfg) / e["file"]
        path.write_text(path.read_text().replace("Twelve", "Eleven"))
        with pytest.raises(corpus.CorpusChanged):
            corpus.all_sessions(cfg, "mine")

    def test_the_whole_flow_over_http(self, app, cfg):
        import io
        cl = app.test_client()
        r = cl.post("/datasets/upload", data={"file": (io.BytesIO(CSV.encode()), "class.csv")},
                    content_type="multipart/form-data")
        assert r.status_code == 302 and "/datasets/map/" in r.location
        page = cl.get(r.location).get_data(as_text=True)
        assert "Good morning" in page, "the preview shows parsed rows"
        form = {"col_session_id": "conv", "col_speaker": "who", "col_text": "utt",
                "col_order": "seq", "title": "Class", "dataset_id": "class",
                "create": "1"}
        speakers = ["Teacher", "Student", "T", "S", "mouse"]
        roles = ["tutor", "student", "tutor", "student", "drop"]
        # Order on the page is most-common-first; post by value, not position.
        vals = up.speaker_values(up.read_rows(CSV)[1], "who")
        for i, (v, _n) in enumerate(vals):
            form[f"spk__{i}"] = v
            form[f"role__{i}"] = roles[speakers.index(v)]
        r = cl.post(r.location, data=form)
        assert r.status_code == 302, r.get_data(as_text=True)[:500]
        assert "class" in corpus.dataset_ids(cfg)
        assert len(corpus.all_sessions(cfg, "class")) == 2

    def test_delete_refuses_while_a_construct_uses_it(self, app, cfg):
        self._write(cfg)
        _construct(cfg, dataset="mine")
        cl = app.test_client()
        cl.post("/datasets/mine/delete")
        assert "mine" in corpus.dataset_ids(cfg)
        assert "Not deleted" in cl.get("/datasets/mine").get_data(as_text=True)

    def test_delete_works_once_nothing_uses_it(self, app, cfg):
        self._write(cfg)
        app.test_client().post("/datasets/mine/delete")
        assert "mine" not in corpus.dataset_ids(cfg)

    def test_the_bundled_corpus_cannot_be_deleted(self, app, cfg):
        app.test_client().post("/datasets/talkmoves/delete")
        assert "talkmoves" in corpus.dataset_ids(cfg)


# --------------------------------------------------------------------------- #
# Binding and packaging
# --------------------------------------------------------------------------- #

def test_a_non_loopback_bind_is_refused_without_the_flag(monkeypatch):
    from studio import __main__ as m
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv(m.NO_AUTH_FLAG, raising=False)
    monkeypatch.setattr(m, "create_app", lambda cfg: pytest.fail("started"))
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 2


def test_only_talkmoves_ships():
    """The other six corpora must not come back by accident (plan §6.1)."""
    data = REPO / "src" / "studio" / "data"
    assert sorted(p.name for p in data.glob("*.jsonl")) == ["talkmoves_20.jsonl"]
    assert [d["id"] for d in json.loads((data / "datasets.json").read_text())] == ["talkmoves"]


@pytest.mark.skipif(os.environ.get("SKIP_BUILD") == "1", reason="SKIP_BUILD=1")
def test_the_built_wheel_contains_exactly_one_dataset(tmp_path):
    out = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(REPO)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        pytest.skip(f"could not build a wheel here: {out.stderr[-300:]}")
    wheel = next(tmp_path.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    jsonl = [n for n in names if n.endswith(".jsonl")]
    assert jsonl == ["studio/data/talkmoves_20.jsonl"], jsonl
    assert "studio/templates/models.html" in names


class TestMappingGuards:
    def test_one_column_cannot_be_two_fields(self):
        _, rows = up.read_rows(CSV)
        m = _mapping()
        m.text = m.session_id
        with pytest.raises(up.UploadError):
            up.build(rows, m, "mine")

    def test_a_required_column_with_no_guess_is_left_unchosen(self, app):
        """Falling through to the first column once made the session id the
        text of every utterance, and the counts still looked right."""
        import io
        cl = app.test_client()
        r = cl.post("/datasets/upload",
                    data={"file": (io.BytesIO(b"a,b,zzz\n1,T,hi\n"), "x.csv")},
                    content_type="multipart/form-data")
        page = cl.get(r.location).get_data(as_text=True)
        text_select = page.split('name="col_text"')[1].split("</select>")[0]
        assert 'value="" selected' in text_select

    def test_numbered_speakers_are_guessed(self):
        assert up.guess_role("S1") == "student"
        assert up.guess_role("Student 2") == "student"
        assert up.guess_role("T") == "tutor"
        assert up.guess_role("narrator") == "other"
