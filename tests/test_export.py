"""The export bundle.

The load-bearing property is the three-way cell: TRUE, FALSE, and EMPTY for
"never asked". Collapsing the third into FALSE would silently assert that a
criterion did not fire on utterances it was never shown.
"""

import csv
import io
import json
import zipfile

import pytest

from studio import export
from studio.db import connect, init_db
from studio.schema_map import Criterion, Edge

LABELS = ["Revoicing", "PressingForReasoning", "None"]


def test_slug_matches_upstream():
    assert export.slug("Pressing for Reasoning") == "pressing_for_reasoning"
    assert export.slug("  Boats & Fish 1_Grade 4 ") == "boats_fish_1_grade_4"


def test_short_model_drops_the_openrouter_prefix():
    assert export.short_model("openrouter_claude_haiku_4_5") == "claude_haiku_4_5"
    assert export.short_model("gemini_3_8_flash") == "gemini_3_8_flash"


def test_short_model_keeps_what_distinguishes_two_annotators():
    """Local-mode nicknames carry provider, reasoning and the second-sample
    marker. Two annotators bound to one model must still get two columns."""
    a = export.short_model("anthropic/claude-haiku-4-5@low")
    b = export.short_model("anthropic/claude-haiku-4-5@low#2")
    assert a == "anthropic_claude_haiku_4_5_low"
    assert a != b


class TestAssertionIds:
    def _crit(self, cid, *labels, parent=None):
        return Criterion(cid, f"c{cid}", tuple(Edge(l) for l in labels), parent)

    def test_ids_are_category_slug_and_index(self):
        cs = [self._crit(1, "Revoicing"), self._crit(2, "Revoicing"),
              self._crit(3, "PressingForReasoning")]
        got = export.assertion_ids(cs, LABELS)
        assert got == {1: "revoicing__0", 2: "revoicing__1",
                       3: "pressingforreasoning__0"}

    def test_a_multi_label_criterion_gets_exactly_one_id(self):
        """Ids must stay 1:1 with the firing CSV's columns."""
        cs = [self._crit(1, "PressingForReasoning", "Revoicing")]
        got = export.assertion_ids(cs, LABELS)
        assert list(got) == [1]
        # Category is the first label in LABEL SPACE order, not edge order.
        assert got[1] == "revoicing__0"

    def test_a_gate_is_categorised_as_gate(self):
        cs = [self._crit(1)]
        assert export.assertion_ids(cs, LABELS) == {1: "gate__0"}


class TestResolveModels:
    def test_both_is_the_default(self):
        from studio.config import annotators
        MODELS = annotators()
        assert export.resolve_models(None) == list(MODELS)
        assert export.resolve_models("both") == list(MODELS)

    def test_a_single_model_by_nickname_or_short_name(self):
        from studio.config import annotators
        MODELS = annotators()
        assert export.resolve_models(MODELS[0]) == [MODELS[0]]
        assert export.resolve_models(export.short_model(MODELS[1])) == [MODELS[1]]

    def test_an_unknown_model_is_rejected(self):
        with pytest.raises(ValueError):
            export.resolve_models("gpt-4")


# --------------------------------------------------------------------------- #
# End to end, against a real database and a stub corpus
# --------------------------------------------------------------------------- #

SESSION = "s1.xlsx"


class _Utt:
    def __init__(self, index, role, text):
        self.index, self.role, self.text = index, role, text


class _Session:
    session_id = SESSION
    utterances = [
        _Utt(0, "tutor", "why do you think that"),
        _Utt(1, "student", "because six"),
        _Utt(2, "tutor", "say more"),
    ]

    def in_scope(self, scope):
        return [u for u in self.utterances if scope == "all" or u.role == scope]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(export.corpus, "get", lambda cfg, sid: _Session())
    path = tmp_path / "t.sqlite3"
    init_db(path)
    conn = connect(path)
    conn.execute("INSERT INTO reviewer (display_name) VALUES ('t')")
    conn.execute(
        "INSERT INTO construct (id, owner_id, name, description, label_space, scope, version) "
        "VALUES (1, 1, 'Talk Moves', 'd', ?, 'tutor', 3)",
        (json.dumps(LABELS),),
    )
    conn.execute(
        "INSERT INTO round (construct_id, kind, session_id, schema_version, status) "
        "VALUES (1, 'review', ?, 1, 'complete')", (SESSION,),
    )
    conn.commit()
    return conn


def _add_criterion(conn, text, labels, parent_id=None):
    cur = conn.execute(
        "INSERT INTO criterion (construct_id, text, edges, status, origin, parent_id) "
        "VALUES (1, ?, ?, 'active', 'seed', ?)",
        (text, json.dumps([{"label": l} for l in labels]), parent_id),
    )
    conn.commit()
    return cur.lastrowid


def _annotate(conn, text, model, hits):
    conn.execute(
        "INSERT OR REPLACE INTO annotation (session_id, criterion_text, model, true_indices) "
        "VALUES (?,?,?,?)", (SESSION, text, model, json.dumps(hits)),
    )
    conn.commit()


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_firing_csv_shape_and_scope(db):
    _add_criterion(db, "asks why", ["PressingForReasoning"])
    _annotate(db, "asks why", "m1", [0])
    rows = _rows(export.firing_csv(db, None, 1, ["m1"]))

    # scope=tutor, so the student utterance is not a row.
    assert [r["utterance_index"] for r in rows] == ["0", "2"]
    assert rows[0]["session_id"] == SESSION
    assert rows[0]["role"] == "tutor"
    assert rows[0]["utterance"] == "why do you think that"
    # One model, so the column is the bare shortname.
    assert rows[0]["pressingforreasoning__0"] == "TRUE"
    assert rows[1]["pressingforreasoning__0"] == "FALSE"


def test_never_annotated_is_empty_not_false(db):
    """The distinction the whole format exists for."""
    _add_criterion(db, "asks why", ["PressingForReasoning"])
    _add_criterion(db, "added later", ["Revoicing"])
    _annotate(db, "asks why", "m1", [0])
    rows = _rows(export.firing_csv(db, None, 1, ["m1"]))
    assert rows[0]["pressingforreasoning__0"] == "TRUE"
    assert rows[0]["revoicing__0"] == "", "a criterion never put to the model is blank"
    assert rows[1]["revoicing__0"] == ""


def test_both_models_suffix_columns_and_split_on_the_last_dot(db):
    _add_criterion(db, "asks why", ["PressingForReasoning"])
    _annotate(db, "asks why", "m.one", [0])
    _annotate(db, "asks why", "m.two", [])
    text = export.firing_csv(db, None, 1, ["m.one", "m.two"])
    header = text.splitlines()[0].split(",")
    assert "pressingforreasoning__0.m_one" in header
    assert "pressingforreasoning__0.m_two" in header
    aid, _, model = header[-1].rpartition(".")
    assert aid == "pressingforreasoning__0" and model == "m_two"
    rows = _rows(text)
    assert rows[0]["pressingforreasoning__0.m_one"] == "TRUE"
    assert rows[0]["pressingforreasoning__0.m_two"] == "FALSE"


def test_a_sub_criterion_is_blank_where_its_own_gate_did_not_fire(db):
    """A child is only defined where its parent fired, per model."""
    gate = _add_criterion(db, "repeats a student", [])
    _add_criterion(db, "wording matches", ["Revoicing"], parent_id=gate)
    _annotate(db, "repeats a student", "m1", [0])          # gate fires on 0 only
    _annotate(db, "wording matches\n⊂ repeats a student", "m1", [0])
    rows = _rows(export.firing_csv(db, None, 1, ["m1"]))
    assert rows[0]["revoicing__0"] == "TRUE"    # utterance 0: gate fired
    assert rows[1]["revoicing__0"] == ""        # utterance 2: never asked
    assert rows[1]["gate__0"] == "FALSE"        # the gate itself was asked


def test_assertions_csv_is_one_row_per_assertion_with_labels_joined(db):
    _add_criterion(db, "asks why", ["PressingForReasoning", "Revoicing"])
    _add_criterion(db, "a gate", [])
    rows = _rows(export.assertions_csv(db, 1))
    assert [r["name"] for r in rows] == ["asks why", "a gate"]
    by_name = {r["name"]: r for r in rows}
    # Labels are in label-space order, joined with "|".
    assert by_name["asks why"]["labels"] == "Revoicing|PressingForReasoning"
    assert by_name["asks why"]["shortname"] == "revoicing__0"
    assert by_name["a gate"]["labels"] == ""


def test_the_two_csvs_join_one_to_one(db):
    _add_criterion(db, "asks why", ["PressingForReasoning"])
    _add_criterion(db, "a gate", [])
    _annotate(db, "asks why", "m1", [0])
    header = export.firing_csv(db, None, 1, ["m1"]).splitlines()[0].split(",")
    shortnames = {r["shortname"] for r in _rows(export.assertions_csv(db, 1))}
    assert set(header[4:]) == shortnames


def test_bundle_is_a_zip_of_both_csvs(db):
    _add_criterion(db, "asks why", ["PressingForReasoning"])
    _annotate(db, "asks why", "m1", [0])
    z = zipfile.ZipFile(io.BytesIO(export.bundle(db, None, 1, ["m1"])))
    names = sorted(z.namelist())
    assert names == [
        "talk_moves_v3/README.txt",
        "talk_moves_v3/annotations.csv",
        "talk_moves_v3/assertions.csv",
    ]
    assert "asks why" in z.read("talk_moves_v3/assertions.csv").decode()


def test_a_session_missing_from_the_corpus_is_skipped_not_crashed(db, monkeypatch):
    """The pinned corpus may not contain a session an old round used."""
    monkeypatch.setattr(
        export.corpus, "get",
        lambda cfg, sid: (_ for _ in ()).throw(KeyError(sid)),
    )
    _add_criterion(db, "asks why", ["PressingForReasoning"])
    text = export.firing_csv(db, None, 1, ["m1"])
    assert len(text.splitlines()) == 1, "header only"


class TestAvailableModels:
    """A construct worked before a model change holds rows under the old
    nicknames; defaulting to config.annotators() then exports an all-blank matrix that
    looks like a bug rather than like a model change."""

    def test_models_with_data_are_offered_even_when_not_configured(self, db):
        from studio.config import annotators
        MODELS = annotators()
        _add_criterion(db, "asks why", ["PressingForReasoning"])
        _annotate(db, "asks why", "retired_model_v1", [0])
        got = export.available_models(db, 1)
        assert got[: len(MODELS)] == list(MODELS), "configured models sort first"
        assert "retired_model_v1" in got

    def test_a_fresh_construct_still_offers_the_configured_pair(self, db):
        from studio.config import annotators
        MODELS = annotators()
        assert export.available_models(db, 1) == list(MODELS)

    def test_default_covers_retired_models_so_history_is_not_blank(self, db):
        _add_criterion(db, "asks why", ["PressingForReasoning"])
        _annotate(db, "asks why", "retired_model_v1", [0])
        models = export.resolve_models(None, export.available_models(db, 1))
        rows = _rows(export.firing_csv(db, None, 1, models))
        col = "pressingforreasoning__0.retired_model_v1"
        assert rows[0][col] == "TRUE", "the retired model's data must still export"
