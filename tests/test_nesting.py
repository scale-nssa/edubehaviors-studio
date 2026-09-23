"""Two-level criteria.

A parent is a pure gate: no edges, and its children are only annotated on the
utterances it selected. The gate carries no edges because if it did, they would
be raised on every match and its children could never narrow the label down —
which is the whole point of nesting.

With one kind of edge, nesting carries more weight than it used to: since no
edge can rule a label out, two narrow sibling criteria are the main way to keep
two confusable labels from competing.
"""

from studio.schema_map import (
    Criterion,
    Edge,
    Outcome,
    apply_map,
    validate_nesting,
)
from studio.service import annotation_key, criteria_keys

GATE = Criterion(1, "repeats a prior student utterance", (), None)
EXACT = Criterion(2, "wording matches exactly", (Edge("Restating"),), 1)
ALTERED = Criterion(3, "wording is altered", (Edge("Revoicing"),), 1)


class TestValidation:
    def test_a_well_formed_nest_passes(self):
        assert validate_nesting([GATE, EXACT, ALTERED]) == []

    def test_a_gate_may_not_carry_edges(self):
        bad = Criterion(1, "gate", (Edge("Restating"),), None)
        problems = validate_nesting([bad, EXACT])
        assert any("must carry no edges" in p for p in problems)

    def test_nesting_is_two_levels_only(self):
        deep = Criterion(4, "deeper", (Edge("Restating"),), 2)
        assert any("two levels only" in p for p in validate_nesting([GATE, EXACT, deep]))

    def test_a_child_needs_edges(self):
        mute = Criterion(5, "no edges", (), 1)
        assert any("no edges" in p for p in validate_nesting([GATE, mute]))

    def test_a_missing_parent_is_reported(self):
        orphan = Criterion(9, "x", (Edge("Restating"),), 77)
        assert any("does not exist" in p for p in validate_nesting([orphan]))

    def test_siblings_pointing_at_each_others_labels_is_legal(self):
        """There is no sibling rule left to break: an edge cannot cancel another,
        so the worst two overlapping siblings can do is over-determine."""
        a = Criterion(2, "exact", (Edge("Restating"), Edge("Revoicing")), 1)
        b = Criterion(3, "altered", (Edge("Revoicing"),), 1)
        assert validate_nesting([GATE, a, b]) == []


class TestMap:
    def test_children_discriminate_where_the_gate_alone_could_not(self):
        cs = [GATE, EXACT, ALTERED]
        r = apply_map(cs, {1: True, 2: True, 3: False})
        assert r.candidates == {"Restating"}
        assert r.outcome is Outcome.DETERMINED

    def test_the_gate_firing_alone_settles_nothing(self):
        r = apply_map([GATE, EXACT, ALTERED], {1: True})
        assert r.candidates == frozenset()
        assert r.outcome is Outcome.UNDER

    def test_both_siblings_firing_over_determines(self):
        """Not a schema error — an unresolved utterance the researcher can look
        at, which is the outcome this design prefers to a contradiction."""
        r = apply_map([GATE, EXACT, ALTERED], {1: True, 2: True, 3: True})
        assert r.candidates == {"Restating", "Revoicing"}
        assert r.outcome is Outcome.OVER


class TestCacheKey:
    def test_a_child_key_carries_its_parent(self):
        """Without this, rewording a gate leaves every child's cached firing
        intact and silently wrong, because the plain key is unchanged."""
        a = annotation_key("wording matches exactly", "repeats a prior utterance")
        b = annotation_key("wording matches exactly", "echoes a prior utterance")
        assert a != b
        assert annotation_key("x", None) == "x"

    def test_criteria_keys_resolves_parent_text(self):
        keys = dict((k[0], k[2]) for k in criteria_keys([GATE, EXACT, ALTERED]))
        assert keys[1] is None
        assert keys[2] == GATE.text
        assert keys[3] == GATE.text


class TestRewire:
    def test_the_search_never_proposes_edges_on_a_gate(self):
        from studio.rewire import search

        cs = [GATE, EXACT, ALTERED]
        consensus = {i: {1: True, 2: True} for i in range(4)}
        moves = search(cs, ["Restating", "Revoicing"], consensus, {0: "Restating"}, list(range(4)))
        assert all(m.criterion_id != GATE.id for m in moves), \
            "giving a gate edges would stop its children discriminating"

    def test_the_search_never_proposes_cutting_a_childs_last_edge(self):
        """A sub-criterion with no edges is invalid, and validate_edges rejects
        the empty set, so the move is not a candidate."""
        from studio.rewire import search

        consensus = {i: {1: True, 2: True} for i in range(4)}
        moves = search([GATE, EXACT, ALTERED], ["Restating", "Revoicing"],
                       consensus, {0: "Restating"}, list(range(4)))
        assert not any(
            m.criterion_id == EXACT.id and m.label == "Restating" and not m.add
            for m in moves
        )
