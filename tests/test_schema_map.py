"""The map is the part of this system most likely to be subtly wrong. These
tests pin the behaviour of the pared-down map: one kind of edge, so the
candidate set is the union of what fired and nothing subtracts.
"""

from studio.schema_map import (
    Criterion,
    Edge,
    Outcome,
    apply_map,
    flip_sensitivity,
    validate_edges,
)

LABELS = ["REVOICE", "PRESS", "OTHER"]


def crit(cid, *labels):
    return Criterion(id=cid, text=f"c{cid}", edges=tuple(Edge(l) for l in labels))


def test_one_firing_edge_determines():
    cs = [crit(1, "REVOICE")]
    r = apply_map(cs, {1: True})
    assert r.candidates == {"REVOICE"}
    assert r.outcome is Outcome.DETERMINED


def test_several_criteria_one_label_still_determines():
    """Presence, not count: agreement among criteria is not a stronger claim."""
    cs = [crit(1, "REVOICE"), crit(2, "REVOICE"), crit(3, "REVOICE")]
    r = apply_map(cs, {1: True, 2: True, 3: True})
    assert r.candidates == {"REVOICE"}
    assert r.outcome is Outcome.DETERMINED


def test_nothing_subtracts_a_label_another_criterion_raised():
    """The defining property of this branch. There is no edge that could remove
    REVOICE here, so an extra firing criterion can only ever add a candidate."""
    cs = [crit(1, "REVOICE"), crit(2, "PRESS")]
    r = apply_map(cs, {1: True, 2: True})
    assert r.candidates == {"REVOICE", "PRESS"}
    assert r.outcome is Outcome.OVER


def test_a_criterion_with_two_edges_raises_both():
    cs = [crit(1, "REVOICE", "PRESS")]
    r = apply_map(cs, {1: True})
    assert r.candidates == {"REVOICE", "PRESS"}
    assert r.outcome is Outcome.OVER


def test_under_determined_when_nothing_fires():
    cs = [crit(1, "REVOICE")]
    r = apply_map(cs, {1: False})
    assert r.candidates == frozenset()
    assert r.outcome is Outcome.UNDER


def test_absent_from_firing_means_false():
    cs = [crit(1, "REVOICE"), crit(2, "PRESS")]
    r = apply_map(cs, {2: True})
    assert r.candidates == {"PRESS"}


def test_presence_not_count_five_for_one_label_ties_with_one_for_another():
    cs = [crit(i, "REVOICE") for i in range(1, 6)]
    cs.append(crit(9, "PRESS"))
    r = apply_map(cs, dict.fromkeys([1, 2, 3, 4, 5, 9], True))
    assert r.candidates == {"REVOICE", "PRESS"}
    assert r.outcome is Outcome.OVER


def test_a_gate_carries_no_edges_and_raises_nothing():
    gate = Criterion(id=1, text="gate", edges=())
    r = apply_map([gate, crit(2, "PRESS")], {1: True, 2: False})
    assert r.candidates == frozenset()
    assert r.outcome is Outcome.UNDER


def test_flip_sensitivity_counts_outcome_changing_flips():
    cs = [crit(1, "REVOICE"), crit(2, "PRESS")]
    assert flip_sensitivity(cs, {1: True, 2: False}) == 2
    # Flipping 2 changes nothing: REVOICE is already a candidate via 1, and 2
    # points at the same label.
    cs2 = [crit(1, "REVOICE"), crit(2, "REVOICE")]
    assert flip_sensitivity(cs2, {1: True, 2: True}) == 0


class TestValidation:
    def test_rejects_two_edges_on_one_label(self):
        assert validate_edges([Edge("REVOICE"), Edge("REVOICE")], LABELS)

    def test_rejects_empty(self):
        assert validate_edges([], LABELS)

    def test_rejects_unknown_label(self):
        assert validate_edges([Edge("NOPE")], LABELS)

    def test_allows_several_labels(self):
        assert not validate_edges([Edge("REVOICE"), Edge("PRESS")], LABELS)

    def test_allows_one(self):
        assert not validate_edges([Edge("REVOICE")], LABELS)
