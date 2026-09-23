"""Edge-space search.

Rests on the firing matrix being fixed: whether a criterion fires depends on its
TEXT, not its edges, so every rewiring is scorable against cached annotations
with no model calls.

With untyped edges a move is a single bit — wire this criterion to this label, or
cut that edge — so the space is small and no two candidates for one slot can tie.
"""

from studio.rewire import Move, recommend, search
from studio.schema_map import Criterion, Edge

LABELS = ["A", "B", "Other"]


def crit(cid, *labels):
    return Criterion(cid, f"c{cid}", tuple(Edge(l) for l in labels))


def test_finds_the_over_firing_edge_to_cut():
    """The real defect from a live round, in this branch's terms: a criterion is
    wired to B, fires on utterances that are actually A, and cutting that edge
    leaves A alone as the candidate."""
    criteria = [crit(1, "A"), crit(2, "B")]
    # Both fire on every utterance; gold says they are all A.
    consensus = {i: {1: True, 2: True} for i in range(6)}
    gold = {i: "A" for i in range(6)}

    moves, base = recommend(criteria, LABELS, consensus, gold, list(range(6)))
    assert base["gold"] == 0, "over-determined at baseline, so nothing is exact"
    top = moves[0]
    assert top.criterion_id == 2 and top.label == "B"
    assert top.add is False
    assert top.d_gold == 6


def test_a_move_is_add_or_cut_nothing_else():
    criteria = [crit(1, "A", "B")]
    consensus = {0: {1: True}}
    for m in search(criteria, LABELS, consensus, {0: "A"}, [0]):
        labels = [e["label"] for e in m.edges]
        assert len(set(labels)) == len(labels)
        assert labels, "this criterion has two edges, so no move can empty it"
        assert (m.label in labels) is m.add


def test_cutting_the_only_edge_is_offered_as_a_removal():
    """There is no demotion left, so on a single-edge criterion the one
    edge-space repair available is to cut that edge — which retires the
    criterion, and is reported as such rather than as a rewire."""
    criteria = [crit(1, "A")]
    moves = search(criteria, LABELS, {0: {1: True}}, {0: "A"}, [0])
    cut = [m for m in moves if m.label == "A" and not m.add]
    assert len(cut) == 1
    assert cut[0].removes and cut[0].edges == []


def test_a_sub_criterion_is_never_emptied():
    """An edgeless sub-criterion is invalid, not a retirement."""
    from studio.schema_map import Criterion, Edge

    gate = Criterion(1, "gate", ())
    child = Criterion(2, "child", (Edge("A"),), 1)
    moves = search([gate, child], LABELS, {0: {1: True, 2: True}}, {0: "A"}, [0])
    assert not any(m.removes for m in moves)


def test_helps_both_requires_gold_and_session_agreement():
    """Gold is ~10 utterances, well-definedness is ~300 and needs no gold.
    Requiring both is the guard against fitting noise in a tiny sample."""
    good = Move(1, "t", "A", False, [], 2, 1)
    gold_only = Move(1, "t", "A", False, [], 2, -3)
    no_gold_gain = Move(1, "t", "A", False, [], 0, 5)

    assert good.helps_both
    assert not gold_only.helps_both
    assert not no_gold_gain.helps_both


def test_one_row_per_criterion_label_slot():
    criteria = [crit(1, "A"), crit(2, "B")]
    consensus = {i: {1: True, 2: True} for i in range(6)}
    moves, _ = recommend(criteria, LABELS, consensus, {i: "A" for i in range(6)}, list(range(6)))
    slots = [(m.criterion_id, m.label) for m in moves]
    assert len(slots) == len(set(slots))


def test_a_clean_single_criterion_schema_yields_nothing():
    """Nothing to fix: no gold to fit and nothing left unresolved."""
    criteria = [crit(1, "A")]
    moves, base = recommend(criteria, LABELS, {0: {1: True}}, {}, [0])
    assert moves == []
    assert base["n_gold"] == 0


def test_search_is_linear_not_combinatorial():
    """Single-edge moves only: criteria x labels, never the joint space."""
    criteria = [crit(i, "A") for i in range(1, 11)]
    consensus = {0: {i: True for i in range(1, 11)}}
    moves = search(criteria, LABELS, consensus, {0: "A"}, [0])
    assert len(moves) <= 10 * len(LABELS)


class TestConsistencyOnlyMoves:
    """Under/over-determination is a property of the schema alone, so it is
    measurable on every utterance with no labelling. Requiring a gold improvement
    hid those fixes entirely on runs where gold was thin.
    """

    def test_an_over_determination_fix_is_proposed_without_any_gold(self):
        # Both criteria fire everywhere, so every utterance has two candidates.
        criteria = [crit(1, "A"), crit(2, "B")]
        consensus = {i: {1: True, 2: True} for i in range(5)}
        moves, base = recommend(criteria, LABELS, consensus, {}, list(range(5)))
        assert base["n_gold"] == 0
        assert base["welldef"] == 0
        assert moves, "no gold is not a reason to stay silent about competing labels"
        assert moves[0].d_welldef > 0

    def test_gold_backed_moves_still_come_first(self):
        criteria = [crit(1, "A"), crit(2, "B")]
        consensus = {i: {1: True, 2: True} for i in range(6)}
        moves, _ = recommend(criteria, LABELS, consensus, {i: "A" for i in range(6)}, list(range(6)))
        assert moves[0].helps_both

    def test_a_move_that_hurts_gold_is_never_consistency_only(self):
        m = Move(1, "t", "A", False, [], -1, 9)
        assert not m.helps_consistency
        assert not m.helps_both

    def test_the_two_classes_are_disjoint(self):
        m = Move(1, "t", "A", False, [], 2, 3)
        assert m.helps_both and not m.helps_consistency
