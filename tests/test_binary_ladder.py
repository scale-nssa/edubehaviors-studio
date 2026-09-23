"""The binary selection ladder.

The general ladder ranks by how badly the schema failed, so a cleanly resolved
positive is DETERMINED — last, reached only as backfill. On a binary construct
that means a round can contain no positives at all, which is what happened:
188 of 292 in-scope utterances predicted positive, zero of the ten drawn.
"""

import itertools

import pytest

from studio.schema_map import Criterion, Edge, LabelRules, Outcome, apply_map
from studio.selection import (
    BIN_CONFLICT_BOTH,
    BIN_CONFLICT_ONE,
    BIN_NEG_AGREE,
    BIN_POS_AGREE,
    BIN_POS_DIFFER,
    BIN_TIER_LABEL,
    _binary_tier,
)

P = "Meta"
RULES = LabelRules(other_label="None", priorities=())
POSCRIT = Criterion(1, "positive", (Edge(P),))
NEGCRIT = Criterion(2, "catch-all-ish", (Edge("None"),))


def raw(pos: bool, neg: bool) -> frozenset:
    """One model's candidate set, computed WITHOUT the rules."""
    return apply_map([POSCRIT, NEGCRIT], {1: pos, 2: neg}).candidates


class TestOverIsUnreachable:
    """Why 'competing labels' has to be detected before the rules run."""

    def test_the_catch_all_rule_collapses_two_candidates_to_one(self):
        r = apply_map([POSCRIT, NEGCRIT], {1: True, 2: True}, RULES)
        assert r.candidates == {P}
        assert r.outcome is Outcome.TIER3

    def test_so_a_binary_construct_can_never_reach_OVER(self):
        for pos, neg in itertools.product([False, True], repeat=2):
            r = apply_map([POSCRIT, NEGCRIT], {1: pos, 2: neg}, RULES)
            assert r.outcome is not Outcome.OVER

    def test_but_the_conflict_is_visible_pre_rule(self):
        assert raw(True, True) == {P, "None"}


class TestPartition:
    def test_every_combination_lands_in_exactly_one_tier(self):
        """Two models × (positive fired?, catch-all fired?) = 16 states."""
        seen = {}
        for a in itertools.product([False, True], repeat=2):
            for b in itertools.product([False, True], repeat=2):
                t = _binary_tier([raw(*a), raw(*b)], P)
                assert t in BIN_TIER_LABEL
                seen[(a, b)] = t
        assert len(seen) == 16
        # and every tier is actually reachable
        assert set(seen.values()) == set(BIN_TIER_LABEL)

    def test_ordering_is_the_one_asked_for(self):
        assert [
            BIN_POS_AGREE, BIN_POS_DIFFER,
            BIN_CONFLICT_BOTH, BIN_CONFLICT_ONE, BIN_NEG_AGREE,
        ] == [0, 1, 2, 3, 4]

    def test_both_models_positive_only(self):
        assert _binary_tier([raw(True, False), raw(True, False)], P) == BIN_POS_AGREE

    def test_one_model_positive(self):
        assert _binary_tier([raw(True, False), raw(False, False)], P) == BIN_POS_DIFFER
        assert _binary_tier([raw(False, True), raw(True, False)], P) == BIN_POS_DIFFER

    def test_both_models_conflicted(self):
        assert _binary_tier([raw(True, True), raw(True, True)], P) == BIN_CONFLICT_BOTH

    def test_one_model_conflicted(self):
        """Conflict is checked first, so this does not get absorbed into the
        positive tiers even though both models resolve to positive."""
        assert _binary_tier([raw(True, True), raw(True, False)], P) == BIN_CONFLICT_ONE

    def test_both_catch_all(self):
        assert _binary_tier([raw(False, False), raw(False, False)], P) == BIN_NEG_AGREE
        assert _binary_tier([raw(False, True), raw(False, True)], P) == BIN_NEG_AGREE

    def test_silence_and_a_catch_all_criterion_tier_together(self):
        """Nothing fired and 'a None criterion fired' are both the negative
        answer; the ladder does not separate them."""
        assert (_binary_tier([raw(False, False), raw(False, True)], P)
                == BIN_NEG_AGREE)


class TestPresentationOrder:
    """Which ten get picked, and what order they are read in, are different
    questions. Ranking by fragility is right for the first and wrong for the
    second: it walks the transcript at random and throws away the context the
    reviewer just built."""

    def _c(self, index, tier, flips):
        from studio.schema_map import Outcome
        from studio.selection import Candidate
        return Candidate(
            index=index, outcome=Outcome.DETERMINED, tier=tier, disagree=False,
            flips=flips, words=10, candidates=frozenset(), involved=frozenset(),
        )

    def test_within_a_tier_ordered_by_utterance_index(self):
        from studio.selection import choose
        got = choose([self._c(50, 0, 9), self._c(10, 0, 1), self._c(30, 0, 5)], n=3)
        assert [c.index for c in got] == [10, 30, 50]

    def test_tiers_still_come_in_ladder_order(self):
        from studio.selection import choose
        got = choose([self._c(1, 2, 1), self._c(99, 0, 1), self._c(50, 1, 1)], n=3)
        assert [c.index for c in got] == [99, 50, 1]

    def test_fragility_still_decides_who_gets_in(self):
        """Ordering is presentational — it must not change the selection."""
        from studio.selection import choose
        cands = [self._c(1, 0, 1), self._c(2, 0, 9), self._c(3, 0, 5)]
        assert {c.index for c in choose(cands, n=2)} == {2, 3}

    def test_regressions_stay_at_the_front(self):
        from studio.selection import choose
        got = choose([self._c(5, 0, 1), self._c(90, 4, 1)], n=2, prioritise=[90])
        assert [c.index for c in got] == [90, 5]
