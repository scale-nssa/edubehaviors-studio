"""Boundary rules — the third tier.

Consulted only when the evidence has already failed. The provenance split
matters more than it looks: folding rule-resolved utterances into the
well-definedness figure would make a schema with NO criteria perfectly
well-defined, because everything falls through to the catch-all.
"""

import pytest

from studio.metrics import schema_logic
from studio.rewire import _score, rule_load, search_rules
from studio.schema_map import (
    Criterion,
    Edge,
    LabelRules,
    Outcome,
    apply_map,
)

LABELS = ["PFA", "PFR", "None"]
A = Criterion(1, "asks for a fact", (Edge("PFA"),))
B = Criterion(2, "asks why", (Edge("PFR"),))
O = Criterion(3, "logistics", (Edge("None"),))
RULES = LabelRules(other_label="None", priorities=(("PFR", "PFA"),))


class TestTierThree:
    def test_priority_settles_a_tie(self):
        r = apply_map([A, B], {1: True, 2: True}, RULES)
        assert r.candidates == {"PFR"}
        assert r.outcome is Outcome.TIER3

    def test_silence_routes_to_the_catch_all(self):
        """A label, but by default rather than by decision — its own outcome."""
        r = apply_map([A, B], {}, RULES)
        assert r.candidates == {"None"}
        assert r.outcome is Outcome.DEFAULT

    def test_the_catch_all_never_beats_a_real_label(self):
        r = apply_map([A, O], {1: True, 3: True}, RULES)
        assert r.candidates == {"PFA"}

    def test_rules_never_overrule_evidence(self):
        """The evidence settled it, so the rules are not consulted at all."""
        r = apply_map([A, B], {1: True}, RULES)
        assert r.candidates == {"PFA"}
        assert r.outcome is Outcome.DETERMINED

    def test_no_catch_all_means_the_defaults_do_not_apply(self):
        bare = LabelRules(priorities=(("PFR", "PFA"),))
        assert apply_map([A, B], {}, bare).outcome is Outcome.UNDER

    def test_no_rules_at_all_behaves_exactly_as_before(self):
        assert apply_map([A, B], {1: True, 2: True}).outcome is Outcome.OVER


class TestCycles:
    def test_a_two_cycle_is_detected(self):
        assert LabelRules(priorities=(("A", "B"), ("B", "A"))).cycles()

    def test_a_longer_cycle_is_detected(self):
        assert LabelRules(priorities=(("A", "B"), ("B", "C"), ("C", "A"))).cycles()

    def test_a_chain_is_not_a_cycle(self):
        assert LabelRules(priorities=(("A", "B"), ("B", "C"))).cycles() == []


class TestProvenance:
    def test_the_catch_all_fall_through_is_never_counted_as_well_defined(self):
        firings = {i: {} for i in range(10)}   # nothing fires anywhere
        got = schema_logic([A, B], firings, list(range(10)), RULES)
        assert got["counts"]["default"] == 10
        assert got["well_defined_overall"] == 0.0, \
            "a schema whose criteria never fire is not well-defined"
        assert got["by_default"] == 1.0

    def test_a_precedence_rule_does_count_as_well_defined(self):
        """The rules are schema. A stated precedence, applied to candidates the
        edges raised, is the schema deciding — not the absence of a decision."""
        firings = {i: {1: True, 2: True} for i in range(10)}
        got = schema_logic([A, B], firings, list(range(10)), RULES)
        assert got["counts"]["tier3"] == 10
        assert got["well_defined_overall"] == 1.0
        assert got["by_default"] == 0.0

    def test_an_empty_schema_is_not_well_defined_just_because_of_the_catch_all(self):
        got = schema_logic([], {i: {} for i in range(5)}, list(range(5)), RULES)
        assert got["well_defined_overall"] == 0.0

    def test_the_rewire_score_ignores_rule_resolutions(self):
        """Otherwise deleting an edge looks like an improvement: fewer firings,
        more fall-through, higher score."""
        consensus = {i: {1: True} for i in range(6)}
        without = Criterion(1, "asks for a fact", ())
        wd_with = _score([A], consensus, {}, list(range(6)), RULES)[1]
        wd_without = _score([without], consensus, {}, list(range(6)), RULES)[1]
        assert wd_with > wd_without, "removing an edge must not raise the score"


class TestRuleSearch:
    def test_finds_the_rule_that_matches_gold(self):
        consensus = {i: {1: True, 2: True} for i in range(8)}
        gold = {i: "PFR" for i in range(8)}
        moves = search_rules([A, B], LABELS, consensus, gold, list(range(8)))
        assert moves[0].winner == "PFR" and moves[0].loser == "PFA"
        assert moves[0].d_gold == 8

    def test_never_proposes_a_rule_that_would_cycle(self):
        consensus = {i: {1: True, 2: True} for i in range(4)}
        base = LabelRules(priorities=(("PFA", "PFR"),))
        moves = search_rules([A, B], LABELS, consensus, {}, list(range(4)), base)
        assert not any(m.winner == "PFR" and m.loser == "PFA" for m in moves)

    def test_rule_load_reports_how_much_work_a_rule_does(self):
        consensus = {i: {1: True, 2: True} for i in range(8)}
        load = rule_load([A, B], consensus, list(range(8)), RULES)
        assert load[0]["decides"] == 8
        assert load[0]["share"] == 1.0
