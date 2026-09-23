import pytest

from studio.annotate import parse_true_indices, validate_indices
from studio.metrics import predictive, schema_logic, wilson
from studio.schema_map import Criterion, Edge, Outcome
from studio.selection import Candidate, TIER_RANK, choose


# --------------------------------------------------------------------------- #
# Sparse positive-only parsing (spec §3.1)
# --------------------------------------------------------------------------- #

class TestParse:
    def test_bare_ints(self):
        assert parse_true_indices({"true_indices": [3, 1, 2]}) == [3, 1, 2]

    def test_upstream_idx_objects(self):
        assert parse_true_indices({"true_indices": [{"idx": 4, "confidence": 5}]}) == [4]

    def test_empty_is_valid_and_common(self):
        assert parse_true_indices({"true_indices": []}) == []

    def test_missing_key_raises_so_caller_retries(self):
        with pytest.raises(ValueError):
            parse_true_indices({"indices": [1]})

    def test_bool_is_not_an_index(self):
        with pytest.raises(ValueError):
            parse_true_indices({"true_indices": [True]})

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            parse_true_indices({"true_indices": "1,2,3"})


class TestValidate:
    def test_drops_out_of_range_and_out_of_scope_with_counts(self):
        kept, warn = validate_indices([1, 2, 99, 3], candidates={1, 3}, all_indices={1, 2, 3})
        assert kept == [1, 3]
        assert warn["out_of_range"] == 1   # 99
        assert warn["out_of_scope"] == 1   # 2 is in range but not a candidate
        assert warn["duplicate"] == 0

    def test_dedupes(self):
        kept, warn = validate_indices([1, 1, 1], {1}, {1})
        assert kept == [1]
        assert warn["duplicate"] == 2

    def test_never_raises_on_garbage_indices(self):
        kept, _ = validate_indices([-5, 10**9], {1}, {1})
        assert kept == []


# --------------------------------------------------------------------------- #
# Selection ladder and caps (docs/selection.md)
# --------------------------------------------------------------------------- #

def cand(index, outcome, *, disagree=False, flips=0, words=10, involved=()):
    return Candidate(
        index=index,
        outcome=outcome,
        tier=TIER_RANK[outcome],
        disagree=disagree,
        flips=flips,
        words=words,
        candidates=frozenset(),
        involved=frozenset(involved),
    )


class TestChoose:
    def test_ladder_order_competing_labels_first(self):
        """Contradiction used to head the ladder; with one kind of edge it cannot
        happen, so competing labels lead."""
        scored = [
            cand(1, Outcome.DETERMINED),
            cand(2, Outcome.UNDER),
            cand(3, Outcome.TIER3),
            cand(4, Outcome.OVER),
            cand(5, Outcome.DEFAULT),
        ]
        # Reading order is (tier, index); the tiers are what this pins down.
        got = [c.index for c in choose(scored, n=5)]
        assert got == [4, 2, 5, 3, 1]

    def test_tier_cap_stops_one_tier_eating_the_round(self):
        scored = [cand(i, Outcome.UNDER) for i in range(20)]
        scored += [cand(100 + i, Outcome.DETERMINED) for i in range(20)]
        picked = choose(scored, n=10, tier_cap=6)
        n_under = sum(1 for c in picked if c.outcome is Outcome.UNDER)
        assert n_under == 6
        assert len(picked) == 10

    def test_criterion_cap_stops_one_criterion_monopolising(self):
        scored = [cand(i, Outcome.OVER, involved=(7,)) for i in range(10)]
        scored += [cand(100 + i, Outcome.OVER, involved=(8,)) for i in range(10)]
        picked = choose(scored, n=6, tier_cap=10, criterion_cap=3)
        assert sum(1 for c in picked if 7 in c.involved) == 3
        assert sum(1 for c in picked if 8 in c.involved) == 3

    def test_criterion_cap_relaxes_only_as_a_last_resort(self):
        """Filling the round matters less than not letting one criterion own it,
        so the cap is the last thing to give."""
        scored = [cand(i, Outcome.OVER, involved=(7,)) for i in range(10)]
        picked = choose(scored, n=10, tier_cap=10, criterion_cap=3)
        assert len(picked) == 10  # backfilled, but only after everything else

    def test_backfill_when_caps_leave_us_short(self):
        scored = [cand(i, Outcome.UNDER) for i in range(20)]
        assert len(choose(scored, n=10, tier_cap=6)) == 10

    def test_agreement_leads_within_a_tier(self):
        """If both models fire the same way and the schema still can't resolve
        it, that is an unambiguous schema defect. Disagreement may just be one
        annotator, which is assertion-sharpening and explicitly secondary.

        n=1 because `choose` returns the batch in READING order (tier, then
        utterance index), not ranking order — so with n=2 both get in and the
        order says nothing about which ranked higher. Taking one forces the
        ranking to show itself.
        """
        scored = [
            cand(1, Outcome.OVER, disagree=True, flips=9),
            cand(2, Outcome.OVER, disagree=False, flips=1),
        ]
        assert [c.index for c in choose(scored, n=1)] == [2]

    def test_under_tier_breaks_ties_on_length_not_flips(self):
        """Flip-sensitivity is degenerate when nothing fired: everything ties at n,
        so a naive sort fills the round with one-word utterances."""
        scored = [
            cand(1, Outcome.UNDER, disagree=True, flips=10, words=2),
            cand(2, Outcome.UNDER, disagree=True, flips=10, words=25),
        ]
        assert [c.index for c in choose(scored, n=1)] == [2]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class TestMetrics:
    def test_schema_logic_counts(self):
        cs = [
            Criterion(1, "a", (Edge("A"),)),
            Criterion(2, "b", (Edge("B"),)),
        ]
        firings = {1: {1: True}, 2: {2: True}, 3: {}}
        got = schema_logic(cs, firings, [1, 2, 3])
        assert got["counts"] == {
            "determined": 2, "tier3": 0, "default": 0, "under": 1, "over": 0
        }
        assert got["well_defined_by_evidence"] == pytest.approx(2 / 3)
        assert got["well_defined_overall"] == pytest.approx(2 / 3)

    def test_predictive_matches_the_readme_worked_example(self):
        """spec §6.1: 4×({A},A), 2×({A},B), 3×({A,B},A), 1×({B},B)."""
        pairs = (
            [(frozenset({"A"}), "A")] * 4
            + [(frozenset({"A"}), "B")] * 2
            + [(frozenset({"A", "B"}), "A")] * 3
            + [(frozenset({"B"}), "B")] * 1
        )
        got = predictive(pairs, ["A", "B"])
        a = got["per_label"]["A"]
        assert a["optimistic_precision"] == pytest.approx(4 / 6)
        assert a["pessimistic_precision"] == pytest.approx(4 / 9)
        assert a["optimistic_recall"] == pytest.approx(1.0)
        assert a["pessimistic_recall"] == pytest.approx(4 / 7)
        # gold ∈ pred for 4 + 3 + 1 = 8 of 10
        assert got["optimistic_accuracy"] == pytest.approx(8 / 10)
        assert got["pessimistic_accuracy"] == pytest.approx(5 / 10)

    def test_pessimistic_precision_never_exceeds_optimistic(self):
        pairs = (
            [(frozenset({"A"}), "A")] * 3
            + [(frozenset({"A", "B"}), "A")] * 2
            + [(frozenset({"B"}), "B")] * 2
        )
        for v in predictive(pairs, ["A", "B"]).values():
            if isinstance(v, dict) and "optimistic_precision" in v:
                o, p = v["optimistic_precision"], v["pessimistic_precision"]
                if o is not None and p is not None:
                    assert p <= o + 1e-12

    def test_wilson_widens_at_small_n(self):
        lo10, hi10 = wilson(7, 10)
        lo100, hi100 = wilson(70, 100)
        assert (hi10 - lo10) > (hi100 - lo100)


class TestSalvage:
    """Malformed JSON showed up ~1 call in 20 live; the shape is constrained
    enough to recover without paying for a full retry."""

    def test_recovers_from_broken_array_syntax(self):
        from studio.annotate import salvage_true_indices

        assert salvage_true_indices('{"true_indices": [1, 2 3, 4]}') == [1, 2, 3, 4]

    def test_recovers_from_truncation(self):
        from studio.annotate import salvage_true_indices

        assert salvage_true_indices('{"true_indices": [10, 20, 30') == [10, 20, 30]

    def test_empty_array_is_recovered_as_empty_not_none(self):
        from studio.annotate import salvage_true_indices

        assert salvage_true_indices('{"true_indices": []}') == []

    def test_returns_none_when_unrecognisable_so_caller_retries(self):
        from studio.annotate import salvage_true_indices

        assert salvage_true_indices("I'm sorry, I cannot help with that.") is None
        assert salvage_true_indices("") is None


class TestAtomicityLinter:
    """Mechanically-detectable subset of the atomicity spec in
    ../atoms/schematize-codebook. Advisory only — the paraphrase test itself
    needs a human or a model."""

    def test_the_four_style_checks_are_gone(self):
        """Fused verbs, carve-outs, quoted phrases and exclusion-defined
        residuals no longer warn.

        The advice is still right, and still delivered — in the seeding prompt,
        where it shapes criteria before they exist. `_ATOMICITY` (4) forbids
        literal matchers by name and (4a) forbids exclusion-defined residuals;
        `_DECOMPOSE` covers carve-outs. A warning on Revise arrived after the
        fact, when the researcher is reading evidence rather than style, and
        each of these fired on wording that was often deliberate.
        """
        from studio.schema_map import lint_criterion_text
        for t in [
            # fused verbs, and the participle false positive it could not avoid
            "Does the tutor walk through a procedure or define a term?",
            "The utterance repeats a prior student utterance word-for-word, "
            "with no added or changed wording.",
            # carve-outs
            "The student states a fact without commenting on their own thinking.",
            "The utterance asks for an answer rather than an explanation.",
            # quoted phrase
            'Does the utterance contain the word "why" (or a direct equivalent)?',
            # defined by exclusion
            "Does the utterance contain none of the following: a question, "
            "a restatement?",
        ]:
            assert lint_criterion_text(t) == [], t

    def test_what_the_linter_still_catches(self):
        """Not stylistic: unusable, or nearly always several criteria fused."""
        from studio.schema_map import lint_criterion_text
        assert lint_criterion_text("   ") == ["Empty criterion."]
        assert any("Very long" in o for o in lint_criterion_text("word " * 60))

    def test_allows_illustrative_parenthetical(self):
        from studio.schema_map import lint_criterion_text
        assert lint_criterion_text(
            "Does the tutor ask the student to justify a claim (e.g., asking for reasons)?"
        ) == []

    def test_flags_very_long_criteria(self):
        from studio.schema_map import lint_criterion_text
        assert any("Very long" in o for o in lint_criterion_text("word " * 60))

    def test_empty_is_reported(self):
        from studio.schema_map import lint_criterion_text
        assert lint_criterion_text("   ") == ["Empty criterion."]


class TestFireRateFlags:
    """Consistency scoring cannot catch a mis-wired edge — cutting one that fires
    everywhere *reduces* well-definedness — so it has to be visible."""

    def _row(self, fires):
        from studio.schema_view import Row
        return Row(criterion_id=1, text="t", fires=fires)

    def test_a_high_but_plausible_fire_rate_is_not_flagged(self):
        """35%+ used to raise "possibly mis-wired"; it cried wolf."""
        assert self._row(0.63).flag is None
        assert self._row(0.40).flag is None

    def test_does_not_flag_a_normal_fire_rate(self):
        assert self._row(0.08).flag is None

    def test_flags_criteria_that_never_fire(self):
        assert self._row(0.0).flag == "never fires — unseen behavior"

    def test_flags_criteria_that_fire_on_everything(self):
        assert "almost no information" in self._row(0.95).flag

    def test_silent_without_annotation_data(self):
        assert self._row(None).flag is None
