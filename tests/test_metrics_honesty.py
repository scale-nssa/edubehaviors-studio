"""Metrics that were reporting agreement they had not measured.

Both defects showed the same way in a live construct: a column of confident
numbers that no amount of schema work could move, because nothing in the data
fed them.
"""

from studio.metrics import _alpha, _rollups
from studio.schema_view import Row


class TestAlphaOnDegenerateInput:
    def test_neither_model_ever_fired_is_undefined_not_perfect(self):
        """A criterion nothing matched used to score 1.00 and sort to the top
        of "best agreement". There is nothing to agree about."""
        assert _alpha([0] * 20, [0] * 20) is None

    def test_both_fired_everywhere_is_still_unanimous(self):
        """Degenerate too, but there is a real observation behind it, and the
        fire-rate flag already calls out a criterion at 90%+."""
        assert _alpha([1] * 20, [1] * 20) == 1.0

    def test_real_disagreement_still_computed(self):
        a = [1, 1, 0, 0, 1, 0, 1, 0]
        b = [1, 0, 0, 0, 1, 0, 1, 1]
        got = _alpha(a, b)
        assert got is not None and 0.0 < got < 1.0

    def test_rollups_no_longer_inflated_by_silent_criteria(self):
        """Twenty never-firing criteria at 1.00 apiece used to drag the macro
        average up to near-perfect regardless of the ones that fire."""
        per = {1: {"alpha": _alpha([0] * 10, [0] * 10), "fires": 0},
               2: {"alpha": 0.2, "fires": 10}}
        assert _rollups(per)["macro"] == 0.2


class TestFlipsRatherThanAgreement:
    """Review pre-checks every criterion at the model's verdict, so a reviewer
    who touches nothing "agrees" with everything — which is how 24 criteria all
    read 100% (5). Flips make no claim the data cannot support."""

    def test_untouched_reads_as_zero_flips_not_full_agreement(self):
        assert Row(criterion_id=1, text="t", human_flips=0, human_total=5).human == "0 / 5"

    def test_overridden_criterion_is_visible(self):
        assert Row(criterion_id=1, text="t", human_flips=4, human_total=5).human == "4 / 5"

    def test_never_reviewed_says_nothing(self):
        assert Row(criterion_id=1, text="t").human is None
