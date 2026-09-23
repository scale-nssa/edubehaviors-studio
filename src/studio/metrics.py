"""Metrics. Spec §6.

Everything is per round (Q78). Gold is human review labels only (Q121); there is
no held-out test set (Q122), so every gold-dependent number here is computed on
a deliberately-informative sample and is not a dataset-wide estimate.
"""

from __future__ import annotations

import krippendorff
import numpy as np

from .schema_map import Criterion, Outcome, apply_map


# --------------------------------------------------------------------------- #
# Schema-logic metrics (spec §2.3)
# --------------------------------------------------------------------------- #

def schema_logic(
    criteria: list[Criterion],
    firings: dict[int, dict[int, bool]],
    universe: list[int],
    rules=None,
) -> dict:
    """Well-definedness over a set of utterances.

    Boundary rules count toward well-definedness — they are schema, and an
    utterance a precedence rule settles is one the schema decided. What is held
    out is the empty fall-through (`default`): nothing fired and the catch-all
    supplied a label. Counting that would make a schema with no criteria at all
    perfectly well-defined.

    There is no validity figure here. Validity measured contradictions, and with
    a single kind of edge nothing can contradict anything — the only ways the
    map fails are silence and competition, both counted below.
    """
    n = len(universe) or 1
    determined = tier3 = default = under = over = 0
    for idx in universe:
        res = apply_map(criteria, firings.get(idx, {}), rules)
        if res.outcome is Outcome.DETERMINED:
            determined += 1
        elif res.outcome is Outcome.TIER3:
            tier3 += 1
        elif res.outcome is Outcome.DEFAULT:
            default += 1
        elif res.outcome is Outcome.UNDER:
            under += 1
        else:
            over += 1
    return {
        "n": len(universe),
        "well_defined_by_evidence": determined / n,
        # Everything the schema decided: firings plus the rules that adjudicate
        # between them. Excludes the catch-all fall-through, reported alongside.
        "well_defined_overall": (determined + tier3) / n,
        "resolved_by_rule": tier3 / n,
        "by_default": default / n,
        "counts": {
            "determined": determined,
            "tier3": tier3,
            "default": default,
            "under": under,
            "over": over,
        },
    }


# --------------------------------------------------------------------------- #
# Agreement (Q73–Q76): Krippendorff's alpha, per criterion
# --------------------------------------------------------------------------- #

def _alpha(rater_a: list[int], rater_b: list[int]) -> float | None:
    """Nominal alpha for two raters. Returns None when undefined."""
    if not rater_a or len(rater_a) != len(rater_b):
        return None
    if not any(rater_a) and not any(rater_b):
        # Neither model ever fired. Alpha is formally undefined here, and the
        # old answer — 1.00, "perfect agreement" — was actively misleading:
        # it put every criterion nothing matched at the top of a table sorted
        # by agreement, which is exactly where a criterion with no evidence
        # behind it should not be. There is nothing to agree about. The fire
        # rate column says "never fires"; this one says nothing.
        return None
    if len(set(rater_a) | set(rater_b)) < 2:
        # Both fired everywhere. Degenerate too, but unlike the all-zero case
        # there is a real, unanimous observation behind it (Q76), and the
        # fire-rate flag already calls out a criterion at 90%+.
        return 1.0
    try:
        return float(
            krippendorff.alpha(
                reliability_data=[rater_a, rater_b], level_of_measurement="nominal"
            )
        )
    except Exception:  # noqa: BLE001
        return None


def model_agreement(
    criteria: list[Criterion],
    firings_by_model: dict[str, dict[int, dict[int, bool]]],
    universe: list[int],
) -> dict:
    """LLM-vs-LLM alpha per criterion, plus both roll-ups (Q75)."""
    nicks = list(firings_by_model)
    if len(nicks) != 2:
        return {"per_criterion": {}, "macro": None, "weighted": None}
    a_f, b_f = firings_by_model[nicks[0]], firings_by_model[nicks[1]]

    per: dict[int, dict] = {}
    for c in criteria:
        a = [int(bool(a_f.get(i, {}).get(c.id))) for i in universe]
        b = [int(bool(b_f.get(i, {}).get(c.id))) for i in universe]
        fires = sum(a) + sum(b)
        per[c.id] = {"alpha": _alpha(a, b), "fires": fires}
    return {"per_criterion": per, **_rollups(per)}


def human_agreement(
    criteria: list[Criterion],
    model_firings: dict[int, dict[int, bool]],
    human_verdicts: dict[int, dict[int, bool]],
) -> dict:
    """LLM-vs-human alpha per criterion over reviewed utterances only."""
    universe = sorted(human_verdicts)
    per: dict[int, dict] = {}
    for c in criteria:
        a, b = [], []
        for i in universe:
            if c.id not in human_verdicts[i]:
                continue
            a.append(int(bool(model_firings.get(i, {}).get(c.id))))
            b.append(int(bool(human_verdicts[i][c.id])))
        if not a:
            continue
        per[c.id] = {"alpha": _alpha(a, b), "fires": sum(a) + sum(b)}
    return {"per_criterion": per, **_rollups(per)}


def _rollups(per: dict[int, dict]) -> dict:
    vals = [(v["alpha"], v["fires"]) for v in per.values() if v["alpha"] is not None]
    if not vals:
        return {"macro": None, "weighted": None}
    macro = sum(a for a, _ in vals) / len(vals)
    total_w = sum(w for _, w in vals)
    weighted = (sum(a * w for a, w in vals) / total_w) if total_w else macro
    return {"macro": macro, "weighted": weighted}


# --------------------------------------------------------------------------- #
# Optimistic / pessimistic family (spec §6.1) — the README's construction (Q303)
# --------------------------------------------------------------------------- #

def predictive(pairs: list[tuple[frozenset[str], str]], label_space: list[str]) -> dict:
    """`pairs` is (candidate_set, gold_label) over reviewed utterances.

    Definitions are exactly as the README gives them. Note that the optimistic F1
    harmonically averages a precision measured over singleton predictions with a
    recall measured over any-containing predictions, so it does not correspond to
    a single confusion matrix; read the accuracies and the per-class P/R rather
    than leaning on the F1 (spec §6.1).
    """
    n = len(pairs) or 1
    opt_acc = sum(1 for c, g in pairs if g in c) / n
    pess_acc = sum(1 for c, g in pairs if c == {g}) / n

    per: dict[str, dict] = {}
    for A in label_space:
        only_a = [(c, g) for c, g in pairs if c == {A}]
        any_a = [(c, g) for c, g in pairs if A in c]
        gold_a = [(c, g) for c, g in pairs if g == A]

        num = sum(1 for c, g in only_a if g == A)
        opt_p = num / len(only_a) if only_a else None
        pess_p = num / len(any_a) if any_a else None
        opt_r = (sum(1 for c, g in gold_a if A in c) / len(gold_a)) if gold_a else None
        pess_r = (sum(1 for c, g in gold_a if c == {A}) / len(gold_a)) if gold_a else None

        per[A] = {
            "optimistic_precision": opt_p,
            "pessimistic_precision": pess_p,
            "optimistic_recall": opt_r,
            "pessimistic_recall": pess_r,
            "optimistic_f1": _f1(opt_p, opt_r),
            "pessimistic_f1": _f1(pess_p, pess_r),
            "support": len(gold_a),
        }

    return {
        "optimistic_accuracy": opt_acc,
        "pessimistic_accuracy": pess_acc,
        "per_label": per,
        "optimistic_macro_f1": _macro(per, "optimistic_f1"),
        "pessimistic_macro_f1": _macro(per, "pessimistic_f1"),
        "n": len(pairs),
    }


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def _macro(per: dict[str, dict], key: str) -> float | None:
    vals = [v[key] for v in per.values() if v[key] is not None]
    return sum(vals) / len(vals) if vals else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — n is ~10 per round, so a naive interval misleads (Q129)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))
