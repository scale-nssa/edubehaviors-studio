"""Which session, and which ten utterances. Implements docs/selection.md."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .config import CRITERION_CAP, TIER_CAP, UTTERANCES_PER_ROUND, annotators
from .corpus import Session, eligible_session_ids
from .schema_map import Criterion, Outcome, apply_map, flip_sensitivity

# Priority ladder (docs/selection.md §3.2). Lower sorts first.
#
# Contradiction used to head the ladder. With one kind of edge it cannot happen,
# so competing labels lead: they are the loudest remaining schema defect.
TIER_RANK = {
    Outcome.OVER: 0,
    Outcome.UNDER: 1,
    Outcome.DEFAULT: 2,
    Outcome.TIER3: 3,
    Outcome.DETERMINED: 4,
}
# The badge says why an utterance was PICKED, not what the schema currently says
# about it. Those differ: selection tiers by the worst case across the two models
# (docs/selection.md §3.1) while Review displays the union of their firings. On a
# real round 3 of 10 items showed a badge reading "under-determined" next to a
# displayed label, which reads as a bug even though both halves are correct.
TIER_LABEL = {
    Outcome.OVER: "competing labels",
    Outcome.UNDER: "no label at all",
    Outcome.DEFAULT: "nothing fired; the catch-all applied",
    Outcome.TIER3: "settled by a boundary rule",
    Outcome.DETERMINED: "one label, but fragile",
}

# ---------------------------------------------------------------- binary ladder
#
# The general ladder ranks by how badly the schema failed, so a positive it
# resolved cleanly is DETERMINED — dead last, reached only as backfill. That is
# right for a seven-label construct and wrong for a binary one, where the
# positive class IS the question and a round that never shows you one cannot
# tell you whether the schema is right.
#
# Measured on a real round: 188 of 292 in-scope utterances were predicted
# positive and the queue of ten contained none of them.
#
# "Competing labels" needs care. With the boundary rules applied, a binary
# construct can never produce two candidates: the catch-all never beats a real
# label, so {positive, catch-all} collapses to {positive} and OVER is
# unreachable. The interesting state is therefore detected BEFORE the rules —
# both a positive criterion and a catch-all criterion fired, and the rule
# silently picked a winner. That is a schema conflict the resolved label hides.
BIN_POS_AGREE = 0     # both models: positive evidence only
BIN_POS_DIFFER = 1    # exactly one model says positive, neither conflicted
BIN_CONFLICT_BOTH = 2  # both models fired positive AND catch-all evidence
BIN_CONFLICT_ONE = 3
BIN_NEG_AGREE = 4     # both models: catch-all

BIN_TIER_LABEL = {
    BIN_POS_AGREE: "positive, both models",
    BIN_POS_DIFFER: "positive, models differed",
    BIN_CONFLICT_BOTH: "positive and catch-all evidence, both models",
    BIN_CONFLICT_ONE: "positive and catch-all evidence, one model",
    BIN_NEG_AGREE: "catch-all, both models",
}


def name_tier(tier: str, positive: str | None, catch_all: str | None) -> str:
    """Swap the generic words in a stored tier for the construct's own labels.

    The tier is stored generically ("positive, models differed") so the row
    still reads if the label space is renamed, and so one vocabulary covers
    every binary construct. It is named at display time, where "Metacognition,
    models differed" says something and "positive" does not.

    Applied to stored strings rather than at selection time on purpose: rounds
    materialised before this existed get named too, without rewriting history.
    """
    if not positive:
        return tier
    out = tier.replace("catch-all", catch_all or "catch-all")
    return out.replace("positive", positive)


def _binary_tier(per_model_raw: list[frozenset[str]], positive_label: str) -> int:
    """Tier one utterance in a binary construct.

    `per_model_raw` is each model's candidate set computed WITHOUT the boundary
    rules, so a set of size two means that model saw both kinds of evidence.

    Exhaustive and disjoint: conflicted models are counted first, and what is
    left is partitioned by how many models saw the positive.
    """
    conflicted = sum(1 for r in per_model_raw if len(r) > 1)
    if conflicted == len(per_model_raw):
        return BIN_CONFLICT_BOTH
    if conflicted:
        return BIN_CONFLICT_ONE
    positive = sum(1 for r in per_model_raw if positive_label in r)
    if positive == len(per_model_raw):
        return BIN_POS_AGREE
    if positive:
        return BIN_POS_DIFFER
    return BIN_NEG_AGREE


def pick_session(cfg, scope: str, used: set[str], dataset: str | None = None) -> str:
    """The longest unused session in this construct's dataset.

    Deterministic and shared: everyone working the same dataset walks the same
    order, which makes a room comparable to itself and lets the annotation
    cache absorb the overlap when twenty people draw the same transcript. It
    replaces a per-construct seeded draw plus a hand-pinned first session,
    neither of which survived having seven datasets of wildly different
    lengths.

    The cost consequence is deliberate but real: the most expensive session
    comes first.
    """
    pool = eligible_session_ids(cfg, scope, used, dataset)
    if not pool:
        raise RuntimeError("no eligible sessions remain for this construct")
    return pool[0]


@dataclass
class Candidate:
    index: int
    outcome: Outcome
    tier: int
    disagree: bool
    flips: int
    words: int
    candidates: frozenset[str]
    involved: frozenset[int]  # criterion ids that fired here
    # Set only for binary constructs, where the tier is not an Outcome.
    tier_label: str | None = None

    @property
    def sort_key(self) -> tuple:
        """Within a tier: disagreement, then fragility, then substance.

        Flip-sensitivity is degenerate in the under-determined tier — nothing
        fired, so flipping *any* criterion changes the outcome and every
        utterance ties at n. Observed on a real round: four of ten slots went to
        "Travis", "Were going to do some", and similar, all with flips=10.
        Word count breaks that tie toward utterances with enough content to
        actually diagnose. Elsewhere flips is meaningful and leads.
        """
        # Agreeing utterances come FIRST within a tier. If both models fire the
        # same way and the schema still can't resolve it, that is an unambiguous
        # schema defect. If they differ, the defect may be an artifact of one
        # annotator, which is assertion-sharpening — explicitly secondary.
        if self.tier_label is not None:
            # The binary ladder already encodes agreement in the tier itself,
            # so sorting on it again inside a tier would do nothing.
            return (self.tier, -self.flips, -self.words, self.index)
        if self.outcome is Outcome.UNDER:
            return (self.tier, self.disagree, -self.words, self.index)
        return (self.tier, self.disagree, -self.flips, -self.words, self.index)

    @property
    def why(self) -> str:
        """What the badge says. Selection now tiers on consensus, so this no
        longer contradicts the candidate set shown alongside it."""
        if self.tier_label is not None:
            return self.tier_label   # already says what the models did
        base = TIER_LABEL[self.outcome]
        return f"{base} (models differed)" if self.disagree else base


def score_utterances(
    session: Session,
    scope: str,
    criteria: list[Criterion],
    firings: dict[str, dict[int, dict[int, bool]]],
    rules=None,
    positive_label: str | None = None,
) -> list[Candidate]:
    """`firings` is model nick -> utterance index -> {criterion_id: bool}.

    `rules` are the construct's boundary rules. Passing them matters: without
    them every nothing-fired utterance tiers as UNDER ("no label at all")
    rather than falling through to the catch-all, which both overstates the
    number of schema defects and made Review contradict the session board.

    `positive_label` switches on the binary ladder — see `_binary_tier`.
    """
    out: list[Candidate] = []
    for u in session.in_scope(scope):
        per_model, per_model_raw = [], []
        for nick in annotators():
            f = firings.get(nick, {}).get(u.index, {})
            per_model.append((apply_map(criteria, f, rules), f))
            # Pre-rule, so a positive/catch-all conflict is still visible.
            per_model_raw.append(apply_map(criteria, f).candidates)

        # Tier on the CONSENSUS map, not the worst case across models.
        #
        # Worst-case tiering promoted an utterance whenever *either* model
        # stumbled, which measured 124 of 297 utterances (42%) on a real session
        # — 69% of the transcript looked "under-determined" against 31% on
        # consensus. That buries schema defects under annotator noise. Schema
        # sharpening comes first; disagreement is a tie-break and a flag, never
        # a promotion.
        cons_f = {
            c.id: any(f.get(c.id, False) for _, f in per_model) for c in criteria
        }
        cons_res = apply_map(criteria, cons_f, rules)
        outcome = cons_res.outcome
        disagree = len({r.candidates for r, _ in per_model}) > 1
        involved = {cid for cid, v in cons_f.items() if v}

        binary = positive_label is not None
        tier = (
            _binary_tier(per_model_raw, positive_label)
            if binary else TIER_RANK[outcome]
        )
        out.append(
            Candidate(
                index=u.index,
                outcome=outcome,
                tier=tier,
                tier_label=BIN_TIER_LABEL[tier] if binary else None,
                disagree=disagree,
                flips=flip_sensitivity(criteria, cons_f),
                words=len(u.text.split()),
                candidates=cons_res.candidates,
                involved=frozenset(involved),
            )
        )
    return out


def choose(
    scored: list[Candidate],
    n: int = UTTERANCES_PER_ROUND,
    tier_cap: int = TIER_CAP,
    criterion_cap: int = CRITERION_CAP,
    prioritise: list[int] | None = None,
) -> list[Candidate]:
    """Greedy fill down the ladder, subject to the two caps.

    `prioritise` jumps the queue — used for utterances that regressed since the
    last schema change. They sit outside the ladder because the ladder reads the
    schema's current state and cannot see that something used to be right.
    """
    ordered = sorted(scored, key=lambda c: c.sort_key)
    if prioritise:
        first = set(prioritise)
        ordered = (
            [c for c in ordered if c.index in first]
            + [c for c in ordered if c.index not in first]
        )

    picked: list[Candidate] = []
    per_tier: dict[int, int] = {}
    per_criterion: dict[int, int] = {}

    def take(c: Candidate) -> None:
        picked.append(c)
        per_tier[c.tier] = per_tier.get(c.tier, 0) + 1
        for cid in c.involved:
            per_criterion[cid] = per_criterion.get(cid, 0) + 1

    for c in ordered:
        if len(picked) >= n:
            break
        if per_tier.get(c.tier, 0) >= tier_cap:
            continue
        if c.involved and all(
            per_criterion.get(cid, 0) >= criterion_cap for cid in c.involved
        ):
            continue
        take(c)

    # Backfill in two stages. The tier cap is a diversity preference and is
    # relaxed first; the criterion cap exists to stop one bad criterion
    # monopolising the round, which matters more than filling exactly n, so it is
    # only relaxed once nothing else is left.
    chosen = {c.index for c in picked}
    for relax_criterion in (False, True):
        for c in ordered:
            if len(picked) >= n:
                break
            if c.index in chosen:
                continue
            if not relax_criterion and c.involved and all(
                per_criterion.get(cid, 0) >= criterion_cap for cid in c.involved
            ):
                continue
            take(c)
            chosen.add(c.index)

    # Selection order and reading order are different jobs. The ranking above
    # decides WHICH ten — fragility first, so the round is worth the money.
    # Reading them in that order means jumping around the transcript, which
    # costs the reviewer the context they just built up.
    #
    # So: present the batch grouped by tier, as the ladder intends, but walk
    # the transcript in order inside each group. Prioritised (regressed)
    # utterances keep their place at the front — they jumped the ladder
    # deliberately, and the point is to look at them first.
    first = set(prioritise or ())
    return sorted(
        picked[:n],
        key=lambda c: (0 if c.index in first else 1, c.tier, c.index),
    )


def select_round(
    conn: sqlite3.Connection,
    session: Session,
    scope: str,
    criteria: list[Criterion],
    firings: dict[str, dict[int, dict[int, bool]]],
    prioritise: list[int] | None = None,
    rules=None,
    positive_label: str | None = None,
) -> list[Candidate]:
    return choose(
        score_utterances(
            session, scope, criteria, firings, rules, positive_label
        ),
        prioritise=prioritise,
    )
