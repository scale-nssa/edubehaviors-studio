"""The map from criterion firings to a candidate label set.

Spec §2.3, in this branch's pared-down form: **every edge is evidentiary**.
There are no edge kinds. An edge from a criterion to a label means "when this
criterion is true, that is evidence for that label", and nothing else.

What follows from that:

  * The candidate set is just the union of the labels touched by the criteria
    that fired. There is no tier 1 / tier 2 split to make, because there is only
    one kind of evidence.
  * Nothing can subtract a label another edge put in, so a contradiction — a
    label simultaneously asserted and ruled out — is not expressible. The
    failure modes left are silence (no candidate) and competition (several).
  * Membership is presence-based; evidence is never counted or weighed.

Boundary rules (`LabelRules`) survive as a third tier, consulted only once the
evidence has failed to settle an utterance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    DETERMINED = "determined"  # exactly one label, from the evidence alone
    TIER3 = "tier3"  # a boundary rule adjudicated between candidates evidence raised
    DEFAULT = "default"  # nothing fired at all; the catch-all applied
    UNDER = "under"  # nothing fired and there is no catch-all
    OVER = "over"  # >= 2 candidates


# Outcomes where the schema actually discriminated. Boundary rules are part of
# the schema — a precedence rule is a researcher's judgement, stated once and
# applied consistently, exactly like an edge is. What is NOT discrimination is
# the empty fall-through: no criterion fired, so the catch-all applied by
# default. That branch is a constant function, and it is the only one held out.
DETERMINED_OUTCOMES = (Outcome.DETERMINED, Outcome.TIER3)


@dataclass(frozen=True)
class LabelRules:
    """Boundary rules — the third tier, consulted only when evidence has failed.

    Deliberately small. Two defaults come free once a label is designated the
    catch-all, plus whatever pairwise precedence the researcher states.

    These rules ARE schema. A stated precedence is a researcher's judgement
    applied consistently, no less part of the artifact than an edge, and it
    counts toward well-definedness.

    The one branch that does not is rule 3: nothing fired, so the catch-all
    applies. That is not a decision, it is the absence of one, and counting it
    would make a schema with NO criteria perfectly well-defined.
    """

    other_label: str | None = None
    priorities: tuple[tuple[str, str], ...] = ()   # (winner, loser)

    def resolve(self, candidates: frozenset[str]) -> frozenset[str]:
        c = set(candidates)

        # 1. Explicit precedence, to a fixpoint. Only stated pairs apply; no
        #    transitive closure, so the result is predictable from the list.
        changed = True
        while changed and len(c) > 1:
            changed = False
            for winner, loser in self.priorities:
                if winner in c and loser in c:
                    c.discard(loser)
                    changed = True

        # 2. The catch-all never wins against a real label.
        if self.other_label and len(c) > 1:
            c.discard(self.other_label)

        # 3. Silence means the catch-all.
        if not c and self.other_label:
            c = {self.other_label}
        return frozenset(c)

    def cycles(self) -> list[tuple[str, ...]]:
        """Precedence cycles. Without stated pairs only, A>B plus B>A resolves
        by list order — deterministic but arbitrary, so reject it up front."""
        adj: dict[str, set[str]] = {}
        for w, l in self.priorities:
            adj.setdefault(w, set()).add(l)
        found, seen = [], set()

        def walk(node, path):
            if node in path:
                found.append(tuple(path[path.index(node):] + [node]))
                return
            if node in seen:
                return
            seen.add(node)
            for nxt in adj.get(node, ()):
                walk(nxt, path + [node])

        for start in list(adj):
            walk(start, [])
        return found


@dataclass(frozen=True)
class Edge:
    """A criterion bears on a label. There is nothing more to say about it —
    every edge is evidentiary, so the kind is not a field."""

    label: str


@dataclass(frozen=True)
class Criterion:
    id: int
    text: str
    edges: tuple[Edge, ...]
    parent_id: int | None = None

    @property
    def labels(self) -> set[str]:
        return {e.label for e in self.edges}

    @property
    def is_child(self) -> bool:
        return self.parent_id is not None

    @property
    def is_gate(self) -> bool:
        """A parent carries no edges — it only decides whether its children are
        asked at all. If it carried edges to the labels its children
        discriminate between, those edges would be raised on every firing and
        the children could never narrow it down."""
        return not self.edges


@dataclass(frozen=True)
class MapResult:
    candidates: frozenset[str]
    outcome: Outcome

    @property
    def determined(self) -> str | None:
        return next(iter(self.candidates)) if len(self.candidates) == 1 else None


def apply_map(
    criteria: list[Criterion],
    firing: dict[int, bool],
    rules: "LabelRules | None" = None,
) -> MapResult:
    """Run the map for one utterance.

    `firing` maps criterion id -> bool. Criteria absent from `firing` are false.
    `rules` adds a third tier, consulted only when evidence leaves the utterance
    unresolved — never to override a determination the evidence already made.
    """
    candidates = {
        e.label
        for c in criteria
        if firing.get(c.id, False)
        for e in c.edges
    }

    if len(candidates) == 1:
        return MapResult(frozenset(candidates), Outcome.DETERMINED)
    outcome = Outcome.OVER if candidates else Outcome.UNDER

    # Tier 3 — boundary rules, only on what evidence could not settle.
    #
    # Two cases, and they are not the same thing. Adjudicating between labels
    # that evidence actually raised is the schema deciding something. Filling in
    # a label where nothing fired is the schema declining to, so it gets its own
    # outcome and is kept out of the well-definedness figures.
    if rules is not None:
        nothing_fired = not candidates
        after = rules.resolve(frozenset(candidates))
        if len(after) == 1:
            settled = Outcome.DEFAULT if nothing_fired else Outcome.TIER3
            return MapResult(after, settled)
        if after != frozenset(candidates):
            candidates = set(after)
            outcome = Outcome.OVER if len(candidates) >= 2 else Outcome.UNDER

    return MapResult(frozenset(candidates), outcome)


def flip_sensitivity(criteria: list[Criterion], firing: dict[int, bool]) -> int:
    """How many single-criterion flips change the candidate set.

    Cheap: n map evaluations over an in-memory boolean vector.
    """
    base = apply_map(criteria, firing).candidates
    n = 0
    for c in criteria:
        perturbed = dict(firing)
        perturbed[c.id] = not perturbed.get(c.id, False)
        if apply_map(criteria, perturbed).candidates != base:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Validation (spec §2.3.3)
# --------------------------------------------------------------------------- #

def lint_criterion_text(text: str) -> list[str]:
    """Advisory atomicity checks (response plan §4, from atoms/schematize-codebook).

    Only the mechanically-detectable subset — the paraphrase test itself needs a
    human or a model. Warnings, never hard blocks: each of these has legitimate
    exceptions, and a false block is worse than a false warning.

    Four checks were removed after living on the Revise page for a while. All
    four gave advice that is still correct; the objection was to where it was
    delivered. A warning on Revise arrives after the criterion exists, when the
    researcher is looking at evidence and not at style, and every one of them
    fired on wording that was often deliberate.

    - fused verbs (`"or" joins two different actions`) — had a documented false
      positive on participle adjectives ("added or changed wording").
    - carve-outs (`without`, `rather than`, `excluding`).
    - quoted phrases — the paraphrase test.
    - defined by exclusion — a residual that cannot be judged without first
      judging every other label.

    The last two are stated in the seeding prompt verbatim: `_ATOMICITY` (4)
    forbids literal matchers by name, and (4a) forbids exclusion-defined
    residuals. Carve-outs are in `_DECOMPOSE`. That is the right place — it
    shapes criteria before they exist instead of nagging afterwards.

    What is left is not stylistic: an empty criterion is unusable, and a very
    long one is nearly always several fused together.
    """
    out: list[str] = []
    t = (text or "").strip()
    if not t:
        return ["Empty criterion."]

    if len(t.split()) > 45:
        out.append(f"Very long ({len(t.split())} words) — often a sign of several "
                   "criteria fused into one.")
    return out


def validate_nesting(criteria: list[Criterion]) -> list[str]:
    """Structural rules for two-level criteria."""
    problems: list[str] = []
    by_id = {c.id: c for c in criteria}
    parents = {c.parent_id for c in criteria if c.parent_id is not None}

    for c in criteria:
        if c.parent_id is not None:
            parent = by_id.get(c.parent_id)
            if parent is None:
                problems.append(f"Criterion {c.id} names a parent that does not exist.")
            elif parent.parent_id is not None:
                problems.append(
                    f"Criterion {c.id} is nested under {parent.id}, which is itself "
                    "a sub-criterion. Nesting is two levels only."
                )
            if not c.edges:
                problems.append(f"Sub-criterion {c.id} has no edges, so it can never "
                                "affect a label.")
            # There is no sibling rule left to check. Siblings that both fire
            # simply raise both labels, which shows up as an over-determined
            # utterance you can look at — visible and adjudicable rather than a
            # schema error.
        elif c.id in parents and c.edges:
            problems.append(
                f"Criterion {c.id} has sub-criteria, so it must carry no edges of its "
                "own — otherwise its edges fire on every match and its children "
                "cannot discriminate between labels."
            )
    return problems


def validate_edges(edges: list[Edge], label_space: list[str]) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    With one kind of edge the cardinality rules collapse to: a criterion needs
    at least one edge, every edge points at a real label, and a label is touched
    at most once (a second edge to the same label would say nothing new).
    """
    problems: list[str] = []
    if not edges:
        return ["A criterion must have at least one edge."]

    for e in edges:
        if e.label not in label_space:
            problems.append(f"Label {e.label!r} is not in the label space.")

    seen: set[str] = set()
    for e in edges:
        if e.label in seen:
            problems.append(
                f"Label {e.label!r} carries two edges from this criterion; "
                "one edge per label."
            )
        seen.add(e.label)
    return problems
