"""Search the edge space for rewirings that fit the evidence.

The whole thing rests on one property: **the firing matrix is fixed**. Whether
criterion 9 fires on utterance 234 depends on the criterion's *text*, not on its
edges. So every possible rewiring can be scored against already-cached
annotations with no model calls at all.

That gives a clean division of labour, and it is the inverse of what the app did
before: the **search** proposes edge changes, because those are pure
combinatorics; the **model** proposes text changes, because those need meaning.
Two rounds of prompt engineering failed to get P4 to rewire; this makes the
question not arise.

Measured on a real round (14 criteria, 7 labels, 10 gold, 297 utterances): 392
candidate moves, 139 legal after cardinality rules, and the top-ranked move was
the exact defect a human reviewer had diagnosed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema_map import (
    DETERMINED_OUTCOMES,
    Criterion,
    Edge,
    apply_map,
    validate_edges,
    validate_nesting,
)


@dataclass
class Move:
    criterion_id: int
    criterion_text: str
    label: str
    add: bool                  # True: wire this criterion to the label; False: cut it
    edges: list[dict]          # the criterion's complete new edge set
    d_gold: int                # change in exactly-right gold routings
    d_welldef: int             # change in utterances resolving to exactly one label

    @property
    def removes(self) -> bool:
        """Cutting the last edge — the criterion would bear on nothing."""
        return not self.edges

    @property
    def helps_both(self) -> bool:
        """Improves on gold AND does not hurt the whole session.

        Gold is ~10 utterances; well-definedness is ~300 and needs no gold at
        all. Requiring both is the cheap guard against fitting noise in a tiny
        labelled sample — they are close to independent signals.
        """
        return self.d_gold > 0 and self.d_welldef >= 0

    @property
    def helps_consistency(self) -> bool:
        """Tidies the schema's own resolution without costing anything on gold.

        Under/over-determination is a property of the schema alone — no
        labelling required — so this is measured on ~300 utterances rather than
        ~10 and carries no overfitting risk. It matters most exactly when gold is
        thin or uninformative, which is when `helps_both` finds nothing.

        **These are NOT evidence of correctness.** Well-definedness rewards
        confident wrongness — a schema that assigns one label to everything is
        perfectly well-defined and useless. Observed live: with randomised gold
        the search happily proposed cutting the edge from a criterion that asks
        "explain why" to PressingForReasoning, because it tidied the counts. Only
        the gold objective can tell right from tidy, so consistency-only moves
        rank below gold-backed ones and carry an explicit caveat.
        """
        return self.d_gold >= 0 and self.d_welldef > 0 and not self.helps_both

    def describe(self) -> str:
        if self.removes:
            return f"cut its only edge ({self.label}) — retires the criterion"
        return f"{'wire to' if self.add else 'cut edge to'} {self.label}"


def _score(criteria, consensus, gold, universe, rules=None) -> tuple[int, int]:
    """Gold accuracy counts rules — a rule that routes correctly is a real gain.

    Well-definedness counts a precedence rule too — it only ever fires on
    candidates the edges raised, so it moves with them. What it excludes is the
    catch-all fall-through, which is constant with respect to the edges: count
    that and deleting an edge looks like an improvement (fewer firings, more
    fall-through, higher score) and the search learns to delete the schema.
    """
    exact = 0
    for idx, label in gold.items():
        if set(apply_map(criteria, consensus.get(idx, {}), rules).candidates) == {label}:
            exact += 1
    welldef = 0
    for idx in universe:
        if apply_map(criteria, consensus.get(idx, {}), rules).outcome in DETERMINED_OUTCOMES:
            welldef += 1
    return exact, welldef


def search(
    criteria: list[Criterion],
    labels: list[str],
    consensus: dict[int, dict[int, bool]],
    gold: dict[int, str],
    universe: list[int],
    rules=None,
) -> list[Move]:
    """Every legal single-edge change, scored. Cheap: linear in criteria×labels.

    With untyped edges each (criterion, label) slot has exactly one alternative
    to its current state — wired or not — so the space is a quarter of what it
    was and no two candidates for a slot can tie.

    One case needs care. There is no demotion any more, so on a criterion with a
    single edge the only edge-space repair available is to cut that edge, which
    leaves the criterion bearing on nothing at all. That is allowed for a
    top-level criterion and reported as such (`Move.removes`); the caller stages
    it as a removal, because a criterion with no edges cannot affect a label.
    Gates and sub-criteria are excluded: a gate is already edgeless by design and
    an edgeless sub-criterion is invalid.
    """
    base = _score(criteria, consensus, gold, universe, rules)
    raw: list[Move] = []

    parents = {c.parent_id for c in criteria if c.parent_id is not None}
    nested = bool(parents)
    for i, c in enumerate(criteria):
        # A parent is a pure gate: giving it edges would fire on every match and
        # stop its children discriminating, so it is not a candidate for rewiring.
        if c.id in parents:
            continue
        current = c.labels
        for label in labels:
            add = label not in current
            proposed = (current | {label}) if add else (current - {label})
            edges = [Edge(l) for l in sorted(proposed)]
            if not proposed:
                # Cutting the last edge. Legal only at the top level, where it
                # amounts to retiring the criterion.
                if c.parent_id is not None:
                    continue
            elif validate_edges(edges, labels):
                continue  # not a candidate

            trial = list(criteria)
            trial[i] = Criterion(c.id, c.text, tuple(edges), c.parent_id)
            # Edge legality is per-criterion; nesting rules are about the whole
            # family, so a legal edge set can still break them.
            if (c.parent_id is not None or nested) and validate_nesting(trial):
                continue
            e, w = _score(trial, consensus, gold, universe, rules)
            raw.append(
                Move(
                    criterion_id=c.id,
                    criterion_text=c.text,
                    label=label,
                    add=add,
                    edges=[{"label": l} for l in sorted(proposed)],
                    d_gold=e - base[0],
                    d_welldef=w - base[1],
                )
            )

    return sorted(raw, key=lambda m: (-m.d_gold, -m.d_welldef, m.criterion_id))


def recommend(
    criteria, labels, consensus, gold, universe, *, top_n: int = 5, rules=None
) -> tuple[list[Move], dict]:
    """The moves worth showing, plus the baseline they are measured against.

    Two classes, gold-backed first: moves that improve the utterances you judged,
    then moves that only clean up the schema's own resolution. The second class
    needs no gold, so it still has something to say on a construct nobody has
    reviewed much yet.
    """
    all_moves = search(criteria, labels, consensus, gold, universe, rules)
    base = _score(criteria, consensus, gold, universe, rules)

    seen: set[tuple[int, str]] = set()
    picked: list[Move] = []
    for pool in (
        [m for m in all_moves if m.helps_both],
        sorted(
            (m for m in all_moves if m.helps_consistency),
            key=lambda m: (-m.d_welldef, m.criterion_id),
        ),
    ):
        for m in pool:
            slot = (m.criterion_id, m.label)
            if slot in seen:
                continue
            seen.add(slot)
            picked.append(m)
            if len(picked) >= top_n:
                break
        if len(picked) >= top_n:
            break

    return picked, {
        "gold": base[0],
        "n_gold": len(gold),
        "welldef": base[1],
        "n_utterances": len(universe),
        "n_legal_moves": len(all_moves),
        "n_improving": sum(1 for m in all_moves if m.d_gold > 0),
        "n_consistency_only": sum(1 for m in all_moves if m.helps_consistency),
    }


# --------------------------------------------------------------------------- #
# Priority-rule search
#
# Same trick as the edge search: the firing matrix is fixed, so every candidate
# rule can be scored against cached annotations with no model calls. The space
# is even smaller — L×(L−1) ordered pairs, 42 for seven labels.
# --------------------------------------------------------------------------- #

@dataclass
class RuleMove:
    winner: str
    loser: str
    d_gold: int
    d_resolved: int       # utterances the rule settles that evidence could not
    decides: int          # how many utterances the rule is load-bearing on

    @property
    def helps(self) -> bool:
        return self.d_gold > 0 or (self.d_gold == 0 and self.d_resolved > 0)

    def describe(self) -> str:
        return f"{self.winner} wins over {self.loser}"


def search_rules(
    criteria, labels, consensus, gold, universe, rules=None, *, top_n: int = 5
) -> list[RuleMove]:
    """Ordered label pairs that would settle utterances evidence leaves open."""
    from .schema_map import LabelRules, Outcome, apply_map

    base = rules or LabelRules()
    base_gold, _ = _score(criteria, consensus, gold, universe, base)

    out: list[RuleMove] = []
    for winner in labels:
        for loser in labels:
            if winner == loser or (winner, loser) in base.priorities:
                continue
            trial = LabelRules(
                other_label=base.other_label,
                priorities=base.priorities + ((winner, loser),),
            )
            if trial.cycles():
                continue  # a precedence loop resolves by list order, arbitrarily
            g, _ = _score(criteria, consensus, gold, universe, trial)

            decides = resolved = 0
            for idx in universe:
                before = apply_map(criteria, consensus.get(idx, {}), base)
                after = apply_map(criteria, consensus.get(idx, {}), trial)
                if before.candidates != after.candidates:
                    decides += 1
                    if (
                        len(after.candidates) == 1
                        and before.outcome is not Outcome.DETERMINED
                        and len(before.candidates) != 1
                    ):
                        resolved += 1
            if decides:
                out.append(RuleMove(winner, loser, g - base_gold, resolved, decides))

    out.sort(key=lambda m: (-m.d_gold, -m.d_resolved, m.winner, m.loser))
    return [m for m in out if m.helps][:top_n]


def rule_load(criteria, consensus, universe, rules) -> list[dict]:
    """How many utterances each rule is actually deciding.

    A rule carrying a large share of a session is not resolving a boundary, it
    is masking two criteria that overlap badly.
    """
    from .schema_map import LabelRules, apply_map

    if rules is None:
        return []
    out = []
    for w, l in rules.priorities:
        without = LabelRules(
            other_label=rules.other_label,
            priorities=tuple(p for p in rules.priorities if p != (w, l)),
        )
        n = sum(
            1
            for idx in universe
            if apply_map(criteria, consensus.get(idx, {}), rules).candidates
            != apply_map(criteria, consensus.get(idx, {}), without).candidates
        )
        out.append({"winner": w, "loser": l, "decides": n,
                    "share": n / len(universe) if universe else 0.0})
    return sorted(out, key=lambda d: -d["decides"])
