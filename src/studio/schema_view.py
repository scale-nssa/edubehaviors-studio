"""One description of the schema, rendered as a table and as a graph.

Both views read `build()`, so they can never disagree about what the schema is.
Response plan §1.

The row unit is a **(label, criterion) pair**, not a criterion: a criterion that
bears on both A and B appears under each, so reading down a label tells you
everything that can raise it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema_map import Criterion


@dataclass
class Row:
    criterion_id: int
    text: str
    parent_text: str | None = None
    fires: float | None = None        # fraction of in-scope utterances
    alpha: float | None = None        # inter-model agreement
    human_flips: int = 0            # times a reviewer overrode the models
    human_total: int = 0            # times it was in front of a reviewer

    @property
    def human(self) -> str | None:
        """How often a reviewer corrected this criterion, as "flips / seen".

        This column used to show *agreement*, and it read 100% on all 24
        criteria of a live construct. That was not a bug in the arithmetic:
        Review pre-checks every criterion at the model's own verdict, so a
        reviewer who touches nothing agrees with everything. The metric could
        not tell "confirmed" from "not looked at".

        Flips can. Zero flips out of twenty is the same underlying data but
        makes no claim about agreement, and a criterion reviewers keep
        overriding is the one worth rewording — which is the decision this
        table exists to inform.
        """
        if not self.human_total:
            return None
        return f"{self.human_flips} / {self.human_total}"

    @property
    def flag(self) -> str | None:
        """Fire-rate problems the numbers make obvious but the eye skips over.

        Only the two extremes are flagged. A middling-to-high fire rate used to
        raise "possibly mis-wired" at 35%, but a criterion legitimately firing
        on a third of a transcript is common and the warning cried wolf.

        Zero is reported as *unseen behavior* rather than dead weight: a
        criterion nothing matched may be badly worded, or the behaviour may
        simply not occur in the sessions annotated so far, and the fire rate
        alone cannot tell those apart.
        """
        if self.fires is None:
            return None
        if self.fires == 0:
            return "never fires — unseen behavior"
        if self.fires >= 0.9:
            return f"fires on {self.fires:.0%} of utterances — carries almost no information"
        return None


@dataclass
class LabelGroup:
    label: str
    rows: list[Row] = field(default_factory=list)

    @property
    def warning(self) -> str | None:
        """Structural problems visible from the schema alone."""
        if not self.rows:
            return "unreachable — no criterion points at this label"
        return None


@dataclass
class SchemaView:
    labels: list[LabelGroup]
    criteria: list[Criterion]
    orphans: list[Criterion] = field(default_factory=list)  # no edges at all

    @property
    def n_edges(self) -> int:
        return sum(len(g.rows) for g in self.labels)

    def summary_lines(self) -> list[str]:
        """Structured facts for P4. It currently sees criteria only as prose and
        never learns 'label X is unreachable' — the fact that should trigger a
        rewire (response plan §3)."""
        out = []
        for g in self.labels:
            n = len(g.rows)
            out.append(f"- {g.label}: {n} criteri{'on' if n == 1 else 'a'} point here"
                       if n else f"- {g.label}: NO EDGES")
            if g.warning:
                out[-1] += f"  [{g.warning}]"
        for c in self.orphans:
            out.append(f"- criterion #{c.id} has no edges and can never fire on any label")
        dead = [r for g in self.labels for r in g.rows
                if r.fires is not None and r.fires <= 0.0]
        for r in dead:
            out.append(f"- criterion #{r.criterion_id} never fires in annotated data")
        return out


def build(
    criteria: list[Criterion],
    label_space: list[str],
    *,
    stats: dict[int, dict] | None = None,
) -> SchemaView:
    """`stats` maps criterion id -> {fires, alpha, human_flips, human_total}."""
    stats = stats or {}
    groups = {lbl: LabelGroup(label=lbl) for lbl in label_space}

    by_id = {c.id: c for c in criteria}
    gates = {c.parent_id for c in criteria if c.parent_id is not None}

    orphans = []
    for c in criteria:
        if not c.edges:
            # A gate legitimately has no edges — it is a precondition, not a
            # dead criterion, so it does not belong in the orphan list.
            if c.id not in gates:
                orphans.append(c)
            continue
        for e in c.edges:
            if e.label not in groups:
                # An edge into a label that no longer exists — surface it rather
                # than dropping it silently, since it means the label space was
                # edited out from under the criterion.
                groups[e.label] = LabelGroup(label=e.label)
            st = stats.get(c.id, {})
            parent = by_id.get(c.parent_id) if c.parent_id else None
            groups[e.label].rows.append(
                Row(
                    criterion_id=c.id,
                    text=c.text,
                    parent_text=parent.text if parent else None,
                    fires=st.get("fires"),
                    alpha=st.get("alpha"),
                    human_flips=st.get("human_flips", 0),
                    human_total=st.get("human_total", 0),
                )
            )

    for g in groups.values():
        g.rows.sort(key=lambda r: r.criterion_id)

    ordered = [groups[l] for l in label_space if l in groups]
    ordered += [g for l, g in groups.items() if l not in label_space]
    return SchemaView(labels=ordered, criteria=criteria, orphans=orphans)
