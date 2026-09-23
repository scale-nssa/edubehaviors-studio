"""Prompt construction. Spec §7.

P2 (annotate) is lifted from `../atoms/annotate-assertions`, with one change:
the transcript goes in the *system* prompt ahead of the criterion, so the
session is the shared cacheable prefix across every criterion in a round
(spec §3.3). Upstream had them the other way round.

Output for P2 is sparse and positive-only — the model lists the indices where
the criterion fires and nothing else. There is no per-utterance enumeration, so
there is no alignment problem to solve.
"""

from __future__ import annotations

import json

from .providers import Prompt

from .corpus import Session
from .schema_map import Criterion

_SCOPE_NOUN = {"tutor": "TUTOR", "student": "STUDENT", "all": "ANY-SPEAKER"}


# --------------------------------------------------------------------------- #
# P2 — annotate one criterion over one session
# --------------------------------------------------------------------------- #

def annotate(session: Session, criterion_text: str, scope: str) -> Prompt:
    noun = _SCOPE_NOUN.get(scope, "ANY-SPEAKER")
    eligible = (
        "Every utterance is a candidate."
        if scope == "all"
        else f"Only {noun} utterances are eligible answers; "
        f"other speakers are context only, never candidates."
    )
    system = (
        "You are an expert annotation assistant for educational research. You read "
        "the COMPLETE transcript of one tutoring or classroom conversation and "
        "identify every utterance where a given criterion is true.\n\n"
        "<transcript>\n"
        f"{session.render()}\n"
        "</transcript>\n\n"
        "Each line begins with its utterance index, then the speaker in brackets. "
        "Use the whole transcript to understand what is happening across the session."
    )
    user = (
        f'Criterion: "{criterion_text}"\n\n'
        "Instructions:\n"
        f"- {eligible}\n"
        "- Read the criterion as literally as possible. Do not infuse meaning "
        "beyond what the utterance's text supports.\n"
        "- An index belongs in your answer if and only if the criterion is true "
        "for that specific utterance.\n"
        "- It is common and expected for a criterion to be true for zero, one, or "
        "many utterances in the same session.\n\n"
        'Respond with ONLY a single JSON object of the form {"true_indices": [12, 47, 203]}, '
        "in ascending order. If the criterion is true for no utterance, respond with "
        '{"true_indices": []}. No markdown fences, no explanation, no text before or after.'
    )
    return Prompt(system_prompt=system, model_prompt=user)


def annotate_child(
    session: Session,
    child_text: str,
    parent_text: str,
    candidates: list[int],
    scope: str,
) -> Prompt:
    """P2 for a sub-criterion: judge only where the parent already fired.

    The whole point of nesting. Instead of scanning 300 utterances for a
    distinction that only means anything on 14 of them, the model is handed the
    14 and asked a narrower question.
    """
    system = (
        "You are an expert annotation assistant for educational research. You read "
        "the COMPLETE transcript of one tutoring or classroom conversation and "
        "make a fine-grained distinction among a short list of utterances that "
        "have already been identified as matching a broader criterion.\n\n"
        "<transcript>\n"
        f"{session.render()}\n"
        "</transcript>\n\n"
        "Each line begins with its utterance index, then the speaker in brackets."
    )
    listing = "\n".join(
        f"  {i}: {next((u.text for u in session.utterances if u.index == i), '')}"
        for i in candidates
    )
    user = (
        f'A broader criterion already holds for these utterances:\n'
        f'  "{parent_text}"\n\n'
        f"{listing}\n\n"
        f'Among ONLY those utterances, identify every one for which this more '
        f'specific criterion also holds:\n  "{child_text}"\n\n'
        "Instructions:\n"
        "- Consider ONLY the indices listed above. Never return any other index.\n"
        "- Read the criterion as literally as possible.\n"
        "- It is common for the answer to be none, some, or all of them.\n\n"
        'Respond with ONLY a single JSON object of the form {"true_indices": [12, 47]}, '
        "in ascending order, and nothing else."
    )
    return Prompt(system_prompt=system, model_prompt=user)


# --------------------------------------------------------------------------- #
# P1 — seed criteria from the description
# --------------------------------------------------------------------------- #


_ATOMICITY = """
Every criterion must satisfy all of the following. These are adapted from the
codebook-decomposition spec in ../atoms/schematize-codebook and are not
negotiable.

(1) BOOLEAN. It can be evaluated TRUE or FALSE against a single utterance.

(2) ATOMIC — exactly ONE observable property. Never fuse behaviours with
    "and"/"or". The test: if a clause could be TRUE for one reason and FALSE for
    another, or names two things a coder would judge separately, split it into
    that many criteria. A short parenthetical "(e.g., ...)" illustrating one
    behaviour is fine; a disjunction that widens what counts as TRUE is not.

(3) UNAMBIGUOUS. Worded precisely enough that two coders resolve it identically.

(4) CODES FUNCTION, NOT WORDING. Apply the PARAPHRASE TEST: if the same
    communicative act were expressed in entirely different words, the criterion
    must still evaluate TRUE. Never match exact phrases, specific words,
    spellings, punctuation, tag words, or names — e.g. NOT 'contains the word
    "why"', NOT 'ends with "right?"', NOT 'is a single name'. Literal matchers
    memorise these particular transcripts and will not transfer. Prefer
    observable communicative acts judged from meaning alone.

(4a) THE RESIDUAL LABEL IS NOT AN ESCAPE HATCH. Any catch-all label
    (Other/None) needs POSITIVE, independently-checkable criteria exactly like
    every other label — e.g. "The utterance is off-topic and unrelated to the
    lesson content." NEVER define one by exclusion ("does not fit any other
    category", "contains none of the above"): such a criterion cannot be
    evaluated without first evaluating every other label, so it is invalid.

(5) NON-REDUNDANT. Within a label, no two criteria that fire on the same
    utterances or restate the same behaviour.

Referring to context is fine: "The tutor is replying to a question from the
student." or "The student made a mistake within the last few utterances."
""".strip()


_EDGE_RULES = """
Each criterion carries one or more edges into the label space. There is exactly
ONE kind of edge and it is evidentiary: an edge from a criterion to a label means
"when this criterion is TRUE, that is evidence for that label". Nothing more.

There is no way to say a criterion SETTLES a label, and no way to say it RULES
one OUT. Do not try to express either; there is no vocabulary for it.

How an utterance gets its label: every label with at least one firing edge is a
candidate. Exactly one candidate means the schema decided. None means the schema
is silent on that utterance. Two or more means the criteria do not yet separate
those labels.

That has a direct consequence for how you write criteria. Since a label cannot
be ruled out, the only way to stop two labels competing is for their criteria to
be narrow enough that they do not both fire on the same utterance. Precision in
the wording is the whole mechanism.

Hard rules you must obey:
1. At most one edge per label on a given criterion.
2. Every criterion must have at least one edge.
3. An edge is written {"label": "SomeLabel"}. It has no other fields.
""".strip()


def seed_criteria(description: str, label_space: list[str], scope: str, n: int = 0) -> Prompt:
    system = (
        "You are an expert in educational discourse analysis and annotation schema "
        "design. You turn a researcher's free-form description of a behavioural "
        "construct into an itemized, machine-executable schema of yes/no criteria "
        "over single utterances.\n\n" + _ATOMICITY + "\n\n" + _BOUNDARIES_ARE_SCHEMA
        + "\n\n" + _DECOMPOSE + "\n\n" + _NESTING + "\n\n" + _EDGE_RULES
    )
    user = (
        f"Construct description (this also defines the labels):\n\n{description}\n\n"
        f"Label space: {json.dumps(label_space)}\n"
        f"Scope: only {_SCOPE_NOUN.get(scope, 'ANY-SPEAKER')} utterances are annotated.\n\n"
        # min, not max. The floor is now "four per label, capped at 20" rather
        # than "20, or four per label if that is more". A binary construct
        # asked for 20 criteria over one real label and a catch-all, which is
        # where over-splitting and mutually-firing criteria come from; a
        # seven-label one asked for 28, which is a bigger seed than anyone
        # wants to walk through on Approve and a bigger annotation bill every
        # round. "More is better than fewer" still follows, so this is a floor
        # the model is free to exceed.
        f"Propose at least {n or min(20, 4 * len(label_space))} criteria — more is "
        "better than fewer, because splitting is cheap and fusing is not. Each must "
        "be a yes/no question decidable from "
        "a single utterance read in the context of its session. Prefer concrete, "
        "observable, literal wording over abstract judgement.\n\n"
        "Because every edge is mere evidence, two criteria that both fire on the "
        "same utterance leave it with two competing labels and no way to choose. "
        "Write each criterion narrowly enough that it fires only where its label "
        "is genuinely the right reading.\n\n"
        'Respond with ONLY JSON of the form:\n'
        '{"criteria": [\n'
        '  {"text": "...", "edges": [{"label": "..."}]},\n'
        '  {"text": "a gate", "edges": [], "children": [\n'
        '     {"text": "a fine distinction", "edges": [{"label": "..."}]}\n'
        '  ]}\n'
        ']}'
    )
    return Prompt(system_prompt=system, model_prompt=user)


# --------------------------------------------------------------------------- #
# P3 / P4 — clarifying questions and schema deltas
# --------------------------------------------------------------------------- #

_CONTEXT_RULES = """
The context is a transcript of work already done, not a puzzle. Two things in
it are commonly misread:

- Criterion ids in the review history may refer to criteria that have since
  been REMOVED. Any such criterion is listed explicitly under "Criteria
  REMOVED from the schema", with its wording and edges. Never ask what a
  criterion id means, and never treat a removed criterion as part of the
  schema — it is there so the history reads, nothing more.
- Human criterion verdicts record what the reviewer said at the time, against
  the schema as it then stood. A verdict disagreeing with a criterion that no
  longer exists is history, not a defect to fix.
""".strip()


def clarifying_questions(context: str, max_q: int) -> Prompt:
    system = (
        "You are helping a researcher refine an annotation schema. Before proposing "
        "changes, you may ask questions whose answers would most change what you "
        "propose. Ask only what you genuinely cannot infer.\n\n" + _CONTEXT_RULES
    )
    user = (
        f"{context}\n\n"
        f"Ask at most {max_q} questions. Fewer is better; zero is acceptable if the "
        "evidence is unambiguous. A question is only worth asking if the answer "
        "lives in the researcher's head — their intent, where they want a "
        "boundary drawn, what a label is for. Anything answerable from the "
        "context above is not a question, it is reading.\n\n"
        'Respond with ONLY JSON: {"questions": ["...", "..."]}'
    )
    return Prompt(system_prompt=system, model_prompt=user)


_BOUNDARIES_ARE_SCHEMA = """
WHERE LABEL BOUNDARIES LIVE. If a criterion is firing on utterances that belong
to a different label, the fix is a SCHEMA change, never a longer criterion.

Do NOT write exclusions into a criterion's text. This is wrong:

  "The utterance requests a short factual answer ... — excluding any utterance
   that asks a student to explain reasoning, justify a decision, or relate to
   another student's contribution (those are PressingForReasoning or
   GettingStudentsToRelate instead), and excluding trailing fragments where the
   teacher pauses mid-thought."

That criterion has absorbed the entire label space into one sentence. It is
unreadable, untestable, impossible to revise, and it duplicates work the edges
already do. Instead, choose one of:

  - narrow the original criterion to the ONE behaviour it is really about, and
    let the other criteria handle the rest;
  - add a new, separate criterion for the behaviour being excluded, wired to the
    label that behaviour really belongs to;
  - cut the edge that is firing where it should not.

A criterion describes one observable behaviour. The relationships BETWEEN
behaviours and labels are carried by edges. Keep them separate.
""".strip()

_BOUNDARIES_REWORD_TAIL = """
Rewordings should usually get SHORTER or stay about the same length. A proposed
reword that is substantially longer than the original is almost always this
mistake.
""".strip()

# Generation-time version. The same failure shows up earlier than revision: on a
# real run, 6 of 14 seeded criteria carried a negated carve-out clause
# ("without addressing math content", "with no added or changed wording"), each
# of which is a second behaviour hidden inside the first one's text.
_DECOMPOSE = """
DO NOT WRITE CARVE-OUTS INTO A CRITERION. Words like "without", "excluding",
"with no", "not requiring", "rather than", "other than" almost always mean you
have fused two behaviours into one sentence. Split them.

Wrong, one criterion doing two jobs:
  "The utterance repeats a prior student utterance word-for-word, with no added
   or changed wording."

Right, a gate with two sub-criteria (see SUB-CRITERIA below):
  gate: "The utterance repeats or references a prior student utterance."
    sub: "The repeated wording matches the student's wording."   -> Restating
    sub: "The repeated wording is altered, corrected, or translated." -> Revoicing

Why this matters mechanically: a clause buried in a criterion's text is
invisible to everything downstream. It cannot be rewired, its firing rate cannot
be measured separately, and no metric can attribute a mistake to it. A criterion
with its own edge can be all three.

Note what is NOT available: you cannot write one broad criterion pointing at both
labels and then subtract one of them, because no edge subtracts. Two edges from
one criterion to two labels means that criterion raises BOTH every time it fires,
and the utterance ends up with competing labels. If two labels need separating,
separate the criteria.

Prefer many small criteria over a few large ones. Each should be a single
behaviour a coder can judge in isolation, in roughly 8-20 words.
""".strip()

_NESTING = """
SUB-CRITERIA. When several labels are distinguished by a fine detail of the SAME
underlying behaviour, express that as one gate criterion with sub-criteria under
it, rather than as several overlapping top-level criteria.

  gate:  "The utterance repeats content from a prior student utterance."
    sub: "The repeated wording matches the student's wording exactly."
           -> Restating
    sub: "The repeated wording is changed, corrected, or extended."
           -> Revoicing

Rules:
- A gate carries NO edges. Its only job is to decide whether its sub-criteria
  get asked at all. If a gate had edges to the labels its children distinguish
  between, those edges would fire on every match and the children could never
  narrow it down.
- Every sub-criterion needs at least one edge.
- Exactly two levels. A sub-criterion may not have sub-criteria of its own.
- Sub-criteria are only evaluated where the gate is true, so they can be worded
  narrowly and assume the gate's context. "The wording matches exactly" needs no
  restatement of what is being repeated.

This is the main tool you have for separating labels that are easy to confuse.
Since no edge can rule a label out, the distinction has to live in the wording of
two narrow sibling criteria that will not both fire. Word siblings so that they
are genuinely mutually exclusive readings of the same behaviour. If both fire
anyway, the result is an unresolved utterance the researcher can look at.
""".strip()


def propose_deltas(context: str) -> Prompt:
    system = (
        "You are helping a researcher refine an annotation schema of yes/no criteria "
        "over utterances. You propose concrete, minimal changes justified by review "
        "evidence.\n\n" + _ATOMICITY + "\n\n" + _BOUNDARIES_ARE_SCHEMA
        + "\n\n" + _BOUNDARIES_REWORD_TAIL + "\n\n" + _EDGE_RULES
        + "\n\n" + _CONTEXT_RULES
    )
    user = (
        f"{context}\n\n"
        "Propose schema changes, most important first. Each delta is one of:\n"
        '  {"kind":"add_criterion","text":"...","edges":[...],"rationale":"..."}\n'
        '  {"kind":"remove_criterion","criterion_id":N,"rationale":"..."}\n'
        '  {"kind":"reword","criterion_id":N,"text":"...","rationale":"..."}\n'
        '  {"kind":"rewire","criterion_id":N,"edges":[{"label":"..."}],"rationale":"..."}\n'
        '  {"kind":"edit_description","description":"...","rationale":"..."}\n\n'
        "Pay particular attention to utterances left with two or more competing "
        "labels. Since no edge can rule a label out, the fix is always either to "
        "narrow the wording of a criterion that is firing too widely, or to cut one "
        "of its edges — never to add an edge that cancels another.\n\n"
        'Respond with ONLY JSON: {"deltas": [ ... ]}'
    )
    return Prompt(system_prompt=system, model_prompt=user)


def reword(criterion_text: str, agreements: list[str], disagreements: list[str]) -> Prompt:
    system = (
        "You reword a single annotation criterion so that two independent annotators "
        "resolve it the same way, WITHOUT changing what the researcher meant by it. "
        "Never make a criterion trivially decidable by draining its meaning.\n\n"
        "Rule (4) is the one that bites here. Optimising for annotator agreement "
        "pushes toward literal string matching, because exact words are the easiest "
        "thing for two annotators to agree on — and a phrase matcher is worthless. "
        "A rewording that raises agreement by naming specific words is a FAILURE, "
        "not a success.\n\n" + _ATOMICITY + "\n\n" + _BOUNDARIES_ARE_SCHEMA
        + "\n\n" + _BOUNDARIES_REWORD_TAIL
    )
    user = (
        f'Current criterion: "{criterion_text}"\n\n'
        f"Utterances where annotators agreed it was true:\n"
        + ("\n".join(f"  - {t}" for t in agreements[:8]) or "  (none)")
        + "\n\nUtterances where annotators disagreed:\n"
        + ("\n".join(f"  - {t}" for t in disagreements[:8]) or "  (none)")
        + "\n\nPropose one rewording that removes the ambiguity causing the "
        "disagreements while preserving the intended notion.\n\n"
        'Respond with ONLY JSON: {"text": "...", "rationale": "..."}'
    )
    return Prompt(system_prompt=system, model_prompt=user)


def scratchpad(context: str, current: str) -> Prompt:
    """Maintain a running note on outstanding schema gaps (note 21)."""
    system = (
        "You keep a short working note about what is still wrong with an "
        "annotation schema: which labels are under-served, which criteria are "
        "unreliable, which distinctions the schema cannot yet make. It is your "
        "own memory across rounds, not a report to the user — be concrete and "
        "terse, and drop items once the evidence says they are resolved."
    )
    user = (
        f"{context}\n\n"
        f"Your current note:\n{current or '(empty)'}\n\n"
        "Rewrite the note to reflect the latest evidence. At most 8 bullets. "
        "Keep unresolved items, drop resolved ones, add anything new.\n\n"
        'Respond with ONLY JSON: {"scratchpad": "- ...\\n- ..."}'
    )
    return Prompt(system_prompt=system, model_prompt=user)
