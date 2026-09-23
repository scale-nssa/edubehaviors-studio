"""Pre-flight cost estimates for every paid action (docs/local-mode-plan.md §4.4).

"This round is ~N calls, ~M input tokens, about $X." Shown before the action
runs, with a confirm, because it is the user's own money and they will be
scared of it — rightly, since the first session drawn is the longest one.

How good the numbers are, stated plainly because the page repeats it:

  * **Calls** are exact for what the cache can see. Annotation is cached on
    (session, criterion text, model), so only the pairs not yet in the cache
    are counted. A sub-criterion is counted even though it is skipped when its
    parent never fires, so this errs high.
  * **Input tokens** are the real prompt text divided by four — a rule of
    thumb, not the provider's tokenizer. Typically within ±25%.
  * **Output tokens** are this data directory's own average for that model and
    purpose once it has any history, and a deliberately generous guess before
    that. Reasoning is billed as output and is most of it.
  * **Price** is whatever the Models page says, or a built-in list price.
    When neither is known the dollar figure is missing, not zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import corpus, model_settings, prompts, spend
from .config import MAX_CHAINED_SESSIONS, annotators, reasoning_model
from .db import jload

CHARS_PER_TOKEN = 4

# Before there is any history. Generous on purpose: an estimate that comes in
# under the bill teaches the user to distrust it.
GUESS_OUTPUT = {"annotate": 2500, "default": 6000}


@dataclass
class Line:
    label: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None


@dataclass
class Estimate:
    lines: list[Line] = field(default_factory=list)
    cached_calls: int = 0
    notes: list[str] = field(default_factory=list)
    # Extra sessions a review round may chain onto automatically, each
    # estimated like this one. Informational: the cap is re-checked per call.
    chain_sessions: int = 0
    guessed_output: bool = False

    @property
    def calls(self) -> int:
        return sum(l.calls for l in self.lines)

    @property
    def input_tokens(self) -> int:
        return sum(l.input_tokens for l in self.lines)

    @property
    def output_tokens(self) -> int:
        return sum(l.output_tokens for l in self.lines)

    @property
    def unpriced(self) -> bool:
        return any(l.usd is None and l.calls for l in self.lines)

    @property
    def usd(self) -> float:
        return sum(l.usd or 0.0 for l in self.lines)

    @property
    def usd_high(self) -> float:
        """Including any sessions that may chain on automatically."""
        return self.usd * (1 + self.chain_sessions)


def _tokens(prompt) -> int:
    return (len(prompt.system_prompt or "") + len(prompt.model_prompt)) // CHARS_PER_TOKEN


def _output_per_call(est: Estimate, nick: str, profile: str) -> int:
    avg = spend.averages(nick, profile)
    if avg is None:
        est.guessed_output = True
        return GUESS_OUTPUT[profile]
    return int(avg[1])


def _line(est: Estimate, label: str, nick: str, profile: str, calls: int,
          input_tokens: int) -> None:
    out = _output_per_call(est, nick, profile) * calls
    est.lines.append(Line(
        label=label, model=model_settings.title(nick), calls=calls,
        input_tokens=input_tokens, output_tokens=out,
        usd=spend.cost(nick, input_tokens, out) if calls else 0.0,
    ))


def annotation(conn, cfg, construct_id: int, kind: str, session_id: str,
               *, chain: bool = True) -> Estimate:
    """One annotation pass over one session, as `service.start_round` runs it."""
    from .service import annotation_key, load_criteria, sessions_remaining

    row = conn.execute(
        "SELECT scope FROM construct WHERE id=?", (construct_id,)
    ).fetchone()
    scope = row["scope"]
    criteria = load_criteria(
        conn, construct_id, status=None if kind == "approve" else "active"
    )
    by_id = {c.id: c for c in criteria}
    session = corpus.get(cfg, session_id)
    est = Estimate()
    for nick in annotators():
        calls = tokens = 0
        for c in criteria:
            parent = by_id.get(c.parent_id) if c.parent_id else None
            key = annotation_key(c.text, parent.text if parent else None)
            hit = conn.execute(
                "SELECT 1 FROM annotation WHERE session_id=? AND criterion_text=? "
                "AND model=?", (session_id, key, nick),
            ).fetchone()
            if hit:
                est.cached_calls += 1
                continue
            calls += 1
            tokens += _tokens(prompts.annotate(session, c.text, scope))
        _line(est, f"Annotate {len(session.utterances)} utterances", nick,
              "annotate", calls, tokens)
    if kind == "review" and chain and est.calls:
        remaining, _ = sessions_remaining(conn, cfg, construct_id)
        est.chain_sessions = max(0, min(MAX_CHAINED_SESSIONS, remaining - 1))
        if est.chain_sessions:
            est.notes.append(
                f"If some label still has too few firings afterwards, up to "
                f"{est.chain_sessions} more session(s) are annotated automatically "
                "— the upper figure includes them. The spend cap is checked "
                "before every call, so it still holds."
            )
    return est


def seeding(description: str, labels: list[str], scope: str) -> Estimate:
    est = Estimate()
    nick = reasoning_model()
    _line(est, "Propose criteria from your description", nick, "default", 1,
          _tokens(prompts.seed_criteria(description, labels, scope)))
    return est


def revise(conn, cfg, construct_id: int, what: str) -> Estimate:
    """P3 (questions) or P4 (deltas, plus the scratchpad refresh)."""
    from .config import MAX_CLARIFYING_QUESTIONS
    from .service import build_schema_view, revise_context

    ctx = revise_context(conn, cfg, construct_id)
    ctx += "\n" + "\n".join(build_schema_view(conn, cfg, construct_id).summary_lines())
    est = Estimate()
    nick = reasoning_model()
    if what == "questions":
        _line(est, "Ask clarifying questions", nick, "default", 1,
              _tokens(prompts.clarifying_questions(ctx, MAX_CLARIFYING_QUESTIONS)))
    else:
        pad = conn.execute(
            "SELECT scratchpad FROM construct WHERE id=?", (construct_id,)
        ).fetchone()["scratchpad"]
        _line(est, "Propose schema changes", nick, "default", 1,
              _tokens(prompts.propose_deltas(ctx)))
        _line(est, "Update the running note", nick, "default", 1,
              _tokens(prompts.scratchpad(ctx, pad)))
    return est


def construct_labels(conn, construct_id: int) -> list[str]:
    return jload(conn.execute(
        "SELECT label_space FROM construct WHERE id=?", (construct_id,)
    ).fetchone()["label_space"], [])
