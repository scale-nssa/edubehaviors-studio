"""Runtime configuration.

Model rule (see CLAUDE.md, local branch): the app needs three *roles* — two
annotators and a reasoning model — and the researcher binds them on the Models
page (`/models`, `model_settings`). A role may be called only once Test
connection has verified that exact binding and key; that verification is what
replaced the hosted branch's three-name Vertex allowlist.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from . import model_settings

PACKAGE_DIR = Path(__file__).resolve().parent
# Where the bundled corpus lives. Inside the package, not beside it, so an
# installed wheel (uvx / pipx) finds it too.
BUNDLED_DATA = PACKAGE_DIR / "data"

# --------------------------------------------------------------------------- #
# Roles.
#
# On the hosted branch these were constants — MODELS = (gemini, haiku),
# REASONING_MODEL = sonnet — and an allowlist asserted at import that nothing
# else could be named. Locally the user picks, so they are functions that read
# the current binding. Every caller asks at the point of use, which is what lets
# a change on the Models page take effect without a restart.
#
# Two annotators remain structural (metrics.model_agreement, the alpha in
# service.criterion_stats, the OR in consensus_firing). They may be bound to the
# same model; the second then samples independently under a `#2` nickname.
#
# The hosted measurements that shaped the defaults, kept because they are the
# reason to think before changing them:
#
#   * Cross-family annotators decorrelate errors (spec §3.1); inter-model alpha
#     was only 0.53 macro on the hosted pair, and that disagreement is the
#     signal example selection runs on.
#   * Reasoning OFF for annotation looked cheap and was wrong. It never worked
#     for Gemini (the provider omitted the thinking config, so the server
#     default applied: 206,531 thinking tokens on a 308-call benchmark with the
#     flag "off"), and it broke Haiku: unpinnable temperature plus no thinking
#     made it sample between readings of an ambiguous criterion — 193, 188, 0,
#     189, 188, 189 hits across six identical calls. With thinking: 0, 0, 0.
#     So the default annotator binding keeps reasoning on, and the reasoning
#     setting is part of the nickname and therefore the annotation cache key.
#   * Annotation is ~93% of tokens; the reasoning tier (seed, clarify, propose,
#     reword) is a handful of calls, so capability there is nearly free.
# --------------------------------------------------------------------------- #


def annotators() -> tuple[str, str]:
    """Nicknames of the two annotators, as currently bound."""
    return model_settings.annotators()


def reasoning_model() -> str:
    return model_settings.reasoning_model()


def model_title(nick: str) -> str:
    return model_settings.title(nick)


def is_permitted(nick: str) -> bool:
    """Only a nickname a *verified* role currently holds."""
    return model_settings.is_permitted(nick)

UTTERANCES_PER_ROUND = 10
TIER_CAP = 6
CRITERION_CAP = 3
CONTEXT_WINDOW = 5  # ±5 utterances shown around a reviewed utterance (Q219)
# How much transcript the Review box holds. Rendered in full and scrolled to the
# target rather than clipped, but still bounded — the longest TalkMoves session
# is 2,359 utterances and dumping all of it makes the page unusable.
TRANSCRIPT_WINDOW = 60
# ±N turns around each Fires / Does-not-fire example on the Approve screen.
# Small: enough to judge whether a firing makes sense, not so much that three
# examples a side become a wall of text.
EXAMPLE_WINDOW = 3
MAX_CLARIFYING_QUESTIONS = 3  # Q311

# Session eligibility (docs/selection.md §2)
#
# There is deliberately NO minimum on in-scope utterances. There used to be
# (40, then 25, then 15): the idea was that a session too short to fill a
# ten-utterance round is not worth annotating. But the corpora differ by an
# order of magnitude in session length, so any single floor silently deleted
# most of the shortest dataset — at 25 it took eighteen of Eedi's twenty — and
# a dataset quietly shrinking to a fraction of itself is worse than a round
# that happens to be short.
#
# Consequence to keep in mind: a round can now draw a session with fewer than
# ten in-scope utterances, and will simply review however many there are.
MAX_SESSION_UTTERANCES = 1800

# Chained rounds (docs/next-round.md item 3)
#
# One session is a thin basis for judging a schema, and the labels that need
# looking at are the rare ones: a label the models raise twice in a transcript
# gives you nothing to revise against. So when a round finishes annotating, if
# some label is still short of MIN_POSITIVES_PER_LABEL firings across every
# session this construct has annotated, the next session is annotated
# immediately rather than waiting for the reviewer to ask.
#
# "Positives" means model firings routed to the label, not gold — gold is what
# the reviewer is about to produce, so gating on it would never start.
#
# The cap is the real safety property: each chained session is a full
# annotation pass over every active criterion, so an uncapped chain would walk
# a whole dataset and spend accordingly. Counted over sessions annotated for
# the construct, not per chain, so the ceiling holds across repeated attempts.
# Lowered from 5 to 3: three sessions is already a long unprompted spend, and
# a label the corpus barely contains will not be rescued by two more.
MIN_POSITIVES_PER_LABEL = 3
MAX_CHAINED_SESSIONS = 3

# Per-model concurrency, enforced in `llm.gate` for every caller.
#
# Hosted, these were 16 per annotator, 5 for reasoning and 6 rounds at once —
# tuned for twenty people against a pooled Vertex quota. A personal key
# rate-limits long before that, and the failure mode (429s, backoff, a round
# that crawls) is worse than a round that is merely unhurried. If you get
# 429s, `ask_json` backs off 2/4/8/20s up to ANNOTATION_RETRIES.
ANNOTATION_CONCURRENCY = 4
REASONING_CONCURRENCY = 2
ANNOTATION_RETRIES = 4


def MODEL_CONCURRENCY(nick: str) -> int:
    return REASONING_CONCURRENCY if nick == reasoning_model() else ANNOTATION_CONCURRENCY


# How many rounds may annotate at once. One: this is one person's machine, and
# a chained session queued behind the one being annotated shows its position.
MAX_CONCURRENT_ROUNDS = 1


def estimate_cost(nick: str, input_tokens: int, output_tokens: int) -> float:
    """`output_tokens` must already include reasoning tokens — the invariant
    `providers.Response` states. Gemini reports thoughts separately and
    OpenAI inside completion_tokens; the shim normalises both, and getting it
    wrong understated Gemini ~2.6x on the hosted branch.

    Unknown price counts as 0 here, for the panel's running totals; the spend
    ledger records NULL instead, and the UI says when a price is missing.
    """
    p = model_settings.price(nick)
    if p is None:
        return 0.0
    return (input_tokens / 1e6) * p[0] + (output_tokens / 1e6) * p[1]


def default_data_dir() -> Path:
    """Per-user data directory, so the database does not land wherever the
    user happened to be standing when they ran the command."""
    from platformdirs import user_data_dir

    return Path(user_data_dir("edubehaviors-studio", appauthor=False))


LOOPBACK = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    db_path: Path
    dataset_path: Path
    secret_key: str
    admin_token: str
    # Where models.json, .env and uploaded datasets live. Defaults to the
    # database's directory, which keeps tests (and any explicit DB_PATH)
    # self-contained.
    data_dir: Path | None = None
    # One researcher, no sign-in: the reviewer machinery is kept and bypassed
    # (plan §4.6). Off by default for a directly constructed Config so the
    # multi-user tests keep exercising the identify flow; `from_env` turns it
    # on unless MULTI_USER=1.
    single_user: bool = False

    @property
    def home(self) -> Path:
        return self.data_dir or self.db_path.parent

    @classmethod
    def from_env(cls) -> Config:
        data_dir = Path(os.environ["STUDIO_DATA_DIR"]) if os.environ.get(
            "STUDIO_DATA_DIR") else None
        if os.environ.get("DB_PATH"):
            db_path = Path(os.environ["DB_PATH"])
        else:
            data_dir = data_dir or default_data_dir()
            db_path = data_dir / "studio.sqlite3"
        return cls(
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5057")),
            db_path=db_path,
            dataset_path=Path(
                os.environ.get("DATASET_PATH", BUNDLED_DATA / "talkmoves_20.jsonl")
            ),
            secret_key=os.environ.get("SECRET_KEY", secrets.token_hex(32)),
            admin_token=os.environ.get("ADMIN_TOKEN", secrets.token_urlsafe(12)),
            data_dir=data_dir,
            single_user=os.environ.get("MULTI_USER") != "1",
        )
