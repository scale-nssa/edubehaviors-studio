"""Read-only access to the pinned corpora.

One `Session` per JSONL line, in the `dataset.v1` shape. Two sources:

  * **Bundled**: TalkMoves, a frozen 20-session subset shipped inside the
    package (`studio/data/`). CC BY-NC-SA 4.0 — see the README; annotations
    produced over it inherit those terms.
  * **Uploaded**: whatever the researcher converts on the Datasets page, kept
    in `<data dir>/datasets/` with its own `datasets.json` registry. Written by
    `dataset_upload`, and pinned with a `.sha256` exactly like the bundled one.

Uploaded session ids are namespaced `<dataset id>:<original id>`. The bundled
corpora were checked at build time to have globally unique ids, which is what
lets `get` resolve a session without being told its dataset; a stranger's CSV
promises no such thing, and "session_1" is the most likely id there is.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import MAX_SESSION_UTTERANCES, Config

log = logging.getLogger("studio.corpus")


class CorpusChanged(RuntimeError):
    """The dataset on disk is not the one this app was pinned to."""

# dataset roles -> our scope vocabulary. TalkMoves says "teacher"; the UI says
# "tutor" (spec §1: sources are coerced to a two-role shape).
_ROLE_MAP = {"teacher": "tutor", "tutor": "tutor", "student": "student"}


@dataclass(frozen=True)
class Utterance:
    index: int
    role: str  # "tutor" | "student" | "other"
    text: str
    gold: tuple[str, ...]  # dataset-native gold label keys that are set


@dataclass(frozen=True)
class Session:
    session_id: str
    utterances: tuple[Utterance, ...]

    def in_scope(self, scope: str) -> list[Utterance]:
        if scope == "all":
            return list(self.utterances)
        return [u for u in self.utterances if u.role == scope]

    def window(self, index: int, span: int) -> list[Utterance]:
        lo = max(0, index - span)
        return [u for u in self.utterances[lo : index + span + 1]]

    def render(self) -> str:
        """The whole transcript, one indexed line per utterance."""
        return "\n".join(
            f"{u.index}: [{u.role.capitalize()}] {u.text}" for u in self.utterances
        )


def digest_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _verify(path: Path) -> None:
    """Fail loudly if the pinned corpus has been edited.

    Annotations are cached on (session, criterion text, model) with no record of
    what the transcript said at the time, so a corpus that changes silently
    invalidates every cached answer and every gold label keyed by utterance
    index — without anything in the UI looking wrong. A digest beside the data
    turns that into an error at startup instead.

    Only enforced when a `.sha256` sits next to the data, so pointing
    DATASET_PATH at some other file for a one-off still works.
    """
    digest_file = digest_path(path)
    if not digest_file.exists():
        return
    expected = digest_file.read_text().strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise CorpusChanged(
            f"{path} does not match {digest_file.name}.\n"
            f"  expected {expected}\n  actual   {actual}\n"
            "The corpus is pinned: cached annotations and gold labels are keyed "
            "by utterance index, so editing it invalidates them silently. "
            "An uploaded dataset cannot be edited in place: delete it on the "
            "Datasets page and upload it again under a new name."
        )


DEFAULT_DATASET = "talkmoves"
SESSION_NS = ":"   # uploaded session ids are "<dataset>:<original id>"


@lru_cache(maxsize=4)
def registry(root_str: str) -> tuple[dict, ...]:
    """The bundled datasets, from `datasets.json` beside them.

    Falls back to a single TalkMoves entry when the registry is absent.
    """
    path = Path(root_str) / "datasets.json"
    if not path.exists():
        entries = ({"id": DEFAULT_DATASET, "title": "TalkMoves", "description": "",
                    "file": "talkmoves_20.jsonl", "sessions": 0},)
    else:
        entries = tuple(json.loads(path.read_text()))
    return tuple({**e, "dir": root_str, "bundled": True} for e in entries)


def user_dir(cfg: Config) -> Path:
    return cfg.home / "datasets"


_user_cache: dict[str, tuple[float, tuple[dict, ...]]] = {}


def user_registry(cfg: Config) -> tuple[dict, ...]:
    """Uploaded datasets. Re-read when the file changes, not on every call:
    `get` consults the registry for every session lookup."""
    path = user_dir(cfg) / "datasets.json"
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return ()
    hit = _user_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    entries = tuple(
        {**e, "dir": str(path.parent), "bundled": False}
        for e in json.loads(path.read_text())
    )
    _user_cache[str(path)] = (mtime, entries)
    return entries


def datasets(cfg: Config) -> tuple[dict, ...]:
    return registry(str(cfg.dataset_path.parent)) + user_registry(cfg)


def dataset_ids(cfg: Config) -> list[str]:
    return [d["id"] for d in datasets(cfg)]


def entry(cfg: Config, dataset: str) -> dict:
    for d in datasets(cfg):
        if d["id"] == dataset:
            return d
    raise KeyError(f"unknown dataset {dataset!r}")


def _path_for(cfg: Config, dataset: str | None) -> Path:
    """Where one dataset's JSONL lives.

    `DATASET_PATH` still points at TalkMoves and still wins for it, so an
    existing deployment that overrides it keeps working.
    """
    if dataset in (None, DEFAULT_DATASET):
        return cfg.dataset_path
    d = entry(cfg, dataset)
    return Path(d["dir"]) / d["file"]


@lru_cache(maxsize=16)
def _load_at(path_str: str, _mtime: float) -> tuple[Session, ...]:
    # Keyed on mtime too, so a dataset deleted and re-uploaded under the same
    # name is re-read rather than served stale from the cache.
    _verify(Path(path_str))
    sessions = []
    with open(path_str, encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            utts = tuple(
                Utterance(
                    index=int(u["index"]),
                    role=_ROLE_MAP.get(u["speaker"]["role"], "other"),
                    text=str(u["text"]),
                    gold=tuple(k for k, v in (u.get("gold") or {}).items() if v),
                )
                for u in raw["utterances"]
            )
            sessions.append(Session(session_id=str(raw["session_id"]), utterances=utts))
    return tuple(sessions)


def _load(path_str: str) -> tuple[Session, ...]:
    return _load_at(path_str, Path(path_str).stat().st_mtime)


def all_sessions(cfg: Config, dataset: str | None = None) -> tuple[Session, ...]:
    return _load(str(_path_for(cfg, dataset)))


def get(cfg: Config, session_id: str) -> Session:
    """Resolve a session by id, across every dataset.

    Bundled session ids are globally unique, and uploaded ones carry their
    dataset as a prefix, so a construct's transcripts resolve without the
    caller having to know which dataset it chose. That is what keeps the
    dataset parameter out of the fifteen call sites that only ever hold a
    session id.
    """
    ds_hint, sep, _ = session_id.partition(SESSION_NS)
    order = dataset_ids(cfg)
    if sep and ds_hint in order:
        order = [ds_hint] + [d for d in order if d != ds_hint]
    for ds in order:
        try:
            for s in all_sessions(cfg, ds):
                if s.session_id == session_id:
                    return s
        except (FileNotFoundError, KeyError):
            continue   # a registry entry whose file is missing must not break the rest
    raise KeyError(session_id)


def eligible_session_ids(
    cfg: Config, scope: str, used: set[str], dataset: str | None = None
) -> list[str]:
    """Sessions a round may draw from, within one dataset (docs/selection.md §2).

    **Ordered longest-first by in-scope utterance count** — that ordering *is*
    the draw order; see `selection.pick_session`. In-scope rather than total,
    because in-scope is how much there is to review; the two diverge sharply
    (PLUS_13 is 160 utterances but 7 tutor ones).

    No minimum on in-scope utterances — see the note in `config`. The only
    filters are "not already used" and the length ceiling.
    """
    out = [
        s for s in all_sessions(cfg, dataset)
        if s.session_id not in used and len(s.utterances) <= MAX_SESSION_UTTERANCES
    ]
    out.sort(key=lambda s: (-len(s.in_scope(scope)), s.session_id))
    return [s.session_id for s in out]
