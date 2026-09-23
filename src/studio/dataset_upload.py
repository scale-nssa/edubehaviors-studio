"""CSV -> pinned `dataset.v1`, for the Datasets page.

Follows `scripts/build_datasets.py`, which encoded the decisions the bundled
corpora needed, and generalises the one part of it that cannot travel: its
hardcoded speaker map. A stranger's data will not say "teacher" or
"student_ui"; it will say "T", "Ms. Lopez", "S2" or "narrator". So the speaker
mapping is not inferred — the page shows every distinct speaker value and the
researcher assigns each one tutor, student, other, or drop. A guess is only
ever a pre-selected default.

Decisions carried over, and why they are not cosmetic:

  * **`index` is renumbered 0-based per session**, in the file's order (or by
    an order column, if one is named). `Session.window()` slices
    `utterances[lo:index+span+1]`, so index MUST equal list position or every
    Review window is mis-centred.
  * **Dropped speakers and blank text are removed before renumbering** —
    machine events ("mouse", "page_scroll") and empty rows are not utterances,
    and keeping them costs prompt tokens on every annotation call.
  * **The file is written, then digested.** `corpus._verify` refuses to load
    it if a byte changes afterwards: annotations and gold labels are keyed by
    utterance index with no record of what the transcript said.
  * **Session ids are namespaced** `<dataset>:<id>` (see corpus).

Optional gold: a column holding the dataset's own label for an utterance. Kept
in `utterance.gold` in the TalkMoves shape (`{"label": 1}`), several labels
split on `|` or `;`. Nothing in the review flow reads it; it rides along into
the corpus for reference.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("tutor", "student", "other", "drop")

# Pre-selected guesses for the speaker step — the build script's map plus the
# obvious abbreviations. Only defaults: the researcher sees and can change all.
_GUESS = {
    "tutor": "tutor", "teacher": "tutor", "t": "tutor", "instructor": "tutor",
    "student": "student", "s": "student", "student_ui": "student",
    "learner": "student", "pupil": "student",
    "system": "other", "unknown": "other", "": "other",
}

MAX_SPEAKER_VALUES = 60
PREVIEW_ROWS = 8
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,40}$")

# Column-name guesses for the mapping step.
_COLUMN_GUESS = {
    "session_id": ("session_id", "session", "sessionid", "conversation_id",
                   "transcript_id", "transcript", "lesson", "file"),
    "speaker": ("speaker", "role", "speaker_role", "participant", "who"),
    "text": ("text", "content", "utterance", "utt", "message", "transcript_text",
             "line", "said", "transcript"),
    "order": ("sequence_id", "index", "turn", "order", "line_no", "utterance_id"),
    "gold": ("gold", "label", "code", "gold_label"),
}


class UploadError(ValueError):
    pass


def decode(raw: bytes) -> tuple[str, str]:
    """(text, encoding used). UTF-8 first; Latin-1 never fails, so it is the
    fallback — and the page says which was used, since a wrong guess shows up
    as mojibake in the preview rather than as an error."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UploadError("could not decode the file as text")


def read_rows(text: str) -> tuple[list[str], list[dict]]:
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise UploadError("the file has no header row")
    header = [h.strip() for h in reader.fieldnames]
    if len(set(header)) != len(header):
        raise UploadError("two columns have the same name; rename one and re-upload")
    reader.fieldnames = header
    rows = [r for r in reader]
    if not rows:
        raise UploadError("the file has a header but no rows")
    return header, rows


def guess_columns(header: list[str]) -> dict[str, str]:
    low = {h.lower().strip(): h for h in header}
    out = {}
    for field_, names in _COLUMN_GUESS.items():
        for n in names:
            if n in low:
                out[field_] = low[n]
                break
    return out


def speaker_values(rows: list[dict], column: str) -> list[tuple[str, int]]:
    counts = Counter((r.get(column) or "").strip() for r in rows)
    return counts.most_common(MAX_SPEAKER_VALUES)


_NUMBERED = re.compile(r"^(t|teacher|tutor|s|student)[ _-]?\d+$")


def guess_role(value: str) -> str:
    v = value.strip().lower()
    if v in _GUESS:
        return _GUESS[v]
    if m := _NUMBERED.match(v):   # "S1", "Student 2", "T3"
        return "tutor" if m.group(1) in ("t", "teacher", "tutor") else "student"
    return "other"


@dataclass
class Mapping:
    session_id: str
    speaker: str
    text: str
    order: str | None = None
    gold: str | None = None
    roles: dict[str, str] = field(default_factory=dict)   # speaker value -> role


@dataclass
class Built:
    sessions: list[dict]
    dropped_speaker: int
    dropped_blank: int
    unmapped_speakers: list[str]
    role_counts: dict[str, int]

    @property
    def n_utterances(self) -> int:
        return sum(len(s["utterances"]) for s in self.sessions)


def _order_key(value: str, fallback: int):
    try:
        return (0, float(value), fallback)
    except (TypeError, ValueError):
        return (1, 0.0, fallback)


def build(rows: list[dict], m: Mapping, dataset_id: str) -> Built:
    """Convert rows to dataset.v1 sessions. Pure: writes nothing."""
    for col in (m.session_id, m.speaker, m.text):
        if not col:
            raise UploadError("session id, speaker and text columns are all required")
    if len({m.session_id, m.speaker, m.text}) < 3:
        raise UploadError("session id, speaker and text must be three different columns")
    by_session: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    dropped_speaker = dropped_blank = 0
    unmapped: set[str] = set()
    for pos, r in enumerate(rows):
        sid = (r.get(m.session_id) or "").strip()
        if not sid:
            dropped_blank += 1
            continue
        spk = (r.get(m.speaker) or "").strip()
        role = m.roles.get(spk)
        if role is None:
            # A value the page never showed (past the display limit): not a
            # silent tutor or student, but visible context.
            unmapped.add(spk)
            role = "other"
        if role == "drop":
            dropped_speaker += 1
            continue
        text = (r.get(m.text) or "").strip()
        if not text:
            dropped_blank += 1
            continue
        by_session[sid].append((pos, {**r, "_role": role, "_text": text}))

    sessions = []
    role_counts: Counter = Counter()
    for sid in sorted(by_session):
        items = by_session[sid]
        if m.order:
            items = sorted(items, key=lambda t: _order_key(t[1].get(m.order), t[0]))
        utts = []
        for i, (_pos, r) in enumerate(items):
            gold = {}
            if m.gold and (g := (r.get(m.gold) or "").strip()):
                gold = {lbl.strip(): 1 for lbl in re.split(r"[|;]", g) if lbl.strip()}
            role_counts[r["_role"]] += 1
            utts.append({
                # 0-based position, NOT any order column — see module docstring.
                "index": i,
                "speaker": {"name": (r.get(m.speaker) or "").strip() or None,
                            "role": r["_role"]},
                "text": r["_text"],
                "turn": (r.get(m.order) if m.order else str(i + 1)),
                "gold": gold,
            })
        sessions.append({
            "schema_version": "dataset.v1",
            "session_id": f"{dataset_id}:{sid}",
            "source": dataset_id,
            "metadata": {"original_session_id": sid},
            "utterances": utts,
        })
    if not sessions:
        raise UploadError("no utterances survived the mapping — check the columns and speaker roles")
    return Built(sessions, dropped_speaker, dropped_blank, sorted(unmapped),
                 dict(role_counts))


def write(built: Built, dest_dir: Path, dataset_id: str, title: str,
          description: str, source_name: str) -> dict:
    """Write JSONL + digest and add the dataset to the user registry."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    registry_path = dest_dir / "datasets.json"
    entries = json.loads(registry_path.read_text()) if registry_path.exists() else []
    if any(e["id"] == dataset_id for e in entries):
        raise UploadError(f"a dataset called {dataset_id!r} already exists")
    dest = dest_dir / f"{dataset_id}.jsonl"
    if dest.exists():
        raise UploadError(f"{dest.name} already exists; pick another name")
    with dest.open("w", encoding="utf-8") as fh:
        for s in built.sessions:
            fh.write(json.dumps(s, sort_keys=True, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(".sha256").write_text(digest + "\n")
    entry = {
        "id": dataset_id,
        "title": title or dataset_id,
        "description": description,
        "file": dest.name,
        "sessions": len(built.sessions),
        "utterances": built.n_utterances,
        "tutor_utterances": built.role_counts.get("tutor", 0),
        "student_utterances": built.role_counts.get("student", 0),
        "source_file": source_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries.append(entry)
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    tmp.replace(registry_path)
    return entry


def remove(dest_dir: Path, dataset_id: str) -> None:
    registry_path = dest_dir / "datasets.json"
    entries = json.loads(registry_path.read_text()) if registry_path.exists() else []
    keep = [e for e in entries if e["id"] != dataset_id]
    gone = [e for e in entries if e["id"] == dataset_id]
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(keep, indent=2) + "\n")
    tmp.replace(registry_path)
    for e in gone:
        p = dest_dir / e["file"]
        p.unlink(missing_ok=True)
        p.with_suffix(".sha256").unlink(missing_ok=True)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
