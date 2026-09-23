# EduBehaviors Studio — local edition

Turn a prose description of a behaviour you care about into an itemised,
testable annotation schema over tutoring and classroom transcripts. Runs on
your own computer, with your own AI provider and API key. Nobody else hosts it
or pays for it, and your transcripts go only to the model provider you choose.

---

## Getting started

You need two things: a way to run it, and an API key from an AI provider.

### 1. Install and start it

Install [`uv`](https://docs.astral.sh/uv/) once. On macOS or Linux, open a
terminal and paste:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(On Windows, in PowerShell: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.)
Then close and reopen the terminal.

Now start the app, with no separate install step:

```bash
uvx --from git+https://github.com/scale-nssa/edubehaviors-studio@local edubehaviors-studio
```

A browser tab opens at <http://127.0.0.1:5057>. Leave the terminal window open
while you work; closing it stops the app. Run the same command to start it
again next time. Your work is kept between runs.

If you have a copy of the code instead (a clone or a download), from inside
that folder:

```bash
uv run studio
```

### 2. Set up your models (the Models page)

The first thing you see is the **Models** page. The app uses three models:

| Role | What it does | Share of the bill |
|---|---|---|
| Annotator A | Reads a whole transcript and marks every utterance where a criterion holds | most of it |
| Annotator B | Does the same, independently, so disagreements can be found | most of it |
| Reasoning | Proposes criteria from your description, and revisions after review | small |

1. **Paste an API key** for your provider and press Save. Supported: Anthropic,
   OpenAI, Google Gemini, Google Vertex AI (uses `gcloud` credentials, no key),
   and any **OpenAI-compatible endpoint**: OpenRouter, Together, vLLM, and
   local servers such as Ollama or LM Studio.
2. **Pick a model for each role.** The defaults work with a single Anthropic
   key: Claude Haiku 4.5 for both annotators and Claude Sonnet 5 for reasoning.
3. Press **Save & test** on each role. That makes one tiny real call. If it
   fails, the page shows the provider's own error message: a wrong key, a model
   name the provider doesn't recognise, or a region problem. The app will not
   call a model until its test has passed.

**Keys are stored in a `.env` file in your data folder, readable only by you,
never in the database.** So you can send someone your database without sending
your key.

Two annotators from **different** model families (say, a Claude and a Gemini)
is the stronger setup: their disagreements point at weak criteria. Both can be
the same model if that's all you have. The second is then an independent
sample, and agreement measures how consistently one model answers rather than
whether two models agree. The page says so.

### 3. Cost: it's your money

- **Before anything that costs money**, the app shows an estimate: how many
  calls, how many tokens, and roughly how many dollars. Nothing runs until you
  press **Go ahead**. Work the app has done before is cached and free.
- A **spend counter** in the top-right corner shows the estimated all-time
  total. The Models page breaks it down.
- A **spend cap** (default **$10**, change it on the Models page) blocks
  anything that would take the total past it, including sessions the app adds
  on its own.
- All figures are **estimates** from the provider's token counts and the prices
  on the Models page. The provider's billing page has the real number.

The first session is the longest one in the dataset, so the first round is
usually the most expensive. With the default models a round typically costs
from a few cents to a dollar or two, depending on transcript length and how
many criteria you have. The estimate page gives the figure for yours.

---

## Using your own transcripts (the Datasets page)

Open **datasets** in the top bar. It lists what's installed, with session and
utterance counts, and lets you preview any transcript exactly as the models
will see it.

To add your own, upload a **CSV with one row per utterance**:

| session | speaker | text | order (optional) | gold (optional) |
|---|---|---|---|---|
| lesson_03 | Teacher | What do you notice about these two fractions? | 1 | |
| lesson_03 | S2 | They have the same bottom number. | 2 | |

The column names don't matter: after uploading, you:

1. **Check the preview** to confirm the file was read correctly.
2. **Say which column is which:** session id, speaker, text, and optionally an
   order column and a gold label column.
3. **Say what each speaker is.** Every distinct speaker value in your file is
   listed. Map each one to **tutor**, **student**, **other** (shown to the
   models as context, never labelled), or **drop** (removed entirely; use this
   for non-speech rows such as clicks or page views). The app can't guess these
   reliably, so check them.
4. **Name the dataset**, check the counts, and press **Create dataset**.

Then choose it on **New construct**. A construct is tied to one dataset for
life.

The app keeps a checksum of every dataset and **refuses to load one that has
been edited afterwards**, because your annotations and judgements point at
utterance positions within it. To change a dataset, upload a corrected copy
under a new name. A dataset can't be deleted while any construct uses it.

Your uploads stay in your data folder on this computer. Transcripts are only
sent anywhere when a round sends them to the models you chose.

---

## How it works

You describe a behaviour in prose. The tool proposes a schema of yes/no
**criteria** over utterances, annotates a real transcript with two independent
models, picks the ten utterances most likely to expose a problem, and asks you
to adjudicate. Your judgements drive a revision pass. Repeat until the schema
says what you mean. The schema is the artifact; a classifier is an export.

```
  Describe ──► Approve ──►┌─► Annotate ──► Review ──► Revise ─┐
                          │                                    │
                          └────────────────────────────────────┘
```

- **Describe.** Paste notes, a paragraph from a paper, or stream of
  consciousness. Define your labels inside the description; the label list
  itself is just names. Name a catch-all (`Other`) so every utterance lands
  somewhere.
- **Approve.** The reasoning model proposes criteria. One session is annotated
  so you can see real examples, and you accept, reword, rewire or exclude each
  criterion.
- **Review.** Ten utterances per round, chosen to be informative: utterances
  where the schema offers several labels, none, or where the two models
  disagree. For each one you give the right label and correct any criterion the
  models got wrong. Whether the models disagreed stays hidden until you submit,
  so it can't anchor you.
- **Revise.** The reasoning model reads your judgements, optionally asks
  clarifying questions, and proposes changes. Every change you apply saves a
  new schema version you can roll back to.

Every edge is evidence: a criterion that fires is evidence for the labels it
points at. Two criteria pointing different ways leave an utterance with two
candidate labels; the fix is narrower wording, never a cancelling rule.
Criteria can be nested: a gate criterion with sub-criteria under it, which are
only asked where the gate fired, is the main way to separate labels that are
easy to confuse.

**Metrics** (the construct page and the session board): well-definedness (how
many utterances the schema resolves to exactly one label), Krippendorff's α
between the models and between models and you, and accuracy against your
labels. The ten reviewed utterances are chosen to be hard, not at random, so
these are diagnostics rather than dataset-wide performance estimates.

**Export**: `export.zip` holds the firing matrix (one row per utterance, one
column per criterion and model) and the assertion key. Exporting only reads
the cache, so it costs nothing.

---

## Where things are kept

Everything lives in one per-user data folder, printed in the terminal at
startup:

| OS | Folder |
|---|---|
| macOS | `~/Library/Application Support/edubehaviors-studio/` |
| Linux | `~/.local/share/edubehaviors-studio/` |
| Windows | `%LOCALAPPDATA%\edubehaviors-studio\` |

| File | Holds |
|---|---|
| `studio.sqlite3` | your constructs, rounds, judgements and the annotation cache |
| `models.json` | which model plays which role, prices, spend cap |
| `.env` | API keys (owner-only permissions) |
| `datasets/` | your uploaded datasets |

To back up your work, copy the folder. To start fresh, move it aside.

Settings (environment variables, all optional):

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `5057` | port to listen on |
| `STUDIO_DATA_DIR` | per-user folder above | where everything is kept |
| `DB_PATH` | `<data dir>/studio.sqlite3` | database file (its folder then holds the rest) |
| `NO_BROWSER` | unset | `1` = don't open a browser tab |
| `HOST` | `127.0.0.1` | see below |

**There is no login.** That's safe while the app only listens on your own
computer (`127.0.0.1`, the default). Anyone who can reach the port can read
your data and spend your key, so the app **refuses** any other `HOST` unless
you also set `I_UNDERSTAND_THERE_IS_NO_AUTH=1`. Don't do that on a network you
don't control.

---

## Local models (Ollama, LM Studio)

For full privacy and no per-token cost, run a model server on your own machine
and pick **OpenAI-compatible endpoint** on the Models page, with base URL
`http://localhost:11434/v1` (Ollama) or `http://localhost:1234/v1` (LM Studio)
and no key. Set its price to `0`. Small local models may simply be too weak for
the annotation task, which asks a model to read a whole transcript and list
every matching utterance. Check a round's firings before trusting them. Reasoning
settings can't be controlled portably on these servers, so the server's default
applies.

---

## Development

From a copy of the code:

```bash
uv sync         # install dependencies
uv run studio   # run the app
```

The model layer is `src/studio/providers.py`, a thin shim over the official
`anthropic`, `openai` and `google-genai` SDKs. Its token accounting has one
invariant: `output_tokens` is everything billed as output, **reasoning
included**. Providers report reasoning differently; see the module docstring
before adding one.

---

## ⚠️ Data licence: read this before you publish anything

The app ships with one dataset, a 20-session subset of **TalkMoves**:

> **The TalkMoves Dataset: K-12 mathematics lesson transcripts annotated for
> teacher and student discursive moves**, by the SumnerLab at the University
> of Colorado Boulder and collaborators.
> Source: <https://github.com/SumnerLab/TalkMoves>. The authors' publications,
> which are the right things to cite, are listed in that repository's README.

TalkMoves is licensed under the **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International** licence (CC BY-NC-SA 4.0):
<https://creativecommons.org/licenses/by-nc-sa/4.0/>.

If you use the bundled TalkMoves data, you are bound by three conditions:

1. **Attribution.** Credit the TalkMoves authors and link the source whenever
   you share or publish anything that uses the data.
2. **Non-commercial use only.** You may not use it, or anything derived from
   it, for commercial purposes.
3. **Share-alike.** Anything you derive from it must be released under the
   same licence.

**Annotations, schemas and exports you produce over TalkMoves are derived from
it, so they carry these same terms.** That includes the `export.zip` firing
matrix. Your own uploaded transcripts are not affected: what you produce over
them is subject only to whatever terms govern your data.
