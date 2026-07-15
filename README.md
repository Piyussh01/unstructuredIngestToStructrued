# stous — a knowledge compiler

Unstructured data in, typed structure out. Companion code to
[*Structure is a cache: compile your data, don't retrieve it*](./article-draft.md).

The premise: don't pay an LLM per document forever — pay it once to build artifacts
that do the work cheaply afterwards. The raw layer is immutable; every structured
artifact above it (parsers, entity store, wiki) is model-generated, provenance-linked,
and disposable by design. When a better model ships, `stous recompile` and rebuild
overnight.

## How it works

```
stous add ./pile/           # 1. everything lands in an immutable, content-addressed raw layer
stous interview             # 2. chat with the model: what are you trying to extract?
stous compile --workers 8   # 3. the compiler passes
stous lint                  # 4. contradiction / staleness / orphan hunt
stous ask "which clients pay late and what did they negotiate?"
```

### The compile passes

Every document is routed by two questions — **canonical?** (does a field mean the same
thing everywhere → a parsing problem, not an interpretation problem) and **connected?**
(does its meaning live in links to other documents). That routes each doc down one of
three paths:

| Path | Data | Mechanism | Cost model |
|---|---|---|---|
| `parser` | canonical (templated invoices, exports, logs) | frontier agent writes a Python parser once, tested against samples; code runs over the group in a sandbox | LLM cost ≈ O(doc *types*), not O(docs) |
| `extract` | non-canonical, self-contained (emails, contracts, tickets) | pool of concurrent **Haiku** extractors with structured outputs; every fact carries a verbatim quote grounded to a char offset; low-confidence facts go to an **Opus adjudicator** | small-model batch price; frontier only on disputes |
| `wiki` | non-canonical, connected (threads, transcripts, strategy docs) | frontier wiki agent: ingest pass (one source may touch many pages), lint pass (contradictions, stale claims, orphans) | frontier, but only for the connective tissue |

Parser failures (format variance — the Evaporate caveat) automatically fall through to
the extract path. Everything lands in a typed entity store with per-fact provenance;
conflicting values are kept visible for the lint pass, never silently overwritten.

### Query time

No embeddings anywhere. `stous ask` runs an agent that writes Python against the entity
store in a subprocess (only results return to context) and reads the wiki index-first.

## Install

```sh
pip install -e .            # core: anthropic + pydantic
pip install -e '.[all]'     # + pypdf, beautifulsoup4, pandas
export ANTHROPIC_API_KEY=sk-ant-...
```

Config via env: `STOUS_FRONTIER_MODEL` (default `claude-opus-4-8`), `STOUS_FAST_MODEL`
(default `claude-haiku-4-5`), `STOUS_ADJUDICATION_THRESHOLD` (default `0.7`),
`STOUS_WORKERS` (default `8`).

## Workspace layout

```
.stous/
  raw/               immutable source of truth (content-addressed; never modified)
  manifest.jsonl     document records
  schema.json        output of the interview
  routes.jsonl       canonical?/connected? decisions per doc
  parsers/           LLM-written parsers (human-reviewable Python)
  extractions.jsonl  per-doc extraction records
  entities.json      typed entity store with provenance + conflicts
  wiki/              LLM-owned markdown (index.md, _schema.md, entity pages)
  lint_report.md     latest lint pass output
```

## Prior art (researched July 2026)

Everything here is either stolen from or validated by existing systems:

- **[Evaporate](https://arxiv.org/abs/2304.09433)** — LLM-written extraction functions,
  110× token reduction; the parser path, including the fall-through for format variance.
- **[LangExtract](https://github.com/google/langextract)** (Google) — char-offset source
  grounding as a first-class feature; our quote→offset provenance is the same idea.
- **[Reducto Deep Extract](https://reducto.ai/blog/reducto-deep-extract-agent)** —
  extract → verify → re-extract with per-field citations; our confidence-gated
  adjudication is the budget version.
- **[LlamaExtract](https://docs.llamaindex.ai/en/stable/use_cases/extraction/)** — infers
  schemas from sample documents; our interview grounds its draft in a corpus sample.
- **[DocETL](https://github.com/ucbepic/docetl)** (Berkeley), **[LOTUS](https://github.com/lotus-data/lotus)**,
  **[Palimpzest](https://github.com/mitdbg/palimpzest)** (MIT) — declarative LLM pipelines
  with cost/quality optimizers; the "compile once, execute cheap" economics.
- **Karpathy's [llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** —
  the three-layer wiki (immutable raw / LLM-owned pages / schema doc) and the
  ingest / file-back / lint passes.
- **Contradiction handling convergence** — GraphRAG *gleaning*, mem0 *consolidation*,
  Graphiti *temporal edge invalidation*: everyone independently discovers that extraction
  without a lint pass produces a confidently wrong database.

### Why so few dependencies

The 2026 landscape review concluded: JSON validity is commoditized (Anthropic's native
structured outputs do constrained decoding server-side, so `instructor`/`outlines` add
little); orchestration frameworks (LangGraph, CrewAI) are ceremony for a deterministic
pipeline — a hand-rolled loop is clearer; and heavy ingestion is best delegated when you
need it. So: `anthropic` + `pydantic` core, optional `pypdf`/`bs4`/`pandas`. If you hit
hard PDFs (scans, complex tables), swap the ingest layer for
[docling](https://github.com/docling-project/docling) or
[markitdown](https://github.com/microsoft/markitdown) — `stous/ingest.py:_normalize` is
the single seam.

What's *not* commoditized — and where this code spends its effort — is semantic
validity: are the values actually right? That's the confidence scores, the adjudicator,
the provenance quotes, and the lint pass.

## Module map

```
stous/
  config.py     model tiers, thresholds, workspace layout
  ingest.py     raw layer: content-addressed copies + text normalization
  llm.py        three call shapes: structured(), agent_loop(), stream_chat()
  interview.py  schema-definition chat; ends when the model calls finalize_schema
  router.py     canonical? / connected? → parser | extract | wiki (concurrent Haiku)
  parsers.py    parser-writer agent + sandboxed execution + fall-through
  extract.py    concurrent batch extraction + confidence-gated Opus adjudication
  store.py      typed entity store: facts, relations, provenance, visible conflicts
  wiki.py       wiki agent: ingest pass + lint pass over LLM-owned markdown
  query.py      ask: agent writes code against the store, reads wiki index-first
  cli.py        init / add / interview / compile / lint / ask / status / recompile
```
