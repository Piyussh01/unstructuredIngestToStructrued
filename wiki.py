"""Path 3: connected prose. The wiki passes.

Three layers: raw sources (immutable, elsewhere), the wiki (LLM-owned
markdown the agent maintains), and _schema.md (tells the agent how the
wiki is organized). Two passes:

- ingest: one connected document at a time; the agent updates every page
  it touches (a client page, a project page, the index) — a single
  source may touch many pages.
- lint: periodic; hunts contradictions between pages, claims superseded
  by newer sources, orphan pages, missing cross-links — and fixes them.

Extraction into rows loses relationships; the wiki keeps them.
"""

from __future__ import annotations

import json

from .config import DOC_CHAR_LIMIT, FRONTIER_MODEL, Workspace
from .ingest import load_manifest
from .llm import agent_loop
from .router import Route

SCHEMA_PAGE = """# Wiki schema

This wiki is maintained by an LLM agent. Layout:

- `index.md` — the entry point. Every page must be reachable from here.
  Agents answering queries read this FIRST and drill in; keep it a tight
  table of contents with one-line summaries.
- One page per significant entity (`client_acme.md`, `project_apollo.md`)
  or theme (`pricing_decisions.md`).
- Every claim cites its source doc id inline like `[doc:1a2b3c4d5e6f7890]`.
- Newer sources supersede older ones: rewrite the claim, keep the citation.
- Cross-link related pages with normal markdown links.
"""

WIKI_TOOLS = [
    {
        "name": "list_pages",
        "description": "List all wiki pages with sizes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_page",
        "description": "Read a wiki page by filename (e.g. 'index.md').",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "write_page",
        "description": "Create or overwrite a wiki page with full new content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_source",
        "description": "Read the raw text of a source document by doc_id (read-only; sources are immutable).",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
]

INGEST_SYSTEM = """You maintain the wiki layer of a knowledge compiler. Raw sources are \
immutable; the wiki is yours entirely. Follow _schema.md conventions strictly.

Extraction goal (what the user cares about): {goal}

You are given ONE new source document. Integrate it:
1. Read index.md (and _schema.md if unsure of conventions).
2. Read any existing pages this document touches.
3. Update or create every page whose content this source changes — entity pages, theme
   pages, and index.md. A single source often touches several pages.
4. Cite the source doc id for every claim you add. If it contradicts an existing claim,
   the NEWER source wins: rewrite the claim, keep both citations, and note the supersession.
Work with tools; when fully integrated, reply with a one-line summary of pages touched."""

LINT_SYSTEM = """You are the lint pass of a knowledge compiler's wiki. Hunt for:
- contradictions between pages (or within one page)
- stale claims superseded by newer sources
- orphan pages unreachable from index.md
- missing cross-references between obviously related pages

Fix what you can directly (rewrite pages, add links, update the index). Verify against
raw sources with read_source when two pages disagree. When done, reply with a markdown
lint report: issues found, fixes applied, and anything needing human attention."""


def _tool_impls(ws: Workspace):
    docs_by_id = {d.doc_id: d for d in load_manifest(ws)}

    def list_pages() -> str:
        pages = sorted(ws.wiki.glob("*.md"))
        if not pages:
            return "(wiki is empty)"
        return "\n".join(f"{p.name} ({p.stat().st_size}B)" for p in pages)

    def read_page(filename: str) -> str:
        path = (ws.wiki / filename).resolve()
        if ws.wiki.resolve() not in path.parents and path != ws.wiki.resolve():
            return "Refused: path escapes the wiki directory."
        if not path.exists():
            return f"(no such page: {filename})"
        return path.read_text(encoding="utf-8")

    def write_page(filename: str, content: str) -> str:
        path = (ws.wiki / filename).resolve()
        if ws.wiki.resolve() not in path.parents:
            return "Refused: path escapes the wiki directory."
        path.write_text(content, encoding="utf-8")
        return f"wrote {filename} ({len(content)} chars)"

    def read_source(doc_id: str) -> str:
        doc = docs_by_id.get(doc_id)
        if doc is None:
            return f"(no such source: {doc_id})"
        return doc.read_text()[:DOC_CHAR_LIMIT]

    return {
        "list_pages": list_pages,
        "read_page": read_page,
        "write_page": write_page,
        "read_source": read_source,
    }


def _ensure_scaffold(ws: Workspace) -> None:
    schema_page = ws.wiki / "_schema.md"
    if not schema_page.exists():
        schema_page.write_text(SCHEMA_PAGE, encoding="utf-8")
    index = ws.wiki / "index.md"
    if not index.exists():
        index.write_text("# Index\n\n(empty — populated by ingest passes)\n", encoding="utf-8")


def _ingested_marker(ws: Workspace) -> set[str]:
    marker = ws.wiki / ".ingested.json"
    if marker.exists():
        return set(json.loads(marker.read_text(encoding="utf-8")))
    return set()


def _save_marker(ws: Workspace, ids: set[str]) -> None:
    (ws.wiki / ".ingested.json").write_text(json.dumps(sorted(ids)), encoding="utf-8")


def run_wiki_ingest(ws: Workspace, routes: list[Route], schema: dict) -> int:
    """Ingest pass: integrate each connected document into the wiki."""
    _ensure_scaffold(ws)
    docs_by_id = {d.doc_id: d for d in load_manifest(ws)}
    done = _ingested_marker(ws)
    todo = [r.doc_id for r in routes if r.path == "wiki" and r.doc_id in docs_by_id and r.doc_id not in done]
    if not todo:
        return 0

    impls = _tool_impls(ws)
    system = INGEST_SYSTEM.format(goal=schema["goal"])

    for i, doc_id in enumerate(todo, 1):
        doc = docs_by_id[doc_id]
        print(f"  wiki ingest {i}/{len(todo)}: {doc.filename}")
        summary = agent_loop(
            system=system,
            user_message=(
                f"New source document [doc:{doc_id}] ({doc.filename}):\n\n"
                f"{doc.read_text()[:DOC_CHAR_LIMIT]}"
            ),
            tools=WIKI_TOOLS,
            tool_impls=impls,
            verbose=True,
        )
        print(f"    -> {summary[:200]}")
        done.add(doc_id)
        _save_marker(ws, done)

    return len(todo)


def run_wiki_lint(ws: Workspace) -> str:
    """Lint pass over the whole wiki. Also reviews entity-store conflicts."""
    _ensure_scaffold(ws)
    from .store import load_store

    store = load_store(ws)
    conflicts = {
        key: node["conflicts"]
        for key, node in store["entities"].items()
        if node.get("conflicts")
    }
    conflict_note = (
        f"\n\nThe entity store also has unresolved fact conflicts you should mention "
        f"in the report:\n{json.dumps(conflicts, indent=2)[:8000]}"
        if conflicts else ""
    )

    report = agent_loop(
        system=LINT_SYSTEM,
        user_message="Run a full lint pass over the wiki now." + conflict_note,
        tools=WIKI_TOOLS,
        tool_impls=_tool_impls(ws),
        verbose=True,
    )
    ws.lint_report.write_text(report, encoding="utf-8")
    return report
