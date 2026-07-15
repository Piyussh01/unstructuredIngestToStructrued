"""Query time: an agent writes code against the compiled artifacts.

No embeddings. The agent reads the wiki index-first and drills into
pages, and runs Python against the entity store in a subprocess —
returning only results, not raw data, into its context.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import Workspace
from .ingest import load_manifest
from .llm import agent_loop

QUERY_SYSTEM = """You answer questions over a compiled knowledge base. Available layers:
- entity store: typed entities with facts, relations, and provenance (query it with run_python)
- wiki: LLM-maintained markdown for connected knowledge (read index.md FIRST, then drill in)
- raw sources: immutable originals (read_source, by doc_id — use for verification only)

Strategy: for aggregations/lookups over structured facts, write code against the store.
For "why/how/history" questions, go index-first into the wiki. Cite doc ids from
provenance in your answer. If the store and wiki disagree, say so.

run_python environment: `store` is the loaded entities dict; pandas is available as pd
if installed. print() what you want to see — only stdout comes back to you."""

QUERY_TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python against the entity store in a sandbox. Variable `store` holds "
            "{'entities': {key: {type, facts, relations, provenance, conflicts, updated_at}}}. "
            "pandas may be available as pd. Only printed output is returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "read_wiki_page",
        "description": "Read a wiki page ('index.md' first).",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "read_source",
        "description": "Read a raw source document by doc_id.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
]

SANDBOX = """\
import json, sys
store = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    import pandas as pd
except ImportError:
    pd = None
exec(open(sys.argv[2], encoding="utf-8").read())
"""


def ask(ws: Workspace, question: str) -> str:
    docs_by_id = {d.doc_id: d for d in load_manifest(ws)}

    def run_python(code: str) -> str:
        if not ws.entities.exists():
            return "Entity store is empty — run `stous compile` first."
        with tempfile.TemporaryDirectory() as td:
            sandbox = Path(td) / "sandbox.py"
            user_code = Path(td) / "user_code.py"
            sandbox.write_text(SANDBOX, encoding="utf-8")
            user_code.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(sandbox), str(ws.entities), str(user_code)],
                    capture_output=True, text=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                return "Code timed out (30s)."
        out = proc.stdout[-6000:]
        if proc.returncode != 0:
            return f"Error:\n{proc.stderr[-2000:]}"
        return out or "(no output — did you print()?)"

    def read_wiki_page(filename: str) -> str:
        path = (ws.wiki / filename).resolve()
        if ws.wiki.resolve() not in path.parents:
            return "Refused: path escapes the wiki."
        if not path.exists():
            pages = ", ".join(p.name for p in sorted(ws.wiki.glob("*.md")))
            return f"(no such page; available: {pages or 'none'})"
        return path.read_text(encoding="utf-8")

    def read_source(doc_id: str) -> str:
        doc = docs_by_id.get(doc_id)
        return doc.read_text()[:40000] if doc else f"(no such source: {doc_id})"

    return agent_loop(
        system=QUERY_SYSTEM,
        user_message=question,
        tools=QUERY_TOOLS,
        tool_impls={
            "run_python": run_python,
            "read_wiki_page": read_wiki_page,
            "read_source": read_source,
        },
        verbose=True,
    )
