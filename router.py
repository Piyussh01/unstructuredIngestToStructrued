"""The router: two questions per document — canonical? connected?

canonical  = fields mean the same thing everywhere they appear; a
             parsing problem, not an interpretation problem.
connected  = understanding this document requires other documents.

Routing table:
    canonical                -> "parser"   (LLM writes code once)
    not canonical, isolated  -> "extract"  (small-model batch extraction)
    not canonical, connected -> "wiki"     (frontier wiki passes)

Runs concurrently on the fast model — this is a per-document cost, so it
has to be cheap. Documents of the same doc_type on the parser path get
grouped so one parser serves the whole group.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass

from .config import DOC_CHAR_LIMIT, FAST_MODEL, Workspace
from .ingest import Document, load_manifest
from .llm import structured_async

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {
            "type": "string",
            "description": "Short snake_case label for the document's kind, e.g. 'invoice', 'email_thread', 'meeting_transcript'. Documents sharing a template must share a doc_type.",
        },
        "canonical": {
            "type": "boolean",
            "description": "True if this document is templated/machine-generated enough that a regex/code parser written once would extract its fields reliably.",
        },
        "connected": {
            "type": "boolean",
            "description": "True if this document's meaning depends on other documents (references prior conversations, contracts, decisions).",
        },
        "reason": {"type": "string"},
    },
    "required": ["doc_type", "canonical", "connected", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You route documents in a knowledge compiler. Answer two questions:
1. canonical: could code parse this? (templated invoices, CSV exports, machine logs: yes. \
Freeform prose: no.)
2. connected: does its meaning live in links to other documents? (negotiation email \
threads, meeting notes referencing decisions: yes. A self-contained contract or ticket: no.)

Corpus context from the user's schema interview:
{hints}
"""


@dataclass
class Route:
    doc_id: str
    doc_type: str
    canonical: bool
    connected: bool
    path: str  # "parser" | "extract" | "wiki"
    reason: str


def _path(canonical: bool, connected: bool) -> str:
    if canonical:
        return "parser"
    return "wiki" if connected else "extract"


async def _route_one(sem: asyncio.Semaphore, doc: Document, system: str) -> Route:
    async with sem:
        text = doc.read_text()[:4000]  # an excerpt is enough to classify
        result = await structured_async(
            model=FAST_MODEL,
            system=system,
            prompt=f"Filename: {doc.filename}\n\n{text}",
            schema=ROUTE_SCHEMA,
            max_tokens=1024,
        )
    return Route(
        doc_id=doc.doc_id,
        doc_type=result["doc_type"],
        canonical=result["canonical"],
        connected=result["connected"],
        path=_path(result["canonical"], result["connected"]),
        reason=result["reason"],
    )


async def _route_all(docs: list[Document], hints: str, workers: int) -> list[Route]:
    sem = asyncio.Semaphore(workers)
    system = SYSTEM.format(hints=hints)
    return list(await asyncio.gather(*(_route_one(sem, d, system) for d in docs)))


def run_router(ws: Workspace, schema: dict, workers: int) -> list[Route]:
    docs = load_manifest(ws)
    already = {r.doc_id for r in load_routes(ws)}
    todo = [d for d in docs if d.doc_id not in already]
    if not todo:
        return load_routes(ws)

    print(f"routing {len(todo)} documents ({workers} concurrent)...")
    routes = asyncio.run(_route_all(todo, schema.get("routing_hints", ""), workers))

    with ws.routes.open("a", encoding="utf-8") as f:
        for r in routes:
            f.write(json.dumps(asdict(r)) + "\n")

    return load_routes(ws)


def load_routes(ws: Workspace) -> list[Route]:
    if not ws.routes.exists():
        return []
    return [Route(**json.loads(line)) for line in ws.routes.read_text(encoding="utf-8").splitlines() if line.strip()]


def reroute(ws: Workspace, doc_ids: set[str], new_path: str) -> None:
    """Fall-through: rewrite routes for docs whose parser failed."""
    routes = load_routes(ws)
    with ws.routes.open("w", encoding="utf-8") as f:
        for r in routes:
            if r.doc_id in doc_ids:
                r.path = new_path
                r.canonical = False
            f.write(json.dumps(asdict(r)) + "\n")
