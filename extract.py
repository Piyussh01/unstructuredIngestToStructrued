"""Path 2: non-canonical, self-contained data. Small-model batch extraction.

A pool of concurrent extractor agents (fast model, structured outputs)
runs over every document on the extract path. Each fact comes back with
a verbatim quote (grounded to a char offset at merge time) and a
confidence score.

Facts below the confidence threshold are re-examined by a frontier
adjudicator that sees the document plus the disputed extraction — the
frontier model's only per-document job in the whole compiler.
"""

from __future__ import annotations

import asyncio
import json

from .config import (
    ADJUDICATION_THRESHOLD,
    DOC_CHAR_LIMIT,
    FAST_MODEL,
    FRONTIER_MODEL,
    Workspace,
)
from .ingest import Document, load_manifest
from .interview import extraction_json_schema
from .llm import structured_async
from .router import Route

EXTRACT_SYSTEM = """You are a batch extraction worker in a knowledge compiler. Extract \
entities from the document into the required JSON shape.

Extraction goal: {goal}

Entity schema:
{schema}

Rules:
- Only extract what the document actually supports. Every fact needs a short VERBATIM
  quote copied exactly from the document text — this becomes provenance.
- Derive keys per each type's identity_hint so the same real-world entity gets the same
  key across documents (lowercase snake_case).
- confidence is your honest probability the value is correct (0-1). Ambiguous or inferred
  values get low confidence; clearly stated ones get high.
- If the document contains none of the target entities, return an empty list.
"""

ADJUDICATE_SYSTEM = """You are the adjudicator in a knowledge compiler. A fast extraction \
pass produced low-confidence facts for this document. Re-read carefully and return the
corrected extraction for ONLY the entities listed. Drop facts the document does not
actually support; fix wrong values; keep correct ones with raised confidence.

Extraction goal: {goal}

Entity schema:
{schema}
"""


async def _extract_one(
    sem: asyncio.Semaphore, doc: Document, schema: dict, out_schema: dict
) -> dict:
    text = doc.read_text()[:DOC_CHAR_LIMIT]
    system = EXTRACT_SYSTEM.format(goal=schema["goal"], schema=json.dumps(schema["entity_types"], indent=2))

    async with sem:
        result = await structured_async(
            model=FAST_MODEL,
            system=system,
            prompt=f"Filename: {doc.filename}\n\n{text}",
            schema=out_schema,
        )

    record = {"doc_id": doc.doc_id, "method": f"extract:{FAST_MODEL}", "entities": result["entities"]}

    # Frontier adjudication for low-confidence facts only.
    disputed = [
        e for e in result["entities"]
        if any(f["confidence"] < ADJUDICATION_THRESHOLD for f in e["facts"])
    ]
    if disputed:
        adj_system = ADJUDICATE_SYSTEM.format(
            goal=schema["goal"], schema=json.dumps(schema["entity_types"], indent=2)
        )
        prompt = (
            f"Filename: {doc.filename}\n\nDocument:\n{text}\n\n"
            f"Low-confidence extraction to re-check:\n{json.dumps(disputed, indent=2)}"
        )
        async with sem:
            fixed = await structured_async(
                model=FRONTIER_MODEL, system=adj_system, prompt=prompt, schema=out_schema
            )
        fixed_by_key = {(e["type"], e["key"]): e for e in fixed["entities"]}
        merged = []
        for e in record["entities"]:
            k = (e["type"], e["key"])
            if any(f["confidence"] < ADJUDICATION_THRESHOLD for f in e["facts"]) :
                if k in fixed_by_key:
                    fe = fixed_by_key[k]
                    fe["adjudicated"] = True
                    merged.append(fe)
                # adjudicator dropped it entirely -> it was unsupported
            else:
                merged.append(e)
        record["entities"] = merged
        record["method"] += "+adjudicated"

    return record


async def _extract_all(docs: list[Document], schema: dict, workers: int) -> list[dict]:
    sem = asyncio.Semaphore(workers)
    out_schema = extraction_json_schema(schema)
    tasks = [_extract_one(sem, d, schema, out_schema) for d in docs]
    records = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        try:
            rec = await coro
            records.append(rec)
            print(f"  extracted {i}/{len(tasks)}: doc {rec['doc_id']} -> {len(rec['entities'])} entities")
        except Exception as e:
            print(f"  ! extraction failed ({i}/{len(tasks)}): {e}")
    return records


def run_extract_path(ws: Workspace, routes: list[Route], schema: dict, workers: int) -> list[dict]:
    docs_by_id = {d.doc_id: d for d in load_manifest(ws)}
    done = _already_extracted(ws)
    todo = [
        docs_by_id[r.doc_id]
        for r in routes
        if r.path == "extract" and r.doc_id in docs_by_id and r.doc_id not in done
    ]
    if not todo:
        return []
    print(f"extraction pass: {len(todo)} documents, {workers} concurrent agents")
    records = asyncio.run(_extract_all(todo, schema, workers))
    with ws.extractions.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def _already_extracted(ws: Workspace) -> set[str]:
    if not ws.extractions.exists():
        return set()
    return {
        json.loads(line)["doc_id"]
        for line in ws.extractions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_extractions(ws: Workspace) -> list[dict]:
    if not ws.extractions.exists():
        return []
    return [
        json.loads(line)
        for line in ws.extractions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
