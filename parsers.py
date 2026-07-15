"""Path 1: canonical data. The LLM writes the parser once; code runs forever.

For each canonical doc_type group, a frontier agent studies sample
documents, writes a Python `parse(text) -> dict` function, and iterates
against a test tool until it passes on the samples. The saved parser
then runs over every document of that type in a subprocess — zero LLM
cost per document.

Documents where the parser returns nothing (format variance, the
Evaporate caveat) fall through to the extraction path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from .config import FRONTIER_MODEL, Workspace
from .ingest import Document, load_manifest
from .llm import agent_loop
from .router import Route, reroute

# Harness executed in a subprocess: isolates the generated code from this
# process and enforces a timeout. The parser gets stdlib only.
RUNNER = """\
import json, sys
parser_path, text_path = sys.argv[1], sys.argv[2]
ns = {}
exec(open(parser_path).read(), ns)
text = open(text_path, encoding="utf-8", errors="replace").read()
result = ns["parse"](text)
print(json.dumps(result, default=str))
"""

PARSER_TOOLS = [
    {
        "name": "read_sample",
        "description": "Read the full text of one of the sample documents by index.",
        "input_schema": {
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    },
    {
        "name": "test_parser",
        "description": (
            "Run your candidate parser against ALL sample documents. The code must define "
            "parse(text: str) -> dict returning {'entities': [{'type', 'key', 'facts': {field: value}, "
            "'relations': [{'name', 'target_key'}]}]}. stdlib only (re, json, datetime, ...). "
            "Returns per-sample output or the error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "save_parser",
        "description": "Save the final parser. Call once, after test_parser shows correct output on every sample.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
]

SYSTEM = """You write extraction parsers inside a knowledge compiler. You get sample \
documents that all share one template ('{doc_type}'). Write ONE Python function \
parse(text) -> dict that extracts the schema fields below from any document of this type.

Target schema (extract only what applies to this doc_type):
{schema}

Requirements:
- stdlib only. Regex is your friend; be tolerant of whitespace and minor format drift.
- Return {{'entities': [...]}} as described in the test_parser tool. Keys must follow the
  schema's identity_hint so the same real-world entity gets the same key across documents.
- If a field is absent, omit it — never guess values.
- If the document doesn't match the template at all, return {{'entities': []}} so the
  compiler can fall this document through to LLM extraction.
- Iterate: read samples, test, fix, test again. Save only when output is right on all samples.
"""


def _run_parser(parser_path: Path, text_path: Path, timeout: int = 20) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(RUNNER)
        runner_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, runner_path, str(parser_path), str(text_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "parser timed out"
    finally:
        Path(runner_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip()[-2000:]
    return True, proc.stdout.strip()


def write_parser(ws: Workspace, doc_type: str, samples: list[Document], schema: dict) -> Path | None:
    """Run the parser-writer agent for one canonical doc_type. Returns saved path."""
    parser_path = ws.parsers / f"{doc_type}.py"
    if parser_path.exists():
        return parser_path

    saved: list[str] = []

    def read_sample(index: int) -> str:
        return samples[index].read_text()[:30000]

    def test_parser(code: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            candidate = Path(f.name)
        try:
            reports = []
            for i, doc in enumerate(samples):
                ok, out = _run_parser(candidate, Path(doc.text_path))
                reports.append(f"--- sample {i} ({doc.filename}) ---\n{'OK' if ok else 'FAIL'}: {out[:3000]}")
            return "\n".join(reports)
        finally:
            candidate.unlink(missing_ok=True)

    def save_parser(code: str) -> str:
        parser_path.write_text(code, encoding="utf-8")
        saved.append(code)
        return f"Saved to {parser_path}"

    print(f"  parser agent: doc_type={doc_type} ({len(samples)} samples)")
    agent_loop(
        system=SYSTEM.format(doc_type=doc_type, schema=json.dumps(schema, indent=2)),
        user_message=f"There are {len(samples)} samples (index 0..{len(samples) - 1}). Begin.",
        tools=PARSER_TOOLS,
        tool_impls={"read_sample": read_sample, "test_parser": test_parser, "save_parser": save_parser},
        model=FRONTIER_MODEL,
        verbose=True,
    )
    return parser_path if saved else None


def run_parser_path(ws: Workspace, routes: list[Route], schema: dict) -> tuple[list[dict], set[str]]:
    """Write parsers per canonical group and run them over all group docs.

    Returns (extraction records, doc_ids that failed and fall through).
    """
    docs_by_id = {d.doc_id: d for d in load_manifest(ws)}
    groups: dict[str, list[Document]] = defaultdict(list)
    for r in routes:
        if r.path == "parser" and r.doc_id in docs_by_id:
            groups[r.doc_type].append(docs_by_id[r.doc_id])

    records: list[dict] = []
    fallthrough: set[str] = set()

    for doc_type, docs in groups.items():
        parser_path = write_parser(ws, doc_type, docs[:3], schema)
        if parser_path is None:
            print(f"  ! agent produced no parser for {doc_type}; falling {len(docs)} docs through")
            fallthrough.update(d.doc_id for d in docs)
            continue

        for doc in docs:
            ok, out = _run_parser(parser_path, Path(doc.text_path))
            entities = []
            if ok:
                try:
                    entities = json.loads(out).get("entities", [])
                except json.JSONDecodeError:
                    ok = False
            if ok and entities:
                records.append({
                    "doc_id": doc.doc_id,
                    "method": f"parser:{doc_type}",
                    "entities": [
                        {
                            "type": e.get("type", doc_type),
                            "key": e.get("key", doc.doc_id),
                            "facts": [
                                {"field": k, "value": str(v), "quote": "", "confidence": 1.0}
                                for k, v in (e.get("facts") or {}).items()
                            ],
                            "relations": e.get("relations", []),
                        }
                        for e in entities
                    ],
                })
            else:
                fallthrough.add(doc.doc_id)

    if fallthrough:
        print(f"  {len(fallthrough)} docs fell through parser path -> extraction")
        reroute(ws, fallthrough, "extract")

    return records, fallthrough
