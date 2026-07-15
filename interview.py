"""The interview: talk with the model to define what to extract.

Instead of asking the user to hand-write a JSON schema, we run a short
conversation. The model sees a sample of the ingested corpus, asks the
user what they're trying to get out of it, and when it has enough it
calls `finalize_schema` — that tool call IS the schema definition.

Output: .stous/schema.json with entity types, per-type fields, and
routing hints the router/extractor/wiki passes all read.
"""

from __future__ import annotations

import json
import random

from .config import FRONTIER_MODEL, Workspace
from .ingest import load_manifest
from .llm import client, stream_chat

FINALIZE_TOOL = {
    "name": "finalize_schema",
    "description": (
        "Call this ONLY when you and the user have agreed on the extraction target. "
        "This ends the interview and writes the schema that all compiler passes use."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "One-paragraph statement of what the user wants extracted and why.",
            },
            "entity_types": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case type name, e.g. 'invoice', 'client'"},
                        "description": {"type": "string"},
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string", "enum": ["string", "number", "boolean", "date", "list"]},
                                    "description": {"type": "string"},
                                },
                                "required": ["name", "type", "description"],
                            },
                        },
                        "identity_hint": {
                            "type": "string",
                            "description": "How to derive a stable key for deduplication, e.g. 'invoice number' or 'lowercased company name'.",
                        },
                    },
                    "required": ["name", "description", "fields", "identity_hint"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_type": {"type": "string"},
                        "to_type": {"type": "string"},
                        "name": {"type": "string", "description": "e.g. 'invoices', 'contracts', 'mentioned_in'"},
                    },
                    "required": ["from_type", "to_type", "name"],
                },
            },
            "routing_hints": {
                "type": "string",
                "description": (
                    "Guidance for the router: which kinds of documents in this corpus are canonical "
                    "(templated, machine-generated, parseable with code) and which are connected prose "
                    "that belongs in the wiki."
                ),
            },
        },
        "required": ["goal", "entity_types", "relations", "routing_hints"],
    },
}

SYSTEM = """You are the schema-definition step of a knowledge compiler. The user has \
ingested a pile of unstructured documents. Your job is a short, sharp interview: \
figure out what structure they want extracted, then call finalize_schema.

Rules:
- You have a sample of their corpus below. Ground your questions in what you actually see.
- Propose a concrete starting schema early — users react better to drafts than blank questions.
- Ask about: what decisions/queries this data should support, which fields matter, how \
entities should be deduplicated (identity), and anything ambiguous in the sample.
- Keep it to a few turns. When the user confirms, call finalize_schema. Do not call it \
before the user has confirmed a draft.

Corpus sample:
{sample}
"""


def _corpus_sample(ws: Workspace, n_docs: int = 6, chars_per_doc: int = 1500) -> str:
    docs = load_manifest(ws)
    if not docs:
        return "(no documents ingested yet)"
    picked = random.sample(docs, min(n_docs, len(docs)))
    parts = []
    for d in picked:
        parts.append(f"--- {d.filename} ({d.n_chars} chars) ---\n{d.read_text()[:chars_per_doc]}")
    return "\n\n".join(parts)


def run_interview(ws: Workspace) -> dict:
    """Interactive loop. Returns the finalized schema (also written to disk)."""
    system = SYSTEM.format(sample=_corpus_sample(ws))
    messages: list[dict] = [
        {"role": "user", "content": "Let's define what to extract from this corpus. Start by telling me what you see and proposing a draft."}
    ]

    print("── interview: describe what you want extracted (Ctrl-C to abort) ──\n")

    while True:
        response = stream_chat(system=system, messages=messages, tools=[FINALIZE_TOOL])

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if tool_uses:
            schema = tool_uses[0].input
            ws.schema.write_text(json.dumps(schema, indent=2), encoding="utf-8")
            # Close the loop with the API so the turn is well-formed.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_uses[0].id,
                    "content": "Schema saved.",
                }],
            })
            client().messages.create(
                model=FRONTIER_MODEL,
                max_tokens=1024,
                system=system,
                tools=[FINALIZE_TOOL],
                messages=messages,
            )
            print(f"\n✓ schema written to {ws.schema}")
            return schema

        messages.append({"role": "assistant", "content": response.content})
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\ninterview aborted — no schema written")
        if not user_input:
            user_input = "(no comment — proceed as you think best)"
        messages.append({"role": "user", "content": user_input})


def load_schema(ws: Workspace) -> dict:
    if not ws.schema.exists():
        raise SystemExit("No schema found. Run `stous interview` first.")
    return json.loads(ws.schema.read_text(encoding="utf-8"))


def extraction_json_schema(schema: dict) -> dict:
    """Build the JSON schema the batch extractor constrains its output to.

    Every extracted fact carries a verbatim quote so provenance can be
    grounded to character offsets in the source (LangExtract-style).
    """
    type_names = [t["name"] for t in schema["entity_types"]] or ["entity"]
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": type_names},
                        "key": {
                            "type": "string",
                            "description": "Stable dedup key per the schema's identity_hint (lowercase, snake_case).",
                        },
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "value": {"type": "string"},
                                    "quote": {
                                        "type": "string",
                                        "description": "Short verbatim quote from the document supporting this value.",
                                    },
                                    "confidence": {"type": "number"},
                                },
                                "required": ["field", "value", "quote", "confidence"],
                            },
                        },
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "target_key": {"type": "string"},
                                },
                                "required": ["name", "target_key"],
                            },
                        },
                    },
                    "required": ["type", "key", "facts", "relations"],
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }
