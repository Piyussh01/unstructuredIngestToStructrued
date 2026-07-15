"""The typed entity store: where paths 1 and 2 land.

Shape (per the framework):
{
  "entities": {
    "acme_corp": {
      "type": "client",
      "facts": {"payment_terms": "net_30"},
      "relations": {"invoices": ["inv_201"]},
      "provenance": {"payment_terms": [{"doc_id": "...", "quote": "...", "offset": 123}]},
      "conflicts": {"payment_terms": [...older contested values...]},
      "updated_at": "..."
    }
  }
}

Every fact points back to the raw documents it came from. Conflicting
values are kept, not silently overwritten — the lint pass adjudicates.
The whole file is a cache: delete it and recompile from raw/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import Workspace


def load_store(ws: Workspace) -> dict:
    if ws.entities.exists():
        return json.loads(ws.entities.read_text(encoding="utf-8"))
    return {"entities": {}}


def save_store(ws: Workspace, store: dict) -> None:
    ws.entities.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def _ground_quote(quote: str, doc_text: str) -> int | None:
    """Char-offset provenance: locate the supporting quote in the source."""
    if not quote:
        return None
    idx = doc_text.find(quote)
    if idx == -1:
        idx = doc_text.lower().find(quote.lower())
    return idx if idx != -1 else None


def merge_extraction(store: dict, record: dict, doc_text: str = "") -> None:
    """Merge one document's extraction record into the store.

    record: {"doc_id", "method", "entities": [{"type","key","facts":[...],"relations":[...]}]}
    """
    now = datetime.now(timezone.utc).isoformat()
    doc_id = record["doc_id"]

    for ent in record.get("entities", []):
        key = ent["key"].strip().lower().replace(" ", "_")
        if not key:
            continue
        node = store["entities"].setdefault(key, {
            "type": ent["type"],
            "facts": {},
            "relations": {},
            "provenance": {},
            "conflicts": {},
            "updated_at": now,
        })
        node["updated_at"] = now

        for fact in ent.get("facts", []):
            field, value = fact["field"], fact["value"]
            prov_entry = {
                "doc_id": doc_id,
                "method": record.get("method", ""),
                "quote": fact.get("quote", ""),
                "offset": _ground_quote(fact.get("quote", ""), doc_text),
                "confidence": fact.get("confidence", 1.0),
            }
            existing = node["facts"].get(field)
            if existing is not None and existing != value:
                # Keep the contradiction visible for the lint pass.
                node["conflicts"].setdefault(field, []).append(
                    {"value": existing, "provenance": node["provenance"].get(field, [])}
                )
            node["facts"][field] = value
            node["provenance"].setdefault(field, []).append(prov_entry)

        for rel in ent.get("relations", []):
            targets = node["relations"].setdefault(rel["name"], [])
            target = rel["target_key"].strip().lower().replace(" ", "_")
            if target and target not in targets:
                targets.append(target)


def store_summary(store: dict) -> str:
    by_type: dict[str, int] = {}
    n_conflicts = 0
    for node in store["entities"].values():
        by_type[node["type"]] = by_type.get(node["type"], 0) + 1
        n_conflicts += sum(len(v) for v in node.get("conflicts", {}).values())
    lines = [f"{count:5d}  {t}" for t, count in sorted(by_type.items())]
    lines.append(f"{n_conflicts:5d}  unresolved fact conflicts")
    return "\n".join(lines) if store["entities"] else "(empty)"
