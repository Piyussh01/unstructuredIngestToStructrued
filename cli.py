"""stous — a knowledge compiler. Structure is a cache.

    stous init                 create a workspace here
    stous add <paths...>       ingest files/dirs into the immutable raw layer
    stous interview            chat with the model to define what to extract
    stous compile [--workers N]  route -> parsers -> batch extract -> wiki ingest
    stous lint                 contradiction/staleness pass over wiki + store
    stous ask "question"       agent answers with code over the store + wiki
    stous status               what's compiled
    stous recompile            delete all derived structure (raw layer survives)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import DEFAULT_WORKERS, Workspace


def cmd_init(args) -> None:
    ws = Workspace(Path(args.dir).resolve())
    ws.init()
    print(f"initialized workspace at {ws.dir}")
    print("next: stous add <files>, then stous interview, then stous compile")


def cmd_add(args) -> None:
    from .ingest import ingest_paths

    ws = Workspace.find()
    added = ingest_paths(ws, [Path(p) for p in args.paths])
    print(f"ingested {len(added)} new documents into the raw layer")


def cmd_interview(args) -> None:
    from .interview import run_interview

    ws = Workspace.find()
    schema = run_interview(ws)
    n_types = len(schema.get("entity_types", []))
    print(f"schema: {n_types} entity types, goal: {schema.get('goal', '')[:100]}")


def cmd_compile(args) -> None:
    from .extract import run_extract_path
    from .ingest import load_manifest
    from .interview import load_schema
    from .parsers import run_parser_path
    from .router import load_routes, run_router
    from .store import load_store, merge_extraction, save_store, store_summary
    from .wiki import run_wiki_ingest

    ws = Workspace.find()
    schema = load_schema(ws)
    docs_by_id = {d.doc_id: d for d in load_manifest(ws)}
    if not docs_by_id:
        raise SystemExit("Nothing ingested. Run `stous add <paths>` first.")

    # Pass 1: route every document down one of the three paths.
    routes = run_router(ws, schema, args.workers)
    counts = {}
    for r in routes:
        counts[r.path] = counts.get(r.path, 0) + 1
    print(f"routes: {counts}")

    store = load_store(ws)

    # Pass 2: canonical -> LLM writes the parser once, code runs for free.
    parser_records, _fallthrough = run_parser_path(ws, routes, schema)
    for rec in parser_records:
        doc = docs_by_id.get(rec["doc_id"])
        merge_extraction(store, rec, doc.read_text() if doc else "")

    # Pass 3: self-contained prose -> concurrent small-model extraction,
    # frontier adjudication on low confidence. (Reload routes: fall-through
    # docs from the parser path are now on the extract path.)
    routes = load_routes(ws)
    extract_records = run_extract_path(ws, routes, schema, args.workers)
    for rec in extract_records:
        doc = docs_by_id.get(rec["doc_id"])
        merge_extraction(store, rec, doc.read_text() if doc else "")

    save_store(ws, store)
    print("\nentity store:")
    print(store_summary(store))

    # Pass 4: connected prose -> wiki ingest passes.
    n = run_wiki_ingest(ws, routes, schema)
    print(f"\nwiki: integrated {n} connected documents")
    print("\ncompile done. consider `stous lint` and then `stous ask '...'`")


def cmd_lint(args) -> None:
    from .wiki import run_wiki_lint

    ws = Workspace.find()
    report = run_wiki_lint(ws)
    print("\n" + report)
    print(f"\n(report saved to {ws.lint_report})")


def cmd_ask(args) -> None:
    from .query import ask

    ws = Workspace.find()
    answer = ask(ws, args.question)
    print("\n" + answer)


def cmd_status(args) -> None:
    from .ingest import load_manifest
    from .router import load_routes
    from .store import load_store, store_summary

    ws = Workspace.find()
    docs = load_manifest(ws)
    routes = load_routes(ws)
    print(f"workspace: {ws.dir}")
    print(f"raw layer: {len(docs)} documents")
    print(f"schema:    {'defined' if ws.schema.exists() else 'MISSING (run stous interview)'}")
    counts = {}
    for r in routes:
        counts[r.path] = counts.get(r.path, 0) + 1
    print(f"routed:    {counts if counts else 'none'}")
    print(f"parsers:   {len(list(ws.parsers.glob('*.py')))}")
    print("entities:")
    print(store_summary(load_store(ws)))
    wiki_pages = [p for p in ws.wiki.glob("*.md") if not p.name.startswith("_")]
    print(f"wiki:      {len(wiki_pages)} pages")


def cmd_recompile(args) -> None:
    ws = Workspace.find()
    if input("delete ALL derived structure (raw layer survives)? [y/N] ").lower() != "y":
        return
    for path in (ws.routes, ws.extractions, ws.entities, ws.lint_report):
        path.unlink(missing_ok=True)
    for d in (ws.parsers, ws.wiki):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    print("derived structure deleted. `stous compile` rebuilds everything from raw/.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="stous", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a workspace")
    p.add_argument("dir", nargs="?", default=".")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("add", help="ingest files or directories")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("interview", help="define the extraction schema in a chat")
    p.set_defaults(fn=cmd_interview)

    p = sub.add_parser("compile", help="run all compiler passes")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("lint", help="contradiction/staleness pass")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("ask", help="query the compiled knowledge base")
    p.add_argument("question")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("status", help="show what's compiled")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("recompile", help="delete derived structure; keep raw layer")
    p.set_defaults(fn=cmd_recompile)

    args = parser.parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
