"""Configuration for the stous knowledge compiler.

Two model tiers, per the framework's economics:
- FRONTIER: writes parsers, runs wiki passes, adjudicates low-confidence
  extractions, and drives the interview. Paid rarely.
- FAST: routes documents and does batch extraction. Paid per-document,
  so it must be cheap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

FRONTIER_MODEL = os.environ.get("STOUS_FRONTIER_MODEL", "claude-opus-4-8")
FAST_MODEL = os.environ.get("STOUS_FAST_MODEL", "claude-haiku-4-5")

# Extraction confidence below this goes to the frontier adjudicator.
ADJUDICATION_THRESHOLD = float(os.environ.get("STOUS_ADJUDICATION_THRESHOLD", "0.7"))

# Max concurrent agents for batch passes.
DEFAULT_WORKERS = int(os.environ.get("STOUS_WORKERS", "8"))

# Truncation limit for document text sent to the fast model (chars).
DOC_CHAR_LIMIT = int(os.environ.get("STOUS_DOC_CHAR_LIMIT", "60000"))

WORKSPACE_DIRNAME = ".stous"


@dataclass
class Workspace:
    """Filesystem layout of a compiled knowledge base.

    raw/ is the immutable source-of-truth layer. Everything else is a
    cache: regenerable from raw/ + schema.json at any time.
    """

    root: Path

    @property
    def dir(self) -> Path:
        return self.root / WORKSPACE_DIRNAME

    @property
    def raw(self) -> Path:
        return self.dir / "raw"

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.jsonl"

    @property
    def schema(self) -> Path:
        return self.dir / "schema.json"

    @property
    def routes(self) -> Path:
        return self.dir / "routes.jsonl"

    @property
    def parsers(self) -> Path:
        return self.dir / "parsers"

    @property
    def extractions(self) -> Path:
        return self.dir / "extractions.jsonl"

    @property
    def entities(self) -> Path:
        return self.dir / "entities.json"

    @property
    def wiki(self) -> Path:
        return self.dir / "wiki"

    @property
    def lint_report(self) -> Path:
        return self.dir / "lint_report.md"

    def init(self) -> None:
        for d in (self.dir, self.raw, self.parsers, self.wiki):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.dir.is_dir()

    @classmethod
    def find(cls, start: Path | None = None) -> "Workspace":
        """Walk up from `start` looking for a .stous directory."""
        cur = (start or Path.cwd()).resolve()
        for candidate in (cur, *cur.parents):
            ws = cls(candidate)
            if ws.exists():
                return ws
        raise SystemExit("No .stous workspace found. Run `stous init` first.")
