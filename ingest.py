"""Ingestion: copy anything into the immutable raw layer.

The raw layer is content-addressed (sha256) and never modified after
write. Every downstream artifact — parsers, extractions, entities, wiki —
points back here via doc ids, and can be deleted and recompiled.

Readers normalize each format to plain text for the LLM passes. The
original bytes are always preserved alongside the normalized text.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Workspace

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".log", ".py", ".js", ".ts", ".yaml", ".yml", ".toml"}


@dataclass
class Document:
    doc_id: str          # sha256[:16] of the raw bytes
    source_path: str     # where it came from
    filename: str
    ext: str
    raw_path: str        # path inside .stous/raw/
    text_path: str       # normalized text alongside raw
    n_chars: int
    ingested_at: str

    def read_text(self) -> str:
        return Path(self.text_path).read_text(encoding="utf-8", errors="replace")


def _normalize(path: Path, data: bytes) -> str:
    """Best-effort conversion of a file to plain text."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit(
                f"{path}: PDF ingestion needs pypdf. Install with: pip install 'stous[pdf]'"
            )
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if ext in {".html", ".htm"}:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Crude fallback: strip tags poorly rather than fail.
            import re
            return re.sub(r"<[^>]+>", " ", data.decode("utf-8", errors="replace"))
        return BeautifulSoup(data, "html.parser").get_text(separator="\n")

    if ext == ".csv":
        text = data.decode("utf-8", errors="replace")
        # Re-render as TSV-ish lines so the LLM sees aligned rows.
        rows = list(csv.reader(io.StringIO(text)))
        return "\n".join("\t".join(row) for row in rows)

    if ext == ".json":
        try:
            return json.dumps(json.loads(data), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return data.decode("utf-8", errors="replace")

    # Everything else: treat as text.
    return data.decode("utf-8", errors="replace")


def ingest_paths(ws: Workspace, paths: list[Path]) -> list[Document]:
    """Copy files into the raw layer and append records to the manifest.

    Idempotent: re-ingesting the same bytes is a no-op (content hash).
    """
    existing = {d.doc_id for d in load_manifest(ws)}
    added: list[Document] = []

    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*")) if f.is_file() and not f.name.startswith("."))
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip (not found): {p}")

    with ws.manifest.open("a", encoding="utf-8") as manifest:
        for f in files:
            data = f.read_bytes()
            doc_id = hashlib.sha256(data).hexdigest()[:16]
            if doc_id in existing:
                continue
            try:
                text = _normalize(f, data)
            except SystemExit:
                raise
            except Exception as e:  # unreadable file — record it, don't crash the batch
                print(f"skip (unreadable): {f} ({e})")
                continue

            raw_dest = ws.raw / f"{doc_id}{f.suffix.lower()}"
            text_dest = ws.raw / f"{doc_id}.txt"
            shutil.copyfile(f, raw_dest)
            text_dest.write_text(text, encoding="utf-8")

            doc = Document(
                doc_id=doc_id,
                source_path=str(f.resolve()),
                filename=f.name,
                ext=f.suffix.lower(),
                raw_path=str(raw_dest),
                text_path=str(text_dest),
                n_chars=len(text),
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
            manifest.write(json.dumps(asdict(doc)) + "\n")
            existing.add(doc_id)
            added.append(doc)

    return added


def load_manifest(ws: Workspace) -> list[Document]:
    if not ws.manifest.exists():
        return []
    docs = []
    for line in ws.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(Document(**json.loads(line)))
    return docs
