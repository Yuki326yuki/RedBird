from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .io_utils import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from .types import Chunk, Document


PARSER_CACHE_VERSION = "paragraph-pages-v2"
CHUNK_CACHE_VERSION = "paragraph-overlap-v2"


def document_cache_fingerprint(
    allowed_paths: Optional[Set[Path]],
    metadata_by_path: Dict[Path, Dict[str, object]],
    countries: Iterable[str],
    manifest_path: Optional[Path],
) -> str:
    paths = sorted(path.resolve() for path in (allowed_paths or set()))
    sources = []
    for path in paths:
        stat = path.stat() if path.exists() else None
        sources.append(
            {
                "path": str(path),
                "exists": stat is not None,
                "size": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
                "metadata": metadata_by_path.get(path, {}),
            }
        )
    payload = {
        "parser_version": PARSER_CACHE_VERSION,
        "countries": sorted(str(country) for country in countries),
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "sources": sources,
    }
    return stable_hash(payload)


def chunk_cache_fingerprint(document_fingerprint: str, max_chars: int, overlap_chars: int) -> str:
    return stable_hash(
        {
            "chunker_version": CHUNK_CACHE_VERSION,
            "document_fingerprint": document_fingerprint,
            "max_chars": int(max_chars),
            "overlap_chars": int(overlap_chars),
        }
    )


def cache_matches(meta_path: Path, data_path: Path, fingerprint: str) -> bool:
    if not meta_path.exists() or not data_path.exists():
        return False
    try:
        return str(read_json(meta_path).get("fingerprint") or "") == fingerprint
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_cache_meta(meta_path: Path, fingerprint: str, **details: Any) -> None:
    write_json_atomic(meta_path, {"fingerprint": fingerprint, **details})


def document_sources_match(documents: Iterable[Document], allowed_paths: Optional[Set[Path]]) -> bool:
    if allowed_paths is None:
        return True
    expected = {path.resolve() for path in allowed_paths if path.exists()}
    actual = {Path(document.source_file).resolve() for document in documents}
    return actual == expected


def save_documents(path: Path, documents: Iterable[Document]) -> None:
    write_jsonl_atomic(path, [document.to_dict() for document in documents])


def load_chunks(path: Path) -> List[Chunk]:
    return [Chunk(**row) for row in read_jsonl(path)]


def save_chunks(path: Path, chunks: Iterable[Chunk]) -> None:
    write_jsonl_atomic(path, [chunk.to_dict() for chunk in chunks])


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
