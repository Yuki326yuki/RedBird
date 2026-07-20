from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


MANIFEST_METADATA_FIELDS = {
    "country",
    "country_iso3",
    "language",
    "policy_title",
    "policy_type",
    "publication_date",
    "source_filename",
}


def load_manifest_selection(
    manifest_path: Path,
    project_root: Path,
    countries: Optional[Iterable[str]] = None,
) -> Tuple[Set[Path], Dict[Path, Dict[str, Any]]]:
    wanted = {country.strip() for country in countries or [] if country and country.strip()}
    allowed_paths: Set[Path] = set()
    metadata_by_path: Dict[Path, Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if wanted and row.get("country") not in wanted:
                continue
            local_path = row.get("local_path") or ""
            if not local_path:
                continue
            path = Path(local_path)
            if not path.is_absolute():
                path = project_root / path
            resolved = path.resolve()
            allowed_paths.add(resolved)
            metadata_by_path[resolved] = row_to_metadata(row, resolved)
    return allowed_paths, metadata_by_path


def row_to_metadata(row: Dict[str, str], path: Path) -> Dict[str, Any]:
    policy_title = row.get("document_title") or row.get("family_title") or path.stem
    return {
        "country": row.get("country") or None,
        "country_iso3": row.get("country_iso3") or None,
        "language": row.get("language") or None,
        "policy_title": policy_title,
        "policy_type": normalize_policy_type(row.get("document_type") or row.get("family_category") or ""),
        "publication_date": normalize_publication_date(row.get("published_date") or ""),
        "source_filename": path.name,
    }


def normalize_publication_date(value: str) -> Optional[str]:
    if not value:
        return None
    return value.split("T", 1)[0]


def normalize_policy_type(value: str) -> Optional[str]:
    lower = value.strip().lower()
    if not lower:
        return None
    mapping = {
        "act": "law",
        "law": "law",
        "strategy": "strategy",
        "plan": "plan",
        "regulation": "regulation",
        "guideline": "guideline",
        "budget": "budget",
        "report": "report",
    }
    return mapping.get(lower, lower if lower in set(mapping.values()) else "unknown")
