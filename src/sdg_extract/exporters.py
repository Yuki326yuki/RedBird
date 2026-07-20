from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import ensure_dir, write_jsonl
from .translation import PolicyTranslator


POLICY_OUTPUT_FIELDS = [
    "country",
    "country_iso3",
    "policy_title",
    "policy_title_en",
    "publication_date",
    "policy_content",
    "policy_content_en",
    "policy_summary",
    "policy_summary_en",
    "sdg_goal",
    "sdg_targets",
    "source_filename",
    "source_file",
    "record_count",
    "review_status",
    "review_confidence",
    "translation_model",
    "translation_status",
    "validation_status",
]


def export_records(
    records: List[Dict[str, Any]],
    output_dir: Path,
    formats: Iterable[str],
    translator: Optional[PolicyTranslator] = None,
) -> List[Path]:
    ensure_dir(output_dir)
    policy_records = aggregate_policy_records(records)
    if translator:
        policy_records = translator.translate_records(policy_records)
    written: List[Path] = []
    for fmt in formats:
        normalized = fmt.strip().lower()
        if not normalized:
            continue
        if normalized == "jsonl":
            path = output_dir / "records.jsonl"
            write_jsonl(path, project_records(policy_records))
            written.append(path)
        elif normalized == "csv":
            path = output_dir / "records.csv"
            write_csv(path, project_records(policy_records))
            written.append(path)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
    return written


def aggregate_policy_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        filename = str(record.get("source_filename") or "").strip()
        country = str(record.get("country_iso3") or record.get("country") or "").strip()
        source_file = str(record.get("source_file") or "").strip()
        key = "::".join(part for part in [country, filename] if part) or source_file
        if not key:
            key = f"unknown:{len(grouped)}"
        grouped.setdefault(key, []).append(record)

    return [aggregate_group(group_records) for group_records in grouped.values()]


def aggregate_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = records[0]
    statuses = {str(record.get("validation_status") or "") for record in records}
    valid_statuses = {status for status in statuses if status}
    return {
        "country": first_non_empty(records, "country"),
        "country_iso3": first_non_empty(records, "country_iso3"),
        "policy_title": first_non_empty(records, "policy_title"),
        "publication_date": first_non_empty(records, "publication_date"),
        "policy_content": join_unique(records, "policy_content_original"),
        "policy_summary": join_unique(records, "policy_content_summary"),
        "sdg_goal": aggregate_sdg_goal(records),
        "sdg_targets": join_values(sorted_unique(records, "sdg_target")),
        "source_filename": first_non_empty(records, "source_filename"),
        "source_file": first_non_empty(records, "source_file"),
        "record_count": len(records),
        "review_status": aggregate_review_status(records),
        "review_confidence": aggregate_review_confidence(records),
        "validation_status": "valid" if valid_statuses == {"valid"} else "warning:mixed_validation_status",
    }


def aggregate_sdg_goal(records: List[Dict[str, Any]]) -> Any:
    goals = sorted_unique(records, "sdg_goal")
    if not goals:
        return None
    if goals == ["SDG 13"]:
        return "SDG 13"
    return join_values(goals)


def aggregate_review_status(records: List[Dict[str, Any]]) -> str | None:
    statuses = {str(record.get("review_status") or "").strip() for record in records}
    statuses.discard("")
    if not statuses:
        return None
    if statuses == {"reviewed"}:
        return "reviewed"
    if statuses <= {"reviewed", "reviewed_with_revision"}:
        return "reviewed_with_revisions" if "reviewed_with_revision" in statuses else "reviewed"
    return "warning:mixed_review_status"


def aggregate_review_confidence(records: List[Dict[str, Any]]) -> float | None:
    values: List[float] = []
    for record in records:
        value = record.get("review_confidence")
        if value in {None, ""}:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def first_non_empty(records: List[Dict[str, Any]], field: str) -> Any:
    for record in records:
        value = record.get(field)
        if value not in {None, ""}:
            return value
    return None


def sorted_unique(records: List[Dict[str, Any]], field: str) -> List[str]:
    values = {str(record.get(field)).strip() for record in records if record.get(field) not in {None, ""}}
    return sorted(values, key=sort_sdg_target)


def join_unique(records: List[Dict[str, Any]], field: str) -> str | None:
    seen = set()
    values: List[str] = []
    for record in records:
        value = normalize_space(record.get(field))
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return "\n\n---\n\n".join(values) if values else None


def join_values(values: List[str]) -> str | None:
    return "; ".join(values) if values else None


def sort_sdg_target(value: str) -> tuple[int, str]:
    order = {"13.1": 1, "13.2": 2, "13.3": 3, "13.a": 4, "13.b": 5}
    return (order.get(value, 999), value)


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def project_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{field: record.get(field) for field in POLICY_OUTPUT_FIELDS} for record in records]


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POLICY_OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {field: stringify(record.get(field)) for field in POLICY_OUTPUT_FIELDS}
            writer.writerow(row)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={item}" for key, item in value.items())
    return str(value)
