from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


OUTPUT_FIELDS = [
    "country",
    "country_iso3",
    "policy_title",
    "publication_date",
    "policy_content_original",
    "policy_content_summary",
    "sdg_goal",
    "sdg_target",
    "sdg_confidence",
    "evidence_quote",
    "source_filename",
    "source_file",
    "validation_status",
]


INTERNAL_FIELDS = [
    "language",
    "chunk_id",
    "evidence_page",
    "evidence_page_end",
    "evidence_paragraph_ids",
    "extraction_model",
    "target_reason",
    "review_status",
    "review_confidence",
    "review_model",
]


REQUIRED_FIELDS = [
    "country",
    "policy_title",
    "policy_content_original",
    "sdg_goal",
    "sdg_target",
    "evidence_quote",
]


def normalize_records(
    raw_records: Iterable[Dict[str, Any]],
    defaults: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in raw_records:
        record = {field: normalize_empty(raw.get(field)) for field in OUTPUT_FIELDS}
        for field in INTERNAL_FIELDS:
            if field in raw:
                record[field] = normalize_empty(raw.get(field))
        if defaults:
            for key, value in defaults.items():
                if key in OUTPUT_FIELDS + INTERNAL_FIELDS and record.get(key) in {None, ""}:
                    record[key] = normalize_empty(value)
        record["validation_status"] = validate_record(record)
        normalized.append(record)
    return normalized


def normalize_empty(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def validate_record(record: Dict[str, Any]) -> str:
    errors: List[str] = []

    for field in REQUIRED_FIELDS:
        if not record.get(field):
            errors.append(f"missing_{field}")

    if not record.get("source_file"):
        errors.append("missing_source_file")
    if not record.get("source_filename"):
        errors.append("missing_source_filename")

    goal = record.get("sdg_goal")
    if goal and not valid_sdg_goal(str(goal)):
        errors.append("invalid_sdg_goal")

    target = record.get("sdg_target")
    if target and not valid_sdg_target(str(target)):
        errors.append("invalid_sdg_target")

    confidence = record.get("sdg_confidence")
    if confidence is not None:
        try:
            value = float(confidence)
            if not 0 <= value <= 1:
                errors.append("invalid_sdg_confidence")
            record["sdg_confidence"] = value
        except (TypeError, ValueError):
            errors.append("invalid_sdg_confidence")

    if errors:
        return "error:" + ",".join(errors)
    return "valid"


def valid_sdg_goal(value: str) -> bool:
    match = re.fullmatch(r"SDG\s*([1-9]|1[0-7])", value.strip(), flags=re.IGNORECASE)
    return bool(match)


def valid_sdg_target(value: str) -> bool:
    stripped = value.strip()
    return bool(
        re.fullmatch(r"([1-9]|1[0-7])\.[0-9]+[a-z]?", stripped, flags=re.IGNORECASE)
        or re.fullmatch(r"13\.[ab]", stripped, flags=re.IGNORECASE)
    )



def output_schema_prompt() -> str:
    return "\n".join(f"- {field}" for field in OUTPUT_FIELDS)


def evidence_candidate_schema_prompt() -> str:
    return "\n".join(
        f"- {field}"
        for field in [
            "policy_content_original",
            "policy_content_summary",
            "evidence_quote",
        ]
    )
