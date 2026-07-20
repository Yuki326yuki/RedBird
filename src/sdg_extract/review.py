from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import read_jsonl, write_jsonl
from .models import ModelBackend
from .schema import normalize_records
from .types import Chunk


REVIEW_DECISIONS = {"accept", "revise", "reject"}
REVISABLE_FIELDS = {
    "policy_content_original",
    "policy_content_summary",
    "sdg_target",
    "sdg_confidence",
    "evidence_quote",
}
SDG13_TARGETS = {"13.1", "13.2", "13.3", "13.a", "13.b"}


class RecordReviewer:
    def __init__(
        self,
        backend: ModelBackend,
        prompt_path: Path,
        cache_path: Path,
        keep_rejected: bool = False,
    ) -> None:
        self.backend = backend
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.cache_path = cache_path
        self.keep_rejected = keep_rejected
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            for row in read_jsonl(cache_path):
                key = row.get("cache_key")
                if key:
                    self.cache[str(key)] = row

    def review_records(
        self,
        chunk: Chunk,
        records: List[Dict[str, Any]],
        errors_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        reviewed: List[Dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            accepted, review_row = self.review_record(chunk, record, index, errors_path=errors_path)
            self.cache[str(review_row["cache_key"])] = review_row
            if accepted:
                reviewed.append(accepted)
        self.save_cache()
        return reviewed

    def review_record(
        self,
        chunk: Chunk,
        record: Dict[str, Any],
        index: int,
        errors_path: Optional[Path] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        cache_key = self.cache_key(chunk, record)
        cached = self.cache.get(cache_key)
        if cached:
            return self.apply_decision(record, cached), cached

        try:
            response = self.backend.complete_json(
                [
                    {"role": "system", "content": self.prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "record_index": index,
                                "allowed_sdg_targets": sorted(SDG13_TARGETS),
                                "chunk": chunk.to_dict(),
                                "candidate_record": record,
                            },
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        ),
                    },
                ],
                schema_hint="decision, review_confidence, rejection_reason, revised_record",
            )
            review_row = self.normalize_response(cache_key, chunk, record, response)
        except Exception as exc:
            review_row = self.error_response(cache_key, chunk, record, exc)
            append_jsonl(
                errors_path,
                {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stage": "review",
                },
            )
        return self.apply_decision(record, review_row), review_row

    def normalize_response(
        self,
        cache_key: str,
        chunk: Chunk,
        record: Dict[str, Any],
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        decision = str(response.get("decision") or "reject").strip().lower()
        if decision not in REVIEW_DECISIONS:
            decision = "reject"
        confidence = normalize_confidence(response.get("review_confidence"))
        revised_record = response.get("revised_record") if isinstance(response.get("revised_record"), dict) else None
        row = base_review_row(cache_key, chunk, record, self.backend.model_name())
        row.update(
            {
                "decision": decision,
                "review_confidence": confidence,
                "rejection_reason": normalize_text(response.get("rejection_reason")),
                "revised_record": revised_record,
            }
        )
        if decision == "revise" and not self.valid_revised_record(record, revised_record):
            row["decision"] = "reject"
            row["rejection_reason"] = "invalid_revision"
            row["revised_record"] = None
        return row

    def error_response(
        self,
        cache_key: str,
        chunk: Chunk,
        record: Dict[str, Any],
        exc: Exception,
    ) -> Dict[str, Any]:
        row = base_review_row(cache_key, chunk, record, self.backend.model_name())
        row.update(
            {
                "decision": "reject",
                "review_confidence": None,
                "rejection_reason": f"review_error:{type(exc).__name__}",
                "revised_record": None,
            }
        )
        return row

    def apply_decision(self, record: Dict[str, Any], review_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        decision = review_row.get("decision")
        if decision == "accept":
            accepted = dict(record)
            accepted["review_status"] = "reviewed"
            accepted["review_confidence"] = review_row.get("review_confidence")
            accepted["review_model"] = review_row.get("review_model")
            return accepted
        if decision == "revise":
            revised = self.build_revised_record(record, review_row.get("revised_record"))
            revised["review_status"] = "reviewed_with_revision"
            revised["review_confidence"] = review_row.get("review_confidence")
            revised["review_model"] = review_row.get("review_model")
            return revised
        if self.keep_rejected:
            rejected = dict(record)
            rejected["review_status"] = "rejected"
            rejected["review_confidence"] = review_row.get("review_confidence")
            rejected["review_model"] = review_row.get("review_model")
            return rejected
        return None

    def build_revised_record(self, record: Dict[str, Any], revised_record: Any) -> Dict[str, Any]:
        merged = dict(record)
        if isinstance(revised_record, dict):
            for field in REVISABLE_FIELDS:
                if field in revised_record:
                    merged[field] = revised_record[field]
        return normalize_records([merged], defaults=record)[0]

    def valid_revised_record(self, record: Dict[str, Any], revised_record: Any) -> bool:
        revised = self.build_revised_record(record, revised_record)
        return (
            revised.get("validation_status") == "valid"
            and revised.get("sdg_goal") == "SDG 13"
            and revised.get("sdg_target") in SDG13_TARGETS
            and bool(revised.get("policy_content_original"))
            and bool(revised.get("evidence_quote"))
        )

    def cache_key(self, chunk: Chunk, record: Dict[str, Any]) -> str:
        raw = json.dumps(
            {
                "source_filename": record.get("source_filename"),
                "chunk_id": chunk.chunk_id,
                "candidate_record": record,
                "review_model": self.backend.model_name(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def save_cache(self) -> None:
        write_jsonl(self.cache_path, list(self.cache.values()))


def base_review_row(
    cache_key: str,
    chunk: Chunk,
    record: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    return {
        "cache_key": cache_key,
        "review_model": model_name,
        "chunk_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "source_filename": record.get("source_filename"),
        "candidate_record": record,
    }


def normalize_confidence(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
