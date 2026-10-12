from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import read_jsonl, write_jsonl
from .models import ModelBackend
from .multilingual import (
    CLIMATE_TERMS, TARGET_13_1_ACTION_TERMS, TARGET_13_2_INTEGRATION_TERMS,
    TARGET_13_B_CAPACITY_TERMS, TARGET_TERMS,
)
from .types import Chunk


SDG13_TARGET_DEFINITIONS = {
    "13.1": "Strengthen resilience and adaptive capacity to climate-related hazards and natural disasters.",
    "13.2": "Integrate climate change measures into national policies, strategies and planning.",
    "13.3": "Improve climate education, awareness, institutional capacity and early warning.",
    "13.a": "Implement developed-country UNFCCC climate-finance commitments and the Green Climate Fund.",
    "13.b": "Build climate planning and management capacity in least developed countries and small island states.",
}


class SDG13TargetClassifier:
    def __init__(
        self,
        backend: ModelBackend,
        prompt_path: Path,
        cache_path: Path,
        min_confidence: float = 0.75,
        max_tokens: int | None = None,
    ) -> None:
        self.backend = backend
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.prompt_hash = hashlib.sha1(self.prompt.encode("utf-8")).hexdigest()[:12]
        self.cache_path = cache_path
        self.min_confidence = float(min_confidence)
        self.max_tokens = max_tokens
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            for row in read_jsonl(cache_path):
                if row.get("cache_key"):
                    self.cache[str(row["cache_key"])] = row

    def classify(self, chunk: Chunk, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        classified: List[Dict[str, Any]] = []
        for candidate in candidates:
            cache_key = self.cache_key(chunk, candidate)
            row = self.cache.get(cache_key)
            if row is None:
                response = self.backend.complete_json(
                    [
                        {"role": "system", "content": self.prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "target_definitions": SDG13_TARGET_DEFINITIONS,
                                    "source_chunk": chunk.to_prompt_dict(),
                                    "candidate": compact_candidate(candidate),
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    schema_hint="assessments: sdg_target, decision, confidence, evidence_quote, reason",
                    max_tokens=self.max_tokens,
                    stage="target_classification",
                )
                row = {
                    "cache_key": cache_key,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "model": self.backend.model_name(),
                    "candidate": candidate,
                    "assessments": response.get("assessments", []),
                }
                self.cache[cache_key] = row
            classified.extend(self.apply_assessments(chunk, candidate, row.get("assessments")))
        write_jsonl(self.cache_path, list(self.cache.values()))
        return classified

    def apply_assessments(
        self,
        chunk: Chunk,
        candidate: Dict[str, Any],
        assessments: Any,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for assessment in assessments if isinstance(assessments, list) else []:
            if not isinstance(assessment, dict):
                continue
            target = str(assessment.get("sdg_target") or "").strip().lower()
            decision = str(assessment.get("decision") or "").strip().lower()
            confidence = normalize_confidence(assessment.get("confidence"))
            quote = normalize_text(assessment.get("evidence_quote")) or normalize_text(candidate.get("evidence_quote"))
            if target not in SDG13_TARGET_DEFINITIONS or decision != "supported":
                continue
            if confidence is None or confidence < self.min_confidence:
                continue
            record = dict(candidate)
            record.update(
                {
                    "sdg_goal": "SDG 13",
                    "sdg_target": target,
                    "sdg_confidence": confidence,
                    "evidence_quote": quote,
                    "target_reason": normalize_text(assessment.get("reason")),
                }
            )
            evidence_text = " ".join(
                [str(record.get("policy_content_original") or ""), str(quote or "")]
            )
            if evidence_is_traceable(chunk.text, quote) and target_has_required_signal(target, evidence_text):
                records.append(record)
        return records

    def cache_key(self, chunk: Chunk, candidate: Dict[str, Any]) -> str:
        raw = json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "candidate": candidate,
                "model": self.backend.model_name(),
                "prompt": self.prompt_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def evidence_is_traceable(source_text: str, quote: Any) -> bool:
    needle = normalize_for_match(quote)
    return bool(needle and needle in normalize_for_match(source_text))


def target_has_required_signal(target: str, text: str) -> bool:
    value = text.lower()
    climate = has_any(value, CLIMATE_TERMS)
    if target == "13.1":
        return climate and has_any(value, TARGET_13_1_ACTION_TERMS)
    if target == "13.2":
        return climate and has_any(value, TARGET_13_2_INTEGRATION_TERMS)
    if target == "13.b":
        return climate and has_any(value, TARGET_TERMS["13.b"]) and has_any(value, TARGET_13_B_CAPACITY_TERMS)
    return target in TARGET_TERMS and climate and has_any(value, TARGET_TERMS[target])


def has_any(text: str, values: List[str]) -> bool:
    return any(value in text for value in values)


def normalize_for_match(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def normalize_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def normalize_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    fields = ["policy_content_original", "policy_content_summary", "evidence_quote"]
    return {field: candidate.get(field) for field in fields}
