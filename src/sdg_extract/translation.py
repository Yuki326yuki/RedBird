from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import read_jsonl, write_jsonl
from .models import ModelBackend


TRANSLATED_FIELDS = {
    "policy_title": "policy_title_en",
    "policy_content": "policy_content_en",
    "policy_summary": "policy_summary_en",
}


class PolicyTranslator:
    def __init__(
        self,
        backend: ModelBackend,
        prompt_path: Path,
        cache_path: Path,
        output_language: str = "English",
    ) -> None:
        self.backend = backend
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.cache_path = cache_path
        self.output_language = output_language
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            for row in read_jsonl(cache_path):
                key = row.get("cache_key")
                if key:
                    self.cache[str(key)] = row

    def translate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        translated: List[Dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            translated.append(self.translate_record(record, index))
        self.save_cache()
        return translated

    def translate_record(self, record: Dict[str, Any], index: int) -> Dict[str, Any]:
        row = dict(record)
        payload = {field: row.get(field) for field in TRANSLATED_FIELDS}
        if not any(payload.values()):
            return add_empty_translation(row, self.backend.model_name(), "skipped_empty")

        cache_key = self.cache_key(row, payload)
        cached = self.cache.get(cache_key)
        if cached:
            return apply_translation(row, cached, self.backend.model_name(), str(cached.get("translation_status") or "cached"))

        try:
            response = self.backend.complete_json(
                [
                    {"role": "system", "content": self.prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "output_language": self.output_language,
                                "record_index": index,
                                **payload,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ],
                schema_hint=", ".join(TRANSLATED_FIELDS.values()),
            )
            translated = {
                target: normalize_text(response.get(target))
                for target in TRANSLATED_FIELDS.values()
            }
            translated["cache_key"] = cache_key
            translated["translation_model"] = self.backend.model_name()
            translated["translation_status"] = "translated"
            self.cache[cache_key] = translated
            return apply_translation(row, translated, self.backend.model_name(), "translated")
        except Exception as exc:
            failed = {
                "cache_key": cache_key,
                "translation_model": self.backend.model_name(),
                "translation_status": f"error:{type(exc).__name__}",
            }
            self.cache[cache_key] = failed
            return apply_translation(row, failed, self.backend.model_name(), failed["translation_status"])

    def cache_key(self, record: Dict[str, Any], payload: Dict[str, Any]) -> str:
        raw = json.dumps(
            {
                "source_filename": record.get("source_filename"),
                "translation_model": self.backend.model_name(),
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def save_cache(self) -> None:
        write_jsonl(self.cache_path, list(self.cache.values()))


def apply_translation(
    record: Dict[str, Any],
    translated: Dict[str, Any],
    model_name: str,
    status: str,
) -> Dict[str, Any]:
    for field in TRANSLATED_FIELDS.values():
        record[field] = translated.get(field)
    record["translation_model"] = translated.get("translation_model") or model_name
    record["translation_status"] = status
    return record


def add_empty_translation(record: Dict[str, Any], model_name: str, status: str) -> Dict[str, Any]:
    for field in TRANSLATED_FIELDS.values():
        record[field] = None
    record["translation_model"] = model_name
    record["translation_status"] = status
    return record


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

