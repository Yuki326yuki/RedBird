from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from .chunking import chunk_documents
from .io_utils import read_jsonl, write_jsonl
from .models import ModelBackend
from .multilingual import CLIMATE_TERMS, TARGET_TERMS
from .review import RecordReviewer
from .schema import OUTPUT_FIELDS, evidence_candidate_schema_prompt, normalize_records, output_schema_prompt
from .targeting import SDG13TargetClassifier, target_has_required_signal
from .types import Chunk, Document, Paragraph


SDG13_TARGETS = {"13.1", "13.2", "13.3", "13.a", "13.b"}
SDG13_TARGET_KEYWORDS = TARGET_TERMS
SDG13_KEYWORDS = CLIMATE_TERMS


def load_documents(path: Path) -> List[Document]:
    documents: List[Document] = []
    for item in read_jsonl(path):
        paragraphs = [Paragraph(**paragraph) for paragraph in item.get("paragraphs", [])]
        documents.append(
            Document(
                document_id=item["document_id"],
                source_file=item["source_file"],
                source_type=item["source_type"],
                title=item.get("title"),
                language=item.get("language"),
                paragraphs=paragraphs,
                metadata=item.get("metadata", {}),
            )
        )
    return documents


def extract_records(
    documents: Iterable[Document],
    backend: ModelBackend,
    prompt_dir: Path,
    max_chars: int,
    overlap_chars: int = 0,
    records_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    errors_path: Optional[Path] = None,
    max_chunks: Optional[int] = None,
    sdg13_filter: bool = False,
    strict_sdg13: bool = False,
    balance_by: Optional[str] = None,
    neighbor_chunks: int = 0,
    reviewer: Optional[RecordReviewer] = None,
    target_classifier: Optional[SDG13TargetClassifier] = None,
) -> List[Dict[str, Any]]:
    system_prompt = read_prompt(prompt_dir / "system.md")
    extraction_prompt = read_prompt(prompt_dir / "extraction.md")
    schema_hint = evidence_candidate_schema_prompt() if target_classifier else output_schema_prompt()
    all_chunks = chunk_documents(documents, max_chars=max_chars, overlap_chars=overlap_chars)
    chunks = select_extraction_chunks(
        all_chunks,
        max_chunks=max_chunks,
        sdg13_filter=sdg13_filter,
        balance_by=balance_by,
        neighbor_chunks=neighbor_chunks,
    )
    total_chunks = len(chunks)
    completed_chunks = load_completed_chunks(checkpoint_path)
    records = load_extracted_records(records_path) if records_path and records_path.exists() else []
    if strict_sdg13:
        records = dedupe_records(filter_strict_sdg13_records(records))
    started_at = time.time()
    if total_chunks:
        print(
            f"[PROGRESS] Starting extraction: {total_chunks} chunks "
            f"(source_chunks={len(all_chunks)}, resume_from={completed_chunks + 1}, "
            f"existing_records={len(records)}, sdg13_filter={sdg13_filter}, strict={strict_sdg13})",
            flush=True,
        )
    for index, chunk in enumerate(chunks, start=1):
        if index <= completed_chunks:
            continue
        chunk_started_at = time.time()
        try:
            normalized = extract_chunk_records(
                chunk=chunk,
                backend=backend,
                system_prompt=system_prompt,
                extraction_prompt=extraction_prompt,
                schema_hint=schema_hint,
                strict_sdg13=strict_sdg13,
                target_classifier=target_classifier,
            )
            if reviewer:
                before_review = len(normalized)
                normalized = reviewer.review_records(chunk, normalized, errors_path=errors_path)
                print(
                    "[REVIEW] "
                    f"chunk {index}/{total_chunks} reviewed "
                    f"accepted={len(normalized)}/{before_review}",
                    flush=True,
                )
        except Exception as exc:
            normalized = []
            append_jsonl(
                errors_path,
                {
                    "chunk_index": index,
                    "chunk_id": chunk.chunk_id,
        "language": chunk.metadata.get("language"),
                    "source_file": chunk.source_file,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(
                "[ERROR] "
                f"chunk {index}/{total_chunks} failed; recorded error and continuing. "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
        records.extend(normalized)
        if strict_sdg13:
            records = dedupe_records(filter_strict_sdg13_records(records))
        if records_path:
            save_extracted_records(records_path, records)
        save_checkpoint(checkpoint_path, index, total_chunks, len(records))
        elapsed = time.time() - started_at
        chunk_elapsed = time.time() - chunk_started_at
        percent = index / total_chunks * 100 if total_chunks else 100.0
        print(
            "[PROGRESS] "
            f"chunk {index}/{total_chunks} ({percent:5.1f}%) "
            f"records+={len(normalized)} total_records={len(records)} "
            f"chunk_sec={chunk_elapsed:.1f} elapsed_min={elapsed / 60:.1f} "
            f"source={chunk.source_file}",
            flush=True,
        )
    return records


def select_extraction_chunks(
    chunks: List[Chunk],
    max_chunks: Optional[int],
    sdg13_filter: bool,
    balance_by: Optional[str],
    neighbor_chunks: int = 0,
) -> List[Chunk]:
    if not sdg13_filter:
        return chunks[: int(max_chunks)] if max_chunks is not None else list(chunks)

    scored = []
    for chunk in chunks:
        scores = score_sdg13_chunk(chunk)
        if scores:
            best_target = max(scores, key=scores.get)
            scored.append((chunk, best_target, scores[best_target]))
    print(f"[FILTER] SDG13 candidate chunks: {len(scored)}/{len(chunks)}", flush=True)
    if balance_by == "country_file":
        ranked = round_robin_scored_chunks(scored)
    else:
        ranked = [item[0] for item in sorted(scored, key=lambda item: item[2], reverse=True)]
    return include_neighbor_chunks(chunks, ranked, max(0, int(neighbor_chunks)), max_chunks)


def score_sdg13_chunk(chunk: Chunk) -> Dict[str, float]:
    text = chunk.text.lower()
    generic_hits = sum(1 for keyword in SDG13_KEYWORDS if keyword.lower() in text)
    scores: Dict[str, float] = {}
    for target, keywords in SDG13_TARGET_KEYWORDS.items():
        specific_hits = sum(1 for keyword in keywords if keyword.lower() in text)
        if specific_hits:
            scores[target] = specific_hits * 3.0 + min(generic_hits, 5) * 0.25
    if not scores and generic_hits:
        scores["general"] = float(generic_hits)
    return scores


def is_sdg13_relevant_chunk(chunk: Chunk) -> bool:
    return bool(score_sdg13_chunk(chunk))


def round_robin_scored_chunks(scored: List[Tuple[Chunk, str, float]]) -> List[Chunk]:
    groups: Dict[Tuple[str, str, str], Deque[Tuple[Chunk, float]]] = defaultdict(deque)
    for chunk, target, score in sorted(scored, key=lambda item: item[2], reverse=True):
        metadata = chunk.metadata or {}
        key = (
            str(metadata.get("country") or ""),
            str(metadata.get("source_filename") or chunk.source_file),
            target,
        )
        groups[key].append((chunk, score))
    queues: Deque[Tuple[str, str, str]] = deque(sorted(groups))
    ordered: List[Chunk] = []
    while queues:
        key = queues.popleft()
        chunk, _ = groups[key].popleft()
        ordered.append(chunk)
        if groups[key]:
            queues.append(key)
    return ordered


def include_neighbor_chunks(
    all_chunks: List[Chunk],
    ranked: List[Chunk],
    neighbor_count: int,
    max_chunks: Optional[int],
) -> List[Chunk]:
    positions = {chunk.chunk_id: index for index, chunk in enumerate(all_chunks)}
    selected: List[Chunk] = []
    seen = set()
    limit = int(max_chunks) if max_chunks is not None else None
    for chunk in ranked:
        index = positions[chunk.chunk_id]
        candidate_indexes = [index]
        for offset in range(1, neighbor_count + 1):
            candidate_indexes.extend([index - offset, index + offset])
        for candidate_index in candidate_indexes:
            if not 0 <= candidate_index < len(all_chunks):
                continue
            candidate = all_chunks[candidate_index]
            if candidate.document_id != chunk.document_id or candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            selected.append(candidate)
            if limit is not None and len(selected) >= limit:
                return selected
    return selected


def round_robin_by_country_file(chunks: List[Chunk]) -> List[Chunk]:
    groups: Dict[Tuple[str, str], Deque[Chunk]] = defaultdict(deque)
    for chunk in chunks:
        metadata = chunk.metadata or {}
        key = (
            str(metadata.get("country") or ""),
            str(metadata.get("source_filename") or chunk.source_file),
        )
        groups[key].append(chunk)
    queues: Deque[Tuple[str, str]] = deque(sorted(groups))
    ordered: List[Chunk] = []
    while queues:
        key = queues.popleft()
        queue = groups[key]
        ordered.append(queue.popleft())
        if queue:
            queues.append(key)
    return ordered


def extract_chunk_records(
    chunk: Chunk,
    backend: ModelBackend,
    system_prompt: str,
    extraction_prompt: str,
    schema_hint: str,
    strict_sdg13: bool = False,
    target_classifier: Optional[SDG13TargetClassifier] = None,
) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_prompt(extraction_prompt, schema_hint, chunk.to_dict()),
        },
    ]
    response = backend.complete_json(messages, schema_hint=schema_hint)
    raw_records = response.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("Model JSON must contain a list field named 'records'.")
    raw_records = [sanitize_raw_record(record) for record in raw_records if isinstance(record, dict)]
    if target_classifier:
        raw_records = target_classifier.classify(chunk, raw_records)
    defaults = {
        "document_id": chunk.document_id,
        "source_file": chunk.source_file,
        "source_filename": Path(chunk.source_file).name,
        "source_type": chunk.source_type,
        "chunk_id": chunk.chunk_id,
        "evidence_page": chunk.page_start or chunk.evidence_page,
        "evidence_page_end": chunk.page_end,
        "evidence_paragraph_ids": chunk.paragraph_ids,
        "evidence_char_start": chunk.evidence_char_start,
        "evidence_char_end": chunk.evidence_char_end,
    }
    authoritative_metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if key in OUTPUT_FIELDS and value not in {None, ""}
    }
    defaults.update(authoritative_metadata)
    normalized = normalize_records(raw_records, defaults=defaults)
    for record in normalized:
        record.update(authoritative_metadata)
    if strict_sdg13:
        normalized = filter_strict_sdg13_records(normalized)
    return dedupe_records(normalized)


def sanitize_raw_record(record: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {key: normalize_empty(value) for key, value in record.items()}
    goal = cleaned.get("sdg_goal")
    if isinstance(goal, str) and re.fullmatch(r"sdg\s*13|13", goal.strip(), flags=re.IGNORECASE):
        cleaned["sdg_goal"] = "SDG 13"
    target = cleaned.get("sdg_target")
    if isinstance(target, str):
        target_clean = target.strip().lower().replace("13.a", "13.a").replace("13.b", "13.b")
        if target_clean in SDG13_TARGETS:
            cleaned["sdg_target"] = target_clean
    for field in ("jurisdiction_level", "policy_type", "date_granularity"):
        value = cleaned.get(field)
        if isinstance(value, str):
            cleaned[field] = normalize_enum_value(field, value)
    return cleaned


def normalize_empty(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def normalize_enum_value(field: str, value: str) -> str:
    lower = value.strip().lower()
    if field == "date_granularity" and lower in {"annual", "yearly"}:
        return "year"
    if field == "jurisdiction_level" and lower == "national":
        return "national"
    if field == "policy_type":
        mapping = {"act": "law", "policy": "unknown"}
        return mapping.get(lower, lower)
    return lower


def filter_strict_sdg13_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if record.get("sdg_goal") != "SDG 13":
            continue
        if record.get("sdg_target") not in SDG13_TARGETS:
            continue
        if not record.get("policy_content_original") or not record.get("evidence_quote"):
            continue
        if not is_record_text_sdg13_relevant(record):
            continue
        filtered.append(record)
    return filtered


def is_record_text_sdg13_relevant(record: Dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(field) or "")
        for field in ["policy_content_original", "policy_content_summary", "evidence_quote"]
    )
    target = str(record.get("sdg_target") or "")
    return target in SDG13_TARGETS and target_has_required_signal(target, text)


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        key = (
            record.get("source_filename"),
            record.get("sdg_target"),
            normalize_space(record.get("policy_content_original")),
            normalize_space(record.get("evidence_quote")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_completed_chunks(path: Optional[Path]) -> int:
    if not path or not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("completed_chunks", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def save_checkpoint(path: Optional[Path], completed_chunks: int, total_chunks: int, total_records: int) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completed_chunks": completed_chunks,
                "total_chunks": total_chunks,
                "total_records": total_records,
                "updated_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def save_extracted_records(path: Path, records: List[Dict[str, Any]]) -> None:
    write_jsonl(path, normalize_records(records))


def load_extracted_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    return normalize_records(read_jsonl(path))


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_user_prompt(extraction_prompt: str, schema_hint: str, chunk: Dict[str, Any]) -> str:
    return (
        f"{extraction_prompt}\n\n"
        f"Output schema fields:\n{schema_hint}\n\n"
        f"Source chunk metadata:\n{json.dumps(chunk, ensure_ascii=False, indent=2)}\n\n"
        f"Source text:\n{chunk['text']}"
    )
