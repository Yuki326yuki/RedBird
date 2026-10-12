from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .cache import (
    cache_matches,
    chunk_cache_fingerprint,
    document_cache_fingerprint,
    document_sources_match,
    load_chunks,
    save_cache_meta,
    save_chunks,
    save_documents,
)
from .chunking import chunk_documents
from .config import load_config, resolve_path
from .database import build_database, database_summary, query_policies
from .exporters import export_records
from .extraction import extract_records, load_documents, load_extracted_records, save_extracted_records
from .io_utils import ensure_dir, write_json, write_jsonl, write_jsonl_atomic
from .manifest import load_manifest_selection
from .models import ModelBackend
from .models import build_backend
from .parsers import parse_directory
from .quality import target_distribution_report
from .review import RecordReviewer
from .targeting import SDG13TargetClassifier
from .translation import PolicyTranslator
from .types import Chunk, Document


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sdg-extract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse local documents into normalized JSONL.")
    ingest_parser.add_argument("--input", required=True, type=Path)
    ingest_parser.add_argument("--workdir", required=True, type=Path)
    ingest_parser.add_argument("--max-chars", type=int, default=6000)
    ingest_parser.add_argument("--overlap-chars", type=int, default=0)

    extract_parser = subparsers.add_parser("extract", help="Run SDG extraction with configured model.")
    extract_parser.add_argument("--config", required=True, type=Path)

    export_parser = subparsers.add_parser("export", help="Export extracted records to JSONL and/or CSV.")
    export_parser.add_argument("--config", type=Path)
    export_parser.add_argument("--workdir", type=Path)
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--format", default="jsonl,csv")

    db_build_parser = subparsers.add_parser("db-build", help="Build a SQLite database from exported records.")
    db_build_parser.add_argument("--input", type=Path)
    db_build_parser.add_argument("--db", type=Path)
    db_build_parser.add_argument("--passages", type=Path)
    db_build_parser.add_argument("--config", type=Path)

    db_query_parser = subparsers.add_parser("db-query", help="Search the SQLite policy database.")
    db_query_parser.add_argument("--db", required=True, type=Path)
    db_query_parser.add_argument("--q", required=True)
    db_query_parser.add_argument("--country", action="append", default=[])
    db_query_parser.add_argument("--sdg-target", action="append", default=[])
    db_query_parser.add_argument("--limit", type=int, default=10)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run extract, export and synchronized database build.")
    pipeline_parser.add_argument("--config", required=True, type=Path)
    pipeline_parser.add_argument("--format", default="jsonl,csv")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args.input, args.workdir, args.max_chars, args.overlap_chars)
    if args.command == "extract":
        return run_extract(args.config)
    if args.command == "export":
        return run_export_command(args.config, args.workdir, args.output, args.format)
    if args.command == "db-build":
        return run_db_build(args.input, args.db, passages_path=args.passages, config_path=args.config)
    if args.command == "db-query":
        return run_db_query(args.db, args.q, args.country, args.sdg_target, args.limit)
    if args.command == "pipeline":
        return run_pipeline(args.config, args.format)
    parser.error(f"Unknown command: {args.command}")
    return 2


def run_ingest(
    input_dir: Path,
    workdir: Path,
    max_chars: int,
    overlap_chars: int = 0,
    allowed_paths: Optional[Set[Path]] = None,
    metadata_by_path: Optional[Dict[Path, Dict[str, object]]] = None,
) -> int:
    ensure_dir(workdir)
    documents = parse_directory(input_dir, allowed_paths=allowed_paths, metadata_by_path=metadata_by_path, show_progress=True)
    chunks = chunk_documents(documents, max_chars=max_chars, overlap_chars=overlap_chars)
    write_jsonl(workdir / "documents.jsonl", [document.to_dict() for document in documents])
    write_jsonl(workdir / "chunks.jsonl", [chunk.to_dict() for chunk in chunks])
    warnings = [
        {
            "document_id": document.document_id,
            "source_file": document.source_file,
            "parse_status": document.metadata.get("parse_status"),
            "low_text_pages": document.metadata.get("low_text_pages", []),
        }
        for document in documents
        if str(document.metadata.get("parse_status") or "ok") != "ok"
    ]
    write_jsonl(workdir / "parse_warnings.jsonl", warnings)
    print(f"Parsed {len(documents)} documents and {len(chunks)} chunks.")
    print(f"Wrote {workdir / 'documents.jsonl'}")
    print(f"Wrote {workdir / 'chunks.jsonl'}")
    return 0


def run_extract(config_path: Path) -> int:
    config = load_config(config_path)
    project_root = infer_project_root(config_path)
    paths = config["paths"]
    workdir = resolve_path(paths["workdir"], project_root)
    documents, cached_chunks, documents_path, chunks_path = prepare_extraction_inputs(
        config, project_root, workdir
    )
    print(f"[INFO] Loaded {len(documents)} documents from {documents_path}", flush=True)
    if cached_chunks is not None:
        print(f"[INFO] Loaded {len(cached_chunks)} chunks from {chunks_path}", flush=True)
    print(
        f"[INFO] Loading model backend: {config['model'].get('backend')} "
        f"name={config['model'].get('name')}",
        flush=True,
    )
    backend = build_backend(config["model"])
    print(f"[INFO] Model backend ready: {backend.model_name()}", flush=True)
    target_classifier: Optional[SDG13TargetClassifier] = None
    reviewer: Optional[RecordReviewer] = None
    try:
        target_classifier = build_target_classifier_for_config(config, project_root, workdir, backend)
        reviewer = build_reviewer_for_config(config, project_root, workdir, backend)
        prompt_dir = resolve_path(paths.get("prompt_dir", "prompts"), project_root)
        records_path = workdir / "records.jsonl"
        experiment = config.get("experiment", {})
        records = extract_records(
            documents=documents,
            backend=backend,
            prompt_dir=prompt_dir,
            max_chars=int(config["chunking"]["max_chars"]),
            overlap_chars=int(config["chunking"].get("overlap_chars", 0)),
            records_path=records_path,
            checkpoint_path=workdir / "checkpoint.json",
            errors_path=workdir / "errors.jsonl",
            max_chunks=experiment.get("max_chunks"),
            sdg13_filter=bool(experiment.get("sdg13_filter", False)),
            strict_sdg13=bool(experiment.get("strict_sdg13", False)),
            balance_by=experiment.get("balance_by"),
            neighbor_chunks=int(experiment.get("neighbor_chunks", 0)),
            reviewer=reviewer,
            target_classifier=target_classifier,
            max_consecutive_oom=int(experiment.get("max_consecutive_oom", 3)),
            preflight_chunks=int(experiment.get("preflight_chunks", 0)),
            preflight_max_seconds_per_chunk=float(
                experiment.get("preflight_max_seconds_per_chunk", 0)
            ),
            fail_on_chunk_errors=bool(experiment.get("fail_on_chunk_errors", False)),
            precomputed_chunks=cached_chunks,
            extraction_max_tokens=int(config["model"].get("max_tokens", 512)),
            target_policies_per_country=int(
                experiment.get("target_policies_per_country", 0)
            ),
            quota_countries=[str(country) for country in config.get("countries") or []],
            require_policy_quota=bool(experiment.get("require_policy_quota", False)),
        )
        save_extracted_records(records_path, records)
        quality_report = target_distribution_report(records)
        quality_path = workdir / "quality_report.json"
        write_json(quality_path, quality_report)
        print(f"Extracted {len(records)} records.")
        print(f"Wrote {workdir / 'records.jsonl'}")
        print(
            f"[QUALITY] targets={quality_report['target_counts']} "
            f"dominant={quality_report['dominant_target']}:{quality_report['dominant_share']:.1%} "
            f"warnings={quality_report['warnings']}",
            flush=True,
        )
        print(f"Wrote {quality_path}")
        minimum_records = int(experiment.get("min_records", 0))
        if len(records) < minimum_records:
            raise RuntimeError(
                f"Extraction produced {len(records)} records, below required minimum {minimum_records}."
            )
        return 0
    finally:
        close_backends(backend, target_classifier, reviewer)


def prepare_extraction_inputs(
    config: dict,
    project_root: Path,
    workdir: Path,
) -> Tuple[List[Document], Optional[List[Chunk]], Path, Path]:
    """Load validated shared parse/chunk caches, rebuilding only stale stages."""
    ensure_dir(workdir)
    paths = config["paths"]
    input_dir = resolve_path(paths["input_dir"], project_root)
    cache_config = config.get("cache") or {}
    if not cache_config.get("enabled"):
        documents_path = workdir / "documents.jsonl"
        chunks_path = workdir / "chunks.jsonl"
        if not documents_path.exists():
            allowed_paths, metadata_by_path = manifest_selection_for_config(config, project_root)
            run_ingest(
                input_dir,
                workdir,
                int(config["chunking"]["max_chars"]),
                int(config["chunking"].get("overlap_chars", 0)),
                allowed_paths=allowed_paths,
                metadata_by_path=metadata_by_path,
            )
        return load_documents(documents_path), None, documents_path, chunks_path

    allowed_paths, metadata_by_path = manifest_selection_for_config(config, project_root)
    if allowed_paths is None:
        raise ValueError("Shared document caching requires manifest_path in the config.")
    manifest_value = config.get("manifest_path")
    manifest_path = resolve_path(str(manifest_value), project_root) if manifest_value else None
    document_fingerprint = document_cache_fingerprint(
        allowed_paths,
        metadata_by_path,
        config.get("countries") or [],
        manifest_path,
    )
    documents_path = resolve_path(
        str(cache_config.get("documents_path") or "data/cache/documents.jsonl"),
        project_root,
    )
    documents_meta_path = resolve_path(
        str(cache_config.get("documents_meta_path") or documents_path.with_name("documents.meta.json")),
        project_root,
    )
    if cache_matches(documents_meta_path, documents_path, document_fingerprint):
        documents = load_documents(documents_path)
        print("[CACHE] documents hit; PDF parsing skipped.", flush=True)
    else:
        documents: List[Document] = []
        seed_value = cache_config.get("seed_documents_path")
        seed_path = resolve_path(str(seed_value), project_root) if seed_value else None
        if seed_path and seed_path.exists():
            seed_documents = load_documents(seed_path)
            if document_sources_match(seed_documents, allowed_paths):
                documents = seed_documents
                print(
                    f"[CACHE] Migrating {len(documents)} validated documents from {seed_path}; "
                    "PDF parsing skipped.",
                    flush=True,
                )
            else:
                print("[CACHE] Seed documents do not match current sources; reparsing PDFs.", flush=True)
        if not documents:
            documents = parse_directory(
                input_dir,
                allowed_paths=allowed_paths,
                metadata_by_path=metadata_by_path,
                show_progress=True,
            )
        if not document_sources_match(documents, allowed_paths):
            raise RuntimeError("Parsed document set does not match the current manifest selection.")
        save_documents(documents_path, documents)
        save_cache_meta(
            documents_meta_path,
            document_fingerprint,
            document_count=len(documents),
            countries=config.get("countries") or [],
        )
        write_jsonl_atomic(
            documents_path.with_name("parse_warnings.jsonl"),
            parse_warning_rows(documents),
        )
        print(f"[CACHE] Wrote shared document cache: {documents_path}", flush=True)

    max_chars = int(config["chunking"]["max_chars"])
    overlap_chars = int(config["chunking"].get("overlap_chars", 0))
    chunk_fingerprint = chunk_cache_fingerprint(document_fingerprint, max_chars, overlap_chars)
    chunks_path = resolve_path(
        str(cache_config.get("chunks_path") or documents_path.with_name("chunks.jsonl")),
        project_root,
    )
    chunks_meta_path = resolve_path(
        str(cache_config.get("chunks_meta_path") or chunks_path.with_name("chunks.meta.json")),
        project_root,
    )
    if cache_matches(chunks_meta_path, chunks_path, chunk_fingerprint):
        chunks = load_chunks(chunks_path)
        print("[CACHE] chunks hit; document splitting skipped.", flush=True)
    else:
        chunks = chunk_documents(documents, max_chars=max_chars, overlap_chars=overlap_chars)
        save_chunks(chunks_path, chunks)
        save_cache_meta(
            chunks_meta_path,
            chunk_fingerprint,
            chunk_count=len(chunks),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            document_fingerprint=document_fingerprint,
        )
        print(f"[CACHE] Wrote shared chunk cache: {chunks_path}", flush=True)
    return documents, chunks, documents_path, chunks_path


def parse_warning_rows(documents: List[Document]) -> List[Dict[str, object]]:
    return [
        {
            "document_id": document.document_id,
            "source_file": document.source_file,
            "parse_status": document.metadata.get("parse_status"),
            "low_text_pages": document.metadata.get("low_text_pages", []),
        }
        for document in documents
        if str(document.metadata.get("parse_status") or "ok") != "ok"
    ]


def manifest_selection_for_config(config: dict, project_root: Path) -> Tuple[Optional[Set[Path]], Dict[Path, Dict[str, object]]]:
    manifest_path_value = config.get("manifest_path")
    if not manifest_path_value:
        return None, {}
    manifest_path = resolve_path(str(manifest_path_value), project_root)
    countries = config.get("countries") or []
    return load_manifest_selection(manifest_path, project_root, countries=countries)


def infer_project_root(config_path: Path) -> Path:
    config_based_root = config_path.resolve().parent.parent
    if (config_based_root / "prompts").exists():
        return config_based_root
    cwd = Path.cwd().resolve()
    if (cwd / "prompts").exists():
        return cwd
    return config_path.resolve().parent


def run_export_command(
    config_path: Optional[Path],
    workdir_arg: Optional[Path],
    output_arg: Optional[Path],
    formats: str,
) -> int:
    if config_path:
        config = load_config(config_path)
        project_root = infer_project_root(config_path)
        paths = config["paths"]
        workdir = workdir_arg or resolve_path(paths["workdir"], project_root)
        output_dir = output_arg or resolve_path(paths.get("output_dir", "data/output"), project_root)
    else:
        workdir = workdir_arg or Path("data/work")
        output_dir = output_arg or Path("data/output")
    translator = build_translator_for_config(config, project_root, workdir) if config_path else None
    try:
        return run_export(workdir, output_dir, formats, translator=translator)
    finally:
        if translator is not None:
            translator.backend.close()



def build_target_classifier_for_config(
    config: dict,
    project_root: Path,
    workdir: Path,
    extraction_backend: ModelBackend,
) -> Optional[SDG13TargetClassifier]:
    classifier_config = config.get("target_classification") or {}
    if not classifier_config.get("enabled"):
        return None
    model_config = classifier_config.get("model") or config.get("model")
    if not model_config:
        raise ValueError("target_classification.model is required when enabled.")
    prompt_path = resolve_path(
        classifier_config.get("prompt_path", "prompts/targeting/sdg13_targets.md"),
        project_root,
    )
    cache_path = Path(str(classifier_config.get("cache_path") or "target_classifications.jsonl"))
    if not cache_path.is_absolute():
        cache_path = workdir / cache_path
    if same_model_config(model_config, config.get("model") or {}):
        backend = extraction_backend
        print(f"[INFO] Reusing extraction backend for target classification: {backend.model_name()}", flush=True)
    else:
        backend = build_backend(model_config)
    return SDG13TargetClassifier(
        backend=backend,
        prompt_path=prompt_path,
        cache_path=cache_path,
        min_confidence=float(classifier_config.get("min_confidence", 0.75)),
        max_tokens=int(model_config.get("max_tokens", 700)),
    )

def build_reviewer_for_config(
    config: dict,
    project_root: Path,
    workdir: Path,
    extraction_backend: ModelBackend,
) -> Optional[RecordReviewer]:
    review_config = config.get("review") or {}
    if not review_config.get("enabled"):
        return None
    model_config = review_config.get("model") or config.get("model")
    if not model_config:
        raise ValueError("review.model is required when review.enabled is true.")
    prompt_path = resolve_path(
        review_config.get("prompt_path", "prompts/review/sdg13_strict.md"),
        project_root,
    )
    cache_path_value = str(review_config.get("cache_path") or "reviews.jsonl")
    cache_path = Path(cache_path_value)
    if not cache_path.is_absolute():
        cache_path = workdir / cache_path
    if same_model_config(model_config, config.get("model") or {}):
        backend = extraction_backend
        print(f"[INFO] Reusing extraction backend for review: {backend.model_name()}", flush=True)
    else:
        print(
            f"[INFO] Loading review backend: {model_config.get('backend')} "
            f"name={model_config.get('name')}",
            flush=True,
        )
        backend = build_backend(model_config)
        print(f"[INFO] Review backend ready: {backend.model_name()}", flush=True)
    return RecordReviewer(
        backend=backend,
        prompt_path=prompt_path,
        cache_path=cache_path,
        keep_rejected=bool(review_config.get("keep_rejected", False)),
        min_confidence=float(review_config.get("min_confidence", 0.75)),
        enforce_traceable_evidence=bool(review_config.get("enforce_traceable_evidence", False)),
        max_tokens=int(model_config.get("max_tokens", 350)),
    )


def same_model_config(left: dict, right: dict) -> bool:
    left_backend = str(left.get("backend") or "api")
    right_backend = str(right.get("backend") or "api")
    if left_backend != right_backend:
        return False
    if left_backend == "local_transformers":
        keys = [
            "name", "local_model_path", "torch_dtype", "attn_implementation",
        ]
    else:
        keys = [
            "name", "api_base", "api_key_env", "api_protocol", "temperature",
            "max_tokens", "json_mode", "reasoning_effort",
        ]
    if not all(str(left.get(key) or "") == str(right.get(key) or "") for key in keys):
        return False
    return left_backend != "local_transformers" or (
        normalized_quantization(left) == normalized_quantization(right)
    )


def normalized_quantization(config: dict) -> str:
    value = config.get("quantization") or {}
    if isinstance(value, dict):
        value = value.get("mode", "none")
    return str(value or "none").strip().lower()


def close_backends(
    extraction_backend: ModelBackend,
    target_classifier: Optional[SDG13TargetClassifier],
    reviewer: Optional[RecordReviewer],
) -> None:
    backends = [
        extraction_backend,
        getattr(target_classifier, "backend", None),
        getattr(reviewer, "backend", None),
    ]
    closed = set()
    for backend in backends:
        if backend is None or id(backend) in closed:
            continue
        closed.add(id(backend))
        backend.close()


def build_translator_for_config(config: dict, project_root: Path, workdir: Path) -> Optional[PolicyTranslator]:
    translation_config = config.get("translation") or {}
    if not translation_config.get("enabled"):
        return None
    model_config = translation_config.get("model")
    if not model_config:
        raise ValueError("translation.model is required when translation.enabled is true.")
    prompt_path = resolve_path(
        translation_config.get("prompt_path", "prompts/translation/en_policy.md"),
        project_root,
    )
    cache_path_value = str(translation_config.get("cache_path") or "translations.jsonl")
    cache_path = Path(cache_path_value)
    if not cache_path.is_absolute():
        cache_path = workdir / cache_path
    print(
        f"[INFO] Loading translation backend: {model_config.get('backend')} "
        f"name={model_config.get('name')}",
        flush=True,
    )
    backend = build_backend(model_config)
    print(f"[INFO] Translation backend ready: {backend.model_name()}", flush=True)
    return PolicyTranslator(
        backend=backend,
        prompt_path=prompt_path,
        cache_path=cache_path,
        output_language=str(translation_config.get("output_language") or "English"),
        max_tokens=int(model_config.get("max_tokens", 1800)),
    )


def run_export(
    workdir: Path,
    output_dir: Path,
    formats: str,
    translator: Optional[PolicyTranslator] = None,
) -> int:
    records_path = workdir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"No extracted records found at {records_path}")
    records = load_extracted_records(records_path)
    written = export_records(records, output_dir, formats.split(","), translator=translator)
    for path in written:
        print(f"Wrote {path}")
    return 0


def run_db_build(
    input_path: Optional[Path],
    db_path: Optional[Path],
    passages_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> int:
    if config_path:
        config = load_config(config_path)
        project_root = infer_project_root(config_path)
        paths = config["paths"]
        output_dir = resolve_path(paths.get("output_dir", "data/output"), project_root)
        workdir = resolve_path(paths.get("workdir", "data/work"), project_root)
        database_config = config.get("database") or {}
        chatbot_config = config.get("chatbot") or {}
        if input_path is None:
            input_path = output_dir / "records.jsonl"
        if passages_path is None:
            passages_path = workdir / "records.jsonl"
        if db_path is None:
            db_value = database_config.get("db_path") or chatbot_config.get("db_path")
            if db_value:
                db_path = resolve_path(str(db_value), project_root)
    if input_path is None or db_path is None:
        raise ValueError("db-build requires --input and --db unless --config supplies their paths.")
    count = build_database(input_path, db_path, passages_path=passages_path)
    summary = database_summary(db_path)
    print(f"Imported {count} policy records and {summary.get('passages', 0)} passages.")
    print(f"Wrote {db_path}")
    return 0


def run_pipeline(config_path: Path, formats: str) -> int:
    config = load_config(config_path)
    project_root = infer_project_root(config_path)
    paths = config["paths"]
    workdir = resolve_path(paths["workdir"], project_root)
    output_dir = resolve_path(paths.get("output_dir", "data/output"), project_root)
    run_extract(config_path)
    run_export_command(config_path, workdir, output_dir, formats)
    database_config = config.get("database") or {}
    chatbot_config = config.get("chatbot") or {}
    db_value = database_config.get("db_path") or chatbot_config.get("db_path") or "data/db/redbird_sdg13.sqlite"
    db_path = resolve_path(str(db_value), project_root)
    input_path = output_dir / "records.jsonl"
    passages_path = workdir / "records.jsonl"
    count = build_database(
        input_path,
        db_path,
        passages_path=passages_path,
    )
    summary = database_summary(db_path)
    print(
        f"[PIPELINE] Database synchronized: policies={count} passages={summary.get('passages', 0)} db={db_path}",
        flush=True,
    )
    return 0


def run_db_query(
    db_path: Path,
    query: str,
    countries: List[str],
    sdg_targets: List[str],
    limit: int,
) -> int:
    rows = query_policies(
        db_path,
        query=query,
        countries=countries,
        sdg_targets=sdg_targets,
        limit=limit,
    )
    preview = [
        {
            "id": row.get("id"),
            "country": row.get("country"),
            "policy_title": row.get("policy_title_en") or row.get("policy_title"),
            "sdg_targets": row.get("sdg_targets"),
            "source_filename": row.get("source_filename"),
        }
        for row in rows
    ]
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
