from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .chunking import chunk_documents
from .config import load_config, resolve_path
from .database import build_database, query_policies
from .exporters import export_records
from .extraction import extract_records, load_documents, load_extracted_records, save_extracted_records
from .io_utils import ensure_dir, write_jsonl
from .manifest import load_manifest_selection
from .models import ModelBackend
from .models import build_backend
from .parsers import parse_directory
from .review import RecordReviewer
from .translation import PolicyTranslator


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sdg-extract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse local documents into normalized JSONL.")
    ingest_parser.add_argument("--input", required=True, type=Path)
    ingest_parser.add_argument("--workdir", required=True, type=Path)
    ingest_parser.add_argument("--max-chars", type=int, default=6000)

    extract_parser = subparsers.add_parser("extract", help="Run SDG extraction with configured model.")
    extract_parser.add_argument("--config", required=True, type=Path)

    export_parser = subparsers.add_parser("export", help="Export extracted records to JSONL and/or CSV.")
    export_parser.add_argument("--config", type=Path)
    export_parser.add_argument("--workdir", type=Path)
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--format", default="jsonl,csv")

    db_build_parser = subparsers.add_parser("db-build", help="Build a SQLite database from exported records.")
    db_build_parser.add_argument("--input", required=True, type=Path)
    db_build_parser.add_argument("--db", required=True, type=Path)

    db_query_parser = subparsers.add_parser("db-query", help="Search the SQLite policy database.")
    db_query_parser.add_argument("--db", required=True, type=Path)
    db_query_parser.add_argument("--q", required=True)
    db_query_parser.add_argument("--country", action="append", default=[])
    db_query_parser.add_argument("--sdg-target", action="append", default=[])
    db_query_parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args.input, args.workdir, args.max_chars)
    if args.command == "extract":
        return run_extract(args.config)
    if args.command == "export":
        return run_export_command(args.config, args.workdir, args.output, args.format)
    if args.command == "db-build":
        return run_db_build(args.input, args.db)
    if args.command == "db-query":
        return run_db_query(args.db, args.q, args.country, args.sdg_target, args.limit)
    parser.error(f"Unknown command: {args.command}")
    return 2


def run_ingest(
    input_dir: Path,
    workdir: Path,
    max_chars: int,
    allowed_paths: Optional[Set[Path]] = None,
    metadata_by_path: Optional[Dict[Path, Dict[str, object]]] = None,
) -> int:
    ensure_dir(workdir)
    documents = parse_directory(input_dir, allowed_paths=allowed_paths, metadata_by_path=metadata_by_path, show_progress=True)
    chunks = chunk_documents(documents, max_chars=max_chars)
    write_jsonl(workdir / "documents.jsonl", [document.to_dict() for document in documents])
    write_jsonl(workdir / "chunks.jsonl", [chunk.to_dict() for chunk in chunks])
    print(f"Parsed {len(documents)} documents and {len(chunks)} chunks.")
    print(f"Wrote {workdir / 'documents.jsonl'}")
    print(f"Wrote {workdir / 'chunks.jsonl'}")
    return 0


def run_extract(config_path: Path) -> int:
    config = load_config(config_path)
    project_root = infer_project_root(config_path)
    paths = config["paths"]
    workdir = resolve_path(paths["workdir"], project_root)
    documents_path = workdir / "documents.jsonl"
    if not documents_path.exists():
        input_dir = resolve_path(paths["input_dir"], project_root)
        allowed_paths, metadata_by_path = manifest_selection_for_config(config, project_root)
        run_ingest(
            input_dir,
            workdir,
            int(config["chunking"]["max_chars"]),
            allowed_paths=allowed_paths,
            metadata_by_path=metadata_by_path,
        )
    documents = load_documents(documents_path)
    print(f"[INFO] Loaded {len(documents)} documents from {documents_path}", flush=True)
    print(
        f"[INFO] Loading model backend: {config['model'].get('backend')} "
        f"name={config['model'].get('name')}",
        flush=True,
    )
    backend = build_backend(config["model"])
    print(f"[INFO] Model backend ready: {backend.model_name()}", flush=True)
    reviewer = build_reviewer_for_config(config, project_root, workdir, backend)
    prompt_dir = resolve_path(paths.get("prompt_dir", "prompts"), project_root)
    records_path = workdir / "records.jsonl"
    experiment = config.get("experiment", {})
    records = extract_records(
        documents=documents,
        backend=backend,
        prompt_dir=prompt_dir,
        max_chars=int(config["chunking"]["max_chars"]),
        records_path=records_path,
        checkpoint_path=workdir / "checkpoint.json",
        errors_path=workdir / "errors.jsonl",
        max_chunks=experiment.get("max_chunks"),
        sdg13_filter=bool(experiment.get("sdg13_filter", False)),
        strict_sdg13=bool(experiment.get("strict_sdg13", False)),
        balance_by=experiment.get("balance_by"),
        reviewer=reviewer,
    )
    save_extracted_records(records_path, records)
    print(f"Extracted {len(records)} records.")
    print(f"Wrote {workdir / 'records.jsonl'}")
    return 0


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
    return run_export(workdir, output_dir, formats, translator=translator)


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
    )


def same_model_config(left: dict, right: dict) -> bool:
    keys = ["backend", "name", "local_model_path", "api_base", "api_key_env"]
    return all(str(left.get(key) or "") == str(right.get(key) or "") for key in keys)


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


def run_db_build(input_path: Path, db_path: Path) -> int:
    count = build_database(input_path, db_path)
    print(f"Imported {count} policy records.")
    print(f"Wrote {db_path}")
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
