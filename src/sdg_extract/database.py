from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .exporters import POLICY_OUTPUT_FIELDS
from .io_utils import ensure_dir, read_jsonl


SCHEMA_VERSION = "2"
POLICY_TABLE_FIELDS = ["id", *POLICY_OUTPUT_FIELDS]
POLICY_FTS_FIELDS = [
    "policy_title", "policy_title_en", "policy_content", "policy_content_en",
    "policy_summary", "policy_summary_en", "country", "sdg_goal", "sdg_targets",
]
PASSAGE_FTS_FIELDS = [
    "policy_title", "policy_title_en", "country", "sdg_goal", "sdg_target",
    "text_original", "text_en", "summary_original", "summary_en",
]
FTS_STOPWORDS = {
    "a", "an", "and", "are", "about", "does", "for", "in", "is", "mention",
    "mentions", "of", "on", "policies", "policy", "the", "to", "what", "which", "with",
}
CJK_STOP_PHRASES = [
    "请帮我", "请问", "帮我", "总结", "全部", "所有", "哪些", "什么", "有关", "相关",
    "政策", "数据库", "记录", "内容", "介绍", "一下", "中国", "美国",
]
DOMAIN_TERMS = [
    "气候适应", "气候韧性", "气候融资", "温室气体", "碳中和", "碳达峰", "碳市场",
    "灾害风险", "早期预警", "减排", "排放", "甲烷", "气候", "适应", "韧性", "融资",
]


def build_database(
    input_path: Path,
    db_path: Path,
    passages_path: Optional[Path] = None,
) -> int:
    records = load_policy_records(input_path)
    passage_records = load_passage_records(passages_path) if passages_path and passages_path.exists() else []
    ensure_dir(db_path.parent)
    temp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with sqlite3.connect(temp_path) as conn:
            initialize_database(conn)
            policy_ids = insert_policies(conn, records)
            passages = prepare_passages(records, passage_records, policy_ids)
            insert_passages(conn, passages)
            rebuild_policy_targets(conn)
            write_database_metadata(conn, input_path, passages_path, len(records), len(passages))
        os.replace(temp_path, db_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return len(records)


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS policies_fts;
        DROP TABLE IF EXISTS passages_fts;
        DROP TABLE IF EXISTS policy_targets;
        DROP TABLE IF EXISTS passages;
        DROP TABLE IF EXISTS policies;
        DROP TABLE IF EXISTS db_meta;
        """
    )
    columns = ", ".join(f"{field} TEXT" for field in POLICY_OUTPUT_FIELDS)
    conn.execute(f"CREATE TABLE policies (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})")
    conn.execute(f"CREATE VIRTUAL TABLE policies_fts USING fts5({', '.join(POLICY_FTS_FIELDS)})")
    conn.execute(
        """
        CREATE TABLE passages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            policy_id INTEGER NOT NULL,
            chunk_id TEXT,
            page_start INTEGER,
            page_end INTEGER,
            paragraph_ids TEXT,
            language TEXT,
            text_original TEXT,
            text_en TEXT,
            summary_original TEXT,
            summary_en TEXT,
            sdg_goal TEXT,
            sdg_target TEXT,
            review_status TEXT,
            review_confidence REAL,
            source_filename TEXT,
            text_hash TEXT NOT NULL,
            FOREIGN KEY(policy_id) REFERENCES policies(id)
        )
        """
    )
    conn.execute(f"CREATE VIRTUAL TABLE passages_fts USING fts5({', '.join(PASSAGE_FTS_FIELDS)})")
    conn.execute(
        """
        CREATE TABLE policy_targets (
            policy_id INTEGER NOT NULL,
            sdg_target TEXT NOT NULL,
            confidence REAL,
            evidence_count INTEGER NOT NULL,
            review_status TEXT,
            PRIMARY KEY(policy_id, sdg_target),
            FOREIGN KEY(policy_id) REFERENCES policies(id)
        )
        """
    )
    conn.execute("CREATE TABLE db_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE INDEX idx_policies_country ON policies(country)")
    conn.execute("CREATE INDEX idx_policies_validation_status ON policies(validation_status)")
    conn.execute("CREATE INDEX idx_policies_review_status ON policies(review_status)")
    conn.execute("CREATE INDEX idx_passages_policy_id ON passages(policy_id)")
    conn.execute("CREATE INDEX idx_passages_target ON passages(sdg_target)")
    conn.execute("CREATE INDEX idx_passages_review ON passages(review_status)")


def insert_policies(conn: sqlite3.Connection, records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    fields_sql = ", ".join(POLICY_OUTPUT_FIELDS)
    placeholders = ", ".join("?" for _ in POLICY_OUTPUT_FIELDS)
    ids: Dict[str, int] = {}
    for record in records:
        values = [to_db_value(record.get(field)) for field in POLICY_OUTPUT_FIELDS]
        cursor = conn.execute(f"INSERT INTO policies ({fields_sql}) VALUES ({placeholders})", values)
        rowid = int(cursor.lastrowid)
        ids[policy_key(record)] = rowid
        fts_values = [to_db_value(record.get(field)) for field in POLICY_FTS_FIELDS]
        conn.execute(
            f"INSERT INTO policies_fts (rowid, {', '.join(POLICY_FTS_FIELDS)}) VALUES ({', '.join('?' for _ in ['rowid', *POLICY_FTS_FIELDS])})",
            [rowid, *fts_values],
        )
    return ids


def insert_passages(conn: sqlite3.Connection, passages: Iterable[Dict[str, Any]]) -> None:
    for passage in passages:
        cursor = conn.execute(
            """
            INSERT INTO passages (
                policy_id, chunk_id, page_start, page_end, paragraph_ids, language,
                text_original, text_en, summary_original, summary_en,
                sdg_goal, sdg_target, review_status, review_confidence,
                source_filename, text_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [passage.get(field) for field in [
                "policy_id", "chunk_id", "page_start", "page_end", "paragraph_ids", "language",
                "text_original", "text_en", "summary_original", "summary_en",
                "sdg_goal", "sdg_target", "review_status", "review_confidence",
                "source_filename", "text_hash",
            ]],
        )
        passage_id = int(cursor.lastrowid)
        conn.execute(
            f"INSERT INTO passages_fts (rowid, {', '.join(PASSAGE_FTS_FIELDS)}) VALUES ({', '.join('?' for _ in ['rowid', *PASSAGE_FTS_FIELDS])})",
            [passage_id, *[to_db_value(passage.get(field)) for field in PASSAGE_FTS_FIELDS]],
        )


def prepare_passages(
    policies: List[Dict[str, Any]],
    passage_records: List[Dict[str, Any]],
    policy_ids: Dict[str, int],
) -> List[Dict[str, Any]]:
    policy_by_key = {policy_key(record): record for record in policies}
    passages: List[Dict[str, Any]] = []
    seen = set()
    for record in passage_records:
        key = policy_key(record)
        policy = policy_by_key.get(key)
        if not policy or key not in policy_ids:
            continue
        if str(record.get("validation_status") or "") != "valid":
            continue
        if str(record.get("review_status") or "").startswith("rejected"):
            continue
        text = normalize_text(record.get("policy_content_original"))
        if not text:
            continue
        passage = passage_from_chunk_record(record, policy, policy_ids[key])
        dedupe_key = (passage["policy_id"], passage["sdg_target"], passage["text_hash"])
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            passages.append(passage)
    if passages:
        return passages
    for policy in policies:
        key = policy_key(policy)
        if key not in policy_ids:
            continue
        original_parts = split_policy_content(policy.get("policy_content"))
        english_parts = split_policy_content(policy.get("policy_content_en"))
        for index, text in enumerate(original_parts or english_parts, start=1):
            original = original_parts[index - 1] if index <= len(original_parts) else None
            english = english_parts[index - 1] if index <= len(english_parts) else None
            for target in split_targets(policy.get("sdg_targets")) or [None]:
                passages.append(
                    make_passage(
                        policy_id=policy_ids[key],
                        policy=policy,
                        text_original=original,
                        text_en=english,
                        summary_original=policy.get("policy_summary"),
                        summary_en=policy.get("policy_summary_en"),
                        sdg_target=target,
                        chunk_id=f"legacy-{policy_ids[key]}-{index}",
                    )
                )
    return passages


def passage_from_chunk_record(record: Dict[str, Any], policy: Dict[str, Any], policy_id: int) -> Dict[str, Any]:
    return make_passage(
        policy_id=policy_id,
        policy=policy,
        text_original=record.get("policy_content_original"),
        text_en=None,
        summary_original=record.get("policy_content_summary"),
        summary_en=None,
        sdg_target=record.get("sdg_target"),
        chunk_id=record.get("chunk_id"),
        page_start=record.get("evidence_page"),
        page_end=record.get("evidence_page_end"),
        paragraph_ids=record.get("evidence_paragraph_ids"),
        review_status=record.get("review_status"),
        review_confidence=record.get("review_confidence"),
        language=record.get("language"),
    )


def make_passage(
    policy_id: int,
    policy: Dict[str, Any],
    text_original: Any,
    text_en: Any,
    summary_original: Any,
    summary_en: Any,
    sdg_target: Any,
    chunk_id: Any,
    page_start: Any = None,
    page_end: Any = None,
    paragraph_ids: Any = None,
    review_status: Any = None,
    review_confidence: Any = None,
    language: Any = None,
) -> Dict[str, Any]:
    original = normalize_text(text_original)
    english = normalize_text(text_en)
    digest = hashlib.sha256((original or english or "").encode("utf-8")).hexdigest()
    return {
        "policy_id": policy_id,
        "policy_title": policy.get("policy_title"),
        "policy_title_en": policy.get("policy_title_en"),
        "country": policy.get("country"),
        "chunk_id": to_db_value(chunk_id),
        "page_start": to_int(page_start),
        "page_end": to_int(page_end),
        "paragraph_ids": to_db_value(paragraph_ids),
        "language": normalize_text(language),
        "text_original": original,
        "text_en": english,
        "summary_original": normalize_text(summary_original),
        "summary_en": normalize_text(summary_en),
        "sdg_goal": policy.get("sdg_goal") or "SDG 13",
        "sdg_target": normalize_text(sdg_target),
        "review_status": normalize_text(review_status) or normalize_text(policy.get("review_status")),
        "review_confidence": to_float(review_confidence) if review_confidence not in {None, ""} else to_float(policy.get("review_confidence")),
        "source_filename": policy.get("source_filename"),
        "text_hash": digest,
    }


def rebuild_policy_targets(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT policy_id, sdg_target, AVG(review_confidence), COUNT(*),
               CASE WHEN SUM(CASE WHEN review_status LIKE 'reviewed%' THEN 1 ELSE 0 END) > 0
                    THEN 'reviewed' ELSE NULL END
        FROM passages
        WHERE sdg_target IS NOT NULL AND sdg_target != ''
        GROUP BY policy_id, sdg_target
        """
    ).fetchall()
    conn.executemany(
        "INSERT INTO policy_targets (policy_id, sdg_target, confidence, evidence_count, review_status) VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def load_policy_records(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return [project_policy_record(record) for record in read_jsonl(input_path)]
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            return [project_policy_record(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported database input format: {input_path}")


def load_passage_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Passage input must be JSONL chunk records.")
    return read_jsonl(path)


def project_policy_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {field: normalize_empty(record.get(field)) for field in POLICY_OUTPUT_FIELDS}


def query_policies(
    db_path: Path,
    query: Optional[str] = None,
    countries: Optional[List[str]] = None,
    sdg_targets: Optional[List[str]] = None,
    publication_year: Optional[str] = None,
    review_statuses: Optional[List[str]] = None,
    translation_statuses: Optional[List[str]] = None,
    validation_status: Optional[str] = None,
    require_reviewed: bool = False,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    normalized = normalize_sdg_references(query or "").strip()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for broad in (False, True):
            sql, params = policy_query_sql(
                normalized, countries, sdg_targets, publication_year, review_statuses,
                translation_statuses, validation_status, require_reviewed, limit, broad,
            )
            rows = conn.execute(sql, params).fetchall()
            if rows or not normalized:
                return [dict(row) for row in rows]
    return []


def policy_query_sql(
    query: str,
    countries: Optional[List[str]],
    sdg_targets: Optional[List[str]],
    publication_year: Optional[str],
    review_statuses: Optional[List[str]],
    translation_statuses: Optional[List[str]],
    validation_status: Optional[str],
    require_reviewed: bool,
    limit: int,
    broad: bool,
) -> Tuple[str, List[Any]]:
    where: List[str] = []
    params: List[Any] = []
    joins = ""
    use_like = bool(query and contains_cjk(query))
    if query and not use_like:
        joins = "JOIN policies_fts ON policies_fts.rowid = p.id"
        where.append("policies_fts MATCH ?")
        params.append(to_fts_query(query, use_or=broad))
    elif query:
        add_cjk_like_search(where, params, query, "p")
    add_policy_filters(where, params, countries, sdg_targets, publication_year, review_statuses, translation_statuses, validation_status, require_reviewed)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    if query and not use_like:
        order_sql = "ORDER BY bm25(policies_fts), CASE WHEN p.review_status LIKE 'reviewed%' THEN 0 ELSE 1 END"
    else:
        order_sql = "ORDER BY CASE WHEN p.review_status LIKE 'reviewed%' THEN 0 ELSE 1 END, p.country, p.publication_date DESC"
    sql = f"SELECT p.* FROM policies p {joins} {where_sql} {order_sql} LIMIT ?"
    params.append(int(limit))
    return sql, params


def query_passages(
    db_path: Path,
    query: str,
    countries: Optional[List[str]] = None,
    sdg_targets: Optional[List[str]] = None,
    publication_year: Optional[str] = None,
    validation_status: Optional[str] = "valid",
    require_reviewed: bool = False,
    limit: int = 8,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = database_summary(db_path)
    diagnostics: Dict[str, Any] = {
        "query": query,
        "eligible_policies": summary.get("valid_policies", summary.get("policies", 0)),
        "passages": summary.get("passages", 0),
        "lexical_hits": 0,
        "fallback_stage": None,
    }
    if not summary.get("has_passages"):
        rows = query_policies(
            db_path, query=query, countries=countries, sdg_targets=sdg_targets,
            publication_year=publication_year, validation_status=validation_status,
            require_reviewed=require_reviewed, limit=limit,
        )
        diagnostics["fallback_stage"] = "legacy_policy_search"
        return [legacy_policy_passage(row) for row in rows], diagnostics

    lexical = lexical_passage_search(
        db_path, query, countries, sdg_targets, publication_year,
        validation_status, require_reviewed, max(50, limit),
    )
    bridge = policy_bridge_passage_search(
        db_path, query, countries, sdg_targets, publication_year,
        validation_status, require_reviewed, max(50, limit),
    )
    diagnostics["lexical_hits"] = len(lexical)
    diagnostics["policy_bridge_hits"] = len(bridge)
    fused = reciprocal_rank_fusion(lexical, bridge)
    if not fused and query.strip():
        diagnostics["fallback_stage"] = "metadata_only"
        fused = lexical_passage_search(
            db_path, "", countries, sdg_targets, publication_year,
            validation_status, require_reviewed, max(limit, 20),
        )
    return fused[: int(limit)], diagnostics


def lexical_passage_search(
    db_path: Path,
    query: str,
    countries: Optional[List[str]],
    sdg_targets: Optional[List[str]],
    publication_year: Optional[str],
    validation_status: Optional[str],
    require_reviewed: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for broad in (False, True):
            where, params = passage_filters(countries, sdg_targets, publication_year, validation_status, require_reviewed)
            joins = "JOIN policies pol ON pol.id = ps.policy_id"
            score_sql = "0.0 AS lexical_score"
            normalized = normalize_sdg_references(query).strip()
            if normalized and not contains_cjk(normalized):
                joins += " JOIN passages_fts ON passages_fts.rowid = ps.id"
                where.append("passages_fts MATCH ?")
                params.append(to_fts_query(normalized, use_or=broad))
                score_sql = "bm25(passages_fts) AS lexical_score"
            elif normalized:
                add_cjk_passage_search(where, params, normalized)
            sql = passage_select_sql(joins, where, score_sql) + " ORDER BY lexical_score, ps.id LIMIT ?"
            rows = conn.execute(sql, [*params, int(limit)]).fetchall()
            if rows or not normalized:
                return [dict(row) for row in rows]
    return []


def policy_bridge_passage_search(
    db_path: Path,
    query: str,
    countries: Optional[List[str]],
    sdg_targets: Optional[List[str]],
    publication_year: Optional[str],
    validation_status: Optional[str],
    require_reviewed: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    """Use translated/original policy-level FTS as a cross-language bridge to passages."""
    policies = query_policies(
        db_path,
        query=query or None,
        countries=countries,
        sdg_targets=sdg_targets,
        publication_year=publication_year,
        validation_status=validation_status,
        require_reviewed=require_reviewed,
        limit=min(max(limit, 20), 100),
    )
    if not policies:
        return []
    policy_rank = {int(row["id"]): rank for rank, row in enumerate(policies, start=1)}
    placeholders = ", ".join("?" for _ in policy_rank)
    where = [f"ps.policy_id IN ({placeholders})"]
    params: List[Any] = list(policy_rank)
    if sdg_targets:
        add_in_filter(where, params, "ps.sdg_target", sdg_targets)
    if require_reviewed:
        where.append("ps.review_status LIKE 'reviewed%'")
    sql = passage_select_sql("JOIN policies pol ON pol.id = ps.policy_id", where, "0.0 AS lexical_score")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    rows.sort(
        key=lambda row: (
            policy_rank.get(int(row["policy_id"]), len(policy_rank) + 1),
            0 if str(row.get("review_status") or "").startswith("reviewed") else 1,
            int(row.get("passage_id") or 0),
        )
    )
    return rows[:limit]


def passage_select_sql(joins: str, where: List[str], score_sql: str) -> str:
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    return f"""
        SELECT ps.id AS passage_id, ps.*, pol.country, pol.country_iso3,
               pol.policy_title, pol.policy_title_en, pol.publication_date,
               pol.source_file, pol.validation_status, {score_sql}
        FROM passages ps {joins} {where_sql}
    """


def reciprocal_rank_fusion(*rankings: Sequence[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    combined: Dict[int, Dict[str, Any]] = {}
    scores: Dict[int, float] = {}
    channels: Dict[int, List[str]] = {}
    for ranking_index, ranking in enumerate(rankings):
        channel = f"lexical_{ranking_index + 1}"
        for rank, row in enumerate(ranking, start=1):
            passage_id = int(row["passage_id"])
            combined.setdefault(passage_id, dict(row))
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (k + rank)
            channels.setdefault(passage_id, []).append(channel)
    for passage_id, row in combined.items():
        review_bonus = 0.002 if str(row.get("review_status") or "").startswith("reviewed") else 0.0
        row["retrieval_score"] = round(scores[passage_id] + review_bonus, 6)
        row["retrieval_channels"] = ",".join(channels[passage_id])
    return sorted(combined.values(), key=lambda row: row["retrieval_score"], reverse=True)


def passage_filters(
    countries: Optional[List[str]], sdg_targets: Optional[List[str]], publication_year: Optional[str],
    validation_status: Optional[str], require_reviewed: bool,
) -> Tuple[List[str], List[Any]]:
    where: List[str] = []
    params: List[Any] = []
    add_in_filter(where, params, "pol.country", countries)
    add_in_filter(where, params, "ps.sdg_target", sdg_targets)
    if publication_year:
        where.append("substr(pol.publication_date, 1, 4) = ?")
        params.append(str(publication_year))
    if validation_status:
        where.append("pol.validation_status = ?")
        params.append(validation_status)
    if require_reviewed:
        where.append("ps.review_status LIKE 'reviewed%'")
    return where, params


def add_policy_filters(
    where: List[str], params: List[Any], countries: Optional[List[str]], sdg_targets: Optional[List[str]],
    publication_year: Optional[str], review_statuses: Optional[List[str]], translation_statuses: Optional[List[str]],
    validation_status: Optional[str], require_reviewed: bool,
) -> None:
    add_in_filter(where, params, "p.country", countries)
    add_sdg_target_filter(where, params, sdg_targets)
    add_in_filter(where, params, "p.review_status", review_statuses)
    add_in_filter(where, params, "p.translation_status", translation_statuses)
    if publication_year:
        where.append("substr(p.publication_date, 1, 4) = ?")
        params.append(str(publication_year))
    if validation_status:
        where.append("p.validation_status = ?")
        params.append(validation_status)
    if require_reviewed:
        where.append("p.review_status LIKE 'reviewed%'")


def distinct_values(db_path: Path, field: str) -> List[str]:
    if field not in POLICY_TABLE_FIELDS:
        raise ValueError(f"Unsupported field: {field}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"SELECT DISTINCT {field} FROM policies WHERE {field} IS NOT NULL AND {field} != '' ORDER BY {field}").fetchall()
    return [str(row[0]) for row in rows]


def database_summary(db_path: Path) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        countries = conn.execute("SELECT COUNT(DISTINCT country) FROM policies").fetchone()[0]
        valid = conn.execute("SELECT COUNT(*) FROM policies WHERE validation_status = 'valid'").fetchone()[0]
        reviewed = conn.execute("SELECT COUNT(*) FROM policies WHERE review_status LIKE 'reviewed%'").fetchone()[0]
        has_passages = table_exists(conn, "passages")
        passages = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0] if has_passages else 0
        meta = dict(conn.execute("SELECT key, value FROM db_meta").fetchall()) if table_exists(conn, "db_meta") else {}
    return {
        "policies": total, "countries": countries, "valid_policies": valid,
        "reviewed_policies": reviewed, "passages": passages, "has_passages": has_passages,
        "schema_version": meta.get("schema_version"), "source_hash": meta.get("source_hash"),
        "built_at": meta.get("built_at"),
    }


def database_is_stale(db_path: Path, input_path: Path) -> bool:
    if not db_path.exists() or not input_path.exists():
        return True
    summary = database_summary(db_path)
    return summary.get("source_hash") != file_sha256(input_path)


def write_database_metadata(
    conn: sqlite3.Connection, input_path: Path, passages_path: Optional[Path],
    policy_count: int, passage_count: int,
) -> None:
    values = {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(input_path.resolve()),
        "source_hash": file_sha256(input_path),
        "passages_source_path": str(passages_path.resolve()) if passages_path and passages_path.exists() else "",
        "passages_source_hash": file_sha256(passages_path) if passages_path and passages_path.exists() else "",
        "policy_count": str(policy_count),
        "passage_count": str(passage_count),
        "built_at": str(time.time()),
        "retrieval_backend": "sqlite_fts5_passages",
    }
    conn.executemany("INSERT INTO db_meta (key, value) VALUES (?, ?)", values.items())


def legacy_policy_passage(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **row,
        "passage_id": f"legacy-{row.get('id')}",
        "policy_id": row.get("id"),
        "text_original": row.get("policy_content"),
        "text_en": row.get("policy_content_en"),
        "summary_original": row.get("policy_summary"),
        "summary_en": row.get("policy_summary_en"),
        "sdg_target": row.get("sdg_targets"),
        "page_start": None,
        "page_end": None,
        "retrieval_score": None,
        "retrieval_channels": "legacy",
    }


def add_in_filter(where: List[str], params: List[Any], column: str, values: Optional[List[str]]) -> None:
    selected = [value for value in values or [] if value]
    if not selected:
        return
    where.append(f"{column} IN ({', '.join('?' for _ in selected)})")
    params.extend(selected)


def add_sdg_target_filter(where: List[str], params: List[Any], values: Optional[List[str]]) -> None:
    selected = [value for value in values or [] if value]
    if not selected:
        return
    clauses = []
    for value in selected:
        clauses.append("(';' || replace(p.sdg_targets, '; ', ';') || ';') LIKE ?")
        params.append(f"%;{value};%")
    where.append("(" + " OR ".join(clauses) + ")")


def add_cjk_like_search(where: List[str], params: List[Any], query: str, alias: str) -> None:
    terms = cjk_search_terms(query)
    columns = [
        f"{alias}.policy_title", f"{alias}.policy_title_en", f"{alias}.policy_content",
        f"{alias}.policy_content_en", f"{alias}.policy_summary", f"{alias}.policy_summary_en",
        f"{alias}.country", f"{alias}.sdg_goal", f"{alias}.sdg_targets",
    ]
    clauses = []
    for term in terms:
        clauses.append("(" + " OR ".join(f"{column} LIKE ?" for column in columns) + ")")
        params.extend(f"%{term}%" for _ in columns)
    if clauses:
        where.append("(" + " OR ".join(clauses) + ")")


def add_cjk_passage_search(where: List[str], params: List[Any], query: str) -> None:
    terms = cjk_search_terms(query)
    columns = [
        "ps.text_original", "ps.text_en", "ps.summary_original", "ps.summary_en",
        "pol.policy_title", "pol.policy_title_en", "pol.country", "ps.sdg_target",
    ]
    clauses = []
    for term in terms:
        clauses.append("(" + " OR ".join(f"{column} LIKE ?" for column in columns) + ")")
        params.extend(f"%{term}%" for _ in columns)
    if clauses:
        where.append("(" + " OR ".join(clauses) + ")")


def cjk_search_terms(query: str) -> List[str]:
    normalized = normalize_sdg_references(query)
    terms = [term for term in DOMAIN_TERMS if term in normalized]
    cleaned = normalized
    for phrase in CJK_STOP_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"[，。？！、；：,.!?;:的了呢吗和与及中]", " ", cleaned)
    terms.extend(part for part in cleaned.split() if len(part) >= 2)
    if "SDG 13" in normalized:
        terms.append("SDG 13")
    return list(dict.fromkeys(terms)) or [normalized.strip()]


def normalize_sdg_references(query: str) -> str:
    return re.sub(r"(?i)\bsdg\s*[-_]?\s*(\d{1,2}(?:\.[0-9a-z]+)?)", lambda match: f"SDG {match.group(1)}", query)


def to_fts_query(query: str, use_or: bool = False) -> str:
    terms = [
        term.strip().replace('"', '""') for term in re.findall(r"[\w.]+", query, flags=re.UNICODE)
        if term.strip() and term.strip().lower() not in FTS_STOPWORDS
    ]
    if not terms:
        return '""'
    operator = " OR " if use_or else " AND "
    return operator.join(f'"{term}"' for term in terms)


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def policy_key(record: Dict[str, Any]) -> str:
    source_file = normalize_text(record.get("source_file"))
    if source_file:
        return f"path:{source_file}"
    return "::".join([
        normalize_text(record.get("country_iso3")) or normalize_text(record.get("country")) or "",
        normalize_text(record.get("source_filename")) or "",
    ])


def split_policy_content(value: Any, max_chars: int = 1800) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    blocks = [item.strip() for item in re.split(r"\n\n---\n\n|\n{2,}", text) if item.strip()]
    parts: List[str] = []
    for block in blocks:
        cursor = 0
        while cursor < len(block):
            end = min(cursor + max_chars, len(block))
            if end < len(block):
                boundary = max(block.rfind(". ", cursor, end), block.rfind("。", cursor, end))
                if boundary > cursor + max_chars // 2:
                    end = boundary + 1
            parts.append(block[cursor:end].strip())
            cursor = end
    return [part for part in parts if part]


def split_targets(value: Any) -> List[str]:
    return [item.strip() for item in re.split(r"[;,]", str(value or "")) if item.strip()]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", [name]).fetchone() is not None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_db_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    return text or None


def normalize_empty(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def normalize_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def to_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None
