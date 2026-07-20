from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .exporters import POLICY_OUTPUT_FIELDS
from .io_utils import ensure_dir, read_jsonl


POLICY_TABLE_FIELDS = ["id", *POLICY_OUTPUT_FIELDS]
FTS_FIELDS = [
    "policy_title",
    "policy_title_en",
    "policy_content",
    "policy_content_en",
    "policy_summary",
    "policy_summary_en",
    "country",
    "sdg_targets",
]
FTS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "does",
    "for",
    "in",
    "is",
    "mention",
    "mentions",
    "of",
    "on",
    "policies",
    "policy",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def build_database(input_path: Path, db_path: Path) -> int:
    records = load_policy_records(input_path)
    ensure_dir(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        initialize_database(conn)
        insert_policies(conn, records)
    return len(records)


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS policies_fts")
    conn.execute("DROP TABLE IF EXISTS policies")
    columns = ", ".join(f"{field} TEXT" for field in POLICY_OUTPUT_FIELDS)
    conn.execute(
        f"""
        CREATE TABLE policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns}
        )
        """
    )
    fts_columns = ", ".join(FTS_FIELDS)
    conn.execute(f"CREATE VIRTUAL TABLE policies_fts USING fts5({fts_columns}, content='')")
    conn.execute("CREATE INDEX idx_policies_country ON policies(country)")
    conn.execute("CREATE INDEX idx_policies_country_iso3 ON policies(country_iso3)")
    conn.execute("CREATE INDEX idx_policies_validation_status ON policies(validation_status)")
    conn.execute("CREATE INDEX idx_policies_review_status ON policies(review_status)")
    conn.execute("CREATE INDEX idx_policies_translation_status ON policies(translation_status)")


def insert_policies(conn: sqlite3.Connection, records: Iterable[Dict[str, Any]]) -> None:
    fields_sql = ", ".join(POLICY_OUTPUT_FIELDS)
    placeholders = ", ".join("?" for _ in POLICY_OUTPUT_FIELDS)
    fts_fields_sql = ", ".join(["rowid", *FTS_FIELDS])
    fts_placeholders = ", ".join("?" for _ in ["rowid", *FTS_FIELDS])
    for record in records:
        values = [to_db_value(record.get(field)) for field in POLICY_OUTPUT_FIELDS]
        cursor = conn.execute(
            f"INSERT INTO policies ({fields_sql}) VALUES ({placeholders})",
            values,
        )
        rowid = cursor.lastrowid
        fts_values = [rowid, *[to_db_value(record.get(field)) for field in FTS_FIELDS]]
        conn.execute(
            f"INSERT INTO policies_fts ({fts_fields_sql}) VALUES ({fts_placeholders})",
            fts_values,
        )


def load_policy_records(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return [project_policy_record(record) for record in read_jsonl(input_path)]
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            return [project_policy_record(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported database input format: {input_path}")


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
    where: List[str] = []
    params: List[Any] = []
    joins = ""

    use_like_search = bool(query and query.strip() and contains_cjk(query))
    if query and query.strip() and not use_like_search:
        joins = "JOIN policies_fts ON policies_fts.rowid = p.id"
        where.append("policies_fts MATCH ?")
        params.append(to_fts_query(query))
    elif query and query.strip():
        add_like_search(where, params, query)
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
        where.append("p.review_status IS NOT NULL")
        where.append("p.review_status != ''")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    order_sql = (
        "ORDER BY rank"
        if query and query.strip() and not use_like_search
        else "ORDER BY p.country, p.publication_date DESC, p.policy_title"
    )
    sql = f"""
        SELECT p.*
        FROM policies p
        {joins}
        {where_sql}
        {order_sql}
        LIMIT ?
    """
    params.append(int(limit))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def distinct_values(db_path: Path, field: str) -> List[str]:
    if field not in POLICY_TABLE_FIELDS:
        raise ValueError(f"Unsupported field: {field}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {field} FROM policies WHERE {field} IS NOT NULL AND {field} != '' ORDER BY {field}"
        ).fetchall()
    return [str(row[0]) for row in rows]


def database_summary(db_path: Path) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        countries = conn.execute("SELECT COUNT(DISTINCT country) FROM policies").fetchone()[0]
    return {"policies": total, "countries": countries}


def add_in_filter(where: List[str], params: List[Any], column: str, values: Optional[List[str]]) -> None:
    selected = [value for value in values or [] if value]
    if not selected:
        return
    placeholders = ", ".join("?" for _ in selected)
    where.append(f"{column} IN ({placeholders})")
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


def add_like_search(where: List[str], params: List[Any], query: str) -> None:
    pattern = f"%{query.strip()}%"
    columns = [
        "p.policy_title",
        "p.policy_title_en",
        "p.policy_content",
        "p.policy_content_en",
        "p.policy_summary",
        "p.policy_summary_en",
        "p.country",
        "p.sdg_targets",
    ]
    where.append("(" + " OR ".join(f"{column} LIKE ?" for column in columns) + ")")
    params.extend(pattern for _ in columns)


def to_fts_query(query: str) -> str:
    terms = [
        term.strip().replace('"', '""')
        for term in query.split()
        if term.strip() and term.strip().lower() not in FTS_STOPWORDS
    ]
    if not terms:
        terms = [term.strip().replace('"', '""') for term in query.split() if term.strip()]
    if not terms:
        return '""'
    return " OR ".join(f'"{term}"' for term in terms)


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


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
