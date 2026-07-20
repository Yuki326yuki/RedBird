from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from download_sdg13_policy_texts import MANIFEST, MANIFEST_FIELDS, OUT_DIR, SUMMARY


BACKUP = MANIFEST.with_suffix(".raw.csv")


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"No manifest found: {MANIFEST}")
    raw_rows = list(csv.reader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if not raw_rows:
        raise SystemExit("Manifest is empty")
    if not BACKUP.exists():
        BACKUP.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")

    header = raw_rows[0]
    normalized: Dict[str, Dict[str, str]] = {}
    for raw in raw_rows[1:]:
        if not raw:
            continue
        row = normalize_row(header, raw)
        local_path = row.get("local_path", "")
        path = Path(local_path)
        if local_path and path.exists():
            row["file_size_bytes"] = str(path.stat().st_size)
            if not row.get("download_status"):
                row["download_status"] = "exists"
            normalized[local_path] = row

    for path in sorted(p for p in OUT_DIR.rglob("*") if p.is_file()):
        local_path = str(path)
        if local_path not in normalized:
            parts = path.relative_to(OUT_DIR).parts
            slug = parts[0] if parts else ""
            normalized[local_path] = {
                field: "" for field in MANIFEST_FIELDS
            }
            normalized[local_path].update(
                {
                    "country": slug.replace("-", " ").title(),
                    "document_slug": path.stem,
                    "local_path": local_path,
                    "file_size_bytes": str(path.stat().st_size),
                    "download_status": "exists",
                }
            )

    rows = sorted(normalized.values(), key=lambda item: (item.get("country", ""), item.get("local_path", "")))
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "clean_manifest_rows": len(rows),
        "files_on_disk": sum(1 for _ in OUT_DIR.rglob("*") if _.is_file()),
        "bytes_on_disk": sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file()),
        "manifest": str(MANIFEST),
        "backup": str(BACKUP),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def normalize_row(header: List[str], raw: List[str]) -> Dict[str, str]:
    row = {field: "" for field in MANIFEST_FIELDS}
    mapped = {key: raw[index] if index < len(raw) else "" for index, key in enumerate(header)}
    row.update({key: mapped.get(key, "") for key in MANIFEST_FIELDS})
    if mapped.get("download_status") in {"downloaded", "exists"}:
        return row

    shifted = shifted_row(raw)
    local_path = shifted.get("local_path", "")
    if local_path and Path(local_path).exists():
        return shifted
    return row


def shifted_row(raw: List[str]) -> Dict[str, str]:
    row = {field: "" for field in MANIFEST_FIELDS}
    if len(raw) < 18:
        return row
    # Rows written while MANIFEST_FIELDS temporarily placed content_type_downloaded
    # before source_url are shifted relative to the persisted header.
    for index, field in enumerate(MANIFEST_FIELDS):
        if index < len(raw):
            row[field] = raw[index]
    return row


if __name__ == "__main__":
    main()
