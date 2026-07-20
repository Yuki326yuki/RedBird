from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COUNTRY_CSV = Path("data/sdg13_policy_collection/countries_from_excel.csv")
OUT_DIR = Path("data/sdg13_policy_collection/original_documents")
MANIFEST = Path("data/sdg13_policy_collection/download_manifest.csv")
FAILURES = Path("data/sdg13_policy_collection/download_failures.csv")
SUMMARY = Path("data/sdg13_policy_collection/download_summary.json")
ENV_SOURCE = "https://climate-laws.org/geographies/india"
MANIFEST_FIELDS = [
    "country",
    "country_iso3",
    "income_group",
    "family_slug",
    "family_title",
    "family_category",
    "published_date",
    "document_slug",
    "document_title",
    "document_type",
    "document_role",
    "language",
    "content_type",
    "source_url",
    "download_url",
    "local_path",
    "file_size_bytes",
    "download_status",
    "content_type_downloaded",
]
FAILURE_FIELDS = ["country", "country_iso3", "family_slug", "document_slug", "url", "stage", "error"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download original SDG13/climate policy source files.")
    parser.add_argument("--countries", default="", help="Comma-separated country slugs or names. Empty means all.")
    parser.add_argument("--max-per-country", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    env = load_env_config()
    countries = load_countries()
    selected = select_countries(countries, args.countries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    for country in selected:
        print(f"== {country['country']} ({country['slug']}) ==")
        country_manifest: List[Dict[str, Any]] = []
        country_failures: List[Dict[str, Any]] = []
        try:
            families = fetch_country_families(env, country["slug"], limit=args.max_per_country)
        except Exception as exc:
            country_failures.append(failure(country, "", "", "", "country_search", f"{type(exc).__name__}: {exc}"))
            append_csv(FAILURES, country_failures)
            failure_rows.extend(country_failures)
            print(f"country search failed: {exc}")
            continue
        print(f"families selected: {len(families)}")
        downloaded_for_country = 0
        for family in families:
            family_slug = family["family_slug"]
            try:
                family_detail = get_json(env, f"/documents/{family_slug}")
                documents = family_detail.get("documents", [])
            except Exception as exc:
                country_failures.append(failure(country, family_slug, "", "", "family_detail", str(exc)))
                continue
            for document in documents:
                if document.get("document_role") not in {None, "", "MAIN"}:
                    continue
                row, error = download_document(env, country, family_detail, document, overwrite=args.overwrite)
                if row:
                    country_manifest.append(row)
                    downloaded_for_country += int(row["download_status"] == "downloaded")
                if error:
                    country_failures.append(error)
                time.sleep(args.sleep)
        append_csv(MANIFEST, country_manifest)
        append_csv(FAILURES, country_failures)
        manifest_rows.extend(country_manifest)
        failure_rows.extend(country_failures)
        print(f"new downloads: {downloaded_for_country}")
        time.sleep(args.sleep)

    write_summary(selected, manifest_rows, failure_rows)
    print(f"manifest rows written: {len(manifest_rows)}")
    print(f"failures written: {len(failure_rows)}")
    print(f"output directory: {OUT_DIR}")


def load_env_config() -> Dict[str, str]:
    html = fetch_text(ENV_SOURCE)
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not match:
        raise RuntimeError("Could not find __NEXT_DATA__ in climate-laws page.")
    data = json.loads(match.group(1))
    return data["props"]["pageProps"]["envConfig"]


def load_countries() -> List[Dict[str, str]]:
    with COUNTRY_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_countries(countries: List[Dict[str, str]], selector: str) -> List[Dict[str, str]]:
    if not selector.strip():
        return countries
    wanted = {normalize_slug(item) for item in selector.split(",") if item.strip()}
    selected = [
        country
        for country in countries
        if normalize_slug(country["slug"]) in wanted or normalize_slug(country["country"]) in wanted
    ]
    if not selected:
        raise ValueError(f"No countries matched selector: {selector}")
    return selected


def fetch_country_families(env: Dict[str, str], slug: str, limit: int) -> List[Dict[str, Any]]:
    families: List[Dict[str, Any]] = []
    continuation_tokens: List[str] = []
    seen = set()
    while len(families) < limit:
        payload = search_payload(slug, page_size=min(100, max(limit - len(families), 1)), tokens=continuation_tokens)
        response = post_json(env, "/searches", payload)
        for family in response.get("families", []):
            family_slug = family.get("family_slug")
            if family_slug and family_slug not in seen and family_relevant_to_sdg13(family):
                families.append(family)
                seen.add(family_slug)
                if len(families) >= limit:
                    break
        token = response.get("continuation_token")
        if not token or token in continuation_tokens:
            break
        continuation_tokens.append(token)
    return families


def search_payload(slug: str, page_size: int, tokens: List[str]) -> Dict[str, Any]:
    return {
        "query_string": "climate OR emissions OR adaptation OR mitigation OR carbon OR net zero OR NDC OR SDG 13",
        "exact_match": False,
        "keyword_filters": {
            "countries": [slug],
            "regions": [],
            "subdivisions": [],
            "categories": [],
            "sources": [],
            "languages": [],
        },
        "sort_field": "date",
        "sort_order": "desc",
        "page_size": page_size,
        "offset": 0,
        "max_passages_per_doc": 0,
        "continuation_tokens": tokens,
        "corpus_import_ids": [],
        "metadata": [],
    }


def family_relevant_to_sdg13(family: Dict[str, Any]) -> bool:
    text = " ".join(
        str(family.get(key) or "")
        for key in ["family_name", "family_description", "family_category", "family_source", "corpus_type_name"]
    ).lower()
    terms = [
        "climate",
        "emission",
        "greenhouse",
        "carbon",
        "adaptation",
        "mitigation",
        "energy",
        "renewable",
        "resilience",
        "ndc",
        "transparency report",
        "net zero",
    ]
    return any(term in text for term in terms)


def download_document(
    env: Dict[str, str],
    country: Dict[str, str],
    family: Dict[str, Any],
    document: Dict[str, Any],
    overwrite: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    url = document.get("cdn_object") or document.get("document_cdn_object")
    if not url:
        url = document.get("source_url") or document.get("document_source_url")
    if not url:
        return None, failure(country, family.get("slug", ""), document.get("slug", ""), "", "missing_url", "No source URL")
    if isinstance(url, str) and url.startswith("/") is False and not url.startswith("http"):
        url = env.get("CDN_URL", "https://cdn.climatepolicyradar.org").rstrip("/") + "/navigator/" + url.lstrip("/")

    doc_slug = document.get("slug") or document.get("document_slug") or hash_text(document.get("title", "document"))
    ext = extension_for(document, url)
    country_dir = OUT_DIR / country["slug"]
    country_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(f"{doc_slug}{ext}")
    path = country_dir / filename
    if path.exists() and not overwrite:
        return manifest_row(country, family, document, url, path, "exists", path.stat().st_size), None

    try:
        data, final_url, content_type = fetch_binary(url)
        path.write_bytes(data)
        row = manifest_row(country, family, document, final_url, path, "downloaded", len(data))
        row["content_type_downloaded"] = content_type
        return row, None
    except Exception as exc:
        return None, failure(
            country,
            family.get("slug", ""),
            doc_slug,
            url,
            "download",
            f"{type(exc).__name__}: {exc}",
        )


def get_json(env: Dict[str, str], path: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        env["BACKEND_API_URL"].rstrip("/") + path,
        headers={"app-token": env["BACKEND_API_TOKEN"], "Accept": "application/json", "User-Agent": user_agent()},
    )
    with urlopen_with_retry(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(env: Dict[str, str], path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        env["BACKEND_API_URL"].rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "app-token": env["BACKEND_API_TOKEN"],
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": user_agent(),
        },
        method="POST",
    )
    with urlopen_with_retry(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urlopen_with_retry(request, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_binary(url: str) -> Tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urlopen_with_retry(request, timeout=120) as response:
        return response.read(), response.geturl(), response.headers.get("content-type", "")


def urlopen_with_retry(request: urllib.request.Request, timeout: int, attempts: int = 3):
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(1.5 * attempt)
    raise last_error or RuntimeError("request failed")


def manifest_row(
    country: Dict[str, str],
    family: Dict[str, Any],
    document: Dict[str, Any],
    url: str,
    path: Path,
    status: str,
    size: int,
) -> Dict[str, Any]:
    return {
        "country": country["country"],
        "country_iso3": country["iso3"],
        "income_group": country["income_group"],
        "family_slug": family.get("slug") or family.get("family_slug", ""),
        "family_title": family.get("title") or family.get("family_name", ""),
        "family_category": family.get("category") or family.get("family_category", ""),
        "published_date": family.get("published_date") or family.get("family_date", ""),
        "document_slug": document.get("slug") or document.get("document_slug", ""),
        "document_title": document.get("title") or document.get("document_title", ""),
        "document_type": document.get("document_type", ""),
        "document_role": document.get("document_role", ""),
        "language": document.get("language", ""),
        "content_type": document.get("content_type") or document.get("document_content_type", ""),
        "content_type_downloaded": "",
        "source_url": document.get("source_url") or document.get("document_source_url", ""),
        "download_url": url,
        "local_path": str(path),
        "file_size_bytes": size,
        "download_status": status,
    }


def failure(
    country: Dict[str, str],
    family_slug: str,
    document_slug: str,
    url: str,
    stage: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "country": country["country"],
        "country_iso3": country["iso3"],
        "family_slug": family_slug,
        "document_slug": document_slug,
        "url": url,
        "stage": stage,
        "error": error,
    }


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = FAILURE_FIELDS if path == FAILURES else MANIFEST_FIELDS
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_summary(countries: List[Dict[str, str]], manifest_rows: List[Dict[str, Any]], failure_rows: List[Dict[str, Any]]) -> None:
    summary = {
        "countries_requested": len(countries),
        "manifest_rows_this_run": len(manifest_rows),
        "new_downloads_this_run": sum(1 for row in manifest_rows if row["download_status"] == "downloaded"),
        "existing_files_this_run": sum(1 for row in manifest_rows if row["download_status"] == "exists"),
        "failures_this_run": len(failure_rows),
        "output_dir": str(OUT_DIR),
        "manifest": str(MANIFEST),
        "failures": str(FAILURES),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def extension_for(document: Dict[str, Any], url: str) -> str:
    content_type = str(document.get("content_type") or document.get("document_content_type") or "").lower()
    parsed_ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if parsed_ext in {".pdf", ".html", ".htm", ".txt", ".doc", ".docx", ".xls", ".xlsx", ".csv"}:
        return parsed_ext
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    if "word" in content_type:
        return ".docx"
    if "spreadsheet" in content_type or "excel" in content_type:
        return ".xlsx"
    return ".bin"


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value[:180] or "document.bin"


def normalize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def user_agent() -> str:
    return "RedBirdSDG13Research/0.1 (local academic policy research)"


if __name__ == "__main__":
    main()
