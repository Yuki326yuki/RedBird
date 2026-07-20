from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


ACCESS_DATE = "2026-06-18"
OUT_DIR = Path("data/sdg13_policy_collection")

COUNTRIES = [
    {"country": "United States", "iso3": "USA", "income_group": "High-income", "slug": "united-states-of-america", "documents": 43, "targets": 0, "litigation_cases": 348},
    {"country": "Germany", "iso3": "DEU", "income_group": "High-income", "slug": "germany", "documents": 20, "targets": 9, "litigation_cases": 6},
    {"country": "Japan", "iso3": "JPN", "income_group": "High-income", "slug": "japan", "documents": 24, "targets": 19, "litigation_cases": 7},
    {"country": "United Kingdom", "iso3": "GBR", "income_group": "High-income", "slug": "united-kingdom", "documents": 39, "targets": 37, "litigation_cases": 146},
    {"country": "Canada", "iso3": "CAN", "income_group": "High-income", "slug": "canada", "documents": 45, "targets": 10, "litigation_cases": 44},
    {"country": "China", "iso3": "CHN", "income_group": "Upper-middle", "slug": "china", "documents": 55, "targets": 32, "litigation_cases": 2},
    {"country": "Brazil", "iso3": "BRA", "income_group": "Upper-middle", "slug": "brazil", "documents": 35, "targets": 8, "litigation_cases": 50},
    {"country": "South Africa", "iso3": "ZAF", "income_group": "Upper-middle", "slug": "south-africa", "documents": 13, "targets": 10, "litigation_cases": 9},
    {"country": "Mexico", "iso3": "MEX", "income_group": "Upper-middle", "slug": "mexico", "documents": 20, "targets": 9, "litigation_cases": 15},
    {"country": "Indonesia", "iso3": "IDN", "income_group": "Upper-middle", "slug": "indonesia", "documents": 18, "targets": 10, "litigation_cases": 3},
    {"country": "India", "iso3": "IND", "income_group": "Lower-middle", "slug": "india", "documents": 54, "targets": 11, "litigation_cases": 21},
    {"country": "Vietnam", "iso3": "VNM", "income_group": "Lower-middle", "slug": "vietnam", "documents": 17, "targets": 7, "litigation_cases": 0},
    {"country": "Philippines", "iso3": "PHL", "income_group": "Lower-middle", "slug": "philippines", "documents": 16, "targets": 10, "litigation_cases": 4},
    {"country": "Egypt", "iso3": "EGY", "income_group": "Lower-middle", "slug": "egypt", "documents": 12, "targets": 15, "litigation_cases": 1},
    {"country": "Kenya", "iso3": "KEN", "income_group": "Lower-middle", "slug": "kenya", "documents": 13, "targets": 8, "litigation_cases": 3},
    {"country": "Ethiopia", "iso3": "ETH", "income_group": "Low-income", "slug": "ethiopia", "documents": 14, "targets": 11, "litigation_cases": 0},
    {"country": "Uganda", "iso3": "UGA", "income_group": "Low-income", "slug": "uganda", "documents": 13, "targets": 11, "litigation_cases": 2},
    {"country": "Mozambique", "iso3": "MOZ", "income_group": "Low-income", "slug": "mozambique", "documents": 7, "targets": 8, "litigation_cases": 0},
    {"country": "Nepal", "iso3": "NPL", "income_group": "Low-income", "slug": "nepal", "documents": 9, "targets": 3, "litigation_cases": 1},
    {"country": "Madagascar", "iso3": "MDG", "income_group": "Low-income", "slug": "madagascar", "documents": 10, "targets": 3, "litigation_cases": 0},
]

RECENT_DOCUMENTS: Dict[str, List[str]] = {
    "United States": ["United States Mid-Century Strategy for Deep Decarbonization", "United States 2024 Biennial Transparency Report", "New Source Performance Standards for Greenhouse Gas Emissions from New, Modified, and Reconstructed Fossil Fuel-Fired Electric Generating Units"],
    "Germany": ["Federal Climate Change Act", "Climate Action Programme 2030", "Climate Action Plan 2050", "The Energy Concept and its Accelerated Implementation"],
    "Japan": ["Japan's National Greenhouse Gas Emission Reduction Target for 2035 and 2040", "Bill for the Act on Promotion of Supply and Utilisation of Low-Carbon Hydrogen", "Bill on Carbon Dioxide Storage Business"],
    "United Kingdom": ["The Carbon Budget Order 2021", "Net Zero Strategy: Build Back Greener", "Powering Up Britain", "UK Climate Change Act 2008"],
    "Canada": ["2030 Emissions Reduction Plan", "Canadian Net-Zero Emissions Accountability Act", "A Healthy Environment and a Healthy Economy"],
    "China": ["China's Nationally Determined Contribution and Long-term Low Greenhouse Gas Emission Development Strategy", "14th Five-Year Plan for Modern Energy System", "Carbon Peak Action Plan before 2030"],
    "Brazil": ["Brazil's Nationally Determined Contribution", "National Policy on Climate Change", "Amazon Fund and deforestation control plans"],
    "South Africa": ["Climate Change Act", "National Climate Change Adaptation Strategy", "Just Energy Transition Investment Plan"],
    "Mexico": ["General Law on Climate Change", "Special Climate Change Program", "Updated Nationally Determined Contribution"],
    "Indonesia": ["Enhanced Nationally Determined Contribution", "Long-Term Strategy for Low Carbon and Climate Resilience 2050", "Presidential Regulation on Carbon Economic Value"],
    "India": ["India's Updated First Nationally Determined Contribution", "National Action Plan on Climate Change", "Energy Conservation Amendment Act"],
    "Vietnam": ["National Strategy on Climate Change to 2050", "Nationally Determined Contribution 2022", "Power Development Plan VIII"],
    "Philippines": ["Philippine Nationally Determined Contribution", "National Climate Change Action Plan", "Climate Change Act"],
    "Egypt": ["Egypt National Climate Change Strategy 2050", "Updated Nationally Determined Contribution", "Sustainable Energy Strategy 2035"],
    "Kenya": ["Climate Change Act", "National Climate Change Action Plan", "Updated Nationally Determined Contribution"],
    "Ethiopia": ["Ethiopia Updated Nationally Determined Contribution", "Climate Resilient Green Economy Strategy", "National Adaptation Plan"],
    "Uganda": ["Uganda Updated Nationally Determined Contribution", "National Climate Change Act", "National Climate Change Policy"],
    "Mozambique": ["Mozambique Updated Nationally Determined Contribution", "National Climate Change Adaptation and Mitigation Strategy", "Energy Transition Strategy"],
    "Nepal": ["Second Nationally Determined Contribution", "National Climate Change Policy", "Long-term Strategy for Net-zero Emissions"],
    "Madagascar": ["Madagascar Nationally Determined Contribution", "National Adaptation Plan", "Climate Change Policy"],
}

TARGET_KEYWORDS: Dict[str, List[str]] = {
    "United States": ["net-zero by 2050", "2035 power sector carbon pollution standards", "economy-wide emission reductions"],
    "Germany": ["net greenhouse gas neutrality by 2045", "65% reduction by 2030", "88% reduction by 2040"],
    "Japan": ["60% emission reduction by 2035", "73% emission reduction by 2040", "net zero by 2050"],
    "United Kingdom": ["net zero by 2050", "sixth carbon budget", "power sector decarbonisation"],
    "Canada": ["40-45% below 2005 by 2030", "net-zero emissions by 2050", "carbon pricing"],
    "China": ["peak CO2 before 2030", "carbon neutrality before 2060", "non-fossil energy expansion"],
    "Brazil": ["emission reduction under NDC", "deforestation control", "climate adaptation"],
    "South Africa": ["low-emission development", "just energy transition", "carbon budgets"],
    "Mexico": ["emission reduction under NDC", "adaptation measures", "clean energy transition"],
    "Indonesia": ["enhanced NDC emissions target", "FOLU net sink", "low-carbon and climate resilience"],
    "India": ["45% emissions intensity reduction by 2030", "50% installed electric capacity from non-fossil sources", "net zero by 2070"],
    "Vietnam": ["net zero by 2050", "coal transition", "renewable energy expansion"],
    "Philippines": ["NDC mitigation commitment", "climate resilience", "disaster risk reduction"],
    "Egypt": ["NCCS 2050", "renewable energy", "adaptation and resilience"],
    "Kenya": ["32% GHG reduction by 2030", "climate finance", "adaptation and resilience"],
    "Ethiopia": ["climate resilient green economy", "conditional and unconditional NDC targets", "adaptation"],
    "Uganda": ["24.7% emissions reduction by 2030", "adaptation priority sectors", "climate governance"],
    "Mozambique": ["adaptation and mitigation strategy", "renewable energy", "climate resilience"],
    "Nepal": ["net zero by 2045", "electric cooking and transport", "forest cover"],
    "Madagascar": ["adaptation priorities", "mitigation under NDC", "resilience"],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    countries_dir = OUT_DIR / "countries"
    countries_dir.mkdir(exist_ok=True)

    write_countries_csv()
    records = build_policy_records()
    write_csv(OUT_DIR / "sdg13_policy_seed_catalog.csv", records)
    write_jsonl(OUT_DIR / "sdg13_policy_seed_catalog.jsonl", records)
    write_json(OUT_DIR / "country_summaries.json", build_country_summaries())
    write_country_markdown(countries_dir)
    write_readme(records)


def source_url(slug: str) -> str:
    return f"https://climate-laws.org/geographies/{slug}"


def write_countries_csv() -> None:
    rows = [
        {
            **country,
            "source_url": source_url(country["slug"]),
            "access_date": ACCESS_DATE,
        }
        for country in COUNTRIES
    ]
    write_csv(OUT_DIR / "countries_from_excel.csv", rows)


def build_policy_records() -> List[dict]:
    records: List[dict] = []
    for country in COUNTRIES:
        for index, title in enumerate(RECENT_DOCUMENTS[country["country"]], start=1):
            records.append(
                {
                    "record_type": "policy_document_seed",
                    "country": country["country"],
                    "country_iso3": country["iso3"],
                    "income_group": country["income_group"],
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "policy_title": title,
                    "policy_type": infer_policy_type(title),
                    "policy_content_summary": "Potentially relevant to SDG 13 climate action, mitigation, adaptation, climate governance, or low-emission transition.",
                    "evidence_quote": title,
                    "source_name": "Climate Change Laws of the World / Climate Policy Radar",
                    "source_url": source_url(country["slug"]),
                    "source_country_document_count": country["documents"],
                    "source_country_target_count": country["targets"],
                    "confidence": "seed",
                    "access_date": ACCESS_DATE,
                    "notes": "Seed record for follow-up document download and paragraph-level extraction.",
                }
            )
        for index, target in enumerate(TARGET_KEYWORDS[country["country"]], start=1):
            records.append(
                {
                    "record_type": "sdg13_target_seed",
                    "country": country["country"],
                    "country_iso3": country["iso3"],
                    "income_group": country["income_group"],
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.1/13.2/13.3",
                    "policy_title": f"Climate target keyword: {target}",
                    "policy_type": "target",
                    "policy_content_summary": target,
                    "evidence_quote": target,
                    "source_name": "Climate Change Laws of the World / Climate Policy Radar",
                    "source_url": source_url(country["slug"]),
                    "source_country_document_count": country["documents"],
                    "source_country_target_count": country["targets"],
                    "confidence": "seed",
                    "access_date": ACCESS_DATE,
                    "notes": "Target keyword to prioritize during SDG13 extraction.",
                }
            )
    return records


def build_country_summaries() -> List[dict]:
    return [
        {
            **country,
            "source_url": source_url(country["slug"]),
            "recent_documents": RECENT_DOCUMENTS[country["country"]],
            "target_keywords": TARGET_KEYWORDS[country["country"]],
            "access_date": ACCESS_DATE,
        }
        for country in COUNTRIES
    ]


def write_country_markdown(countries_dir: Path) -> None:
    for country in COUNTRIES:
        slug = country["slug"]
        path = countries_dir / f"{slug}.md"
        lines = [
            f"# {country['country']} - SDG13 Policy Seed Data",
            "",
            f"- ISO3: {country['iso3']}",
            f"- Income group from workbook: {country['income_group']}",
            f"- Source: {source_url(slug)}",
            f"- Source document count: {country['documents']}",
            f"- Source target count: {country['targets']}",
            f"- Litigation cases in source: {country['litigation_cases']}",
            f"- Access date: {ACCESS_DATE}",
            "",
            "## Recent / priority policy documents",
        ]
        for title in RECENT_DOCUMENTS[country["country"]]:
            lines.append(f"- {title}")
        lines.extend(["", "## SDG13 target keywords"])
        for target in TARGET_KEYWORDS[country["country"]]:
            lines.append(f"- {target}")
        lines.extend(
            [
                "",
                "## Next extraction step",
                "Use the source URL to download individual policy texts or PDFs, then run the local `sdg-extract` pipeline for paragraph-level JSONL/CSV extraction.",
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(records: List[dict]) -> None:
    by_group: Dict[str, int] = {}
    for country in COUNTRIES:
        by_group[country["income_group"]] = by_group.get(country["income_group"], 0) + 1
    content = [
        "# SDG13 Policy Collection",
        "",
        f"Access date: {ACCESS_DATE}",
        "",
        "This folder contains seed data for collecting SDG 13-related policy documents for the 20 countries listed in the provided workbook.",
        "",
        "## Files",
        "",
        "- `countries_from_excel.csv`: normalized country list from the workbook.",
        "- `sdg13_policy_seed_catalog.csv`: tabular policy and target seed records.",
        "- `sdg13_policy_seed_catalog.jsonl`: JSONL version for pipeline ingestion.",
        "- `country_summaries.json`: country-level metadata, source URLs, document counts, and target keywords.",
        "- `countries/*.md`: one country note per source page.",
        "",
        "## Coverage",
        "",
        f"- Countries: {len(COUNTRIES)}",
        f"- Seed records: {len(records)}",
        f"- Source country policy documents indexed by source pages: {sum(c['documents'] for c in COUNTRIES)}",
        f"- Source country targets indexed by source pages: {sum(c['targets'] for c in COUNTRIES)}",
        "",
        "Income group country counts:",
    ]
    for group, count in sorted(by_group.items()):
        content.append(f"- {group}: {count}")
    content.extend(
        [
            "",
            "## Source",
            "",
            "Primary collection source: Climate Change Laws of the World / Climate Policy Radar country geography pages.",
            "The seed catalog is intended for follow-up document download and paragraph-level extraction with the local framework.",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(content), encoding="utf-8")


def infer_policy_type(title: str) -> str:
    lower = title.lower()
    if "act" in lower or "law" in lower:
        return "law"
    if "strategy" in lower:
        return "strategy"
    if "plan" in lower or "programme" in lower or "program" in lower:
        return "plan"
    if "regulation" in lower or "standards" in lower:
        return "regulation"
    if "contribution" in lower or "ndc" in lower:
        return "ndc"
    if "report" in lower:
        return "report"
    return "policy"


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, rows: List[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
