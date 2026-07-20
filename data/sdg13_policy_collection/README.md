# SDG13 Policy Collection

Access date: 2026-06-18

This folder contains seed data for collecting SDG 13-related policy documents for the 20 countries listed in the provided workbook.

## Files

- `countries_from_excel.csv`: normalized country list from the workbook.
- `sdg13_policy_seed_catalog.csv`: tabular policy and target seed records.
- `sdg13_policy_seed_catalog.jsonl`: JSONL version for pipeline ingestion.
- `country_summaries.json`: country-level metadata, source URLs, document counts, and target keywords.
- `countries/*.md`: one country note per source page.

## Coverage

- Countries: 20
- Seed records: 122
- Source country policy documents indexed by source pages: 477
- Source country targets indexed by source pages: 231

Income group country counts:
- High-income: 5
- Low-income: 5
- Lower-middle: 5
- Upper-middle: 5

## Source

Primary collection source: Climate Change Laws of the World / Climate Policy Radar country geography pages.
The seed catalog is intended for follow-up document download and paragraph-level extraction with the local framework.
