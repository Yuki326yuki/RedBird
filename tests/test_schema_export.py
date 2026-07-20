import csv
import tempfile
import unittest
from pathlib import Path

from sdg_extract.exporters import POLICY_OUTPUT_FIELDS, aggregate_policy_records, export_records
from sdg_extract.io_utils import read_jsonl
from sdg_extract.models import ModelBackend
from sdg_extract.schema import OUTPUT_FIELDS, normalize_records
from sdg_extract.translation import PolicyTranslator


class SchemaExportTests(unittest.TestCase):
    def test_normalize_records_keeps_only_core_fields_and_validates_sdg(self):
        records = normalize_records(
            [
                {
                    "country": "China",
                    "country_iso3": "CHN",
                    "policy_title": "Climate Plan",
                    "publication_date": "2026",
                    "policy_content_original": "Invalid goal example.",
                    "policy_content_summary": "Invalid goal summary.",
                    "sdg_goal": "SDG 19",
                    "sdg_target": "13.2",
                    "sdg_confidence": 0.8,
                    "evidence_quote": "Invalid goal example.",
                    "source_filename": "a.txt",
                    "source_file": "a.txt",
                    "document_id": "doc1",
                    "evidence_paragraph_id": "p000001",
                    "notes": "drop me",
                }
            ],
        )

        self.assertEqual(list(records[0].keys()), OUTPUT_FIELDS)
        self.assertIn("invalid_sdg_goal", records[0]["validation_status"])
        self.assertNotIn("document_id", records[0])
        self.assertNotIn("notes", records[0])

    def test_normalize_records_marks_missing_core_fields(self):
        records = normalize_records(
            [
                {
                    "source_file": "a.txt",
                    "source_filename": "a.txt",
                    "policy_content_original": "Climate policy.",
                    "evidence_quote": "Climate policy.",
                }
            ]
        )

        self.assertIn("missing_country", records[0]["validation_status"])
        self.assertIn("missing_policy_title", records[0]["validation_status"])
        self.assertIn("missing_sdg_goal", records[0]["validation_status"])
        self.assertIn("missing_sdg_target", records[0]["validation_status"])

    def test_aggregate_policy_records_groups_by_source_file(self):
        records = normalize_records(
            [
                {
                    "country": "China",
                    "country_iso3": "CHN",
                    "policy_title": "Climate Plan",
                    "publication_date": "2026",
                    "source_file": "/data/a.pdf",
                    "source_filename": "a.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "sdg_confidence": 0.9,
                    "policy_content_original": "Reduce greenhouse gas emissions.",
                    "policy_content_summary": "Emissions reduction.",
                    "evidence_quote": "greenhouse gas emissions",
                },
                {
                    "country": "China",
                    "country_iso3": "CHN",
                    "policy_title": "Climate Plan",
                    "publication_date": "2026",
                    "source_file": "/data/a.pdf",
                    "source_filename": "a.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.1",
                    "sdg_confidence": 0.8,
                    "policy_content_original": "Build climate resilience.",
                    "policy_content_summary": "Resilience.",
                    "evidence_quote": "climate resilience",
                },
                {
                    "country": "United States",
                    "country_iso3": "USA",
                    "policy_title": "Methane Rule",
                    "publication_date": "2024",
                    "source_file": "/data/b.pdf",
                    "source_filename": "b.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "sdg_confidence": 0.7,
                    "policy_content_original": "Cut methane emissions.",
                    "policy_content_summary": "Methane reduction.",
                    "evidence_quote": "methane emissions",
                },
            ]
        )

        policies = aggregate_policy_records(records)

        self.assertEqual(len(policies), 2)
        self.assertEqual(policies[0]["source_filename"], "a.pdf")
        self.assertEqual(policies[0]["sdg_targets"], "13.1; 13.2")
        self.assertEqual(policies[0]["record_count"], 2)
        self.assertIn("Reduce greenhouse gas emissions.", policies[0]["policy_content"])
        self.assertIn("Build climate resilience.", policies[0]["policy_content"])
        self.assertNotIn("evidence_quotes", policies[0])
        self.assertEqual(policies[1]["source_filename"], "b.pdf")

    def test_aggregate_policy_records_deduplicates_content_and_quotes(self):
        records = normalize_records(
            [
                {
                    "country": "China",
                    "policy_title": "Climate Plan",
                    "source_file": "/data/a.pdf",
                    "source_filename": "a.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "policy_content_original": "Reduce emissions.",
                    "evidence_quote": "Reduce emissions.",
                },
                {
                    "country": "China",
                    "policy_title": "Climate Plan",
                    "source_file": "/data/a.pdf",
                    "source_filename": "a.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "policy_content_original": "Reduce   emissions.",
                    "evidence_quote": "Reduce emissions.",
                },
            ]
        )

        policies = aggregate_policy_records(records)

        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0]["policy_content"], "Reduce emissions.")
        self.assertNotIn("evidence_quotes", policies[0])
        self.assertEqual(policies[0]["record_count"], 2)

    def test_aggregate_policy_records_summarizes_review_fields(self):
        records = normalize_records(
            [
                {
                    "country": "China",
                    "policy_title": "Climate Plan",
                    "source_file": "/data/a.pdf",
                    "source_filename": "a.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "policy_content_original": "Reduce emissions.",
                    "evidence_quote": "Reduce emissions.",
                },
                {
                    "country": "China",
                    "policy_title": "Climate Plan",
                    "source_file": "/data/a.pdf",
                    "source_filename": "a.pdf",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.1",
                    "policy_content_original": "Build climate resilience.",
                    "evidence_quote": "climate resilience",
                },
            ]
        )
        records[0]["review_status"] = "reviewed"
        records[0]["review_confidence"] = 0.9
        records[1]["review_status"] = "reviewed_with_revision"
        records[1]["review_confidence"] = 0.8

        policies = aggregate_policy_records(records)

        self.assertEqual(policies[0]["review_status"], "reviewed_with_revisions")
        self.assertEqual(policies[0]["review_confidence"], 0.85)

    def test_export_jsonl_and_csv_have_policy_level_fields(self):
        records = normalize_records(
            [
                {
                    "country": "China",
                    "country_iso3": "CHN",
                    "policy_title": "Climate Plan",
                    "publication_date": "2026",
                    "source_file": "a.txt",
                    "source_filename": "a.txt",
                    "source_type": "txt",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "sdg_confidence": 0.9,
                    "policy_content_original": "Climate policy.",
                    "policy_content_summary": "Climate summary.",
                    "evidence_quote": "Climate policy.",
                    "evidence_paragraph_id": "p000001",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            written = export_records(records, Path(tmp), ["jsonl", "csv"])

            self.assertEqual({path.name for path in written}, {"records.jsonl", "records.csv"})
            jsonl_records = read_jsonl(Path(tmp) / "records.jsonl")
            with (Path(tmp) / "records.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                csv_rows = list(reader)

            self.assertEqual(list(csv_rows[0].keys()), POLICY_OUTPUT_FIELDS)
            self.assertEqual(set(jsonl_records[0].keys()), set(POLICY_OUTPUT_FIELDS))
            self.assertEqual(csv_rows[0]["source_filename"], "a.txt")
            self.assertEqual(csv_rows[0]["sdg_targets"], "13.2")
            self.assertEqual(csv_rows[0]["record_count"], "1")
            self.assertIn("policy_title_en", csv_rows[0])
            self.assertIn("review_status", csv_rows[0])
            self.assertIn("review_confidence", csv_rows[0])
            self.assertIn("translation_status", csv_rows[0])
            self.assertNotIn("source_type", jsonl_records[0])
            self.assertNotIn("evidence_paragraph_id", jsonl_records[0])
            self.assertNotIn("evidence_quotes", jsonl_records[0])

    def test_export_applies_translation_without_overwriting_original_fields(self):
        class DummyTranslator:
            def translate_records(self, records):
                translated = []
                for record in records:
                    row = dict(record)
                    row["policy_title_en"] = "Climate Plan"
                    row["policy_content_en"] = "Climate policy."
                    row["policy_summary_en"] = "Climate summary."
                    row["translation_model"] = "dummy-translator"
                    row["translation_status"] = "translated"
                    translated.append(row)
                return translated

        records = normalize_records(
            [
                {
                    "country": "China",
                    "policy_title": "气候规划",
                    "source_file": "a.txt",
                    "source_filename": "a.txt",
                    "sdg_goal": "SDG 13",
                    "sdg_target": "13.2",
                    "policy_content_original": "气候政策。",
                    "policy_content_summary": "气候摘要。",
                    "evidence_quote": "气候政策。",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            export_records(records, Path(tmp), ["jsonl"], translator=DummyTranslator())
            jsonl_records = read_jsonl(Path(tmp) / "records.jsonl")

        self.assertEqual(jsonl_records[0]["policy_title"], "气候规划")
        self.assertEqual(jsonl_records[0]["policy_title_en"], "Climate Plan")
        self.assertEqual(jsonl_records[0]["translation_model"], "dummy-translator")
        self.assertEqual(jsonl_records[0]["translation_status"], "translated")


class TranslationTests(unittest.TestCase):
    def test_policy_translator_uses_cache_on_repeated_records(self):
        class CountingBackend(ModelBackend):
            def __init__(self):
                self.calls = 0

            def complete_json(self, messages, schema_hint):
                self.calls += 1
                return {
                    "policy_title_en": "Climate Plan",
                    "policy_content_en": "Climate policy.",
                    "policy_summary_en": "Climate summary.",
                }

            def model_name(self):
                return "counting"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.md"
            prompt.write_text("Translate.", encoding="utf-8")
            backend = CountingBackend()
            translator = PolicyTranslator(backend, prompt, root / "translations.jsonl")
            records = [
                {
                    "source_filename": "a.pdf",
                    "policy_title": "气候规划",
                    "policy_content": "气候政策。",
                    "policy_summary": "气候摘要。",
                }
            ]

            first = translator.translate_records(records)
            second = translator.translate_records(records)

        self.assertEqual(backend.calls, 1)
        self.assertEqual(first[0]["policy_content_en"], "Climate policy.")
        self.assertEqual(second[0]["policy_content_en"], "Climate policy.")
        self.assertEqual(second[0]["translation_status"], "translated")

    def test_policy_translator_marks_errors_without_raising(self):
        class FailingBackend(ModelBackend):
            def complete_json(self, messages, schema_hint):
                raise RuntimeError("boom")

            def model_name(self):
                return "failing"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.md"
            prompt.write_text("Translate.", encoding="utf-8")
            translator = PolicyTranslator(FailingBackend(), prompt, root / "translations.jsonl")

            translated = translator.translate_records(
                [
                    {
                        "source_filename": "a.pdf",
                        "policy_title": "气候规划",
                        "policy_content": "气候政策。",
                        "policy_summary": None,
                    }
                ]
            )

        self.assertIsNone(translated[0]["policy_title_en"])
        self.assertIsNone(translated[0]["policy_content_en"])
        self.assertTrue(translated[0]["translation_status"].startswith("error:"))


if __name__ == "__main__":
    unittest.main()
