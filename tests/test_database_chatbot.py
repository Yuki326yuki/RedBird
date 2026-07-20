import csv
import tempfile
import unittest
from pathlib import Path

from sdg_extract.chatbot import answer_question
from sdg_extract.database import build_database, database_summary, query_policies
from sdg_extract.exporters import POLICY_OUTPUT_FIELDS
from sdg_extract.io_utils import write_jsonl


class DatabaseTests(unittest.TestCase):
    def test_build_database_from_jsonl_and_query_fts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(input_path, [sample_policy(policy_content_en="Methane emissions reduction plan.")])

            count = build_database(input_path, db_path)
            rows = query_policies(db_path, query="methane emissions")

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["country"], "United States")
        self.assertEqual(rows[0]["policy_title_en"], "Methane Rule")

    def test_build_database_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.csv"
            db_path = root / "redbird.sqlite"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=POLICY_OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerow(sample_policy())

            count = build_database(input_path, db_path)
            summary = database_summary(db_path)

        self.assertEqual(count, 1)
        self.assertEqual(summary["policies"], 1)
        self.assertEqual(summary["countries"], 1)

    def test_query_filters_country_target_and_review_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(
                input_path,
                [
                    sample_policy(country="United States", sdg_targets="13.2", review_status="reviewed"),
                    sample_policy(country="China", sdg_targets="13.1", review_status="reviewed_with_revisions"),
                ],
            )
            build_database(input_path, db_path)

            rows = query_policies(
                db_path,
                countries=["China"],
                sdg_targets=["13.1"],
                review_statuses=["reviewed_with_revisions"],
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["country"], "China")
        self.assertEqual(rows[0]["sdg_targets"], "13.1")

    def test_query_chinese_text_uses_like_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(input_path, [sample_policy(policy_content="中国气候政策要求减少排放。")])
            build_database(input_path, db_path)

            rows = query_policies(db_path, query="气候政策")

        self.assertEqual(len(rows), 1)
        self.assertIn("中国气候政策", rows[0]["policy_content"])


class ChatbotTests(unittest.TestCase):
    def test_chatbot_uses_top_k_context_and_sources(self):
        calls = []

        def fake_completion(messages, config):
            calls.append(messages)
            payload = messages[-1]["content"]
            self.assertIn("Methane Rule", payload)
            return "The database indicates methane emissions are addressed."

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Answer from records only.", encoding="utf-8")
            write_jsonl(input_path, [sample_policy(policy_content_en="Methane emissions reduction plan.")])
            build_database(input_path, db_path)

            result = answer_question(
                db_path,
                "What policies mention methane emissions?",
                prompt_path,
                config={"top_k": 1, "model": {"backend": "api", "name": "fake"}},
                completion_fn=fake_completion,
            )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["sources"][0]["source_filename"], "methane-rule.pdf")

    def test_chatbot_returns_no_evidence_without_api_call(self):
        def fake_completion(messages, config):
            raise AssertionError("completion should not be called")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Answer from records only.", encoding="utf-8")
            write_jsonl(input_path, [sample_policy(policy_content_en="Methane emissions reduction plan.")])
            build_database(input_path, db_path)

            result = answer_question(
                db_path,
                "ocean shipping",
                prompt_path,
                completion_fn=fake_completion,
            )

        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["sources"], [])

    def test_chatbot_reports_api_error_with_sources(self):
        def failing_completion(messages, config):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Answer from records only.", encoding="utf-8")
            write_jsonl(input_path, [sample_policy(policy_content_en="Methane emissions reduction plan.")])
            build_database(input_path, db_path)

            result = answer_question(
                db_path,
                "methane emissions",
                prompt_path,
                completion_fn=failing_completion,
            )

        self.assertTrue(result["status"].startswith("error:"))
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("Chatbot API error", result["answer"])


def sample_policy(**overrides):
    record = {
        "country": "United States",
        "country_iso3": "USA",
        "policy_title": "Methane Rule",
        "policy_title_en": "Methane Rule",
        "publication_date": "2024-04-25",
        "policy_content": "Methane policy.",
        "policy_content_en": "Methane policy.",
        "policy_summary": "Methane reduction.",
        "policy_summary_en": "Methane reduction.",
        "sdg_goal": "SDG 13",
        "sdg_targets": "13.2",
        "source_filename": "methane-rule.pdf",
        "source_file": "/data/methane-rule.pdf",
        "record_count": "1",
        "review_status": "reviewed",
        "review_confidence": "0.9",
        "translation_model": "translator",
        "translation_status": "translated",
        "validation_status": "valid",
    }
    record.update(overrides)
    return record


if __name__ == "__main__":
    unittest.main()
