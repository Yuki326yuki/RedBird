import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sdg_extract.extraction import (
    dedupe_records,
    extract_records,
    filter_strict_sdg13_records,
    is_sdg13_relevant_chunk,
    sanitize_raw_record,
)
from sdg_extract.chunking import chunk_document
from sdg_extract.io_utils import read_jsonl
from sdg_extract.models import ModelBackend
from sdg_extract.models import HeuristicBackend
from sdg_extract.parsers import parse_file
from sdg_extract.review import RecordReviewer


class ExtractionTests(unittest.TestCase):
    def test_heuristic_backend_uses_common_model_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "system.md").write_text("Extract facts only.", encoding="utf-8")
            (prompt_dir / "extraction.md").write_text("Return JSON.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text(
                "The national climate plan aligns with SDG 13.2 and mitigation policy.",
                encoding="utf-8",
            )
            document = parse_file(
                policy,
                metadata={
                    "country": "China",
                    "country_iso3": "CHN",
                    "policy_title": "Climate Plan",
                    "source_filename": "policy.txt",
                },
            )
            backend = HeuristicBackend({"name": "heuristic"})

            records = extract_records([document], backend, prompt_dir, max_chars=1000)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["sdg_goal"], "SDG 13")
            self.assertEqual(records[0]["sdg_target"], "13.2")
            self.assertNotIn("extraction_model", records[0])
            self.assertNotIn("document_id", records[0])
            self.assertEqual(records[0]["country"], "China")
            self.assertEqual(records[0]["country_iso3"], "CHN")
            self.assertEqual(records[0]["policy_title"], "Climate Plan")
            self.assertEqual(records[0]["source_filename"], "policy.txt")
            self.assertEqual(
                records[0]["policy_content_original"],
                "The national climate plan aligns with SDG 13.2 and mitigation policy.",
            )

    def test_extract_records_prints_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "system.md").write_text("Extract facts only.", encoding="utf-8")
            (prompt_dir / "extraction.md").write_text("Return JSON.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("The climate plan aligns with SDG 13.2.", encoding="utf-8")
            document = parse_file(policy)
            backend = HeuristicBackend({"name": "heuristic"})
            output = io.StringIO()

            with redirect_stdout(output):
                extract_records([document], backend, prompt_dir, max_chars=1000)

            progress = output.getvalue()
            self.assertIn("[PROGRESS] Starting extraction: 1 chunks", progress)
            self.assertIn("[PROGRESS] chunk 1/1", progress)

    def test_extract_records_writes_checkpoint_and_records_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "system.md").write_text("Extract facts only.", encoding="utf-8")
            (prompt_dir / "extraction.md").write_text("Return JSON.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("The climate plan aligns with SDG 13.2.", encoding="utf-8")
            document = parse_file(policy)
            backend = HeuristicBackend({"name": "heuristic"})
            records_path = root / "records.jsonl"
            checkpoint_path = root / "checkpoint.json"
            errors_path = root / "errors.jsonl"

            records = extract_records(
                [document],
                backend,
                prompt_dir,
                max_chars=1000,
                records_path=records_path,
                checkpoint_path=checkpoint_path,
                errors_path=errors_path,
            )

            self.assertEqual(len(records), 1)
            self.assertTrue(records_path.exists())
            self.assertIn('"completed_chunks": 1', checkpoint_path.read_text(encoding="utf-8"))
            self.assertFalse(errors_path.exists())

    def test_extract_records_continues_after_chunk_error(self):
        class FailingBackend(HeuristicBackend):
            def complete_json(self, messages, schema_hint):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "system.md").write_text("Extract facts only.", encoding="utf-8")
            (prompt_dir / "extraction.md").write_text("Return JSON.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("The climate plan aligns with SDG 13.2.", encoding="utf-8")
            document = parse_file(policy)
            errors_path = root / "errors.jsonl"

            records = extract_records(
                [document],
                FailingBackend({"name": "failing"}),
                prompt_dir,
                max_chars=1000,
                records_path=root / "records.jsonl",
                checkpoint_path=root / "checkpoint.json",
                errors_path=errors_path,
            )

            self.assertEqual(records, [])
            self.assertIn('"error_type": "RuntimeError"', errors_path.read_text(encoding="utf-8"))

    def test_extract_records_respects_max_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "system.md").write_text("Extract facts only.", encoding="utf-8")
            (prompt_dir / "extraction.md").write_text("Return JSON.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text(
                "The first climate plan aligns with SDG 13.2.\n\n"
                "The second climate plan aligns with SDG 13.2.",
                encoding="utf-8",
            )
            document = parse_file(policy)
            backend = HeuristicBackend({"name": "heuristic"})

            records = extract_records([document], backend, prompt_dir, max_chars=50, max_chunks=1)

            self.assertEqual(len(records), 1)
            self.assertIn("first climate plan", records[0]["policy_content_original"])

    def test_sdg13_chunk_filter_rejects_unrelated_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.txt"
            policy.write_text("Digital services and elderly care improvements.", encoding="utf-8")
            document = parse_file(policy)
            chunk = chunk_document(document, max_chars=1000)[0]

            self.assertFalse(is_sdg13_relevant_chunk(chunk))

    def test_sdg13_chunk_filter_accepts_climate_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.txt"
            policy.write_text("The climate mitigation plan reduces greenhouse gas emissions.", encoding="utf-8")
            document = parse_file(policy)
            chunk = chunk_document(document, max_chars=1000)[0]

            self.assertTrue(is_sdg13_relevant_chunk(chunk))

    def test_strict_sdg13_filter_drops_missing_goal_and_target(self):
        records = [
            {
                "sdg_goal": None,
                "sdg_target": None,
                "policy_content_original": "Generic policy.",
                "evidence_quote": "Generic policy.",
            },
            {
                "sdg_goal": "SDG 13",
                "sdg_target": "13.2",
                "policy_content_original": "Climate policy reduces emissions.",
                "evidence_quote": "reduces emissions",
            },
        ]

        filtered = filter_strict_sdg13_records(records)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["sdg_target"], "13.2")

    def test_sanitize_raw_record_normalizes_empty_confidence_and_enums(self):
        record = sanitize_raw_record(
            {
                "sdg_goal": "sdg13",
                "sdg_target": "13.A",
                "sdg_confidence": "",
                "jurisdiction_level": "National",
                "date_granularity": "annual",
            }
        )

        self.assertEqual(record["sdg_goal"], "SDG 13")
        self.assertEqual(record["sdg_target"], "13.a")
        self.assertIsNone(record["sdg_confidence"])
        self.assertEqual(record["jurisdiction_level"], "national")
        self.assertEqual(record["date_granularity"], "year")

    def test_dedupe_records_uses_source_and_evidence_text(self):
        records = [
            {"source_filename": "a.pdf", "policy_content_original": "Climate policy", "evidence_quote": "Climate policy"},
            {"source_filename": "a.pdf", "policy_content_original": "Climate   policy", "evidence_quote": "Climate policy"},
        ]

        self.assertEqual(len(dedupe_records(records)), 1)

    def test_record_reviewer_accepts_supported_record(self):
        class AcceptBackend(ModelBackend):
            def complete_json(self, messages, schema_hint):
                return {
                    "decision": "accept",
                    "review_confidence": 0.91,
                    "rejection_reason": None,
                    "revised_record": None,
                }

            def model_name(self):
                return "reviewer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "review.md"
            prompt.write_text("Review.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("Climate mitigation reduces emissions.", encoding="utf-8")
            chunk = chunk_document(parse_file(policy), max_chars=1000)[0]
            reviewer = RecordReviewer(AcceptBackend(), prompt, root / "reviews.jsonl")

            reviewed = reviewer.review_records(chunk, [valid_candidate_record(policy)])
            review_rows = read_jsonl(root / "reviews.jsonl")

        self.assertEqual(len(reviewed), 1)
        self.assertEqual(reviewed[0]["review_status"], "reviewed")
        self.assertEqual(reviewed[0]["review_confidence"], 0.91)
        self.assertEqual(len(review_rows), 1)

    def test_record_reviewer_rejects_unsupported_record(self):
        class RejectBackend(ModelBackend):
            def complete_json(self, messages, schema_hint):
                return {
                    "decision": "reject",
                    "review_confidence": 0.2,
                    "rejection_reason": "generic policy",
                    "revised_record": None,
                }

            def model_name(self):
                return "reviewer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "review.md"
            prompt.write_text("Review.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("Generic green development.", encoding="utf-8")
            chunk = chunk_document(parse_file(policy), max_chars=1000)[0]
            reviewer = RecordReviewer(RejectBackend(), prompt, root / "reviews.jsonl")

            reviewed = reviewer.review_records(chunk, [valid_candidate_record(policy)])
            review_rows = read_jsonl(root / "reviews.jsonl")

        self.assertEqual(reviewed, [])
        self.assertEqual(review_rows[0]["decision"], "reject")
        self.assertEqual(review_rows[0]["rejection_reason"], "generic policy")

    def test_record_reviewer_applies_valid_revision(self):
        class ReviseBackend(ModelBackend):
            def complete_json(self, messages, schema_hint):
                return {
                    "decision": "revise",
                    "review_confidence": 0.84,
                    "rejection_reason": None,
                    "revised_record": {
                        "sdg_target": "13.1",
                        "policy_content_original": "Build climate resilience.",
                        "evidence_quote": "climate resilience",
                    },
                }

            def model_name(self):
                return "reviewer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "review.md"
            prompt.write_text("Review.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("Build climate resilience.", encoding="utf-8")
            chunk = chunk_document(parse_file(policy), max_chars=1000)[0]
            reviewer = RecordReviewer(ReviseBackend(), prompt, root / "reviews.jsonl")

            reviewed = reviewer.review_records(chunk, [valid_candidate_record(policy)])

        self.assertEqual(len(reviewed), 1)
        self.assertEqual(reviewed[0]["sdg_target"], "13.1")
        self.assertEqual(reviewed[0]["review_status"], "reviewed_with_revision")

    def test_record_reviewer_uses_cache(self):
        class CountingBackend(ModelBackend):
            def __init__(self):
                self.calls = 0

            def complete_json(self, messages, schema_hint):
                self.calls += 1
                return {
                    "decision": "accept",
                    "review_confidence": 0.9,
                    "rejection_reason": None,
                    "revised_record": None,
                }

            def model_name(self):
                return "reviewer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "review.md"
            prompt.write_text("Review.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("Climate mitigation reduces emissions.", encoding="utf-8")
            chunk = chunk_document(parse_file(policy), max_chars=1000)[0]
            backend = CountingBackend()
            reviewer = RecordReviewer(backend, prompt, root / "reviews.jsonl")
            record = valid_candidate_record(policy)

            reviewer.review_records(chunk, [record])
            reviewer.review_records(chunk, [record])

        self.assertEqual(backend.calls, 1)

    def test_record_reviewer_failure_rejects_and_logs_error(self):
        class FailingBackend(ModelBackend):
            def complete_json(self, messages, schema_hint):
                raise RuntimeError("boom")

            def model_name(self):
                return "reviewer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "review.md"
            prompt.write_text("Review.", encoding="utf-8")
            policy = root / "policy.txt"
            policy.write_text("Climate mitigation reduces emissions.", encoding="utf-8")
            chunk = chunk_document(parse_file(policy), max_chars=1000)[0]
            reviewer = RecordReviewer(FailingBackend(), prompt, root / "reviews.jsonl")

            reviewed = reviewer.review_records(chunk, [valid_candidate_record(policy)], errors_path=root / "errors.jsonl")
            errors_text = (root / "errors.jsonl").read_text(encoding="utf-8")

        self.assertEqual(reviewed, [])
        self.assertIn('"stage": "review"', errors_text)


def valid_candidate_record(path: Path) -> dict:
    return {
        "country": "China",
        "country_iso3": "CHN",
        "policy_title": "Climate Plan",
        "publication_date": "2026",
        "policy_content_original": "Climate mitigation reduces emissions.",
        "policy_content_summary": "Emissions reduction.",
        "sdg_goal": "SDG 13",
        "sdg_target": "13.2",
        "sdg_confidence": 0.9,
        "evidence_quote": "reduces emissions",
        "source_filename": path.name,
        "source_file": str(path),
        "validation_status": "valid",
    }


if __name__ == "__main__":
    unittest.main()
