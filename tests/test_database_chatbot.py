import csv
import tempfile
import unittest
from pathlib import Path

from sdg_extract.chatbot import answer_question, plan_question, retrieve_for_plan
from sdg_extract.database import build_database, database_summary, query_passages, query_policies
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


    def test_query_normalizes_compact_sdg_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(input_path, [sample_policy(sdg_goal="SDG 13", sdg_targets="13.2")])
            build_database(input_path, db_path)

            rows = query_policies(db_path, query="SDG13")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sdg_goal"], "SDG 13")


    def test_database_builds_passages_and_policy_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            passages_path = root / "chunk-records.jsonl"
            db_path = root / "redbird.sqlite"
            policy = sample_policy(review_status="", review_confidence="")
            write_jsonl(input_path, [policy])
            write_jsonl(
                passages_path,
                [
                    {
                        "country": policy["country"], "country_iso3": policy["country_iso3"],
                        "policy_title": policy["policy_title"], "source_filename": policy["source_filename"],
                        "source_file": policy["source_file"], "policy_content_original": "The methane rule cuts greenhouse gas emissions.",
                        "policy_content_summary": "Methane emissions reduction.", "sdg_goal": "SDG 13",
                        "sdg_target": "13.2", "evidence_quote": "greenhouse gas emissions", "validation_status": "valid",
                        "chunk_id": "chunk-1", "evidence_page": 12, "evidence_page_end": 13,
                    }
                ],
            )

            build_database(input_path, db_path, passages_path=passages_path)
            summary = database_summary(db_path)
            rows, diagnostics = query_passages(db_path, "methane greenhouse gas", limit=8)

        self.assertEqual(summary["passages"], 1)
        self.assertEqual(rows[0]["page_start"], 12)
        self.assertEqual(rows[0]["sdg_target"], "13.2")
        self.assertGreaterEqual(diagnostics["lexical_hits"], 1)

    def test_query_planner_routes_chinese_all_policies_to_sql_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(input_path, [sample_policy(country="China", country_iso3="CHN")])
            build_database(input_path, db_path)
            plan, _ = plan_question("总结中国的全部政策", [])

            rows, diagnostics = retrieve_for_plan(
                db_path,
                plan,
                {"top_k": 8, "aggregate_limit": 200, "require_reviewed": False},
            )

        self.assertEqual(plan["mode"], "database_aggregate")
        self.assertEqual(plan["countries"], ["China"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(diagnostics["fallback_stage"], "structured_sql")

    def test_unreviewed_records_are_searchable_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(input_path, [sample_policy(review_status="", review_confidence="")])
            build_database(input_path, db_path)
            plan, _ = plan_question("哪些政策涉及甲烷减排", [])

            rows, _ = retrieve_for_plan(
                db_path,
                plan,
                {"top_k": 8, "require_reviewed": False},
            )

        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0].get("review_status"), {None, ""})

    def test_passage_search_finds_match_after_policy_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            content = ("General introductory material. " * 120) + "The methane emissions reduction requirement applies immediately."
            write_jsonl(input_path, [sample_policy(policy_content=content, policy_content_en=content)])
            build_database(input_path, db_path)

            rows, _ = query_passages(db_path, "methane emissions reduction", limit=8)

        self.assertTrue(rows)
        self.assertIn("methane emissions", (rows[0].get("text_en") or "").lower())


    def test_query_planner_supports_multilingual_country_and_intent(self):
        cases = [
            ("Resumir todas las políticas de México", "Mexico", "database_aggregate"),
            ("Résumer toutes les politiques de l’Allemagne", "Germany", "database_aggregate"),
            ("Tóm tắt tất cả chính sách của Việt Nam", "Vietnam", "database_aggregate"),
            ("لخص جميع السياسات في مصر", "Egypt", "database_aggregate"),
            ("日本の気候政策を要約", "Japan", "database_aggregate"),
        ]
        available = ["Mexico", "Germany", "Vietnam", "Egypt", "Japan"]
        for question, country, mode in cases:
            plan, _ = plan_question(question, [], available)
            self.assertIn(country, plan["countries"], question)
            self.assertEqual(plan["mode"], mode, question)

    def test_spanish_query_expands_to_english_passage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            write_jsonl(
                input_path,
                [sample_policy(country="Mexico", country_iso3="MEX", policy_content_en="National climate adaptation and resilience plan.")],
            )
            build_database(input_path, db_path)
            plan, _ = plan_question("¿Qué políticas de México tratan la adaptación climática?", [], ["Mexico"])

            rows, _ = retrieve_for_plan(
                db_path,
                plan,
                {"top_k": 8, "require_reviewed": False},
            )

        self.assertTrue(rows)
        self.assertEqual(rows[0]["country"], "Mexico")

    def test_policy_translation_bridges_cross_language_query_to_original_passage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            passages_path = root / "chunk-records.jsonl"
            db_path = root / "redbird.sqlite"
            policy = sample_policy(
                country="Mexico",
                country_iso3="MEX",
                policy_content="Plan nacional de adaptación climática.",
                policy_content_en="National climate adaptation and resilience plan.",
            )
            write_jsonl(input_path, [policy])
            write_jsonl(
                passages_path,
                [{
                    "country": "Mexico", "country_iso3": "MEX",
                    "policy_title": policy["policy_title"], "source_filename": policy["source_filename"],
                    "source_file": policy["source_file"], "language": "spa",
                    "policy_content_original": "Plan nacional de adaptación climática.",
                    "policy_content_summary": "Adaptación climática.", "sdg_goal": "SDG 13",
                    "sdg_target": "13.1", "validation_status": "valid", "chunk_id": "mx-1",
                }],
            )
            build_database(input_path, db_path, passages_path=passages_path)

            rows, diagnostics = query_passages(db_path, "climate adaptation resilience", limit=8)

        self.assertTrue(rows)
        self.assertEqual(rows[0]["language"], "spa")
        self.assertIn("adaptación", rows[0]["text_original"])
        self.assertGreaterEqual(diagnostics["policy_bridge_hits"], 1)

    def test_short_us_alias_does_not_match_unrelated_word(self):
        plan, _ = plan_question("Explain sustainable climate planning", [], ["United States"])

        self.assertEqual(plan["countries"], [])


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

    def test_chatbot_can_include_unreviewed_records_for_testing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Answer from records only.", encoding="utf-8")
            write_jsonl(
                input_path,
                [sample_policy(review_status="", review_confidence="")],
            )
            build_database(input_path, db_path)

            result = answer_question(
                db_path,
                "methane emissions",
                prompt_path,
                config={"require_reviewed": False},
                completion_fn=lambda messages, config: "Test answer",
            )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["sources"]), 1)

    def test_chatbot_answers_general_question_when_no_records_match(self):
        calls = []

        def fake_completion(messages, config):
            calls.append(messages)
            return "General answer, not database evidence."

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            db_path = root / "redbird.sqlite"
            prompt_path = root / "prompt.md"
            prompt_path.write_text("Answer appropriately.", encoding="utf-8")
            write_jsonl(input_path, [sample_policy(policy_content_en="Methane emissions reduction plan.")])
            build_database(input_path, db_path)

            result = answer_question(
                db_path,
                "What is photosynthesis?",
                prompt_path,
                config={"allow_general_knowledge": True},
                completion_fn=fake_completion,
                history=[{"role": "user", "content": "Use a short answer."}],
            )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"], [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["content"], "Use a short answer.")

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
                config={"allow_general_knowledge": False},
                completion_fn=fake_completion,
            )

        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["sources"], [])

    def test_chatbot_reports_empty_api_response(self):
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
                completion_fn=lambda messages, config: "   ",
            )

        self.assertEqual(result["status"], "error:RuntimeError")
        self.assertIn("empty answer", result["answer"])

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
