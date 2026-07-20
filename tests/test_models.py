import unittest

from sdg_extract.models import parse_json_object


class ModelParsingTests(unittest.TestCase):
    def test_parse_json_object_ignores_text_after_first_object(self):
        parsed = parse_json_object('Here is JSON:\n{"records": []}\n{"extra": true}')

        self.assertEqual(parsed, {"records": []})

    def test_parse_json_object_handles_braces_inside_strings(self):
        parsed = parse_json_object('{"records": [{"evidence_quote": "policy {text}"}]} trailing')

        self.assertEqual(parsed["records"][0]["evidence_quote"], "policy {text}")

    def test_parse_json_object_accepts_fenced_json(self):
        parsed = parse_json_object('```json\n{"records": []}\n```')

        self.assertEqual(parsed, {"records": []})

    def test_parse_json_object_wraps_top_level_array_as_records(self):
        parsed = parse_json_object('[{"sdg_goal": "SDG 13"}]')

        self.assertEqual(parsed, {"records": [{"sdg_goal": "SDG 13"}]})


if __name__ == "__main__":
    unittest.main()
