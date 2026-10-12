import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from sdg_extract.cli import (
    build_reviewer_for_config,
    build_target_classifier_for_config,
    same_model_config,
)
from sdg_extract.models import LocalTransformersBackend, normalize_quantization_mode, parse_json_object


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

    def test_quantization_mode_accepts_int8_and_defaults_to_none(self):
        self.assertEqual(normalize_quantization_mode({}), "none")
        self.assertEqual(normalize_quantization_mode({"quantization": {"mode": "int8"}}), "int8")

    def test_quantization_mode_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            normalize_quantization_mode({"quantization": {"mode": "int4"}})

    def test_local_backend_retries_once_after_cuda_oom(self):
        class FakeCuda:
            class OutOfMemoryError(RuntimeError):
                pass

        class FakeTorch:
            cuda = FakeCuda()

        backend = LocalTransformersBackend.__new__(LocalTransformersBackend)
        backend._torch = FakeTorch()
        backend.max_tokens = 900
        backend.oom_retry_max_tokens = 450
        backend.raise_on_parse_error = True
        backend.build_prompt = lambda messages: "prompt"
        backend._clear_cuda_cache = lambda: None
        calls = []

        def generate(prompt, max_new_tokens, stage):
            calls.append((max_new_tokens, stage))
            if len(calls) == 1:
                raise FakeCuda.OutOfMemoryError("CUDA out of memory")
            return '{"records": []}'

        backend._generate = generate

        result = backend.complete_json([], schema_hint="records")

        self.assertEqual(result, {"records": []})
        self.assertEqual(calls, [(900, "generation"), (450, "generation_oom_retry")])

    def test_local_same_model_ignores_stage_token_limits(self):
        extraction = {
            "backend": "local_transformers",
            "name": "Qwen2.5-7B-Instruct",
            "local_model_path": "/models/Qwen2.5-7B-Instruct",
            "torch_dtype": "auto",
            "attn_implementation": "sdpa",
            "max_tokens": 512,
            "oom_retry_max_tokens": 256,
        }
        review = {
            **extraction,
            "max_tokens": 350,
            "oom_retry_max_tokens": 175,
        }

        self.assertTrue(same_model_config(extraction, review))

    def test_target_and_review_share_extraction_model_with_separate_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_prompt = root / "target.md"
            review_prompt = root / "review.md"
            target_prompt.write_text("Target.", encoding="utf-8")
            review_prompt.write_text("Review.", encoding="utf-8")
            base = {
                "backend": "local_transformers",
                "name": "Qwen2.5-7B-Instruct",
                "local_model_path": "/models/Qwen2.5-7B-Instruct",
                "torch_dtype": "auto",
                "attn_implementation": "sdpa",
                "quantization": {"mode": "none"},
            }
            config = {
                "model": {**base, "max_tokens": 512},
                "target_classification": {
                    "enabled": True,
                    "prompt_path": str(target_prompt),
                    "model": {**base, "max_tokens": 700},
                },
                "review": {
                    "enabled": True,
                    "prompt_path": str(review_prompt),
                    "model": {**base, "max_tokens": 350},
                },
            }
            extraction_backend = Mock()
            extraction_backend.model_name.return_value = "Qwen2.5-7B-Instruct"

            classifier = build_target_classifier_for_config(config, root, root, extraction_backend)
            reviewer = build_reviewer_for_config(config, root, root, extraction_backend)

        self.assertIs(classifier.backend, extraction_backend)
        self.assertIs(reviewer.backend, extraction_backend)
        self.assertEqual(classifier.max_tokens, 700)
        self.assertEqual(reviewer.max_tokens, 350)

    def test_local_generate_decodes_only_new_tokens(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __init__(self):
                self.decoded = None

            def __call__(self, prompt, **kwargs):
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

            def decode(self, token_ids, skip_special_tokens):
                self.decoded = token_ids.tolist()
                return '{"records": []}'

        class FakeModel:
            def __init__(self):
                self.options = None

            def get_input_embeddings(self):
                return Mock(weight=torch.zeros(1))

            def generate(self, **kwargs):
                self.options = kwargs
                return torch.tensor([[10, 11, 20, 21]])

        backend = LocalTransformersBackend.__new__(LocalTransformersBackend)
        backend._torch = torch
        backend.tokenizer = FakeTokenizer()
        backend.model = FakeModel()
        backend.max_input_tokens = 6144
        backend.temperature = 0
        backend.generation_config = Mock()
        backend._print_cuda_memory = lambda stage: None

        text = backend._generate("prompt", 512, "extraction")

        self.assertEqual(text, '{"records": []}')
        self.assertEqual(backend.tokenizer.decoded, [20, 21])
        self.assertEqual(backend.model.options["max_new_tokens"], 512)
        self.assertTrue(backend.model.options["use_cache"])

    def test_local_same_model_ignores_inherited_api_defaults(self):
        extraction = {
            "backend": "local_transformers",
            "name": "Qwen2.5-7B-Instruct",
            "local_model_path": "/models/Qwen2.5-7B-Instruct",
            "api_base": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "max_tokens": 900,
            "quantization": {"mode": "int8"},
        }
        reviewer = {
            "backend": "local_transformers",
            "name": "Qwen2.5-7B-Instruct",
            "local_model_path": "/models/Qwen2.5-7B-Instruct",
            "max_tokens": 900,
            "quantization": {"mode": "int8"},
        }

        self.assertTrue(same_model_config(extraction, reviewer))


if __name__ == "__main__":
    unittest.main()
