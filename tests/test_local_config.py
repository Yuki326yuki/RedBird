import unittest
from pathlib import Path

from sdg_extract.config import load_config


class LocalConfigTests(unittest.TestCase):
    def test_qwen25_7b_bf16_sample_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs" / "sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml"
        )

        self.assertEqual(config["countries"], ["China", "United States", "Mexico"])
        self.assertEqual(config["model"]["name"], "Qwen2.5-7B-Instruct")
        self.assertEqual(
            config["model"]["local_model_path"],
            "/hpc2hdd/home/czhang162/models/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(config["model"]["torch_dtype"], "auto")
        self.assertEqual(config["model"]["quantization"]["mode"], "none")
        self.assertEqual(config["model"]["max_tokens"], 512)
        self.assertEqual(config["target_classification"]["model"]["max_tokens"], 700)
        self.assertEqual(config["review"]["model"]["max_tokens"], 350)
        self.assertEqual(config["translation"]["model"]["max_tokens"], 1800)
        self.assertTrue(config["cache"]["enabled"])
        self.assertEqual(config["experiment"]["target_policies_per_country"], 2)
        self.assertTrue(config["experiment"]["require_policy_quota"])


if __name__ == "__main__":
    unittest.main()
