import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sdg_extract.cli import prepare_extraction_inputs
from sdg_extract.extraction import load_documents
from sdg_extract.io_utils import write_jsonl
from sdg_extract.parsers import parse_file


class SharedCacheTests(unittest.TestCase):
    def test_shared_cache_migrates_seed_then_skips_parser_and_chunker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "input"
            source_dir.mkdir()
            source = source_dir / "policy.txt"
            source.write_text("Climate adaptation policy.", encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["country", "country_iso3", "language", "document_title", "local_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "country": "China",
                        "country_iso3": "CHN",
                        "language": "zho",
                        "document_title": "Policy",
                        "local_path": str(source),
                    }
                )
            seed = root / "seed.jsonl"
            document = parse_file(source, metadata={"country": "China"})
            write_jsonl(seed, [document.to_dict()])
            config = {
                "paths": {"input_dir": str(source_dir)},
                "manifest_path": str(manifest),
                "countries": ["China"],
                "chunking": {"max_chars": 2400, "overlap_chars": 300},
                "cache": {
                    "enabled": True,
                    "documents_path": str(root / "cache" / "documents.jsonl"),
                    "documents_meta_path": str(root / "cache" / "documents.meta.json"),
                    "chunks_path": str(root / "cache" / "chunks.jsonl"),
                    "chunks_meta_path": str(root / "cache" / "chunks.meta.json"),
                    "seed_documents_path": str(seed),
                },
            }

            documents, chunks, _, _ = prepare_extraction_inputs(config, root, root / "work")
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(chunks or []), 1)

            with patch("sdg_extract.cli.parse_directory") as parser, patch(
                "sdg_extract.cli.chunk_documents"
            ) as chunker:
                cached_documents, cached_chunks, documents_path, _ = prepare_extraction_inputs(
                    config, root, root / "work"
                )

            parser.assert_not_called()
            chunker.assert_not_called()
            self.assertEqual(len(cached_documents), 1)
            self.assertEqual(len(cached_chunks or []), 1)
            self.assertEqual(len(load_documents(documents_path)), 1)

    def test_changed_chunk_parameters_rebuild_only_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "input"
            source_dir.mkdir()
            source = source_dir / "policy.txt"
            source.write_text("Climate policy sentence. " * 200, encoding="utf-8")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "country,country_iso3,language,document_title,local_path\n"
                f"China,CHN,eng,Policy,{source}\n",
                encoding="utf-8",
            )
            seed = root / "seed.jsonl"
            write_jsonl(seed, [parse_file(source, metadata={"country": "China"}).to_dict()])
            config = {
                "paths": {"input_dir": str(source_dir)},
                "manifest_path": str(manifest),
                "countries": ["China"],
                "chunking": {"max_chars": 2400, "overlap_chars": 300},
                "cache": {
                    "enabled": True,
                    "documents_path": str(root / "cache" / "documents.jsonl"),
                    "chunks_path": str(root / "cache" / "chunks.jsonl"),
                    "seed_documents_path": str(seed),
                },
            }
            prepare_extraction_inputs(config, root, root / "work")
            config["chunking"]["max_chars"] = 1200

            with patch("sdg_extract.cli.parse_directory") as parser:
                _, chunks, _, _ = prepare_extraction_inputs(config, root, root / "work")

            parser.assert_not_called()
            self.assertGreater(len(chunks or []), 1)


if __name__ == "__main__":
    unittest.main()
