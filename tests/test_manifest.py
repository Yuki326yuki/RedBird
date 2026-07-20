import csv
import tempfile
import unittest
from pathlib import Path

from sdg_extract.manifest import load_manifest_selection
from sdg_extract.parsers import parse_directory


class ManifestTests(unittest.TestCase):
    def test_manifest_filters_countries_and_injects_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            china_dir = docs / "china"
            us_dir = docs / "united-states-of-america"
            brazil_dir = docs / "brazil"
            china_dir.mkdir(parents=True)
            us_dir.mkdir(parents=True)
            brazil_dir.mkdir(parents=True)
            china = china_dir / "china-policy.txt"
            us = us_dir / "us-policy-without-extension"
            brazil = brazil_dir / "brazil-policy.txt"
            china.write_text("China climate policy and SDG 13.2.", encoding="utf-8")
            us.write_bytes(b"%PDF-1.4\nBT (United States climate policy and SDG 13.2.) Tj ET\n%%EOF")
            brazil.write_text("Brazil climate policy and SDG 13.2.", encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "country",
                        "country_iso3",
                        "document_title",
                        "document_type",
                        "published_date",
                        "language",
                        "local_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "country": "China",
                        "country_iso3": "CHN",
                        "document_title": "China Policy",
                        "document_type": "Plan",
                        "published_date": "2025-01-02T00:00:00Z",
                        "language": "eng",
                        "local_path": str(china.relative_to(root)),
                    }
                )
                writer.writerow(
                    {
                        "country": "United States",
                        "country_iso3": "USA",
                        "document_title": "US Policy",
                        "document_type": "Report",
                        "published_date": "2025-03-04T00:00:00Z",
                        "language": "eng",
                        "local_path": str(us.relative_to(root)),
                    }
                )
                writer.writerow(
                    {
                        "country": "Brazil",
                        "country_iso3": "BRA",
                        "document_title": "Brazil Policy",
                        "document_type": "Plan",
                        "published_date": "2025-05-06T00:00:00Z",
                        "language": "por",
                        "local_path": str(brazil.relative_to(root)),
                    }
                )

            allowed, metadata = load_manifest_selection(manifest, root, countries=["China", "United States"])
            documents = parse_directory(docs, allowed_paths=allowed, metadata_by_path=metadata)

            self.assertEqual(len(allowed), 2)
            self.assertEqual({document.metadata["country"] for document in documents}, {"China", "United States"})
            self.assertEqual({document.metadata["source_filename"] for document in documents}, {"china-policy.txt", "us-policy-without-extension"})
            self.assertEqual({document.source_type for document in documents}, {"txt", "pdf"})


if __name__ == "__main__":
    unittest.main()
