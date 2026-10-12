import tempfile
import unittest
import zipfile
from pathlib import Path

from sdg_extract.chunking import chunk_document
from sdg_extract.parsers import parse_file


class ParserTests(unittest.TestCase):
    def test_txt_parser_preserves_paragraph_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.txt"
            path.write_text("Climate policy supports SDG 13.2.\n\nSecond paragraph.", encoding="utf-8")

            document = parse_file(path)

            self.assertEqual(document.source_type, "txt")
            self.assertEqual(len(document.paragraphs), 2)
            self.assertEqual(document.paragraphs[0].page_number, 1)
            self.assertEqual(document.paragraphs[0].paragraph_id, "p000001")
            self.assertGreater(document.paragraphs[0].char_end, document.paragraphs[0].char_start)

    def test_html_parser_extracts_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.html"
            path.write_text(
                "<html><body><h1>Policy</h1><p>Health access and SDG 3.8.</p></body></html>",
                encoding="utf-8",
            )

            document = parse_file(path)

            self.assertEqual(len(document.paragraphs), 2)
            self.assertIn("Health access", document.paragraphs[1].text)

    def test_docx_parser_extracts_paragraphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.docx"
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                "<w:p><w:r><w:t>Education strategy for SDG 4.7.</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>Implementation by ministry.</w:t></w:r></w:p>"
                "</w:body></w:document>"
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)

            document = parse_file(path)

            self.assertEqual(len(document.paragraphs), 2)
            self.assertIn("Education strategy", document.paragraphs[0].text)

    def test_pdf_fallback_recovers_simple_literal_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pdf"
            path.write_bytes(b"%PDF-1.4\nBT (Water policy and SDG 6.1.) Tj ET\n%%EOF")

            document = parse_file(path)

            self.assertEqual(document.source_type, "pdf")
            self.assertIn("Water policy", document.paragraphs[0].text)

    def test_extensionless_pdf_is_detected_by_magic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy-without-extension"
            path.write_bytes(b"%PDF-1.4\nBT (Climate policy and SDG 13.2.) Tj ET\n%%EOF")

            document = parse_file(path)

            self.assertEqual(document.source_type, "pdf")
            self.assertIn("Climate policy", document.paragraphs[0].text)

    def test_chunking_keeps_evidence_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.txt"
            path.write_text("A policy for SDG 13.2.\n\nB policy for SDG 7.1.", encoding="utf-8")
            document = parse_file(path)

            chunks = chunk_document(document, max_chars=1000)

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].paragraph_ids, ["p000001", "p000002"])
            self.assertEqual(chunks[0].evidence_char_start, document.paragraphs[0].char_start)
            self.assertEqual(chunks[0].evidence_char_end, document.paragraphs[-1].char_end)


    def test_pdf_parser_records_low_text_page_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.pdf"
            path.write_bytes(b"%PDF-1.4\nBT (Climate.) Tj ET\n%%EOF")

            document = parse_file(path)

        self.assertIn(document.metadata["parse_status"], {"ok", "warning:low_text_pages"})
        self.assertIn("parsed_page_count", document.metadata)


if __name__ == "__main__":
    unittest.main()
