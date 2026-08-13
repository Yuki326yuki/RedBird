from __future__ import annotations

import csv
import hashlib
import re
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from xml.etree import ElementTree

from .types import Document, Paragraph


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".docx", ".csv"}


def parse_directory(
    input_dir: Path,
    allowed_paths: Optional[Set[Path]] = None,
    metadata_by_path: Optional[Dict[Path, Dict[str, Any]]] = None,
    show_progress: bool = False,
) -> List[Document]:
    documents: List[Document] = []
    resolved_allowed = {path.resolve() for path in allowed_paths or set()}
    candidates: List[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved_allowed and resolved not in resolved_allowed:
            continue
        if detect_source_type(path) is None:
            continue
        candidates.append(path)

    total = len(candidates)
    if show_progress:
        print(f"[INGEST] Starting document parsing: {total} files", flush=True)
    for index, path in enumerate(candidates, start=1):
        started_at = time.time()
        source_type = detect_source_type(path) or "unknown"
        if show_progress:
            print(f"[INGEST] parsing {index}/{total} type={source_type} source={path}", flush=True)
        document = parse_file(path, metadata=(metadata_by_path or {}).get(path.resolve(), {}))
        documents.append(document)
        if show_progress:
            elapsed = time.time() - started_at
            print(
                f"[INGEST] parsed {index}/{total} paragraphs={len(document.paragraphs)} "
                f"sec={elapsed:.1f} source={path}",
                flush=True,
            )
    return documents


def parse_file(path: Path, metadata: Optional[Dict[str, Any]] = None) -> Document:
    source_type = detect_source_type(path)
    if source_type is None:
        raise ValueError(f"Unsupported file type: {path}")
    parser: Callable[[Path], List[PageText]]
    if source_type == "txt":
        parser = parse_txt
    elif source_type == "html":
        parser = parse_html
    elif source_type == "docx":
        parser = parse_docx
    elif source_type == "csv":
        parser = parse_csv
    elif source_type == "pdf":
        parser = parse_pdf
    else:
        raise ValueError(f"Unsupported file type: {path}")

    document_id = document_id_for(path)
    pages = parser(path)
    paragraphs = pages_to_paragraphs(
        pages=pages,
        document_id=document_id,
        source_file=str(path),
        source_type=source_type,
    )
    metadata = dict(metadata or {})
    metadata.update(parse_quality_metadata(pages, source_type))
    title = metadata.get("policy_title") or path.stem
    language = metadata.get("language")
    return Document(
        document_id=document_id,
        source_file=str(path),
        source_type=source_type,
        title=title,
        language=language,
        paragraphs=paragraphs,
        metadata=metadata,
    )


def detect_source_type(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".docx":
        return "docx"
    if suffix == ".csv":
        return "csv"
    if suffix == ".pdf":
        return "pdf"
    if not suffix and looks_like_pdf(path):
        return "pdf"
    return None


def looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


class PageText:
    def __init__(self, page_number: Optional[int], text: str) -> None:
        self.page_number = page_number
        self.text = text


def document_id_for(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_txt(path: Path) -> List[PageText]:
    return [PageText(1, read_text_file(path))]


def parse_html(path: Path) -> List[PageText]:
    parser = BlockTextHTMLParser()
    parser.feed(read_text_file(path))
    parser.close()
    return [PageText(1, "\n\n".join(parser.blocks))]


def parse_docx(path: Path) -> List[PageText]:
    paragraphs: List[str] = []
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as handle:
            root = ElementTree.fromstring(handle.read())
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return [PageText(1, "\n\n".join(paragraphs))]


def parse_csv(path: Path) -> List[PageText]:
    rows: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        header: Optional[List[str]] = None
        for index, row in enumerate(reader):
            cleaned = [cell.strip() for cell in row]
            if not any(cleaned):
                continue
            if index == 0:
                header = cleaned
                rows.append(" | ".join(cleaned))
                continue
            if header and len(header) == len(cleaned):
                rows.append("; ".join(f"{key}: {value}" for key, value in zip(header, cleaned)))
            else:
                rows.append(" | ".join(cleaned))
    return [PageText(1, "\n\n".join(rows))]


def parse_pdf(path: Path) -> List[PageText]:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages: List[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(PageText(index, page.extract_text() or ""))
        return pages
    except Exception:
        return [PageText(1, recover_text_from_pdf_bytes(path.read_bytes()))]



def parse_quality_metadata(pages: List[PageText], source_type: str) -> Dict[str, Any]:
    if source_type != "pdf":
        return {"parse_status": "ok", "low_text_pages": []}
    low_text_pages: List[int] = []
    replacement_chars = 0
    total_chars = 0
    for page in pages:
        text = page.text or ""
        total_chars += len(text)
        replacement_chars += text.count("\ufffd")
        visible = sum(1 for char in text if char.isprintable() and not char.isspace())
        if visible < 40 and page.page_number is not None:
            low_text_pages.append(page.page_number)
    replacement_ratio = replacement_chars / max(total_chars, 1)
    status = "warning:low_text_pages" if low_text_pages else "ok"
    if replacement_ratio > 0.02:
        status = "warning:garbled_text"
    return {
        "parse_status": status,
        "low_text_pages": low_text_pages,
        "parsed_page_count": len(pages),
        "parsed_character_count": total_chars,
        "replacement_character_ratio": round(replacement_ratio, 5),
    }

def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


class BlockTextHTMLParser(HTMLParser):
    block_tags = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.blocks: List[str] = []
        self._current: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Iterable[tuple]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in self.block_tags:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.block_tags:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = normalize_inline_whitespace(data)
        if cleaned:
            self._current.append(cleaned)

    def close(self) -> None:
        self._flush()
        super().close()

    def _flush(self) -> None:
        text = normalize_inline_whitespace(" ".join(self._current))
        if text:
            self.blocks.append(text)
        self._current = []


def pages_to_paragraphs(
    pages: List[PageText],
    document_id: str,
    source_file: str,
    source_type: str,
) -> List[Paragraph]:
    paragraphs: List[Paragraph] = []
    global_offset = 0
    for page in pages:
        for local_index, item in enumerate(split_paragraphs(page.text), start=1):
            start = global_offset + item.start
            end = global_offset + item.end
            paragraph_id = f"p{len(paragraphs) + 1:06d}"
            paragraphs.append(
                Paragraph(
                    document_id=document_id,
                    source_file=source_file,
                    source_type=source_type,
                    text=item.text,
                    page_number=page.page_number,
                    paragraph_id=paragraph_id,
                    char_start=start,
                    char_end=end,
                )
            )
        global_offset += len(page.text) + 1
    return paragraphs


class ParagraphSpan:
    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


def split_paragraphs(text: str) -> List[ParagraphSpan]:
    spans: List[ParagraphSpan] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        paragraph = normalize_paragraph(match.group(0))
        if paragraph:
            spans.append(ParagraphSpan(paragraph, match.start(), match.end()))
    if not spans:
        cleaned = normalize_paragraph(text)
        if cleaned:
            spans.append(ParagraphSpan(cleaned, 0, len(text)))
    return spans


def normalize_paragraph(text: str) -> str:
    lines = [normalize_inline_whitespace(line) for line in text.splitlines()]
    return normalize_inline_whitespace(" ".join(line for line in lines if line))


def normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def recover_text_from_pdf_bytes(data: bytes) -> str:
    decoded = data.decode("latin-1", errors="ignore")
    strings = re.findall(r"\(([^()]+)\)", decoded)
    if strings:
        return "\n\n".join(normalize_inline_whitespace(item) for item in strings if item.strip())
    printable = "".join(char if char.isprintable() or char in "\n\r\t" else " " for char in decoded)
    return normalize_inline_whitespace(printable)
