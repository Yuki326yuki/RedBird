from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Iterable, List

from .types import Chunk, Document, Paragraph


def chunk_documents(
    documents: Iterable[Document],
    max_chars: int = 6000,
    overlap_chars: int = 0,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars))
    return chunks


def chunk_document(document: Document, max_chars: int = 6000, overlap_chars: int = 0) -> List[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    overlap_chars = max(0, min(int(overlap_chars), max_chars // 2))
    chunks: List[Chunk] = []
    current: List[Paragraph] = []
    current_len = 0
    paragraphs = [
        part
        for paragraph in document.paragraphs
        for part in split_oversized_paragraph(paragraph, max_chars)
    ]
    for paragraph in paragraphs:
        paragraph_len = len(paragraph.text)
        if current and current_len + paragraph_len + 2 > max_chars:
            chunks.append(build_chunk(document, current, len(chunks) + 1))
            current = overlap_tail(current, overlap_chars)
            current_len = sum(len(item.text) + 2 for item in current)
            while current and current_len + paragraph_len + 2 > max_chars:
                removed = current.pop(0)
                current_len -= len(removed.text) + 2
        current.append(paragraph)
        current_len += paragraph_len + 2
    if current:
        chunks.append(build_chunk(document, current, len(chunks) + 1))
    return chunks


def build_chunk(document: Document, paragraphs: List[Paragraph], index: int) -> Chunk:
    text = "\n\n".join(
        f"[{paragraph.paragraph_id} page={paragraph.page_number}] {paragraph.text}"
        for paragraph in paragraphs
    )
    digest = hashlib.sha1(
        f"{document.document_id}:{index}:{paragraphs[0].paragraph_id}".encode("utf-8")
    ).hexdigest()[:12]
    return Chunk(
        chunk_id=f"{document.document_id}-{digest}",
        document_id=document.document_id,
        source_file=document.source_file,
        source_type=document.source_type,
        text=text,
        paragraph_ids=[paragraph.paragraph_id for paragraph in paragraphs],
        evidence_page=paragraphs[0].page_number,
        evidence_char_start=paragraphs[0].char_start,
        evidence_char_end=paragraphs[-1].char_end,
        metadata=dict(document.metadata),
        page_start=paragraphs[0].page_number,
        page_end=paragraphs[-1].page_number,
    )


def split_oversized_paragraph(paragraph: Paragraph, max_chars: int) -> List[Paragraph]:
    if len(paragraph.text) <= max_chars:
        return [paragraph]
    parts: List[Paragraph] = []
    cursor = 0
    part_index = 1
    text = paragraph.text
    while cursor < len(text):
        end = min(cursor + max_chars, len(text))
        if end < len(text):
            window = text[cursor:end]
            boundaries = [match.end() for match in re.finditer(r"[.!?。！？；;]\s*|\s+", window)]
            if boundaries and boundaries[-1] >= max_chars // 2:
                end = cursor + boundaries[-1]
        raw_segment = text[cursor:end]
        segment = raw_segment.strip()
        if segment:
            leading = len(raw_segment) - len(raw_segment.lstrip())
            segment_start = paragraph.char_start + cursor + leading
            parts.append(
                replace(
                    paragraph,
                    text=segment,
                    paragraph_id=f"{paragraph.paragraph_id}-s{part_index:02d}",
                    char_start=segment_start,
                    char_end=segment_start + len(segment),
                )
            )
            part_index += 1
        cursor = max(end, cursor + 1)
    return parts


def overlap_tail(paragraphs: List[Paragraph], overlap_chars: int) -> List[Paragraph]:
    if not overlap_chars:
        return []
    kept: List[Paragraph] = []
    size = 0
    for paragraph in reversed(paragraphs):
        if kept and size + len(paragraph.text) > overlap_chars:
            break
        kept.append(paragraph)
        size += len(paragraph.text) + 2
    return list(reversed(kept))
