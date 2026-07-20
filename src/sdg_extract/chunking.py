from __future__ import annotations

import hashlib
from typing import Iterable, List

from .types import Chunk, Document, Paragraph


def chunk_documents(documents: Iterable[Document], max_chars: int = 6000) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars))
    return chunks


def chunk_document(document: Document, max_chars: int = 6000) -> List[Chunk]:
    chunks: List[Chunk] = []
    current: List[Paragraph] = []
    current_len = 0
    for paragraph in document.paragraphs:
        paragraph_len = len(paragraph.text)
        if current and current_len + paragraph_len + 2 > max_chars:
            chunks.append(build_chunk(document, current, len(chunks) + 1))
            current = []
            current_len = 0
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
    )
