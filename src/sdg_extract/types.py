from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Paragraph:
    document_id: str
    source_file: str
    source_type: str
    text: str
    page_number: Optional[int]
    paragraph_id: str
    char_start: int
    char_end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    document_id: str
    source_file: str
    source_type: str
    title: Optional[str]
    language: Optional[str]
    paragraphs: List[Paragraph]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["paragraphs"] = [paragraph.to_dict() for paragraph in self.paragraphs]
        return data


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_file: str
    source_type: str
    text: str
    paragraph_ids: List[str]
    evidence_page: Optional[int]
    evidence_char_start: int
    evidence_char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
