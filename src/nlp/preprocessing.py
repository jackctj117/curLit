"""Text preprocessing — boilerplate removal and sentence segmentation."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import spacy


@dataclass
class ProcessedDocument:
    doc_id: str
    cb: str
    doc_type: str
    date: datetime
    sentences: list[str]
    paragraphs: list[str]
    word_count: int
    metadata: dict[str, Any]


class TextPreprocessor:
    BOILERPLATE_PATTERNS = [
        r"For release at.*?ET",
        r"Implementation Note issued.*?$",
        r"Last Update:.*?$",
        r"^\s*\d+\s*$",
        r"[A-Z][A-Z\s]{20,}",
    ]

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
        self.nlp.add_pipe("sentencizer")

    def clean(self, text: str) -> str:
        for pat in self.BOILERPLATE_PATTERNS:
            text = re.sub(pat, "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def segment(self, text: str) -> tuple[list[str], list[str]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentences: list[str] = []
        for para in paragraphs:
            doc = self.nlp(para)
            sentences.extend(s.text.strip() for s in doc.sents if len(s.text.strip()) > 10)
        return sentences, paragraphs

    def process(self, doc: "Document") -> ProcessedDocument:  # type: ignore[name-defined] # noqa: F821
        cleaned = self.clean(doc.raw_text)
        sentences, paragraphs = self.segment(cleaned)
        return ProcessedDocument(
            doc_id=doc.doc_id,
            cb=doc.cb,
            doc_type=doc.doc_type,
            date=doc.date,
            sentences=sentences,
            paragraphs=paragraphs,
            word_count=sum(len(s.split()) for s in sentences),
            metadata=doc.metadata,
        )
