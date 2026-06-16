"""Document extraction and sentence splitting helpers for the Streamlit demo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: int
    text: str
    page: int | None = None


def normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def split_sentences(text: str, page: int | None = None, min_chars: int = 8) -> list[SentenceRecord]:
    parts = [normalize_whitespace(part) for part in SENTENCE_SPLIT_RE.split(text)]
    sentences = [part for part in parts if len(part) >= min_chars]
    return [
        SentenceRecord(sentence_id=index + 1, text=sentence, page=page)
        for index, sentence in enumerate(sentences)
    ]


def renumber_sentences(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    return [
        SentenceRecord(sentence_id=index + 1, text=sentence.text, page=sentence.page)
        for index, sentence in enumerate(sentences)
    ]


def extract_txt(file: BinaryIO) -> list[SentenceRecord]:
    raw = file.read()
    if isinstance(raw, str):
        text = raw
    else:
        text = raw.decode("utf-8", errors="ignore")
    return renumber_sentences(split_sentences(text))


def extract_pdf(file: BinaryIO) -> list[SentenceRecord]:
    import fitz

    data = file.read()
    document = fitz.open(stream=data, filetype="pdf")
    sentences: list[SentenceRecord] = []
    for page_index, page in enumerate(document, start=1):
        page_text = page.get_text("text")
        sentences.extend(split_sentences(page_text, page=page_index))
    return renumber_sentences(sentences)


def extract_docx(file: BinaryIO) -> list[SentenceRecord]:
    from docx import Document

    data = BytesIO(file.read())
    document = Document(data)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    text = "\n".join(paragraphs)
    return renumber_sentences(split_sentences(text))


def extract_uploaded_file(uploaded_file: BinaryIO, filename: str) -> list[SentenceRecord]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return extract_txt(uploaded_file)
    if suffix == ".pdf":
        return extract_pdf(uploaded_file)
    if suffix == ".docx":
        return extract_docx(uploaded_file)
    raise ValueError(f"Unsupported file type: {suffix}")


def sentence_table(sentences: list[SentenceRecord]) -> list[dict[str, object]]:
    return [
        {
            "sentence_id": sentence.sentence_id,
            "page": sentence.page if sentence.page is not None else "",
            "text": sentence.text,
        }
        for sentence in sentences
    ]

