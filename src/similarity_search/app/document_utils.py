"""Document extraction and sentence splitting helpers for the Streamlit demo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO


PARAGRAPH_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: int
    text: str
    page: int | None = None
    paragraph: int = 0


def normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def split_paragraph_into_sentences(
    text: str,
    paragraph: int,
    page: int | None = None,
    min_chars: int = 8,
) -> list[SentenceRecord]:
    parts = [normalize_whitespace(part) for part in PARAGRAPH_SENTENCE_SPLIT_RE.split(text)]
    sentences = [part for part in parts if len(part) >= min_chars]
    return [
        SentenceRecord(sentence_id=0, text=sentence, page=page, paragraph=paragraph)
        for sentence in sentences
    ]


def split_sentences(text: str, page: int | None = None, min_chars: int = 8) -> list[SentenceRecord]:
    return split_paragraph_into_sentences(
        normalize_whitespace(text),
        paragraph=1,
        page=page,
        min_chars=min_chars,
    )


def split_text_into_paragraphs(text: str) -> list[str]:
    blocks = [block.strip() for block in PARAGRAPH_BLANK_LINE_RE.split(text)]
    blocks = [block for block in blocks if block]
    if len(blocks) > 1:
        return [normalize_whitespace(block) for block in blocks]
    lines = [line.strip() for line in text.split("\n")]
    return [normalize_whitespace(line) for line in lines if line]


def renumber_sentences(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    return [
        SentenceRecord(
            sentence_id=index + 1,
            text=sentence.text,
            page=sentence.page,
            paragraph=sentence.paragraph,
        )
        for index, sentence in enumerate(sentences)
    ]


def extract_txt(file: BinaryIO) -> list[SentenceRecord]:
    raw = file.read()
    if isinstance(raw, str):
        text = raw
    else:
        text = raw.decode("utf-8", errors="ignore")
    sentences: list[SentenceRecord] = []
    for paragraph_index, paragraph_text in enumerate(split_text_into_paragraphs(text), start=1):
        sentences.extend(split_paragraph_into_sentences(paragraph_text, paragraph=paragraph_index))
    return renumber_sentences(sentences)


def extract_pdf(file: BinaryIO) -> list[SentenceRecord]:
    import fitz

    data = file.read()
    document = fitz.open(stream=data, filetype="pdf")
    sentences: list[SentenceRecord] = []
    paragraph_index = 0
    for page_index, page in enumerate(document, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            block_text = block[4] if len(block) > 4 else ""
            block_text = normalize_whitespace(block_text)
            if not block_text:
                continue
            paragraph_index += 1
            sentences.extend(
                split_paragraph_into_sentences(block_text, paragraph=paragraph_index, page=page_index)
            )
    return renumber_sentences(sentences)


def extract_docx(file: BinaryIO) -> list[SentenceRecord]:
    from docx import Document

    data = BytesIO(file.read())
    document = Document(data)
    sentences: list[SentenceRecord] = []
    paragraph_index = 0
    for paragraph in document.paragraphs:
        paragraph_text = normalize_whitespace(paragraph.text)
        if not paragraph_text:
            continue
        paragraph_index += 1
        sentences.extend(split_paragraph_into_sentences(paragraph_text, paragraph=paragraph_index))
    return renumber_sentences(sentences)


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
            "paragraph": sentence.paragraph,
            "text": sentence.text,
        }
        for sentence in sentences
    ]
