"""Shared text preprocessing and EDA helpers for the fixed benchmark."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")

BASIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def normalize_text(text: object, lowercase: bool = True) -> str:
    """Normalize Unicode, remove obvious artifacts, lowercase, and collapse spaces."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u00a0", " ")
    normalized = HTML_RE.sub(" ", normalized)
    normalized = URL_RE.sub(" <url> ", normalized)
    normalized = CONTROL_RE.sub(" ", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized.lower() if lowercase else normalized


def simple_tokenize(text: object) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text, lowercase=True))


def token_count(text: object) -> int:
    return len(simple_tokenize(text))


def char_count(text: object) -> int:
    return len(normalize_text(text))


def lexical_overlap(text_a: object, text_b: object) -> float:
    tokens_a = set(simple_tokenize(text_a))
    tokens_b = set(simple_tokenize(text_b))
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def top_words(texts: Iterable[object], top_n: int = 50) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(token for token in simple_tokenize(text) if token not in BASIC_STOPWORDS)
    return counter.most_common(top_n)

