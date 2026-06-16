"""Similarity models and hybrid search logic for the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from similarity_search.app.document_utils import SentenceRecord


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TFIDF_DIR = MODELS_DIR / "tfidf_baseline"
BI_ENCODER_DIR = MODELS_DIR / "allnli-minilm-biencoder" / "final"
CROSS_ENCODER_DIR = MODELS_DIR / "allnli-cross-encoder-nli" / "final"

PRETRAINED_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


@dataclass(frozen=True)
class SearchResult:
    sentence_id: int
    text: str
    page: int | None
    score: float
    cosine_score: float | None = None
    entailment_probability: float | None = None
    nli_label: str | None = None
    contradiction_probability: float | None = None
    neutral_probability: float | None = None


@dataclass(frozen=True)
class MatchResult:
    sentence_id_a: int
    sentence_id_b: int
    text_a: str
    text_b: str
    page_a: int | None
    page_b: int | None
    score: float
    cosine_score: float | None = None
    entailment_probability: float | None = None
    nli_label: str | None = None


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_status() -> dict[str, bool]:
    return {
        "tfidf": (TFIDF_DIR / "vectorizer.joblib").exists(),
        "fine_tuned_biencoder": (BI_ENCODER_DIR / "model.safetensors").exists(),
        "cross_encoder": (CROSS_ENCODER_DIR / "model.safetensors").exists(),
    }


@lru_cache(maxsize=1)
def load_tfidf_vectorizer() -> Any:
    import joblib

    vectorizer_path = TFIDF_DIR / "vectorizer.joblib"
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found: {vectorizer_path}")
    return joblib.load(vectorizer_path)


@lru_cache(maxsize=3)
def load_sentence_transformer(model_choice: str) -> Any:
    from sentence_transformers import SentenceTransformer

    if model_choice == "Fine-tuned MiniLM":
        model_name = str(BI_ENCODER_DIR) if BI_ENCODER_DIR.exists() else PRETRAINED_MINILM
    else:
        model_name = PRETRAINED_MINILM
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_cross_encoder_components() -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = str(CROSS_ENCODER_DIR) if CROSS_ENCODER_DIR.exists() else "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def tfidf_scores(query: str, sentences: list[str]) -> np.ndarray:
    vectorizer = load_tfidf_vectorizer()
    sentence_vectors = vectorizer.transform(sentences)
    query_vector = vectorizer.transform([query])
    return np.asarray(sentence_vectors.dot(query_vector.T).todense()).ravel()


def embedding_scores(model: Any, query: str, sentences: list[str], batch_size: int = 64) -> np.ndarray:
    sentence_embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return sentence_embeddings @ query_embedding


def pair_embedding_matrix(model: Any, left: list[str], right: list[str], batch_size: int = 64) -> np.ndarray:
    left_embeddings = model.encode(
        left,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    right_embeddings = model.encode(
        right,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return left_embeddings @ right_embeddings.T


def tfidf_pair_matrix(left: list[str], right: list[str]) -> np.ndarray:
    vectorizer = load_tfidf_vectorizer()
    left_vectors = vectorizer.transform(left)
    right_vectors = vectorizer.transform(right)
    return left_vectors.dot(right_vectors.T).toarray()


def cross_encoder_predict(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict[str, float | str]]:
    import torch

    if not pairs:
        return []
    tokenizer, model, device = load_cross_encoder_components()
    outputs: list[dict[str, float | str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu().numpy()
        probabilities = softmax(logits)
        labels = probabilities.argmax(axis=1)
        for probs, label in zip(probabilities, labels):
            outputs.append(
                {
                    "entailment_probability": float(probs[0]),
                    "neutral_probability": float(probs[1]),
                    "contradiction_probability": float(probs[2]),
                    "nli_label": LABEL_NAMES[int(label)],
                }
            )
    return outputs


def semantic_search(
    query: str,
    sentences: list[SentenceRecord],
    model_choice: str,
    top_k: int,
    threshold: float,
    rerank_top_k: int = 30,
    alpha: float = 0.55,
) -> list[SearchResult]:
    texts = [sentence.text for sentence in sentences]
    if not texts:
        return []

    if model_choice == "TF-IDF baseline":
        scores = tfidf_scores(query, texts)
        return ranked_search_results(sentences, scores, threshold, top_k)

    if model_choice == "Cross-Encoder NLI":
        cross_outputs = cross_encoder_predict([(text, query) for text in texts])
        scores = np.asarray([float(item["entailment_probability"]) for item in cross_outputs])
        results = []
        for sentence, score, cross in zip(sentences, scores, cross_outputs):
            if score >= threshold:
                results.append(
                    SearchResult(
                        sentence_id=sentence.sentence_id,
                        text=sentence.text,
                        page=sentence.page,
                        score=float(score),
                        entailment_probability=float(cross["entailment_probability"]),
                        neutral_probability=float(cross["neutral_probability"]),
                        contradiction_probability=float(cross["contradiction_probability"]),
                        nli_label=str(cross["nli_label"]),
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    bi_model_choice = "Fine-tuned MiniLM" if model_choice == "Hybrid reranker" else model_choice
    bi_model = load_sentence_transformer(bi_model_choice)
    cosine_scores = embedding_scores(bi_model, query, texts)

    if model_choice != "Hybrid reranker":
        return ranked_search_results(sentences, cosine_scores, threshold, top_k, cosine_scores)

    candidate_count = min(max(top_k, rerank_top_k), len(sentences))
    candidate_indices = np.argsort(-cosine_scores)[:candidate_count]
    pairs = [(sentences[index].text, query) for index in candidate_indices]
    cross_outputs = cross_encoder_predict(pairs)

    results: list[SearchResult] = []
    for index, cross in zip(candidate_indices, cross_outputs):
        entailment = float(cross["entailment_probability"])
        cosine = float(cosine_scores[index])
        score = alpha * entailment + (1.0 - alpha) * cosine
        if score >= threshold:
            sentence = sentences[int(index)]
            results.append(
                SearchResult(
                    sentence_id=sentence.sentence_id,
                    text=sentence.text,
                    page=sentence.page,
                    score=score,
                    cosine_score=cosine,
                    entailment_probability=entailment,
                    neutral_probability=float(cross["neutral_probability"]),
                    contradiction_probability=float(cross["contradiction_probability"]),
                    nli_label=str(cross["nli_label"]),
                )
            )
    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def ranked_search_results(
    sentences: list[SentenceRecord],
    scores: np.ndarray,
    threshold: float,
    top_k: int,
    cosine_scores: np.ndarray | None = None,
) -> list[SearchResult]:
    indices = np.argsort(-scores)
    results: list[SearchResult] = []
    for index in indices:
        score = float(scores[index])
        if score < threshold:
            continue
        sentence = sentences[int(index)]
        results.append(
            SearchResult(
                sentence_id=sentence.sentence_id,
                text=sentence.text,
                page=sentence.page,
                score=score,
                cosine_score=float(cosine_scores[index]) if cosine_scores is not None else score,
            )
        )
        if len(results) >= top_k:
            break
    return results


def compare_documents(
    left_sentences: list[SentenceRecord],
    right_sentences: list[SentenceRecord],
    model_choice: str,
    threshold: float,
    candidate_top_k: int = 5,
    alpha: float = 0.55,
) -> tuple[float, list[MatchResult]]:
    left_texts = [sentence.text for sentence in left_sentences]
    right_texts = [sentence.text for sentence in right_sentences]
    if not left_texts or not right_texts:
        return 0.0, []

    if model_choice == "TF-IDF baseline":
        matrix = tfidf_pair_matrix(left_texts, right_texts)
        matches = greedy_matches(left_sentences, right_sentences, matrix, threshold)
        return similarity_percent(len(left_texts), len(right_texts), len(matches)), matches

    if model_choice == "Cross-Encoder NLI":
        candidate_pairs = [(i, j) for i in range(len(left_texts)) for j in range(len(right_texts))]
        matches = cross_encoder_matches(
            left_sentences,
            right_sentences,
            candidate_pairs,
            threshold=threshold,
        )
        return similarity_percent(len(left_texts), len(right_texts), len(matches)), matches

    bi_model_choice = "Fine-tuned MiniLM" if model_choice == "Hybrid reranker" else model_choice
    bi_model = load_sentence_transformer(bi_model_choice)
    cosine_matrix = pair_embedding_matrix(bi_model, left_texts, right_texts)

    if model_choice != "Hybrid reranker":
        matches = greedy_matches(left_sentences, right_sentences, cosine_matrix, threshold)
        return similarity_percent(len(left_texts), len(right_texts), len(matches)), matches

    candidate_pairs = top_candidates_from_matrix(cosine_matrix, top_k_per_left=candidate_top_k)
    matches = cross_encoder_matches(
        left_sentences,
        right_sentences,
        candidate_pairs,
        threshold=threshold,
        cosine_matrix=cosine_matrix,
        alpha=alpha,
    )
    return similarity_percent(len(left_texts), len(right_texts), len(matches)), matches


def top_candidates_from_matrix(matrix: np.ndarray, top_k_per_left: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left_index in range(matrix.shape[0]):
        right_indices = np.argsort(-matrix[left_index])[: min(top_k_per_left, matrix.shape[1])]
        pairs.extend((left_index, int(right_index)) for right_index in right_indices)
    return pairs


def greedy_matches(
    left_sentences: list[SentenceRecord],
    right_sentences: list[SentenceRecord],
    score_matrix: np.ndarray,
    threshold: float,
) -> list[MatchResult]:
    candidate_indices = np.argwhere(score_matrix >= threshold)
    candidates = sorted(
        (
            (int(left_index), int(right_index), float(score_matrix[left_index, right_index]))
            for left_index, right_index in candidate_indices
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[MatchResult] = []
    for left_index, right_index, score in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        left = left_sentences[left_index]
        right = right_sentences[right_index]
        matches.append(
            MatchResult(
                sentence_id_a=left.sentence_id,
                sentence_id_b=right.sentence_id,
                text_a=left.text,
                text_b=right.text,
                page_a=left.page,
                page_b=right.page,
                score=score,
                cosine_score=score,
            )
        )
        used_left.add(left_index)
        used_right.add(right_index)
    return matches


def cross_encoder_matches(
    left_sentences: list[SentenceRecord],
    right_sentences: list[SentenceRecord],
    candidate_pairs: list[tuple[int, int]],
    threshold: float,
    cosine_matrix: np.ndarray | None = None,
    alpha: float = 0.55,
) -> list[MatchResult]:
    pairs = [(left_sentences[i].text, right_sentences[j].text) for i, j in candidate_pairs]
    cross_outputs = cross_encoder_predict(pairs)
    candidates: list[tuple[int, int, float, dict[str, float | str], float | None]] = []
    for (left_index, right_index), cross in zip(candidate_pairs, cross_outputs):
        entailment = float(cross["entailment_probability"])
        cosine = float(cosine_matrix[left_index, right_index]) if cosine_matrix is not None else None
        score = entailment if cosine is None else alpha * entailment + (1.0 - alpha) * cosine
        if score >= threshold:
            candidates.append((left_index, right_index, score, cross, cosine))

    candidates.sort(key=lambda item: item[2], reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[MatchResult] = []
    for left_index, right_index, score, cross, cosine in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        left = left_sentences[left_index]
        right = right_sentences[right_index]
        matches.append(
            MatchResult(
                sentence_id_a=left.sentence_id,
                sentence_id_b=right.sentence_id,
                text_a=left.text,
                text_b=right.text,
                page_a=left.page,
                page_b=right.page,
                score=float(score),
                cosine_score=cosine,
                entailment_probability=float(cross["entailment_probability"]),
                nli_label=str(cross["nli_label"]),
            )
        )
        used_left.add(left_index)
        used_right.add(right_index)
    return matches


def similarity_percent(num_left: int, num_right: int, matched_pairs: int) -> float:
    denominator = num_left + num_right
    if denominator == 0:
        return 0.0
    return 2.0 * matched_pairs / denominator * 100.0
