"""Streamlit demo for semantic document search and comparison."""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from similarity_search.app.document_utils import (
    SentenceRecord,
    extract_uploaded_file,
    sentence_table,
)
from similarity_search.app.similarity_engine import (
    MODELS_DIR,
    OUTPUTS_DIR,
    compare_documents,
    load_json,
    model_status,
    semantic_search,
)


MODEL_OPTIONS = [
    "TF-IDF baseline",
    "Pretrained MiniLM",
    "Fine-tuned MiniLM",
    "SFT-BE checkpoint",
    "Cross-Encoder NLI",
    "TF-IDF + Cross-Encoder",
    "Pretrained MiniLM + Cross-Encoder",
    "Fine-tuned MiniLM + Cross-Encoder",
    "SFT-BE + Cross-Encoder",
]


LIGHT_THEME_VARS = """
    --word-bg: #eae7d6;
    --pane-bg: #f7f8ef;
    --pane-border: #b0d4b8;
    --text-main: #26332e;
    --text-muted: #60736b;
    --accent: #5d7b6f;
    --accent-soft: #a4c3a2;
    --accent-pale: #d7f9fa;
    --hit: #d7f9fa;
    --hit-active: #b0d4b8;
    --header-bg: #10241c;
    --surface: #ffffff;
    --surface-hover: #eaf3e8;
    --page-bg: #ffffff;
    --page-text: #1f1f1f;
    --page-border: #d6d2c3;
    --page-shadow: rgba(0, 0, 0, 0.14);
    --result-bg: rgba(255, 255, 255, 0.42);
    --result-border: rgba(176, 212, 184, 0.7);
    --empty-text: #8a8a8a;
"""

DARK_THEME_VARS = """
    --word-bg: #161a18;
    --pane-bg: #1d2320;
    --pane-border: #3a4a42;
    --text-main: #e6ebe7;
    --text-muted: #9bb0a6;
    --accent: #7fae9b;
    --accent-soft: #4d6a5e;
    --accent-pale: #234547;
    --hit: #2c4a4b;
    --hit-active: #3f6358;
    --header-bg: #0a120f;
    --surface: #232a27;
    --surface-hover: #2c3a33;
    --page-bg: #20262a;
    --page-text: #e6e8e3;
    --page-border: #3a4138;
    --page-shadow: rgba(0, 0, 0, 0.55);
    --result-bg: rgba(255, 255, 255, 0.05);
    --result-border: rgba(127, 174, 155, 0.4);
    --empty-text: #8aa098;
"""


REST_CSS = """
        .stApp {
            background: var(--word-bg);
            color: var(--text-main);
            transition: background 0.2s ease, color 0.2s ease;
        }
        header[data-testid="stHeader"] {
            background: var(--header-bg);
            border-bottom: 1px solid var(--accent);
        }
        .block-container {
            max-width: 100%;
            padding: 3.25rem 1rem 1rem;
        }
        h1 {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
            margin: 0 0 0.15rem !important;
            color: var(--text-main);
        }
        h2, h3 {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            margin: 0 0 0.4rem !important;
        }
        div[data-testid="stTabs"] > div[role="tablist"] {
            background: var(--pane-bg);
            border-bottom: 1px solid var(--pane-border);
            border-left: 0;
            border-right: 0;
            border-top: 0;
            border-radius: 0;
            padding-left: 0.35rem;
            gap: 0;
        }
        button[role="tab"] {
            padding: 0.45rem 0.85rem !important;
            color: var(--text-muted) !important;
        }
        button[role="tab"][aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            border: 1px solid var(--pane-border) !important;
            color: var(--text-main) !important;
            border-radius: 6px !important;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] div {
            color: var(--accent) !important;
        }
        div.stButton > button,
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFileUploader"] button,
        div[data-testid="stNumberInput"] button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-minimal"],
        button[data-baseweb="button"] {
            background: var(--pane-bg) !important;
            border: 1px solid var(--pane-border) !important;
            color: var(--text-main) !important;
            border-radius: 6px !important;
            min-height: 2.25rem;
        }
        div.stButton > button:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFileUploader"] button:hover,
        div[data-testid="stNumberInput"] button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-minimal"]:hover,
        button[data-baseweb="button"]:hover {
            background: var(--surface-hover) !important;
            border-color: var(--accent-soft) !important;
            color: var(--text-main) !important;
        }
        div.stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #ffffff !important;
        }
        div[data-testid="stNumberInput"] button {
            border-left: 0 !important;
            min-width: 2rem !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--surface) !important;
            border: 1px solid var(--pane-border) !important;
            color: var(--text-main) !important;
            min-height: 44px;
            padding: 0.35rem;
            border-radius: 6px;
        }
        div[data-testid="stFileUploaderDropzone"] * {
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploaderFile"] {
            background: var(--surface-hover) !important;
            border: 1px solid var(--pane-border) !important;
            border-radius: 6px !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploaderFile"] * {
            color: var(--text-main) !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: var(--surface) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--pane-border);
            border-radius: 6px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0;
        }
        div[data-testid="stCheckbox"] label p {
            color: var(--text-muted) !important;
        }
        .word-pane {
            background: var(--pane-bg);
            border: 1px solid var(--pane-border);
            min-height: calc(100vh - 125px);
            padding: 0.75rem;
        }
        .word-pane-title {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 0.55rem;
            text-transform: uppercase;
        }
        .word-search-meta {
            color: var(--text-muted);
            font-size: 0.78rem;
            margin: 0.45rem 0 0.6rem;
        }
        .word-result {
            background: var(--result-bg);
            border: 1px solid var(--result-border);
            border-radius: 6px;
            margin-bottom: 0.55rem;
            padding: 0.6rem 0.65rem;
        }
        .word-result-active {
            background: var(--surface-hover);
            border-left: 3px solid var(--accent);
            padding-left: 0.45rem;
        }
        .word-result-score {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        .word-result-text {
            color: var(--text-main);
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .word-result-page {
            color: var(--text-muted);
            font-size: 0.72rem;
            margin-top: 0.25rem;
        }
        .word-workspace {
            min-height: calc(100vh - 125px);
            overflow: auto;
            padding: 0 1rem 1.25rem;
        }
        .word-page {
            background: var(--page-bg);
            border: 1px solid var(--page-border);
            box-shadow: 0 1px 3px var(--page-shadow);
            color: var(--page-text);
            font-family: Georgia, "Times New Roman", serif;
            font-size: 15.5px;
            line-height: 1.72;
            margin: 0 auto 1rem;
            max-width: 840px;
            min-height: 980px;
            padding: 4.8rem 5.3rem;
        }
        .word-page p {
            margin: 0 0 0.95rem;
        }
        .word-page p:last-child {
            margin-bottom: 0;
        }
        .word-page-meta {
            color: var(--text-muted);
            font-family: Arial, sans-serif;
            font-size: 0.74rem;
            margin-bottom: 1rem;
        }
        .word-empty-page {
            color: var(--empty-text);
            font-family: Arial, sans-serif;
            font-size: 0.92rem;
            text-align: center;
            padding-top: 18rem;
        }
        .sentence-hit {
            background: var(--hit);
            border-radius: 2px;
            padding: 0.08rem 0.1rem;
        }
        .sentence-hit-active {
            background: var(--hit-active);
            outline: 2px solid var(--accent);
            scroll-margin-top: 5.5rem;
        }
        .settings-section {
            border-top: 1px solid var(--pane-border);
            margin-top: 0.75rem;
            padding-top: 0.75rem;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stCaptionContainer"],
        label {
            color: var(--text-muted) !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-main) !important;
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--pane-border);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            box-shadow: 0 1px 2px var(--page-shadow);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-1px);
            border-color: var(--accent-soft);
            box-shadow: 0 4px 10px var(--page-shadow);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.74rem !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .app-header {
            margin-bottom: 0.15rem;
        }
        .app-subtitle {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin: 0 0 0.4rem;
        }
        .section-header {
            align-items: center;
            color: var(--text-main);
            display: flex;
            font-size: 0.92rem;
            font-weight: 600;
            gap: 0.5rem;
            margin: 1.15rem 0 0.6rem;
        }
        .section-header .section-bar {
            background: var(--accent);
            border-radius: 2px;
            display: inline-block;
            height: 0.95rem;
            width: 3px;
        }
        button[role="tab"] {
            border-radius: 6px 6px 0 0 !important;
            font-weight: 600 !important;
            transition: background 0.15s ease, color 0.15s ease;
        }
        button[role="tab"]:hover {
            background: var(--surface-hover) !important;
            color: var(--text-main) !important;
        }
        div.stButton > button,
        div[data-testid="stDownloadButton"] button {
            transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
        }
        div.stButton > button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 8px var(--page-shadow);
        }
        div.stButton > button:active,
        div[data-testid="stDownloadButton"] button:active {
            transform: translateY(0);
            box-shadow: none;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--pane-border) !important;
            border-radius: 10px !important;
            background: var(--pane-bg) !important;
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] details > summary {
            font-weight: 600;
            color: var(--text-main) !important;
        }
        div[data-testid="stExpander"] summary:hover {
            color: var(--accent) !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            overflow: hidden;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        @media (max-width: 1050px) {
            .word-page {
                padding: 3rem 2.2rem;
                min-height: 760px;
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding: 2.5rem 0.5rem 1rem;
            }
            .word-page {
                padding: 1.6rem 1.2rem;
                min-height: auto;
                font-size: 14.5px;
            }
            .word-pane {
                min-height: auto;
            }
            .word-workspace {
                min-height: auto;
                padding: 0 0.4rem 1rem;
            }
        }
        """


def apply_app_style(dark: bool = False) -> None:
    theme_vars = DARK_THEME_VARS if dark else LIGHT_THEME_VARS
    st.markdown(
        f"<style>:root {{{theme_vars}}}{REST_CSS}</style>",
        unsafe_allow_html=True,
    )


def default_threshold(model_choice: str) -> float:
    defaults = {
        "TF-IDF baseline": 0.13,
        "Pretrained MiniLM": 0.58,
        "Fine-tuned MiniLM": 0.56,
        "SFT-BE checkpoint": 0.55,
        "Cross-Encoder NLI": 0.36,
        "TF-IDF + Cross-Encoder": 0.55,
        "Pretrained MiniLM + Cross-Encoder": 0.62,
        "Fine-tuned MiniLM + Cross-Encoder": 0.62,
        "SFT-BE + Cross-Encoder": 0.62,
    }
    return defaults.get(model_choice, 0.6)


def model_help(model_choice: str) -> str:
    descriptions = {
        "TF-IDF baseline": "Keyword baseline. Fast, but weak on paraphrases.",
        "Pretrained MiniLM": "Sentence embedding model without project fine-tuning.",
        "Fine-tuned MiniLM": "Project Bi-Encoder trained on AllNLI pair-score.",
        "SFT-BE checkpoint": "Custom shallow transformer bi-encoder loaded from models/sftbe_checkpoint/stage0_final.pt.",
        "Cross-Encoder NLI": "Project NLI model. Accurate on sentence pairs, slower for many pairs.",
        "TF-IDF + Cross-Encoder": "TF-IDF retrieves candidates, then Cross-Encoder reranks them by entailment probability.",
        "Pretrained MiniLM + Cross-Encoder": "Pretrained MiniLM retrieves candidates, then Cross-Encoder reranks them.",
        "Fine-tuned MiniLM + Cross-Encoder": "Fine-tuned MiniLM retrieves candidates, then Cross-Encoder reranks them.",
        "SFT-BE + Cross-Encoder": "SFT-BE retrieves candidates, then Cross-Encoder reranks them.",
    }
    return descriptions[model_choice]


def uploaded_sentences(uploaded_file) -> list:
    if uploaded_file is None:
        return []
    uploaded_file.seek(0)
    return extract_uploaded_file(uploaded_file, uploaded_file.name)


def document_key(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


def current_search_sentences(uploaded_file) -> list[SentenceRecord]:
    key = document_key(uploaded_file)
    if key is None:
        st.session_state.pop("search_document_key", None)
        st.session_state["search_sentences"] = []
        return []

    if st.session_state.get("search_document_key") != key:
        st.session_state["search_document_key"] = key
        st.session_state["search_sentences"] = uploaded_sentences(uploaded_file)
        st.session_state["search_results"] = []
        st.session_state["active_sentence_id"] = None
    return st.session_state.get("search_sentences", [])


def chunk_sentences(sentences: list[SentenceRecord], chunk_size: int = 34) -> list[tuple[str, list[SentenceRecord]]]:
    if not sentences:
        return []
    if any(sentence.page is not None for sentence in sentences):
        pages: dict[int, list[SentenceRecord]] = {}
        for sentence in sentences:
            page = sentence.page if sentence.page is not None else 1
            pages.setdefault(page, []).append(sentence)
        return [(f"Page {page}", page_sentences) for page, page_sentences in sorted(pages.items())]

    chunks: list[list[SentenceRecord]] = []
    current_chunk: list[SentenceRecord] = []
    for paragraph_sentences in group_by_paragraph(sentences):
        if current_chunk and len(current_chunk) + len(paragraph_sentences) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.extend(paragraph_sentences)
    if current_chunk:
        chunks.append(current_chunk)
    return [(f"Page {index + 1}", chunk) for index, chunk in enumerate(chunks)]


def sentence_html(
    sentence: SentenceRecord,
    result_ids: set[int],
    active_sentence_id: int | None,
) -> str:
    text = escape(sentence.text)
    if sentence.sentence_id not in result_ids:
        return text
    classes = "sentence-hit"
    if sentence.sentence_id == active_sentence_id:
        classes += " sentence-hit-active"
    return f'<span id="sentence-{sentence.sentence_id}" class="{classes}">{text}</span>'


def group_by_paragraph(sentences: list[SentenceRecord]) -> list[list[SentenceRecord]]:
    groups: list[list[SentenceRecord]] = []
    current_paragraph: int | None = None
    for sentence in sentences:
        if current_paragraph is None or sentence.paragraph != current_paragraph:
            groups.append([sentence])
            current_paragraph = sentence.paragraph
        else:
            groups[-1].append(sentence)
    return groups


def render_document(
    sentences: list[SentenceRecord],
    results: list,
    active_sentence_id: int | None,
    filename: str | None,
) -> None:
    result_ids = {result.sentence_id for result in results}
    if not sentences:
        st.markdown(
            '<div class="word-workspace"><div class="word-page">'
            '<div class="word-empty-page">Open a TXT, PDF, or DOCX file</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    pages_html = []
    for page_label, page_sentences in chunk_sentences(sentences):
        paragraphs_html = []
        for paragraph_sentences in group_by_paragraph(page_sentences):
            paragraph_body = " ".join(
                sentence_html(sentence, result_ids, active_sentence_id)
                for sentence in paragraph_sentences
            )
            paragraphs_html.append(f"<p>{paragraph_body}</p>")
        pages_html.append(
            '<div class="word-page">'
            f'<div class="word-page-meta">{escape(filename or "Document")} - {escape(page_label)}</div>'
            f"{''.join(paragraphs_html)}"
            "</div>"
        )
    st.markdown(
        f'<div class="word-workspace">{"".join(pages_html)}</div>',
        unsafe_allow_html=True,
    )
    scroll_to_sentence(active_sentence_id)


def scroll_to_sentence(sentence_id: int | None) -> None:
    if sentence_id is None:
        return
    components.html(
        f"""
        <script>
        const sentenceId = "sentence-{sentence_id}";
        setTimeout(() => {{
          const target = window.parent.document.getElementById(sentenceId);
          if (target) {{
            target.scrollIntoView({{ behavior: "smooth", block: "center" }});
          }}
        }}, 250);
        </script>
        """,
        height=0,
    )


def result_summary(result) -> str:
    page = f"p. {result.page}" if result.page is not None else f"#{result.sentence_id}"
    label = result.nli_label or "match"
    return f"{page} - {result.score:.3f} - {label}"


def render_score_chart(results: list) -> None:
    if len(results) < 2:
        return
    chart_frame = pd.DataFrame(
        {f"#{rank}": result.score for rank, result in enumerate(results, start=1)}.items(),
        columns=["Result", "Score"],
    ).set_index("Result")
    st.bar_chart(chart_frame)


def render_results(results: list) -> None:
    if not results:
        st.markdown('<div class="word-search-meta">0 results</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="word-search-meta">{len(results)} results</div>',
        unsafe_allow_html=True,
    )
    render_score_chart(results)
    for rank, result in enumerate(results, start=1):
        active = st.session_state.get("active_sentence_id") == result.sentence_id
        if st.button(
            result_summary(result),
            key=f"search_result_{result.sentence_id}_{rank}",
            use_container_width=True,
        ):
            st.session_state["active_sentence_id"] = result.sentence_id
            active = True
        active_class = " word-result-active" if active else ""
        page = f"Page {result.page}" if result.page is not None else f"Sentence {result.sentence_id}"
        st.markdown(
            f"""
            <div class="word-result{active_class}">
              <div class="word-result-score">Result {rank} - score {result.score:.4f}</div>
              <div class="word-result-text">{escape(result.text)}</div>
              <div class="word-result-page">{escape(page)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def search_results_frame(results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id": result.sentence_id,
                "page": result.page if result.page is not None else "",
                "score": round(result.score, 4),
                "cosine_score": round(result.cosine_score, 4)
                if result.cosine_score is not None
                else "",
                "entailment_probability": round(result.entailment_probability, 4)
                if result.entailment_probability is not None
                else "",
                "nli_label": result.nli_label or "",
                "sentence": result.text,
            }
            for result in results
        ]
    )


def matches_frame(matches) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id_A": match.sentence_id_a,
                "sentence_id_B": match.sentence_id_b,
                "page_A": match.page_a if match.page_a is not None else "",
                "page_B": match.page_b if match.page_b is not None else "",
                "score": round(match.score, 4),
                "cosine_score": round(match.cosine_score, 4)
                if match.cosine_score is not None
                else "",
                "entailment_probability": round(match.entailment_probability, 4)
                if match.entailment_probability is not None
                else "",
                "nli_label": match.nli_label or "",
                "sentence_A": match.text_a,
                "sentence_B": match.text_b,
            }
            for match in matches
        ]
    )


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def section_header(text: str) -> None:
    st.markdown(
        f'<div class="section-header"><span class="section-bar"></span>{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    st.sidebar.header("Models")
    status = model_status()
    st.sidebar.write(
        {
            "TF-IDF": "ready" if status["tfidf"] else "missing",
            "Fine-tuned MiniLM": "ready" if status["fine_tuned_biencoder"] else "missing",
            "SFT-BE": "ready" if status["sftbe"] else "missing",
            "Cross-Encoder": "ready" if status["cross_encoder"] else "missing",
        }
    )
    st.sidebar.caption(
        "Combined modes use the model before '+' for candidate retrieval and Cross-Encoder NLI for reranking."
    )


def semantic_search_tab() -> None:
    st.session_state.setdefault("search_results", [])
    st.session_state.setdefault("search_sentences", [])
    st.session_state.setdefault("active_sentence_id", None)

    left, center, right = st.columns([0.24, 0.56, 0.20], gap="small")

    with right:
        st.markdown('<div class="word-pane-title">Document</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Open",
            type=["txt", "pdf", "docx"],
            key="search_document",
            label_visibility="collapsed",
        )
        sentences = current_search_sentences(uploaded_file)

        st.markdown('<div class="settings-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="word-pane-title">Search settings</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            index=8,
            help=model_help("SFT-BE + Cross-Encoder"),
        )
        top_k = st.number_input("Results", min_value=1, max_value=100, value=10, step=1)
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold(model_choice),
            step=0.01,
        )
        rerank_top_k = st.number_input("Candidates", min_value=5, max_value=200, value=30, step=5)
        alpha = st.slider("Cross-Encoder weight", 0.0, 1.0, 0.55, 0.05)

        st.markdown('<div class="settings-section"></div>', unsafe_allow_html=True)
        status = model_status()
        st.metric("Sentences", len(sentences))
        st.metric("Matches", len(st.session_state.get("search_results", [])))
        st.caption(
            "TF-IDF {tfidf} | MiniLM {mini} | SFT-BE {sftbe} | Cross-Encoder {cross}".format(
                tfidf="ready" if status["tfidf"] else "missing",
                mini="ready" if status["fine_tuned_biencoder"] else "missing",
                sftbe="ready" if status["sftbe"] else "missing",
                cross="ready" if status["cross_encoder"] else "missing",
            )
        )

    with left:
        st.markdown('<div class="word-pane-title">Navigation</div>', unsafe_allow_html=True)
        query = st.text_input(
            "Search",
            placeholder="Search in document",
            key="word_search_query",
            label_visibility="collapsed",
        )
        search_clicked = st.button("Search", type="primary", use_container_width=True)
        if search_clicked:
            if uploaded_file is None:
                st.warning("Upload a document first.")
            elif not query.strip():
                st.warning("Enter a query first.")
            else:
                with st.spinner("Searching..."):
                    results = semantic_search(
                        query=query.strip(),
                        sentences=sentences,
                        model_choice=model_choice,
                        top_k=int(top_k),
                        threshold=float(threshold),
                        rerank_top_k=int(rerank_top_k),
                        alpha=float(alpha),
                    )
                st.session_state["search_results"] = results
                st.session_state["active_sentence_id"] = results[0].sentence_id if results else None

        render_results(st.session_state.get("search_results", []))
        results = st.session_state.get("search_results", [])
        if results:
            frame = search_results_frame(results)
            st.download_button(
                "Download CSV",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name="semantic_search_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with center:
        render_document(
            sentences=sentences,
            results=st.session_state.get("search_results", []),
            active_sentence_id=st.session_state.get("active_sentence_id"),
            filename=uploaded_file.name if uploaded_file is not None else None,
        )


def compare_documents_tab() -> None:
    st.subheader("Compare Two Documents")
    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Document A", type=["txt", "pdf", "docx"], key="document_a")
    with col_b:
        file_b = st.file_uploader("Document B", type=["txt", "pdf", "docx"], key="document_b")

    col_model, col_threshold, col_candidates = st.columns([2, 1, 1])
    with col_model:
        model_choice = st.selectbox("Comparison model", MODEL_OPTIONS, index=8)
        st.caption(model_help(model_choice))
    with col_threshold:
        threshold = st.slider(
            "Match threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold(model_choice),
            step=0.01,
        )
    with col_candidates:
        candidate_top_k = st.number_input("Hybrid candidates per sentence", min_value=1, max_value=20, value=5)

    alpha = st.slider("Combined Cross-Encoder weight", 0.0, 1.0, 0.55, 0.05, key="compare_alpha")

    if st.button("Compare", type="primary"):
        if file_a is None or file_b is None:
            st.warning("Upload both documents first.")
            return

        with st.spinner("Reading documents and matching sentences..."):
            sentences_a = uploaded_sentences(file_a)
            sentences_b = uploaded_sentences(file_b)
            percent, matches = compare_documents(
                left_sentences=sentences_a,
                right_sentences=sentences_b,
                model_choice=model_choice,
                threshold=float(threshold),
                candidate_top_k=int(candidate_top_k),
                alpha=float(alpha),
            )

        metric_a, metric_b, metric_c = st.columns(3)
        with metric_a:
            metric_card("Similarity", f"{percent:.2f}%")
        with metric_b:
            metric_card("Matched pairs", str(len(matches)))
        with metric_c:
            metric_card("Sentences A / B", f"{len(sentences_a)} / {len(sentences_b)}")

        frame = matches_frame(matches)
        if frame.empty:
            st.info("No sentence pairs passed the selected threshold.")
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download matched pairs CSV",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name="document_similarity_matches.csv",
                mime="text/csv",
            )

        with st.expander("Extracted sentences"):
            left_col, right_col = st.columns(2)
            with left_col:
                st.dataframe(pd.DataFrame(sentence_table(sentences_a)), use_container_width=True, hide_index=True)
            with right_col:
                st.dataframe(pd.DataFrame(sentence_table(sentences_b)), use_container_width=True, hide_index=True)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]


def model_name_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            return column
    return None


def render_comparison_chart(frame: pd.DataFrame, title: str) -> None:
    name_column = model_name_column(frame)
    score_columns = numeric_columns(frame)
    if name_column is None or not score_columns:
        return
    section_header(title)
    chart_frame = frame[[name_column, *score_columns]].melt(
        id_vars=name_column, var_name="Metric", value_name="Value"
    )
    chart = (
        alt.Chart(chart_frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{name_column}:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y("Value:Q", title=None),
            color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[name_column, "Metric", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def benchmark_tab() -> None:
    st.subheader("Benchmark")

    final_summary_path = OUTPUTS_DIR / "tables" / "final_model_summary.csv"
    cross_metadata = load_json(OUTPUTS_DIR / "cross_encoder_outputs" / "cross_encoder_training_metadata.json")

    # --- Main view: headline model comparison ---
    if final_summary_path.exists():
        final_summary = pd.read_csv(final_summary_path)
        render_comparison_chart(final_summary, "Final Model Summary")
        st.dataframe(final_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison table not found yet.")

    # --- Main view: best reranker headline metrics ---
    if cross_metadata:
        section_header("Cross-Encoder reranker — key metrics")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            metric_card("NLI accuracy", f"{cross_metadata['test_nli_metrics']['test_accuracy']:.4f}")
        with col_b:
            metric_card("NLI macro F1", f"{cross_metadata['test_nli_metrics']['test_macro_f1']:.4f}")
        with col_c:
            metric_card("Similarity F1", f"{cross_metadata['test_binary_similarity']['f1']:.4f}")
        with col_d:
            metric_card("Rerank MRR", f"{cross_metadata['retrieval_rerank']['mrr']:.4f}")

    # --- Secondary details tucked away to keep the view clean ---
    with st.expander("More details"):
        ablation_path = OUTPUTS_DIR / "tables" / "tfidf_preprocessing_ablation.csv"
        if ablation_path.exists():
            st.markdown("**TF-IDF Preprocessing Ablation**")
            st.dataframe(pd.read_csv(ablation_path), use_container_width=True, hide_index=True)

        training_summary_path = OUTPUTS_DIR / "tables" / "training_process_summary.csv"
        if training_summary_path.exists():
            st.markdown("**Training Process Summary**")
            st.dataframe(pd.read_csv(training_summary_path), use_container_width=True, hide_index=True)

        comparison_path = OUTPUTS_DIR / "finetuned_minilm" / "model_comparison.csv"
        fallback_comparison_path = OUTPUTS_DIR / "tables" / "model_comparison.csv"
        if comparison_path.exists():
            st.markdown("**MiniLM Comparison**")
            comparison = pd.read_csv(comparison_path)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        elif fallback_comparison_path.exists():
            st.markdown("**Baseline Comparison**")
            comparison = pd.read_csv(fallback_comparison_path)
            st.dataframe(comparison, use_container_width=True, hide_index=True)

        confusion_path = OUTPUTS_DIR / "cross_encoder_outputs" / "cross_encoder_confusion_matrix.csv"
        if confusion_path.exists():
            st.markdown("**Cross-Encoder Confusion Matrix**")
            confusion_frame = pd.read_csv(confusion_path, index_col=0)
            try:
                styled_confusion = confusion_frame.style.background_gradient(cmap="Greens")
                st.dataframe(styled_confusion, use_container_width=True)
            except ImportError:
                st.dataframe(confusion_frame, use_container_width=True)

        sftbe_dir = MODELS_DIR / "sftbe_checkpoint"
        loss_summary = load_json(sftbe_dir / "stage0_interval_loss_summary.json")
        if loss_summary:
            st.markdown("**SFT-BE Training Progress (Stage 0 distillation)**")
            points = pd.DataFrame(loss_summary["points"]).sort_values("global_step")
            chart_frame = points.set_index("global_step")[
                ["interval_loss_reconstructed", "interval_cosine_approx"]
            ]
            chart_frame.columns = ["Loss", "Cosine similarity to teacher"]
            st.line_chart(chart_frame)
            col_a, col_b = st.columns(2)
            with col_a:
                metric_card("Final loss", f"{loss_summary['last_interval_loss_reconstructed']:.4f}")
            with col_b:
                metric_card("Final cosine vs teacher", f"{points.iloc[-1]['interval_cosine_approx']:.4f}")

        tail_eval = load_json(sftbe_dir / "stage0_tail10mb_eval.json")
        if tail_eval:
            st.markdown("**SFT-BE Held-out Eval (tail 10MB Wikipedia, unseen during training)**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                metric_card("Test cosine vs teacher", f"{tail_eval['test_cosine']:.4f}")
            with col_b:
                metric_card("Test loss", f"{tail_eval['test_loss']:.4f}")
            with col_c:
                metric_card("Test samples", f"{tail_eval['test_samples']:,}")

        if cross_metadata:
            st.markdown("**Cross-Encoder metadata**")
            st.json(cross_metadata)


def main() -> None:
    st.set_page_config(
        page_title="Semantic Similarity Search",
        page_icon="",
        layout="wide",
    )
    st.session_state.setdefault("dark_mode", False)
    apply_app_style(dark=st.session_state["dark_mode"])

    title_col, toggle_col = st.columns([0.85, 0.15])
    with title_col:
        st.markdown('<div class="app-header">', unsafe_allow_html=True)
        st.title("Semantic Similarity Search")
        st.markdown(
            '<p class="app-subtitle">Tìm kiếm và đối chiếu câu theo ngữ nghĩa — TF-IDF, MiniLM, SFT-BE và Cross-Encoder NLI.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with toggle_col:
        st.checkbox("Dark mode", key="dark_mode")

    search_tab, compare_tab, benchmarks = st.tabs(
        ["Semantic Search", "Compare Documents", "Benchmark"]
    )
    with search_tab:
        semantic_search_tab()
    with compare_tab:
        compare_documents_tab()
    with benchmarks:
        benchmark_tab()


if __name__ == "__main__":
    main()
