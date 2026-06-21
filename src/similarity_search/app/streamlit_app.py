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

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from similarity_search.app.document_utils import (
    SentenceRecord,
    extract_uploaded_file,
    sentence_table,
)
from similarity_search.app.similarity_engine import (
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


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        :root {
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
        }
        .stApp {
            background: var(--word-bg);
            color: var(--text-main);
        }
        header[data-testid="stHeader"] {
            background: #10241c;
            border-bottom: 1px solid var(--accent);
        }
        .block-container {
            max-width: 100%;
            padding: 3.25rem 1rem 1rem;
        }
        h1 {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin: 0 0 0.35rem !important;
            color: var(--text-main);
        }
        h2, h3 {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            margin: 0 0 0.4rem !important;
        }
        div[data-testid="stTabs"] > div[role="tablist"] {
            background: #f7f8ef;
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
            background: #ffffff !important;
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
            background: #f7f8ef !important;
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
            background: #eaf3e8 !important;
            border-color: var(--accent-soft) !important;
            color: #20352d !important;
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
            background: #ffffff !important;
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
            background: #eaf3e8 !important;
            border: 1px solid var(--pane-border) !important;
            border-radius: 6px !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploaderFile"] * {
            color: var(--text-main) !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: #ffffff !important;
            color: var(--text-main) !important;
            border: 1px solid var(--pane-border);
            border-radius: 6px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0;
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
            background: rgba(255, 255, 255, 0.42);
            border: 1px solid rgba(176, 212, 184, 0.7);
            border-radius: 6px;
            margin-bottom: 0.55rem;
            padding: 0.6rem 0.65rem;
        }
        .word-result-active {
            background: #eaf3e8;
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
            background: #ffffff;
            border: 1px solid #d6d2c3;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.14);
            color: #1f1f1f;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 15.5px;
            line-height: 1.72;
            margin: 0 auto 1rem;
            max-width: 840px;
            min-height: 980px;
            padding: 4.8rem 5.3rem;
        }
        .word-page-meta {
            color: var(--text-muted);
            font-family: Arial, sans-serif;
            font-size: 0.74rem;
            margin-bottom: 1rem;
        }
        .word-empty-page {
            color: #8a8a8a;
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
            border-top: 1px solid rgba(176, 212, 184, 0.65);
            margin-top: 0.75rem;
            padding-top: 0.75rem;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stCaptionContainer"],
        label {
            color: var(--text-muted) !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
        @media (max-width: 1050px) {
            .word-page {
                padding: 3rem 2.2rem;
                min-height: 760px;
            }
        }
        </style>
        """,
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
    return [
        (f"Page {index + 1}", sentences[start : start + chunk_size])
        for index, start in enumerate(range(0, len(sentences), chunk_size))
    ]


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
        body = " ".join(
            sentence_html(sentence, result_ids, active_sentence_id)
            for sentence in page_sentences
        )
        pages_html.append(
            '<div class="word-page">'
            f'<div class="word-page-meta">{escape(filename or "Document")} - {escape(page_label)}</div>'
            f"<p>{body}</p>"
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


def render_results(results: list) -> None:
    if not results:
        st.markdown('<div class="word-search-meta">0 results</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="word-search-meta">{len(results)} results</div>',
        unsafe_allow_html=True,
    )
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


def benchmark_tab() -> None:
    st.subheader("Benchmark")

    final_summary_path = OUTPUTS_DIR / "tables" / "final_model_summary.csv"
    if final_summary_path.exists():
        st.markdown("**Final Model Summary**")
        st.dataframe(pd.read_csv(final_summary_path), use_container_width=True, hide_index=True)

    ablation_path = OUTPUTS_DIR / "tables" / "tfidf_preprocessing_ablation.csv"
    if ablation_path.exists():
        st.markdown("**TF-IDF Preprocessing Ablation**")
        st.dataframe(pd.read_csv(ablation_path), use_container_width=True, hide_index=True)

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
    elif not final_summary_path.exists():
        st.info("Model comparison table not found yet.")

    cross_metadata = load_json(OUTPUTS_DIR / "cross_encoder_outputs" / "cross_encoder_training_metadata.json")
    if cross_metadata:
        st.markdown("**Cross-Encoder Summary**")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            metric_card("NLI accuracy", f"{cross_metadata['test_nli_metrics']['test_accuracy']:.4f}")
        with col_b:
            metric_card("NLI macro F1", f"{cross_metadata['test_nli_metrics']['test_macro_f1']:.4f}")
        with col_c:
            metric_card("Similarity F1", f"{cross_metadata['test_binary_similarity']['f1']:.4f}")
        with col_d:
            metric_card("Rerank MRR", f"{cross_metadata['retrieval_rerank']['mrr']:.4f}")

        with st.expander("Cross-Encoder metadata"):
            st.json(cross_metadata)

    confusion_path = OUTPUTS_DIR / "cross_encoder_outputs" / "cross_encoder_confusion_matrix.csv"
    if confusion_path.exists():
        st.markdown("**Cross-Encoder Confusion Matrix**")
        st.dataframe(pd.read_csv(confusion_path, index_col=0), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Semantic Similarity Search",
        page_icon="",
        layout="wide",
    )
    apply_app_style()
    st.title("Semantic Similarity Search")

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
