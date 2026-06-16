"""Streamlit demo for semantic document search and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from similarity_search.app.document_utils import (
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
    "Cross-Encoder NLI",
    "Hybrid reranker",
]


def default_threshold(model_choice: str) -> float:
    defaults = {
        "TF-IDF baseline": 0.13,
        "Pretrained MiniLM": 0.58,
        "Fine-tuned MiniLM": 0.56,
        "Cross-Encoder NLI": 0.36,
        "Hybrid reranker": 0.62,
    }
    return defaults.get(model_choice, 0.6)


def model_help(model_choice: str) -> str:
    descriptions = {
        "TF-IDF baseline": "Keyword baseline. Fast, but weak on paraphrases.",
        "Pretrained MiniLM": "Sentence embedding model without project fine-tuning.",
        "Fine-tuned MiniLM": "Project Bi-Encoder trained on AllNLI pair-score.",
        "Cross-Encoder NLI": "Project NLI model. Accurate on sentence pairs, slower for many pairs.",
        "Hybrid reranker": "Fine-tuned MiniLM retrieves candidates, Cross-Encoder reranks them.",
    }
    return descriptions[model_choice]


def uploaded_sentences(uploaded_file) -> list:
    if uploaded_file is None:
        return []
    uploaded_file.seek(0)
    return extract_uploaded_file(uploaded_file, uploaded_file.name)


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
            "Cross-Encoder": "ready" if status["cross_encoder"] else "missing",
        }
    )
    st.sidebar.caption(
        "Hybrid uses Fine-tuned MiniLM for fast candidate retrieval and Cross-Encoder NLI for reranking."
    )


def semantic_search_tab() -> None:
    st.subheader("Semantic Ctrl+F")
    uploaded_file = st.file_uploader("Document", type=["txt", "pdf", "docx"], key="search_document")
    query = st.text_input("Query", placeholder="Type a sentence or meaning to search for")

    col_model, col_top_k, col_threshold = st.columns([2, 1, 1])
    with col_model:
        model_choice = st.selectbox("Model", MODEL_OPTIONS, index=4, help="Choose how results are scored.")
        st.caption(model_help(model_choice))
    with col_top_k:
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=10, step=1)
    with col_threshold:
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold(model_choice),
            step=0.01,
        )

    rerank_col, alpha_col = st.columns(2)
    with rerank_col:
        rerank_top_k = st.number_input("Hybrid candidates", min_value=5, max_value=200, value=30, step=5)
    with alpha_col:
        alpha = st.slider("Hybrid Cross-Encoder weight", 0.0, 1.0, 0.55, 0.05)

    if st.button("Search", type="primary"):
        if uploaded_file is None:
            st.warning("Upload a document first.")
            return
        if not query.strip():
            st.warning("Enter a query first.")
            return

        with st.spinner("Reading document and running semantic search..."):
            sentences = uploaded_sentences(uploaded_file)
            results = semantic_search(
                query=query.strip(),
                sentences=sentences,
                model_choice=model_choice,
                top_k=int(top_k),
                threshold=float(threshold),
                rerank_top_k=int(rerank_top_k),
                alpha=float(alpha),
            )

        metric_col_a, metric_col_b = st.columns(2)
        with metric_col_a:
            metric_card("Sentences", str(len(sentences)))
        with metric_col_b:
            metric_card("Matches", str(len(results)))

        if not results:
            st.info("No sentence passed the selected threshold.")
            with st.expander("Extracted sentences"):
                st.dataframe(pd.DataFrame(sentence_table(sentences)), use_container_width=True)
            return

        frame = search_results_frame(results)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="semantic_search_results.csv",
            mime="text/csv",
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
        model_choice = st.selectbox("Comparison model", MODEL_OPTIONS, index=4)
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

    alpha = st.slider("Hybrid Cross-Encoder weight", 0.0, 1.0, 0.55, 0.05, key="compare_alpha")

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

    comparison_path = OUTPUTS_DIR / "finetuned_minilm" / "model_comparison.csv"
    fallback_comparison_path = OUTPUTS_DIR / "tables" / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    elif fallback_comparison_path.exists():
        comparison = pd.read_csv(fallback_comparison_path)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
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
    sidebar()
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
