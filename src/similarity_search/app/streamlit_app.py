"""Streamlit entrypoint for the project demo."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Semantic Similarity Search",
        page_icon="",
        layout="wide",
    )

    st.title("Semantic Similarity Search")

    search_tab, compare_tab, benchmark_tab = st.tabs(
        ["Semantic Search", "Compare Documents", "Benchmark"]
    )

    with search_tab:
        st.file_uploader("Document", type=["txt", "pdf", "docx"], key="search_document")
        st.text_input("Query")
        st.selectbox(
            "Model",
            ["TF-IDF baseline", "Pretrained MiniLM", "Fine-tuned MiniLM", "Hybrid reranker"],
        )
        st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
        st.button("Search")

    with compare_tab:
        col_a, col_b = st.columns(2)
        with col_a:
            st.file_uploader("Document A", type=["txt", "pdf", "docx"], key="document_a")
        with col_b:
            st.file_uploader("Document B", type=["txt", "pdf", "docx"], key="document_b")
        st.selectbox(
            "Comparison model",
            ["TF-IDF baseline", "Pretrained MiniLM", "Fine-tuned MiniLM", "Hybrid reranker"],
        )
        st.slider("Match threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
        st.button("Compare")

    with benchmark_tab:
        st.info("EDA and model metrics will be shown here after experiments are finished.")


if __name__ == "__main__":
    main()

