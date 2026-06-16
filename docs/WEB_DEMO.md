# Web demo guide

## Run locally

Install app dependencies:

```powershell
python -m pip install -r requirements-app.txt
python -m pip install -e .
```

Run Streamlit:

```powershell
streamlit run src/similarity_search/app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Run with Docker

```powershell
docker compose build web
docker compose up web
```

Open:

```text
http://localhost:8501
```

## Required local model folders

The web app can use these local folders when present:

```text
models/tfidf_baseline/vectorizer.joblib
models/allnli-minilm-biencoder/final/
models/allnli-cross-encoder-nli/final/
```

If the fine-tuned Bi-Encoder is missing, the app falls back to
`sentence-transformers/all-MiniLM-L6-v2` for embedding search. The Cross-Encoder
needs the trained local folder for the project result.

## Features

Tab 1: Semantic Search

- Upload one TXT/PDF/DOCX document.
- Enter a query sentence.
- Choose a model:
  - TF-IDF baseline.
  - Pretrained MiniLM.
  - Fine-tuned MiniLM.
  - Cross-Encoder NLI.
  - Hybrid reranker.
- Get ranked matching sentences with scores and NLI labels when available.

Tab 2: Compare Documents

- Upload document A and document B.
- Match semantically similar sentences even if sentence order changes.
- Show similarity percentage:

```text
2 * matched_pairs / (sentences_A + sentences_B) * 100
```

Tab 3: Benchmark

- Shows model comparison tables and Cross-Encoder metrics if available in
  `outputs/`.

## Recommended demo script

Use `Hybrid reranker` as the main model:

```text
Fine-tuned MiniLM retrieves top candidate sentences quickly.
Cross-Encoder NLI reranks candidates and provides entailment probability.
```

Suggested thresholds:

| Model | Suggested threshold |
|---|---:|
| TF-IDF baseline | 0.13 |
| Pretrained MiniLM | 0.58 |
| Fine-tuned MiniLM | 0.56 |
| Cross-Encoder NLI | 0.36 |
| Hybrid reranker | 0.62 |

For long documents, prefer Hybrid over pure Cross-Encoder because pure
Cross-Encoder must score many sentence pairs.

## Files to commit

Commit:

- Source code.
- Small metric CSV/JSON files.
- Docs.
- Notebook source.

Do not commit:

- `models/**/*.safetensors`
- `models/**/optimizer.pt`
- Large checkpoint folders.

Push final models to Hugging Face Hub instead.
