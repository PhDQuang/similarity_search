# Semantic Document Similarity Search

This repo is for an NLP final project about semantic document search and
document-to-document similarity using AllNLI.

## Main dataset

- Hugging Face dataset: `sentence-transformers/all-nli`
- Recommended subsets:
  - `pair-class`: EDA, NLI classification, label distribution.
  - `pair-score`: similarity scoring/regression experiments.
  - `pair`: positive pairs for bi-encoder training.
  - `triplet`: anchor-positive-negative training with hard negatives.

## Repo structure

```text
similarity_search/
  configs/
    allnli_data.json
  data/
    raw/
    processed/
  docs/
    ALLNLI_DATASET.md
    TEAM_WORKFLOW.md
  notebooks/
    README.md
  outputs/
    figures/
    reports/
    tables/
  src/
    similarity_search/
      data/
        eda_allnli.py
        prepare_allnli.py
        text_utils.py
      app/
      evaluation/
      models/
  models/
  requirements.txt
  pyproject.toml
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For later model training:

```powershell
python -m pip install -r requirements-train.txt
```

For the web demo:

```powershell
python -m pip install -r requirements-app.txt
```

## Prepare AllNLI data

Quick local test with a small sample:

```powershell
python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
```

Prepare the recommended data subsets:

```powershell
python -m similarity_search.data.prepare_allnli --subsets pair-class pair-score triplet
```

Outputs are written to:

```text
data/processed/allnli/
```

## Run EDA

```powershell
python -m similarity_search.data.eda_allnli --subset pair-class
```

Outputs are written to:

```text
outputs/tables/allnli/pair-class/
outputs/figures/allnli/pair-class/
outputs/reports/allnli/pair-class/
```

## Run the TF-IDF baseline

Fit TF-IDF on the training split, select the entailment similarity threshold
on the development split, and report pair and retrieval metrics on the test
split:

```powershell
python -m similarity_search.models.tfidf_baseline
```

Outputs are written to:

```text
outputs/tfidf_baseline/metrics.json
outputs/tfidf_baseline/dev_predictions.csv
outputs/tfidf_baseline/test_predictions.csv
models/tfidf_baseline/vectorizer.joblib
```

## Run the pretrained MiniLM baseline

Evaluate `sentence-transformers/all-MiniLM-L6-v2` without fine-tuning:

```powershell
python -m pip install -r requirements-train.txt
python -m similarity_search.models.minilm_baseline --device cpu
python scripts/compare_baselines.py
```

The MiniLM metrics and the shared comparison table are written to:

```text
outputs/minilm_baseline/metrics.json
outputs/tables/model_comparison.csv
```

## Fine-tune MiniLM on Kaggle

The repository includes a Kaggle-ready training CLI using AllNLI `pair` and
`MultipleNegativesRankingLoss`:

```bash
python -m similarity_search.models.train_biencoder \
  --output-dir /kaggle/working/allnli-minilm-biencoder \
  --num-train-epochs 1 \
  --batch-size 64
```

See `docs/KAGGLE_BIENCODER_TRAINING.md` for the full notebook workflow,
evaluation commands, model download, and optional Hugging Face upload.

## Collaboration recommendation

Use GitHub for source code, notebooks, report source, issues, and pull requests.
Use Hugging Face Hub for trained models. Use Google Drive only for large files
that should not be committed, such as raw exports, demo videos, slides, and
temporary checkpoints.

## Docker

Build the shared data/EDA environment:

```powershell
docker compose build dev
```

Run data preparation inside Docker:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
```

Run EDA inside Docker:

```powershell
docker compose run --rm dev python -m similarity_search.data.eda_allnli --subset pair-class
```

Run the web skeleton:

```powershell
docker compose up web
```

Then open:

```text
http://localhost:8501
```

For the full setup and GitHub workflow, see:

```text
docs/DOCKER_AND_GITHUB_WORKFLOW.md
```

The web demo now includes semantic search, document comparison, Cross-Encoder
NLI scoring, and Hybrid reranking. See:

```text
docs/WEB_DEMO.md
```

## Share processed dataset with the team

Do not commit processed datasets to GitHub. Publish processed subsets to
Hugging Face Dataset Hub, then every team member can load the same data with
`datasets.load_dataset()`.

Example:

```powershell
python scripts/push_processed_dataset_to_hub.py --subset pair-class --repo-id <team-name>/allnli-pair-class-processed --private
```

Team members load it with:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-pair-class-processed")
```

See:

```text
docs/DATASET_SHARING_AND_TRAINING.md
```
