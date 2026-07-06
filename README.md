# Semantic Document Similarity Search

This repository implements a complete natural language processing system for semantic document search and document-to-document similarity comparison using AllNLI and modern embedding architectures. The project covers data preprocessing, exploratory data analysis, baseline modeling, neural bi-encoders, cross-encoders, and hybrid reranking pipelines, along with an interactive web application demo.

## Overview and Architecture

The system combines fast dense retrieval with high-precision reranking to achieve both efficiency and accuracy:

1. **Bi-Encoder (Fast Retrieval)**:
   - Uses sentence embedding models such as `sentence-transformers/all-MiniLM-L6-v2` and custom supervised fine-tuned bi-encoders (SFT-BE).
   - Encodes document sentences and input queries into dense 384-dimensional vector representations.
   - Calculates cosine similarity across candidate sentences to rapidly extract the top-M relevant matches.

2. **Cross-Encoder NLI (Precise Reranking)**:
   - Uses a natural language inference (NLI) classification model based on `distilbert-base-uncased`, fine-tuned on sentence pairs.
   - Classifies pairs into three semantic relations: `entailment`, `neutral`, and `contradiction`.
   - By processing query and document sentences simultaneously through self-attention layers, it achieves higher precision than standalone bi-encoders.

3. **Hybrid Reranking System**:
   - Implements a two-stage retrieval pipeline:
     - **Stage 1**: The Bi-Encoder retrieves top-M candidate sentences (e.g., M = 30) based on cosine similarity.
     - **Stage 2**: The Cross-Encoder reranks the retrieved candidates based on their entailment probability.
     - **Final Scoring**: Combines both scores using a weighted sum:
       $$\text{Score} = \alpha \times \text{Entailment\_Prob} + (1 - \alpha) \times \text{Cosine\_Similarity}$$
       where $\alpha$ is a tuning parameter controlling the trade-off between semantic inference and lexical/dense similarity.

## Main Dataset

- **Hugging Face Dataset**: `sentence-transformers/all-nli`
- **Recommended Subsets**:
  - `pair-class`: Contains `premise`, `hypothesis`, and `label` columns. Used for EDA, NLI classification, and Cross-Encoder training.
  - `pair-score`: Contains `sentence1`, `sentence2`, and `score` columns. Used for similarity scoring and regression experiments.
  - `pair`: Contains `anchor` and `positive` columns. Used for bi-encoder training with positive pairs.
  - `triplet`: Contains `anchor`, `positive`, and `negative` columns. Used for triplet loss training with hard negatives.
- **Label Mapping**:
  - `0 (entailment)` -> Semantic positive (Similarity score: 1.0)
  - `1 (neutral)` -> Undetermined relation (Similarity score: 0.5)
  - `2 (contradiction)` -> Semantic conflict (Similarity score: 0.0)

## Repository Structure

```text
similarity_search/
  configs/
    allnli_70_15_15.json
    allnli_data.json
  data/
    raw/
    processed/
  models/
    allnli-cross-encoder-nli/
    allnli-minilm-biencoder/
    sftbe_checkpoint/
    tfidf_baseline/
  notebooks/
    01_train_tfidf_clean_full_eval5k_kaggle.ipynb
    02_train_minilm_clean_full_earlystop_eval5k_kaggle.ipynb
    03_train_cross_encoder_clean_full_earlystop_eval5k_kaggle.ipynb
    04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb
    05_evaluate_hybrid_models_kaggle.ipynb
    README.md
  outputs/
    figures/
    reports/
    tables/
  scripts/
    build_final_model_summary.py
    build_training_process_artifacts.py
    compare_baselines.py
    create_hybrid_notebook.py
    evaluate_sftbe_cross_encoder.py
    push_processed_dataset_to_hub.py
  src/
    similarity_search/
      app/
        streamlit_app.py
      data/
        eda_allnli.py
        prepare_allnli.py
        prepare_allnli_70_15_15_clean.py
        text_utils.py
      evaluation/
      models/
        minilm_baseline.py
        sftbe_evaluation.py
        tfidf_baseline.py
        tfidf_preprocessing_ablation.py
        train_biencoder.py
        train_cross_encoder.py
        train_minilm.py
        train_sftbe.py
        train_tfidf.py
      sftbe/
        evaluate.py
        model/
        prepare_data.py
        train.py
  docker-compose.yml
  Dockerfile
  pyproject.toml
  README.md
  requirements-app.txt
  requirements-kaggle.txt
  requirements-train.txt
  requirements.txt
```

## Environment Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install general dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`pyproject.toml` requires Python >= 3.10. If your interpreter is older (e.g., Python 3.9), skip the editable install and set `PYTHONPATH` before executing any module or application:

```powershell
$env:PYTHONPATH = "src"
```

```bash
export PYTHONPATH=src
```

3. For model training dependencies:

```powershell
python -m pip install -r requirements-train.txt
```

4. For web application dependencies:

```powershell
python -m pip install -r requirements-app.txt
```

## Data Preparation and EDA

Quick local test with a small sample:

```powershell
python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
```

Prepare all recommended subsets:

```powershell
python -m similarity_search.data.prepare_allnli --subsets pair-class pair-score triplet
```

Prepare the standardized clean 70/15/15 split used for benchmark evaluation:

```powershell
python -m similarity_search.data.prepare_allnli_70_15_15_clean
```

Run exploratory data analysis (EDA):

```powershell
python -m similarity_search.data.eda_allnli --subset pair-class
```

Outputs, figures, and summary reports are generated in `outputs/tables/`, `outputs/figures/`, and `outputs/reports/`.

## Models and Training Pipeline

The repository provides modular training and evaluation scripts across five core modeling approaches:

1. **TF-IDF Baseline**: Fits a lexical vectorizer and selects an entailment similarity threshold on the development split:
   ```powershell
   python -m similarity_search.models.train_tfidf
   python -m similarity_search.models.tfidf_baseline
   ```

2. **Pretrained MiniLM Baseline**: Evaluates `sentence-transformers/all-MiniLM-L6-v2` without additional fine-tuning:
   ```powershell
   python -m similarity_search.models.minilm_baseline --device cpu
   ```

3. **Fine-Tuned MiniLM Bi-Encoder**: Fine-tunes the MiniLM architecture on AllNLI positive pairs and triplets:
   ```powershell
   python -m similarity_search.models.train_minilm
   ```
   For Kaggle or GPU cluster training using `MultipleNegativesRankingLoss`:
   ```bash
   python -m similarity_search.models.train_biencoder --output-dir models/allnli-minilm-biencoder --num-train-epochs 1 --batch-size 64
   ```

4. **Cross-Encoder NLI Reranker**: Fine-tunes `distilbert-base-uncased` for pairwise classification across the three NLI classes:
   ```powershell
   python -m similarity_search.models.train_cross_encoder
   ```

5. **Custom SFT-BE Checkpoint**: Fine-tunes and evaluates the custom supervised fine-tuned bi-encoder (expected at `models/sftbe_checkpoint/stage0_final.pt`):
   ```powershell
   python -m similarity_search.models.train_sftbe --checkpoint-path models/sftbe_checkpoint/stage0_final.pt
   python -m similarity_search.sftbe.prepare_data
   python -m similarity_search.sftbe.train
   python -m similarity_search.models.sftbe_evaluation
   ```

### Evaluation and Reporting Scripts

- Compare lexical and dense baselines:
  ```powershell
  python scripts/compare_baselines.py
  ```
- Run preprocessing ablation study for TF-IDF:
  ```powershell
  python -m similarity_search.models.tfidf_preprocessing_ablation
  ```
- Evaluate SFT-BE and Cross-Encoder models:
  ```powershell
  python scripts/evaluate_sftbe_cross_encoder.py
  ```
- Generate final model summary comparison tables:
  ```powershell
  python scripts/build_final_model_summary.py
  ```
- Build training loss figures and process summary tables:
  ```powershell
  python scripts/build_training_process_artifacts.py
  ```

## Kaggle Notebooks

For reproducible experimentation and benchmark evaluations on GPU accelerators, five standalone notebooks are located in `notebooks/`:

- `01_train_tfidf_clean_full_eval5k_kaggle.ipynb`
- `02_train_minilm_clean_full_earlystop_eval5k_kaggle.ipynb`
- `03_train_cross_encoder_clean_full_earlystop_eval5k_kaggle.ipynb`
- `04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb`
- `05_evaluate_hybrid_models_kaggle.ipynb`

Each notebook automates environment setup, dataset loading or generation, model training with early stopping, evaluation on a fixed 5k test sample, and latency/throughput runtime benchmarking. See `notebooks/README.md` for detailed instructions on Kaggle execution and model checkpoint uploading.

## Web Application Demo (Streamlit)

An interactive web demo is provided to showcase real-time semantic retrieval and document comparison.

### Features
- **Semantic Search**: Upload a document (`.txt`, `.pdf`, `.docx`) and enter a natural language query to retrieve semantic matches with similarity scores and page numbers.
- **Document Comparison**: Upload two documents to compute sentence-level semantic matching and estimate overall document similarity percentage.
- **Model Selection**: Switch dynamically between TF-IDF, Pretrained MiniLM, Fine-tuned MiniLM, SFT-BE, Cross-Encoder NLI, and Hybrid Reranking pipelines.

### Running Locally

```powershell
python -m pip install -r requirements-app.txt
streamlit run src/similarity_search/app/streamlit_app.py
```

Open your browser at:
```text
http://localhost:8501
```

### Running with Docker

```powershell
docker compose build web
docker compose up web
```

### Required Local Model Folders
For full functionality in the web demo, ensure the following model checkpoints exist locally:
- `models/tfidf_baseline/vectorizer.joblib`
- `models/allnli-minilm-biencoder/final/`
- `models/allnli-cross-encoder-nli/final/`
- `models/sftbe_checkpoint/stage0_final.pt`

## Docker and Team Workflow

### Development Container
Build and run data preparation or EDA tasks within an isolated Docker environment:

```powershell
docker compose build dev
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
docker compose run --rm dev python -m similarity_search.data.eda_allnli --subset pair-class
```

### Team Collaboration and Dataset Sharing
- **Source Code**: Use GitHub for managing code, configuration files, scripts, notebooks, and issues.
- **Trained Models**: Publish large checkpoints and final models to Hugging Face Model Hub.
- **Processed Datasets**: Do not commit large processed dataset files or raw binaries to GitHub. Instead, push processed datasets to Hugging Face Dataset Hub:
  ```powershell
  python scripts/push_processed_dataset_to_hub.py --subset pair-class --repo-id <team-name>/allnli-pair-class-processed --private
  ```
  Team members can then load the dataset directly in code:
  ```python
  from datasets import load_dataset
  ds = load_dataset("<team-name>/allnli-pair-class-processed")
  ```
- **Large Artifacts**: Use external storage (such as Google Drive) for videos, slide presentations, and temporary raw exports.
