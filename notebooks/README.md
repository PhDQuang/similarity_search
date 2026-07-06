# Kaggle Notebooks for Clean Full AllNLI Training

Each notebook can be executed independently on Kaggle:

1. `01_train_tfidf_clean_full_eval5k_kaggle.ipynb`: Lexical TF-IDF baseline training and evaluation on the clean split without early stopping.
2. `02_train_minilm_clean_full_earlystop_eval5k_kaggle.ipynb`: Fine-tuning pretrained MiniLM (`all-MiniLM-L6-v2`) with early stopping and 5k test evaluation.
3. `03_train_cross_encoder_clean_full_earlystop_eval5k_kaggle.ipynb`: Fine-tuning Cross-Encoder NLI (`distilbert-base-uncased`) with early stopping and pairwise classification scoring.
4. `04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb`: Further fine-tuning the custom SFT-BE Stage 0 checkpoint with early stopping.
5. `05_evaluate_hybrid_models_kaggle.ipynb`: Comprehensive evaluation of hybrid retrieval and reranking pipelines (TF-IDF + Cross-Encoder, MiniLM + Cross-Encoder, SFT-BE + Cross-Encoder) and runtime latency/throughput benchmarking.

## Workflow Overview

When running on Kaggle, each notebook automatically performs the following steps:

- Clones the repository into `/kaggle/working/similarity_search` if not already present.
- Installs dependencies from `requirements-kaggle.txt` and installs the package in editable mode.
- Searches for the uploaded clean dataset in `/kaggle/input/allnli-70-15-15-clean/`. If not found, it automatically generates the clean dataset using `similarity_search.data.prepare_allnli_70_15_15_clean`.
- Trains or evaluates models on the standardized 70/15/15 clean split.
- Applies early stopping for neural models (MiniLM, Cross-Encoder, SFT-BE) based on validation performance.
- Evaluates pair classification metrics on the fixed 5k test sample and measures scoring runtime.
- Packages trained model checkpoints and evaluation outputs into a downloadable archive in `/kaggle/working`.

## Checkpoint Notes

For `04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb` and `05_evaluate_hybrid_models_kaggle.ipynb`, upload the project model checkpoints (such as `stage0_final.pt`, `vectorizer.joblib`, or fine-tuned models) as a Kaggle Dataset into `/kaggle/input/`. The notebooks will automatically scan `/kaggle/input/` to locate these artifacts and add the local codebase to the Python path.
