# Notebooks

Recommended notebooks:

```text
01_eda_allnli.ipynb
02_baseline_tfidf.ipynb
03_train_sentence_transformer.ipynb
03_train_biencoder_kaggle.ipynb
03_train_biencoder_allnli_pair_500k_kaggle.ipynb
04_train_cross_encoder.ipynb
05_document_similarity_evaluation.ipynb
```

Keep notebooks readable:

- Put reusable logic in `src/`.
- Use notebooks for experiments, figures, and explanation.
- Save final metrics and figures to `outputs/`.

`03_train_biencoder_kaggle.ipynb` is a self-contained Kaggle notebook. It
loads `phdquang/allnli-pair-class-processed` directly from Hugging Face,
trains on entailment pairs, evaluates all three models, and exports ZIP files.

Use `03_train_biencoder_allnli_pair_500k_kaggle.ipynb` for the main fine-tuning
experiment. It trains on `sentence-transformers/all-nli` subset `pair` with up
to 500,000 pairs, then evaluates on `phdquang/allnli-pair-class-processed`.
