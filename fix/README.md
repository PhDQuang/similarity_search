# Fixed Comparable AllNLI Pipeline

Folder `fix/` la mot pipeline song song de sua van de cac model dang train/evaluate tren
nhieu dataset khac nhau. Khong thay doi source, model, data, output hien tai cua project goc.

## Muc tieu

- Dung chung dataset: `sentence-transformers/all-nli`, config `pair-class`.
- Gop day du cac split goc `train/dev/test`, khong lay sample 5000 dong.
- Preprocess mot lan: normalize Unicode, lowercase, xu ly URL/HTML/control chars, collapse whitespace.
- Chia lai duy nhat mot lan theo ti le:

```text
train / val / test = 70 / 15 / 15
```

- Tat ca model sau day train/evaluate tren cung split moi:
  - TF-IDF baseline.
  - MiniLM fine-tuned tu `sentence-transformers/all-MiniLM-L6-v2`.
  - Cross-Encoder fine-tuned tu `distilbert-base-uncased`.
  - SFT-BE fine-tune tiep tu checkpoint hien co `models/sftbe_checkpoint/stage0_final.pt`.

## Cau truc

```text
fix/
  configs/
    allnli_70_15_15.json
  docs/
    FIX_WORKFLOW.md
  notebooks/
    00_prepare_allnli_70_15_15_kaggle.ipynb
    01_train_tfidf_fixed_allnli_kaggle.ipynb
    02_train_minilm_fixed_allnli_kaggle.ipynb
    03_train_cross_encoder_fixed_allnli_kaggle.ipynb
    04_train_sftbe_fixed_allnli_kaggle.ipynb
    05_build_barem_tables_fixed_allnli_kaggle.ipynb
  src/
    similarity_search_fix/
      data/
      models/
```

Sau khi chay, artifact se nam trong:

```text
fix/data/processed/allnli_70_15_15/pair-class/
fix/models/
fix/outputs/
```

## Thu tu chay tren Kaggle

1. Chay `00_prepare_allnli_70_15_15_kaggle.ipynb`.
2. Chay `01_train_tfidf_fixed_allnli_kaggle.ipynb`.
3. Chay `02_train_minilm_fixed_allnli_kaggle.ipynb`.
4. Chay `03_train_cross_encoder_fixed_allnli_kaggle.ipynb`.
5. Upload checkpoint SFT-BE hien tai vao Kaggle Dataset, roi chay `04_train_sftbe_fixed_allnli_kaggle.ipynb`.
6. Chay `05_build_barem_tables_fixed_allnli_kaggle.ipynb`.

## Ket qua cho bao cao

Script prepare sinh:

- Row counts, label distribution, numeric descriptive stats.
- Examples by label.
- Token length histogram.
- Lexical overlap by label.
- Top words.

Moi trainer sinh:

- Model/checkpoint.
- `metrics.json`.
- `training_metadata.json` neu co train deep model.
- `val_predictions.csv`, `test_predictions.csv`.
- Confusion matrix / classification report khi phu hop.

Script tong hop sinh:

```text
fix/outputs/tables/final_model_summary.csv
fix/outputs/tables/training_process_summary.csv
fix/outputs/reports/barem_artifact_manifest.json
```

