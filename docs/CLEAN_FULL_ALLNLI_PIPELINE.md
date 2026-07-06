# Clean Full AllNLI Pipeline

Pipeline clean-full nay la workflow chinh de train/evaluate cac model tren cung
mot dataset va cung mot split.

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
- SFT-BE fine-tune tiep tu checkpoint Stage 0 upload tren Kaggle hoac dat tai
  `models/sftbe_checkpoint/stage0_final.pt`.

## Cau truc

```text
configs/allnli_70_15_15.json
notebooks/
  01_train_tfidf_clean_full_eval5k_kaggle.ipynb
  02_train_minilm_clean_full_earlystop_eval5k_kaggle.ipynb
  03_train_cross_encoder_clean_full_earlystop_eval5k_kaggle.ipynb
  04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb
src/similarity_search/
  data/
    prepare_allnli_70_15_15_clean.py
  models/
    train_tfidf.py
    train_minilm.py
    train_cross_encoder.py
    train_sftbe.py
```

Sau khi chay, artifact se nam trong:

```text
data/processed/allnli_70_15_15_clean/pair-class/
models/
outputs/
```

## Thu tu chay tren Kaggle

Moi notebook co the chay doc lap. Chay theo nhu cau:

1. `01_train_tfidf_clean_full_eval5k_kaggle.ipynb`
2. `02_train_minilm_clean_full_earlystop_eval5k_kaggle.ipynb`
3. `03_train_cross_encoder_clean_full_earlystop_eval5k_kaggle.ipynb`
4. `04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb`

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
outputs/tables/final_model_summary.csv
outputs/tables/training_process_summary.csv
outputs/reports/barem_artifact_manifest.json
```


