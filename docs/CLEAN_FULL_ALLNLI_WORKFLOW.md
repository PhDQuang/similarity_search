# Workflow sua benchmark model

## Ly do can pipeline clean-full

Project hien tai co nhieu artifact tot, nhung cac model khong hoan toan train/evaluate tren
cung mot dataset/split:

- TF-IDF va pretrained MiniLM dang dung local processed sample 5000 dong moi split.
- Fine-tuned MiniLM co run rieng tren `pair-score`.
- Cross-Encoder co run rieng tren `pair-class` va benchmark dataset khac.
- SFT-BE xuat phat tu distillation checkpoint.

De so sanh khoa hoc theo barem, pipeline nay tao mot benchmark moi:

```text
AllNLI pair-class full -> preprocess chung -> shuffle -> 70/15/15 -> train/evaluate tat ca model
```

## Dataset va preprocessing

Script:

```powershell
python -m similarity_search.data.prepare_allnli_70_15_15
```

Preprocessing:

- Unicode NFKC.
- Replace non-breaking space.
- Remove HTML tags.
- Replace URL bang `<url>`.
- Remove control characters.
- Collapse whitespace.
- Lowercase.
- Tao feature thong ke: char length, token length, lexical overlap.

Label:

```text
entailment -> 0 -> semantic positive -> score 1.0
neutral -> 1 -> score 0.5
contradiction -> 2 -> score 0.0
```

## Protocol danh gia chung

Tat ca model dung:

- `val` de chon threshold entailment-as-similarity.
- `test` de bao cao metric cuoi.
- Pair binary metrics: accuracy, precision, recall, F1, ROC-AUC, average precision.
- Retrieval metrics: Precision@1, Precision@5, Recall@5, MRR, mean rank.
- Cross-Encoder them NLI classification accuracy, macro F1, confusion matrix.

Retrieval setup:

```text
query = hypothesis cua mot dong entailment
positive candidate = premise cung dong
distractors = premise tu neutral/contradiction trong cung split
```

## Lenh chay nhanh local hoac Kaggle terminal

```powershell
python -m pip install -r requirements-kaggle.txt
python -m pip install -e .

python -m similarity_search.data.prepare_allnli_70_15_15
python -m similarity_search.models.train_tfidf
python -m similarity_search.models.tfidf_preprocessing_ablation
python -m similarity_search.models.train_minilm
python -m similarity_search.models.train_cross_encoder
python -m similarity_search.models.train_sftbe --checkpoint-path models/sftbe_checkpoint/stage0_final.pt
python -m similarity_search.models.build_barem_tables
```

## Ghi chu Kaggle

- MiniLM va Cross-Encoder can bat GPU.
- SFT-BE can upload checkpoint hien tai vao Kaggle Dataset neu notebook khong clone/copy duoc `models/sftbe_checkpoint/stage0_final.pt`.
- Neu het thoi gian GPU, giam `--max-retrieval-queries` tam thoi de test pipeline. Lan final nen de `0` neu du thoi gian de dung all entailment queries.
- Cac output mac dinh ghi vao `outputs/` va model ghi vao `models/`. Hai thu muc
  nay da duoc ignore de khong push nham artifact lon.


