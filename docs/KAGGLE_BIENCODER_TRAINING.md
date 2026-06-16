# Fine-tune MiniLM Bi-Encoder tren Kaggle

Tai lieu nay train `sentence-transformers/all-MiniLM-L6-v2` tren AllNLI
`pair` bang `MultipleNegativesRankingLoss`.

## 1. Tao Kaggle Notebook

1. Vao Kaggle, chon **Create > New Notebook**.
2. Trong **Settings**, dat **Accelerator = GPU**.
3. Bat **Internet = On** de tai source code, dataset va base model.
4. Chay cell kiem tra:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Ket qua dau tien phai la `True`.

## 2. Tai source code

Thay URL ben duoi bang URL GitHub cua nhom:

```bash
!git clone <GITHUB_REPOSITORY_URL> /kaggle/working/similarity_search
%cd /kaggle/working/similarity_search
```

Neu repository private, dung Kaggle Secret hoac upload source code thanh Kaggle
Dataset. Khong ghi token truc tiep trong Notebook.

## 3. Cai thu vien

```bash
!python -m pip install -q -r requirements-train.txt
!python -m pip install -q -e .
```

Sau khi cai dat, neu Kaggle yeu cau restart session thi restart va chay lai cell
`%cd`.

## 4. Chay smoke test

Smoke test dung 10.000 mau va 0,1 epoch de kiem tra pipeline:

```bash
!python -m similarity_search.models.train_biencoder \
  --output-dir /kaggle/working/allnli-minilm-smoke \
  --max-train-samples 10000 \
  --max-eval-samples 1000 \
  --num-train-epochs 0.1 \
  --batch-size 64 \
  --eval-steps 50 \
  --save-steps 50 \
  --logging-steps 10
```

Khi smoke test hoan thanh va co thu muc `final/`, chay full training.

## 5. Train full AllNLI pair

```bash
!python -m similarity_search.models.train_biencoder \
  --output-dir /kaggle/working/allnli-minilm-biencoder \
  --num-train-epochs 1 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --max-seq-length 128 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 100
```

Neu bi CUDA out-of-memory, giam `--batch-size` xuong `32` va tang
`--gradient-accumulation-steps` len `2`.

Model cuoi duoc luu tai:

```text
/kaggle/working/allnli-minilm-biencoder/final/
```

Thong tin tham so va metric evaluator:

```text
/kaggle/working/allnli-minilm-biencoder/training_metadata.json
```

## 6. Danh gia cung benchmark voi TF-IDF

Tao lai sample `pair-class` 5.000 dong cho moi split:

```bash
!python -m similarity_search.data.prepare_allnli \
  --subsets pair-class \
  --max-rows-per-split 5000
```

Chay ba model tren cung sample:

```bash
!python -m similarity_search.models.tfidf_baseline

!python -m similarity_search.models.minilm_baseline \
  --device cuda \
  --output-dir outputs/minilm_baseline

!python -m similarity_search.models.minilm_baseline \
  --model-name /kaggle/working/allnli-minilm-biencoder/final \
  --device cuda \
  --trained-in-project \
  --output-dir outputs/finetuned_minilm

!python scripts/compare_baselines.py
```

Bang so sanh duoc luu tai:

```text
outputs/tables/model_comparison.csv
```

## 7. Tai model va ket qua ve may

Nen nen model, metadata va bang so sanh:

```python
import shutil

shutil.make_archive(
    "/kaggle/working/allnli-minilm-biencoder",
    "zip",
    "/kaggle/working/allnli-minilm-biencoder",
)
shutil.make_archive(
    "/kaggle/working/evaluation-results",
    "zip",
    "/kaggle/working/similarity_search/outputs",
)
```

Hai file ZIP se xuat hien trong tab **Output** sau khi chon **Save Version**:

```text
/kaggle/working/allnli-minilm-biencoder.zip
/kaggle/working/evaluation-results.zip
```

## 8. Tuy chon: day model len Hugging Face

Tao Kaggle Secret ten `HF_TOKEN`, sau do chay:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Train voi cac tham so bo sung:

```text
--push-to-hub
--hub-model-id <username-or-team>/allnli-minilm-biencoder
--hub-private-repo
```

Bo `--hub-private-repo` neu muon repository model cong khai.

## 9. Ket qua can ghi vao bao cao

- Base model va dataset subset.
- GPU Kaggle.
- So mau train va evaluation.
- Epoch, batch size, learning rate va max sequence length.
- Loss `MultipleNegativesRankingLoss`.
- Metric cua TF-IDF, pretrained MiniLM va fine-tuned MiniLM.
- Link Kaggle Notebook va Hugging Face model neu co.
