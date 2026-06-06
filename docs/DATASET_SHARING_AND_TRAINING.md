# Chia se dataset da xu ly cho nhom va dung de train model

## 1. Ket luan ngan gon

Khong nen push dataset da xu ly len GitHub, tru khi chi la file sample rat nho.

Workflow khuyen nghi cho nhom:

```text
GitHub              -> code, notebook, config, README, metric nho
Hugging Face Hub    -> processed dataset va trained models
Google Drive        -> video demo, slide, report, checkpoint tam
DVC                 -> tuy chon nang cao neu nhom muon version data nghiem tuc
```

Voi do an nay, cach gon va dung ML workflow nhat la:

```text
1. Mot nguoi xu ly data local/Docker.
2. Push processed dataset len Hugging Face Dataset Hub.
3. Cac thanh vien khac load dataset bang datasets.load_dataset().
4. Moi nguoi train model tu cung mot dataset repo.
5. Model final push len Hugging Face Model Hub.
```

## 2. Vi sao khong dua processed dataset len GitHub?

GitHub phu hop cho source code, khong phu hop cho dataset lon.

Ly do:

- Repo se nang va clone rat cham.
- Git quan ly binary/parquet/checkpoint khong tot.
- De vuot gioi han dung luong.
- Merge conflict voi data file gan nhu khong co y nghia.
- `.gitignore` trong repo hien tai da chan `data/raw/`, `data/processed/`, `outputs/`, `models/`.

GitHub chi nen chua:

- Code xu ly data.
- Script train.
- Config.
- Notebook.
- Bang metric nho.
- Hinh anh EDA can dua vao bao cao.

## 3. Cach chia se tot nhat: Hugging Face Dataset Hub

Hugging Face Dataset Hub phu hop vi nhom co the load dataset bang cung API voi AllNLI goc:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-pair-class-processed")
```

Neu repo private, moi thanh vien can:

- Co tai khoan Hugging Face.
- Duoc add vao organization/repo.
- Dang nhap bang `huggingface-cli login`.

## 4. Chuan bi truoc khi push

Kiem tra processed data dang co:

```powershell
dir data\processed\allnli\pair-class
```

Thu muc dung dang:

```text
data/processed/allnli/pair-class/
  train.parquet
  dev.parquet
  test.parquet
```

Neu chua co, chay:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class
```

Hoac chay sample nho:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
```

## 5. Dang nhap Hugging Face

Cai dependency local neu chua co:

```powershell
python -m pip install -r requirements.txt
```

Dang nhap:

```powershell
huggingface-cli login
```

Neu dung Docker, co 2 cach:

```powershell
huggingface-cli login
docker compose run --rm dev python scripts/push_processed_dataset_to_hub.py --help
```

Hoac dang nhap trong container:

```powershell
docker compose run --rm dev huggingface-cli login
```

## 6. Push processed dataset len Hugging Face

Khuyen nghi moi subset la mot dataset repo rieng de load va train de hon.

Push `pair-class`:

```powershell
python scripts/push_processed_dataset_to_hub.py --subset pair-class --repo-id <team-name>/allnli-pair-class-processed --private
```

Neu muon split `dev` thanh ten pho bien hon la `validation`:

```powershell
python scripts/push_processed_dataset_to_hub.py --subset pair-class --repo-id <team-name>/allnli-pair-class-processed --private --rename-dev-to-validation
```

Neu chay bang Docker:

```powershell
docker compose run --rm dev python scripts/push_processed_dataset_to_hub.py --subset pair-class --repo-id <team-name>/allnli-pair-class-processed --private
```

Push cac subset khac khi da xu ly xong:

```powershell
python scripts/push_processed_dataset_to_hub.py --subset pair --repo-id <team-name>/allnli-pair-processed --private
python scripts/push_processed_dataset_to_hub.py --subset triplet --repo-id <team-name>/allnli-triplet-processed --private
python scripts/push_processed_dataset_to_hub.py --subset pair-score --repo-id <team-name>/allnli-pair-score-processed --private
```

Sau khi push, dataset se co link:

```text
https://huggingface.co/datasets/<team-name>/allnli-pair-class-processed
```

## 7. Thanh vien khac load dataset de train

Clone repo code:

```powershell
git clone <github-repo-url>
cd similarity_search
```

Dang nhap Hugging Face neu dataset private:

```powershell
huggingface-cli login
```

Load dataset trong notebook/script:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-pair-class-processed")
train_ds = ds["train"]
dev_ds = ds["dev"]       # hoac ds["validation"] neu da rename
test_ds = ds["test"]

print(train_ds)
print(train_ds.column_names)
```

Neu train Cross-Encoder:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-pair-class-processed")

train_ds = ds["train"]
eval_ds = ds["dev"]
test_ds = ds["test"]
```

Neu train Bi-Encoder voi pair:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-pair-processed")
train_ds = ds["train"]
eval_ds = ds["dev"]
```

Neu train Bi-Encoder voi triplet:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-triplet-processed")
train_ds = ds["train"]
eval_ds = ds["dev"]
```

## 8. Load dataset truc tiep tu file local neu khong push Hub

Neu nhom dung chung Google Drive/OneDrive, moi nguoi copy thu muc:

```text
data/processed/allnli/
```

Sau do load local:

```python
from datasets import load_dataset

data_files = {
    "train": "data/processed/allnli/pair-class/train.parquet",
    "dev": "data/processed/allnli/pair-class/dev.parquet",
    "test": "data/processed/allnli/pair-class/test.parquet",
}

ds = load_dataset("parquet", data_files=data_files)
```

Cach nay nhanh nhung de lech version dataset giua cac thanh vien. Chi nen dung tam thoi.

## 9. Cach quan ly version dataset

Moi lan xu ly data moi, dat ten repo/tag ro rang:

```text
allnli-pair-class-processed-v1
allnli-pair-class-processed-v2
```

Hoac giu cung repo va ghi commit message:

```text
Upload processed AllNLI pair-class v1: normalized text + token length + lexical overlap
```

Trong bao cao va notebook train, luon ghi:

- Dataset repo id.
- Dataset version/commit.
- Subset.
- Split.
- So mau train/dev/test.
- Cac cot da them.

## 10. Dung dataset trong Colab/Kaggle

Tren Colab:

```python
!pip install -U datasets sentence-transformers transformers evaluate accelerate
```

Neu dataset private:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Load:

```python
from datasets import load_dataset

ds = load_dataset("<team-name>/allnli-pair-class-processed")
```

Train model tu dataset do nhu binh thuong.

## 11. Phuong an nang cao: DVC

DVC phu hop neu nhom muon version dataset nghiem tuc nhung khong muon dung Hugging Face Dataset Hub.

Mo hinh:

```text
GitHub: luu file .dvc va code
Google Drive/S3/Azure: luu data that
Thanh vien moi: git pull + dvc pull
```

Lenh mau:

```powershell
pip install dvc[gdrive]
dvc init
dvc add data/processed/allnli
git add data/processed/allnli.dvc .gitignore
git commit -m "Track processed AllNLI with DVC"
dvc remote add -d gdrive_remote gdrive://<folder-id>
dvc push
```

Thanh vien khac:

```powershell
git pull
dvc pull
```

Voi do an nay, Hugging Face Hub don gian hon DVC.

## 12. Khuyen nghi cuoi cho nhom

Nen lam theo quy trinh nay:

```text
1. Mot nguoi phu trach data push processed dataset len Hugging Face Hub.
2. Ghi dataset repo id vao README va bao cao.
3. Moi nguoi train model bang load_dataset("<team-name>/...").
4. Moi model final push len Hugging Face Model Hub.
5. GitHub chi giu code, metric, docs, notebook.
```

Vi du repo id de chon:

```text
<team-name>/allnli-pair-class-processed
<team-name>/allnli-pair-processed
<team-name>/allnli-triplet-processed
<team-name>/allnli-pair-score-processed
```

## 13. Nguon tham khao

- Hugging Face Datasets - load from Hub: https://huggingface.co/docs/datasets/loading
- Hugging Face Datasets - Hub usage: https://huggingface.co/docs/datasets/upload_dataset
- DVC remote storage: https://dvc.org/doc/user-guide/data-management/remote-storage
