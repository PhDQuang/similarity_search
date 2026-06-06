# Huong dan tung buoc de train model cho do an Similarity Search

## 1. Muc tieu tai lieu

Tai lieu nay huong dan nhom train cac model cho de tai:

**Document Similarity / Semantic Similarity Search using AllNLI**

He thong cuoi cung can co hai chuc nang:

- Nhap 1 file va 1 cau query de tim cac cau trong file co cung y nghia.
- Nhap 2 file va tinh ti le cac cau giong nghia giua hai file, ke ca khi thu tu cau bi dao.

Dataset chinh:

- Hugging Face: `sentence-transformers/all-nli`
- Link: https://huggingface.co/datasets/sentence-transformers/all-nli

Model nen co trong bao cao:

- Model 1: TF-IDF + cosine similarity baseline.
- Model 2: Pretrained SentenceTransformer.
- Model 3: Fine-tuned SentenceTransformer/Bi-Encoder tren AllNLI.
- Model 4: Cross-Encoder NLI reranker.
- Model 5: Hybrid system = Bi-Encoder retrieval + Cross-Encoder reranking.

## 2. Cong cu lam viec chung cua nhom

Nen dung:

- GitHub: quan ly code, notebook, issues, pull requests.
- Docker: thong nhat moi truong chay giua cac thanh vien.
- Hugging Face Hub: luu model sau khi train.
- Google Drive: luu video demo, slide, checkpoint tam, file Word/PDF lon.

Khong nen commit len GitHub:

- Dataset da tai ve.
- Thu muc `models/` chua checkpoint lon.
- Thu muc `data/raw/`, `data/processed/`.
- Thu muc `outputs/` qua lon.
- Token Hugging Face hoac secret.

## 3. Clone repo tu GitHub

Thay `<repo-url>` bang link GitHub cua nhom.

```powershell
git clone <repo-url>
cd similarity_search
```

Neu da clone roi:

```powershell
git switch main
git pull origin main
```

## 4. Chay bang Docker

Mo Docker Desktop truoc, doi den khi Docker engine da chay.

Build moi truong xu ly data va EDA:

```powershell
docker compose build dev
```

Kiem tra CLI:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --help
docker compose run --rm dev python -m similarity_search.data.eda_allnli --help
```

Build moi truong train:

```powershell
docker compose build train
```

Build moi truong web:

```powershell
docker compose build web
```

Chay web skeleton:

```powershell
docker compose up web
```

Mo trinh duyet:

```text
http://localhost:8501
```

## 5. Chay bang local Python neu can

Docker la moi truong chinh. Local Python/venv chi can neu thanh vien muon debug truc tiep trong IDE.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Neu train model:

```powershell
python -m pip install -r requirements-train.txt
```

Neu chay web:

```powershell
python -m pip install -r requirements-app.txt
```

## 6. Lay data AllNLI

Dataset duoc tai bang Hugging Face Datasets:

```python
from datasets import load_dataset

ds = load_dataset("sentence-transformers/all-nli", "pair-class")
```

Cac subset quan trong:

| Subset | Cot chinh | Dung de lam gi |
|---|---|---|
| `pair-class` | `premise`, `hypothesis`, `label` | EDA, NLI classification, Cross-Encoder |
| `pair-score` | `sentence1`, `sentence2`, `score` | Similarity scoring, threshold |
| `pair` | `anchor`, `positive` | Fine-tune Bi-Encoder voi positive pairs |
| `triplet` | `anchor`, `positive`, `negative` | Fine-tune Bi-Encoder voi hard negatives |

Chay sample nho de kiem tra:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
```

Chay full data de lam do an:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class pair-score pair triplet
```

Output:

```text
data/processed/allnli/
```

## 7. Chay EDA

EDA cho `pair-class`:

```powershell
docker compose run --rm dev python -m similarity_search.data.eda_allnli --subset pair-class
```

EDA cho `pair-score`:

```powershell
docker compose run --rm dev python -m similarity_search.data.eda_allnli --subset pair-score
```

EDA cho `triplet`:

```powershell
docker compose run --rm dev python -m similarity_search.data.eda_allnli --subset triplet
```

Output:

```text
outputs/tables/allnli/
outputs/figures/allnli/
outputs/reports/allnli/
```

Can dua vao bao cao:

- So luong mau theo split.
- Phan bo label/score.
- Histogram do dai cau.
- Lexical overlap.
- Top words.
- Vi du theo tung label.

## 8. Tong quan noi train model

| Model | Noi train phu hop | GPU can khong | Ghi chu |
|---|---|---|---|
| TF-IDF baseline | Local hoac Docker | Khong | Nhanh, dung CPU |
| Pretrained SentenceTransformer | Khong can train | Khong | Dung truc tiep de so sanh |
| Fine-tuned Bi-Encoder | Google Colab/Kaggle | Nen co | T4/P100 la du |
| Cross-Encoder NLI | Google Colab/Kaggle | Nen co | Cham hon Bi-Encoder |
| Hybrid | Khong train rieng | Khong bat buoc | Ket hop model da train |

Khuyen nghi:

- Data/EDA: Docker local.
- Baseline TF-IDF: Docker local.
- Fine-tune Transformer: Google Colab hoac Kaggle GPU.
- Demo: Docker local hoac Hugging Face Spaces.

## 9. Model 1: TF-IDF baseline

### Muc dich

TF-IDF la baseline bat buoc de so sanh voi embedding model. Model nay tim cau dua tren do trung lap tu vung, khong hieu ngu nghia sau.

### Data dung

- `pair-class`: phan tich label va co the tao binary task.
- File demo: cac cau da tach tu document.

### Train o dau

- Local Docker la du.
- Khong can GPU.

### Cach train

Phuong an 1: Retrieval baseline

```text
document sentences -> TF-IDF vectorizer -> cosine similarity -> top-k results
```

Phuong an 2: Pair classification baseline

```text
premise + hypothesis -> TF-IDF features -> Logistic Regression/SVM -> label
```

### Hyperparameter nen chinh

| Tham so | Gia tri thu |
|---|---|
| `ngram_range` | `(1, 1)`, `(1, 2)` |
| `max_features` | 20000, 50000, 100000 |
| `min_df` | 1, 2, 5 |
| `max_df` | 0.9, 0.95 |
| classifier | Logistic Regression, Linear SVM |
| similarity threshold | 0.6, 0.7, 0.75, 0.8 |

### Metric

- Cosine similarity.
- Precision@1, Precision@5.
- MRR.
- Neu classification: Accuracy, Macro F1.

### Luu model

Dung `joblib`:

```python
import joblib

joblib.dump(vectorizer, "models/tfidf_baseline/vectorizer.joblib")
joblib.dump(classifier, "models/tfidf_baseline/classifier.joblib")
```

Khong commit file `.joblib` lon len GitHub. Neu can chia se, upload len Google Drive hoac Hugging Face Hub.

## 10. Model 2: Pretrained SentenceTransformer

### Muc dich

Dung model pretrained de lam doi sanh voi model fine-tuned cua nhom.

### Model de xuat

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Train o dau

Khong can train. Chi load model va encode cau.

### Cach chay

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences, normalize_embeddings=True)
query_embedding = model.encode([query], normalize_embeddings=True)
scores = cosine_similarity(query_embedding, embeddings)[0]
```

### Tham so nen chinh

| Tham so | Gia tri thu |
|---|---|
| `top_k` | 5, 10, 20 |
| similarity threshold | 0.65, 0.7, 0.75, 0.8 |
| `normalize_embeddings` | True |
| batch size encode | 32, 64, 128 |

### Metric

- Precision@k.
- Recall@k.
- MRR.
- Thoi gian encode va search.

### Luu y

Model nay khong duoc train boi nhom, nen trong bao cao nen goi la pretrained comparison model, khong phai proposed model.

## 11. Model 3: Fine-tuned Bi-Encoder SentenceTransformer

### Muc dich

Day la model de xuat chinh. Model hoc de dua cac cau cung nghia lai gan nhau trong vector space va day cac cau khac nghia ra xa.

### Data dung

Co 3 phuong an:

Phuong an A: dung `pair`

- Cot: `anchor`, `positive`.
- Loss: `MultipleNegativesRankingLoss`.
- De train nhat va thuong hieu qua cho retrieval.

Phuong an B: dung `triplet`

- Cot: `anchor`, `positive`, `negative`.
- Loss: `TripletLoss`.
- Tot de dua hard negative vao train.

Phuong an C: dung `pair-score`

- Cot: `sentence1`, `sentence2`, `score`.
- Loss: `CosineSimilarityLoss` hoac `CoSENTLoss`.
- Tot cho similarity score.

Khuyen nghi cho nhom:

- Train ban dau bang Phuong an A.
- Neu con thoi gian, train them Phuong an B de so sanh.

### Train o dau

Khuyen nghi:

- Google Colab T4: de dung, phu hop do an.
- Kaggle Notebook GPU: on dinh neu Colab het quota.
- Docker local chi phu hop neu may co GPU va da cau hinh CUDA.

### Cai dat tren Colab/Kaggle

```python
!pip install -U sentence-transformers datasets accelerate
```

Clone repo:

```python
!git clone <repo-url>
%cd similarity_search
```

Tai data:

```python
from datasets import load_dataset

train_ds = load_dataset("sentence-transformers/all-nli", "pair", split="train")
dev_ds = load_dataset("sentence-transformers/all-nli", "pair", split="dev")
```

### Cau hinh train de xuat

| Tham so | Gia tri ban dau |
|---|---|
| base model | `sentence-transformers/all-MiniLM-L6-v2` |
| max sequence length | 128 |
| epochs | 1-3 |
| batch size | 32 hoac 64 |
| learning rate | 2e-5 |
| warmup ratio | 0.1 |
| optimizer | AdamW |
| mixed precision | fp16 tren GPU |
| evaluation steps | 500 hoac 1000 |

### Code mau voi pair + MultipleNegativesRankingLoss

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.max_seq_length = 128

train_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="dev")

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="models/allnli-minilm-biencoder",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

trainer.train()
model.save_pretrained("models/allnli-minilm-biencoder/final")
```

### Code mau voi triplet + TripletLoss

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import TripletLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.max_seq_length = 128

train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

loss = TripletLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="models/allnli-minilm-triplet",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

trainer.train()
model.save_pretrained("models/allnli-minilm-triplet/final")
```

### Tham so can tuning

- `num_train_epochs`: neu loss giam nhung metric chua tot, tang tu 1 len 2-3.
- `batch_size`: batch lon tot cho MultipleNegativesRankingLoss, nhung can VRAM.
- `learning_rate`: thu `1e-5`, `2e-5`, `3e-5`.
- `max_seq_length`: 128 la du cho AllNLI; neu document co cau dai, thu 256.
- `threshold`: chon tren validation, khong chon tren test.

### Danh gia

Can tao retrieval evaluation:

```text
query = anchor/premise
candidates = positive + negatives
model phai dua positive vao top-k
```

Metric:

- Precision@1.
- Precision@5.
- Recall@5.
- MRR.
- Cosine similarity trung binh cho positive vs negative.

### Luu model local

```python
model.save_pretrained("models/allnli-minilm-biencoder/final")
```

### Push model len Hugging Face Hub

Dang nhap:

```bash
huggingface-cli login
```

Push:

```python
model.push_to_hub("<hf-username-or-org>/allnli-minilm-biencoder")
```

Quy uoc ten model:

```text
<team-name>/allnli-minilm-biencoder
<team-name>/allnli-minilm-triplet
```

## 12. Model 4: Cross-Encoder NLI reranker

### Muc dich

Cross-Encoder nhan vao mot cap cau va du doan quan he:

- entailment
- neutral
- contradiction

Model nay cham hon Bi-Encoder, nhung chinh xac hon khi can xep hang lai top-k ket qua.

### Data dung

Dung subset:

```text
pair-class
```

Cot:

- `premise`
- `hypothesis`
- `label`

### Train o dau

Khuyen nghi:

- Google Colab T4.
- Kaggle GPU.

Khong nen train full Cross-Encoder tren CPU vi rat cham.

### Base model de xuat

| Muc tieu | Model |
|---|---|
| Nhanh, nhe | `distilbert-base-uncased` |
| Tot hon | `microsoft/deberta-v3-small` |
| Demo CPU de hon | `cross-encoder/nli-MiniLM2-L6-H768` neu dung pretrained |

### Cau hinh train de xuat

| Tham so | Gia tri ban dau |
|---|---|
| epochs | 1-3 |
| batch size | 16 hoac 32 |
| learning rate | 2e-5 |
| max length | 128 |
| loss | Cross entropy |
| metric | Accuracy, Macro F1 |

### Code mau voi Transformers Trainer

```python
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("sentence-transformers/all-nli", "pair-class")

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=128,
    )

encoded = dataset.map(tokenize, batched=True)
encoded = encoded.rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

args = TrainingArguments(
    output_dir="models/allnli-distilbert-cross-encoder",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["dev"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(encoded["test"])

trainer.save_model("models/allnli-distilbert-cross-encoder/final")
tokenizer.save_pretrained("models/allnli-distilbert-cross-encoder/final")
```

### Tham so can tuning

- `learning_rate`: `1e-5`, `2e-5`, `3e-5`.
- `batch_size`: tang neu GPU du VRAM.
- `max_length`: 128 hoac 256.
- `base model`: DistilBERT nhanh, DeBERTa tot hon.
- so mau train: neu Colab cham, lay sample 100k-300k truoc.

### Luu va push model

```python
trainer.push_to_hub("<hf-username-or-org>/allnli-distilbert-cross-encoder")
tokenizer.push_to_hub("<hf-username-or-org>/allnli-distilbert-cross-encoder")
```

Neu khong push trong Trainer:

```python
model.save_pretrained("models/allnli-distilbert-cross-encoder/final")
tokenizer.save_pretrained("models/allnli-distilbert-cross-encoder/final")
```

## 13. Model 5: Hybrid reranking

### Muc dich

Hybrid khong nhat thiet train rieng. No ket hop:

```text
Bi-Encoder -> lay top-k nhanh
Cross-Encoder -> rerank top-k
```

### Pipeline

```text
1. Tach document thanh cau.
2. Encode tat ca cau bang Fine-tuned Bi-Encoder.
3. Encode query.
4. Lay top 20 cau bang cosine similarity.
5. Dua 20 cap cau vao Cross-Encoder.
6. Tinh final score.
7. Sap xep va hien thi ket qua.
```

Cong thuc de xuat:

```text
final_score = alpha * cosine_score + (1 - alpha) * entailment_probability
```

### Tham so can tuning

| Tham so | Gia tri thu |
|---|---|
| `top_k_before_rerank` | 10, 20, 50 |
| `alpha` | 0.5, 0.6, 0.7, 0.8 |
| final threshold | 0.65, 0.7, 0.75 |

### Metric

- Precision@1.
- Precision@5.
- MRR.
- Latency trung binh cho moi query.

## 14. Chon threshold cho web

Khong nen chon threshold cam tinh. Hay chon tren validation.

Cach lam:

```text
1. Lay tap validation.
2. Tinh similarity cho cac cap cau.
3. Thu threshold tu 0.50 den 0.90.
4. Voi moi threshold, tinh Precision, Recall, F1.
5. Chon threshold co F1 tot nhat.
6. Chi bao cao ket qua cuoi tren test set.
```

Threshold goi y ban dau:

| Model | Threshold ban dau |
|---|---|
| TF-IDF | 0.45-0.65 |
| Pretrained MiniLM | 0.65-0.75 |
| Fine-tuned MiniLM | 0.70-0.80 |
| Hybrid | 0.70-0.85 |

## 15. Luu model va quan ly version

Thu muc local:

```text
models/
  tfidf_baseline/
  allnli-minilm-biencoder/
  allnli-minilm-triplet/
  allnli-distilbert-cross-encoder/
```

Ten tren Hugging Face Hub:

```text
<team-name>/allnli-minilm-biencoder-v1
<team-name>/allnli-minilm-triplet-v1
<team-name>/allnli-distilbert-cross-encoder-v1
```

Moi lan train can ghi lai:

- Dataset subset.
- So mau train/dev/test.
- Base model.
- Epoch.
- Batch size.
- Learning rate.
- Max sequence length.
- Loss.
- Metric validation.
- Metric test.
- Link model Hugging Face.

Nen luu file ket qua:

```text
outputs/tables/model_results.csv
outputs/reports/training_log_<model_name>.md
```

## 16. Workflow GitHub khi train model

Moi thanh vien tao branch rieng:

```powershell
git switch main
git pull origin main
git switch -c feature/train-sbert
```

Sau khi them notebook/script/log nho:

```powershell
git status
git add notebooks/ src/ docs/ outputs/tables/ outputs/figures/
git commit -m "Train AllNLI bi-encoder experiment"
git push -u origin feature/train-sbert
```

Len GitHub tao Pull Request.

Quy tac:

- Khong push checkpoint lon len GitHub.
- Chi push code, config, metric CSV, hinh anh can dua vao bao cao.
- Model final push len Hugging Face Hub.
- Link model va ket qua ghi vao README hoac report.

## 17. Thu tu train khuyen nghi cho nhom

1. Chay EDA AllNLI.
2. Train TF-IDF baseline.
3. Chay pretrained MiniLM khong fine-tune.
4. Fine-tune Bi-Encoder voi `pair`.
5. Neu con thoi gian, fine-tune Bi-Encoder voi `triplet`.
6. Train Cross-Encoder NLI.
7. Lam Hybrid reranking.
8. So sanh bang metric.
9. Tich hop model tot nhat vao web.

## 18. Bang tong hop can co trong bao cao

| Model | Dataset | Train place | Metric chinh | Diem manh | Diem yeu |
|---|---|---|---|---|---|
| TF-IDF | AllNLI text | Docker/local | P@k, cosine | Nhanh, de giai thich | Kem paraphrase |
| Pretrained MiniLM | pretrained | Khong train | P@k, MRR | Nhanh, semantic tot | Chua fine-tune cho data nhom |
| Fine-tuned MiniLM pair | AllNLI pair | Colab/Kaggle | P@k, MRR | Tot cho retrieval | Can GPU |
| Fine-tuned MiniLM triplet | AllNLI triplet | Colab/Kaggle | P@k, MRR | Co hard negative | Can tuning |
| Cross-Encoder | AllNLI pair-class | Colab/Kaggle | Accuracy, F1 | Chinh xac voi cap cau | Cham |
| Hybrid | Model da train | Local/Docker | P@k, MRR, latency | Can bang toc do va do chinh xac | Pipeline phuc tap hon |

## 19. Checklist truoc khi nop

- Da co EDA va bieu do dataset.
- Da co baseline TF-IDF.
- Da co it nhat mot model fine-tuned.
- Da co bang so sanh model.
- Da co loss chart/training log.
- Da co metric test set.
- Da co model luu local hoac tren Hugging Face Hub.
- Da co web demo chay duoc.
- Da co video demo.
- Da co link GitHub repo.
- Da co link model Hugging Face neu push public/private.

## 20. Tai lieu tham khao

- AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
- SentenceTransformers training overview: https://sbert.net/docs/sentence_transformer/training_overview.html
- Cross-Encoder training overview: https://sbert.net/docs/cross_encoder/training_overview.html
- SentenceTransformers model API: https://www.sbert.net/docs/package_reference/base/model.html
- Hugging Face model sharing: https://huggingface.co/docs/hub/models-uploading
