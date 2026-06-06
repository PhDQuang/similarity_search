# Ke hoach do an NLP: Document Similarity / Semantic Similarity Search

## 1. Ten de tai de xuat

**He thong tim kiem ngu nghia va so sanh do tuong dong tai lieu bang Sentence Embedding ket hop Natural Language Inference**

Ten ngan gon khi thuyet trinh:

**Semantic Document Search using SNLI-trained Sentence Embeddings**

## 2. Dinh nghia bai toan

### Loai bai toan NLP

De tai thuoc nhom:

- **Similarity**: do do tuong dong giua hai cau/van ban.
- **Retrieval**: tim cac cau trong tai lieu gan nghia nhat voi cau truy van.
- **Classification phu tro**: dung Natural Language Inference de du doan quan he `entailment`, `neutral`, `contradiction` giua hai cau.

### Chuc nang 1: tim kiem ngu nghia trong mot file

Input:

- 1 file van ban: `.txt`, `.pdf`, `.docx`.
- 1 cau/cum tu truy van do nguoi dung nhap.
- Model va nguong similarity tuy chon.

Output:

- Danh sach cau trong file co y nghia gan voi truy van.
- Diem tuong dong cosine similarity.
- Neu dung model NLI/cross-encoder: hien thi them label `entailment/neutral/contradiction` va xac suat.
- Vi tri cau trong file, so trang neu trich xuat duoc tu PDF.

Vi du:

| Thanh phan | Noi dung |
|---|---|
| Document sentence | "A soccer game with multiple males playing." |
| Query | "Some men are playing a sport." |
| Output | Cau tren duoc tra ve voi similarity cao, label du kien: entailment |
| Task | Semantic Search / Similarity Retrieval |

### Chuc nang 2: so sanh hai file

Input:

- 2 file van ban: file A va file B.
- Model va nguong similarity.

Output:

- Bang cac cap cau giong nghia giua hai file, ke ca khi thu tu cau bi dao.
- Ty le giong nhau giua hai file.
- Cac cau chi co trong file A hoac file B.
- Bao cao co the tai ve: `.csv` hoac `.json`.

Cong thuc de xuat:

- Tach file A thanh `n` cau, file B thanh `m` cau.
- Tinh ma tran similarity `S[i][j]` giua moi cau A_i va B_j.
- Dung matching 1-1 de tranh mot cau bi dem nhieu lan.
- Cap cau duoc xem la giong nghia neu `similarity >= threshold`, vi du 0.75.
- Phan tram giong nhau:

```text
similarity_percent = 2 * matched_pairs / (num_sentences_A + num_sentences_B) * 100
```

Co the bo sung diem trung binh co trong so:

```text
weighted_score = average(similarity_of_matched_pairs) * similarity_percent / 100
```

## 3. Pham vi ngon ngu

Phuong an khuyen nghi de dam bao nhat quan voi dataset:

- **Ban chinh**: xu ly tai lieu tieng Anh, vi SNLI la dataset tieng Anh.
- **Mo rong neu muon demo tieng Viet**: them ViANLI hoac XNLI tieng Viet, dung model multilingual nhu `paraphrase-multilingual-MiniLM-L12-v2` hoac `xlm-roberta-base`.

Khuyen nghi cho nhom: lam ban tieng Anh that chac chan truoc; neu con thoi gian, them tab demo tieng Viet nhu mot huong phat trien hoac diem cong.

## 4. Dataset

### Dataset chinh: SNLI

Cach tai dung theo yeu cau:

```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/snli")
ds = ds.filter(lambda x: x["label"] != -1)
```

Thong tin can trinh bay trong bao cao:

- Nguon: Hugging Face `stanfordnlp/snli` va Stanford NLP.
- Ten day du: Stanford Natural Language Inference Corpus.
- Ngon ngu: English.
- So mau: khoang 570k cap cau.
- Split:
  - Train: 550,152 mau.
  - Validation: 10,000 mau.
  - Test: 10,000 mau.
- Cot du lieu:
  - `premise`: cau goc.
  - `hypothesis`: cau gia thuyet.
  - `label`: nhan quan he ngu nghia.
- Nhan:
  - `0`: entailment, hai cau co quan he suy ra/gan nghia theo ngu canh.
  - `1`: neutral, khong du thong tin de ket luan.
  - `2`: contradiction, hai cau mau thuan.
  - `-1`: khong co gold label, can loai bo truoc khi train.

### Dataset bo sung nen dung

1. **AllNLI**: `sentence-transformers/all-nli`

- La tap ket hop SNLI va MultiNLI.
- Co cac subset tien loi cho SentenceTransformer: `pair-class`, `pair-score`, `pair`, `triplet`.
- Dung de fine-tune embedding model tot hon cho semantic similarity.

2. **STS-B hoac SICK-R** neu can metric regression

- Dung de danh gia tuong dong bang Pearson/Spearman.
- Khong bat buoc, chi can neu muon bao cao co them metric chuan cho semantic textual similarity.

3. **ViANLI/XNLI** neu muon ho tro tieng Viet

- Dung cho huong mo rong multilingual.
- Khong nen dua vao core pipeline neu thoi gian ngan, vi se tang cong preprocessing va train.

## 5. Thong ke du lieu can lam

Trong notebook `01_eda_snli.ipynb`, can co cac bang/bieu do sau:

- So luong mau theo split.
- So luong mau theo tung label sau khi loai `-1`.
- Ti le label train/validation/test.
- Histogram do dai `premise` va `hypothesis` theo so token.
- Boxplot do dai cau theo label.
- Top 20 tu xuat hien nhieu nhat sau khi clean.
- Ti le lexical overlap giua `premise` va `hypothesis`.
- Vi du minh hoa moi label: entailment, neutral, contradiction.

Bang mau:

| Split | So mau truoc filter | So mau sau filter | Entailment | Neutral | Contradiction |
|---|---:|---:|---:|---:|---:|
| Train | 550,152 | ... | ... | ... | ... |
| Validation | 10,000 | ... | ... | ... | ... |
| Test | 10,000 | ... | ... | ... | ... |

## 6. Tien xu ly

### Voi dataset SNLI

Ap dung chung:

- Loai mau co `label = -1`.
- Loai dong rong, strip whitespace.
- Normalize Unicode va khoang trang.
- Giu nguyen split goc cua Hugging Face de tranh data leakage.

Cho baseline TF-IDF:

- Lowercase.
- Remove URL, HTML artifact neu co.
- Tokenization bang regex hoac scikit-learn.
- Co the remove stopwords de so sanh voi ban khong remove stopwords.
- Co the dung n-gram `(1, 2)`.

Cho Transformer/SentenceTransformer:

- Khong can stopword removal.
- Khong xoa punctuation qua manh, vi punctuation co the anh huong nghia.
- Tokenizer cua model se xu ly tokenization.
- Lowercase tuy thuoc model: `uncased` se lowercase noi bo.

### Voi file nguoi dung upload

Pipeline:

```text
file upload
-> extract text
-> normalize whitespace
-> sentence segmentation
-> remove very short/noisy sentences
-> keep metadata: sentence_id, page, char_start, char_end
-> embedding/indexing
```

Thu vien de xuat:

- PDF: `pymupdf` (`fitz`) de lay text va page.
- DOCX: `python-docx`.
- TXT: doc truc tiep.
- Tach cau tieng Anh: `spaCy` hoac `nltk`.
- Tach cau tieng Viet neu mo rong: `underthesea` hoac `pyvi`.

## 7. Mo hinh de xuat

Nhom 5 nguoi khong can train qua nhieu model. Nen lam 3 model chinh + 1 model ket hop de vua du barem, vua co demo tot.

### Model 0: TF-IDF + Cosine Similarity baseline

Muc dich:

- Baseline bat buoc, de chung minh semantic embedding tot hon keyword matching.
- Chay nhanh tren CPU.

Input/output:

- Input: cau truy van va cac cau trong document.
- Vector hoa bang TF-IDF.
- Output: cosine similarity.

Hyperparameters:

- `ngram_range=(1, 2)`.
- `max_features=50_000`.
- `min_df=2`.
- `max_df=0.95`.

Uu diem:

- Don gian, de giai thich.
- Tot voi cau co nhieu tu trung nhau.

Han che:

- Kem voi paraphrase, dao thu tu, dung tu dong nghia.

### Model 1: Pretrained SentenceTransformer

Model de xuat:

- `sentence-transformers/all-MiniLM-L6-v2`.

Muc dich:

- Lam doi sanh pretrained, chua fine-tune rieng tren SNLI cua nhom.
- Cho web chay nhanh, phu hop demo.

Input/output:

- Input: cau.
- Output: dense vector 384 chieu.
- Similarity: cosine similarity.

Ly do chon:

- Nhe, nhanh, phu hop Streamlit/Gradio va CPU.
- Duoc thiet ke cho sentence similarity/semantic search.

### Model 2: Fine-tuned SentenceTransformer tren SNLI/AllNLI

Day la **mo hinh de xuat chinh**.

Phuong an train:

- Base model: `sentence-transformers/all-MiniLM-L6-v2` hoac `distilbert-base-uncased`.
- Positive pairs: cac cap SNLI co label `entailment`.
- Hard negatives: cac cap co label `contradiction`.
- Loss:
  - Ban don gian: `MultipleNegativesRankingLoss` voi entailment pairs.
  - Ban cai tien: dung triplet `(anchor, positive, negative)` voi contradiction lam hard negative.

Muc tieu:

- Dua cac cau cung y nghia lai gan nhau trong vector space.
- Day cac cau mau thuan/khac y nghia ra xa.

Hyperparameters de xuat tren Google Colab T4:

| Tham so | Gia tri de xuat |
|---|---|
| Epoch | 1-3 |
| Batch size | 32 hoac 64 neu du VRAM |
| Max sequence length | 128 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Optimizer | AdamW |
| Evaluation steps | 500-1000 |
| Early stopping | dua tren validation MRR/F1 |

### Model 3: Cross-Encoder NLI reranker

Muc dich:

- Tang do chinh xac cho top-k ket qua cua semantic search.
- Phu hop voi chuc nang so sanh hai file, nhung cham neu so sanh moi cap cau.

Model de xuat:

- Nhe: `distilbert-base-uncased` fine-tune 3-class tren SNLI.
- Tot hon nhung nang hon: `microsoft/deberta-v3-small` hoac cross-encoder NLI da fine-tune san.

Input/output:

- Input: cap cau `(sentence_a, sentence_b)`.
- Output: 3 xac suat `entailment`, `neutral`, `contradiction`.

Cach dung trong web:

- Khong dung de index toan bo document.
- Dung sau khi bi-encoder lay top 20 ung vien.
- Score ket hop:

```text
final_score = 0.7 * cosine_score + 0.3 * entailment_probability
```

### Model 4: Hybrid proposed system

Day la mo hinh he thong nen trinh bay nhu cai tien:

```text
TF-IDF baseline
vs pretrained SentenceTransformer
vs fine-tuned SentenceTransformer
vs hybrid: fine-tuned SentenceTransformer + Cross-Encoder reranking
```

Hybrid co the dat diem tot vi co:

- Retrieval nhanh bang embedding.
- Kiem tra ngu nghia ky hon bang NLI.
- Giai thich duoc ket qua bang label entailment/neutral/contradiction.

## 8. Pipeline train model

### Buoc 1: tai va lam sach SNLI

```python
from datasets import load_dataset

raw = load_dataset("stanfordnlp/snli")
snli = raw.filter(lambda x: x["label"] != -1)
```

Khong tron split train/validation/test.

### Buoc 2: tao dataset cho tung model

Cho TF-IDF:

```text
premise + hypothesis -> clean text -> TF-IDF -> cosine / pair features
```

Cho SentenceTransformer:

```text
label entailment:
premise -> anchor
hypothesis -> positive
```

Cho triplet/hard negative:

```text
cung premise:
entailment hypothesis -> positive
contradiction hypothesis -> negative
```

Cho Cross-Encoder:

```text
[premise, hypothesis] -> label 0/1/2
```

### Buoc 3: train tren Colab/Kaggle

Khuyen nghi:

- **Google Colab T4**: de dung nhat cho nhom, du cho MiniLM/DistilBERT.
- **Kaggle Notebook GPU T4/P100**: on dinh hon neu Colab het quota.
- Luu checkpoint vao Google Drive va push len Hugging Face Hub.

Thu vien:

```bash
pip install datasets transformers sentence-transformers evaluate scikit-learn faiss-cpu pymupdf python-docx streamlit
```

### Buoc 4: log ket qua

Can luu:

- Config train.
- Loss theo step/epoch.
- Validation metric.
- Test metric.
- Confusion matrix cho NLI.
- Bang so sanh model.
- Vi du dung/sai.

Nen dung:

- TensorBoard hoac Weights & Biases.
- Neu muon don gian: luu CSV log va ve bieu do bang matplotlib.

## 9. Danh gia thuc nghiem

### Metric bat buoc theo barem

Vi de tai co Similarity + Retrieval:

- **Cosine similarity** cho similarity.
- **Precision@k** cho retrieval.
- Them **Recall@k, MRR, MAP** neu co thoi gian.

Vi co NLI classifier:

- Accuracy.
- Macro F1.
- Confusion matrix.

### Danh gia tren SNLI test

1. NLI classification:

- Test set: SNLI test.
- Metric: Accuracy, Macro F1, Precision/Recall/F1 tung label.

2. Similarity threshold:

- Map label thanh score:
  - entailment = 1.0
  - neutral = 0.5
  - contradiction = 0.0
- Tim threshold tren validation de phan biet "same meaning" voi "not same meaning".
- Bao cao ROC-AUC hoac F1 tai threshold tot nhat.

3. Retrieval:

- Voi moi premise trong test, tao candidate pool gom:
  - 1 hypothesis entailment dung.
  - nhieu hypothesis neutral/contradiction lam negative.
- Model phai retrieve dung cau entailment vao top-k.
- Metric: Precision@1, Precision@5, Recall@5, MRR.

### Danh gia chuc nang so sanh hai file

Tao synthetic document evaluation tu SNLI:

- File A: gom 20-50 cau premise.
- File B:
  - gom cac hypothesis entailment cua mot so cau trong A.
  - dao thu tu cau.
  - chen them cau neutral/contradiction lam nhieu.
- Ground truth: cap premise-hypothesis entailment.
- Metric:
  - matching precision.
  - matching recall.
  - matching F1.
  - sai so phan tram similarity.

Cach nay giup bao cao thuyet phuc hon vi chuc nang web duoc danh gia dung kieu "dao thu tu cau nhung van giong nghia".

## 10. Thiet ke web demo

Framework khuyen nghi:

- **Streamlit**: nhanh, phu hop upload file, bang ket qua, chart.
- Hoac **Gradio** neu muon deploy len Hugging Face Spaces nhanh.

Khuyen nghi chon Streamlit cho local/demo lop; Gradio neu muon public tren Hugging Face Spaces.

### Giao dien de xuat

Tab 1: Semantic Ctrl+F

- Upload 1 file.
- Chon model: TF-IDF, pretrained MiniLM, fine-tuned MiniLM, hybrid.
- Nhap query.
- Chon `top_k` va threshold.
- Hien thi:
  - Bang cau match.
  - Similarity score.
  - Page/sentence id.
  - NLI label neu co.

Tab 2: Compare two documents

- Upload file A va file B.
- Chon model va threshold.
- Nut "Compare".
- Hien thi:
  - Phan tram giong nhau.
  - So cap cau matched.
  - Bang matched pairs: cau A, cau B, score, label.
  - Bang unmatched sentences.
  - Nut download CSV.

Tab 3: Benchmark

- Bang so sanh metric cua cac model.
- Confusion matrix.
- Bieu do loss.
- Vi du dung/sai.

### Pipeline inference tren web

Mot file:

```text
upload file
-> extract text
-> sentence split
-> encode all sentences
-> build FAISS index hoac numpy cosine matrix
-> encode query
-> search top-k
-> optional cross-encoder rerank
-> show results
```

Hai file:

```text
upload A, upload B
-> extract + sentence split
-> encode sentences A, B
-> similarity matrix
-> optional top-k candidates per sentence by FAISS
-> optional cross-encoder rerank
-> Hungarian/greedy matching
-> threshold filtering
-> similarity percent
-> show pair table
```

## 11. Noi luu model va deploy

### Luu model

Khuyen nghi:

- Luu checkpoint local trong `models/`.
- Push model tot nhat len **Hugging Face Hub**:
  - `group-name/snli-minilm-semantic-search`
  - `group-name/snli-cross-encoder-reranker`
- Web load model bang model id tu Hugging Face.

Ly do:

- De deploy de hon.
- Thanh vien nhom dung chung checkpoint.
- Bao cao co link model, tang tinh chuyen nghiep.

### Deploy demo

Phuong an 1: Local + Docker, an toan nhat khi bao cao

```text
Docker image
-> Streamlit app
-> models download tu Hugging Face hoac mount local
-> chay localhost:8501
```

Phuong an 2: Hugging Face Spaces

- Dung Gradio/Streamlit.
- Phu hop public demo.
- Chu y gioi han RAM/CPU, nen dung MiniLM va khong upload file qua lon.

Phuong an 3: Render/Railway/Fly.io

- Can kiem soat dung luong model va cold start.
- Khong can thiet neu muc tieu la bao cao mon hoc.

Khuyen nghi cuoi:

- Chuan bi **local Docker** cho demo chinh.
- Chuan bi **video demo 3-5 phut** theo yeu cau barem.
- Neu kip, them Hugging Face Spaces lam link online.

## 12. Cau truc repo de xuat

```text
similarity_search/
  README.md
  requirements.txt
  Dockerfile
  docker-compose.yml
  notebooks/
    01_eda_snli.ipynb
    02_baseline_tfidf.ipynb
    03_train_sentence_transformer.ipynb
    04_train_cross_encoder.ipynb
    05_evaluation.ipynb
  src/
    data/
      prepare_snli.py
      make_synthetic_docs.py
    models/
      train_tfidf.py
      train_sbert.py
      train_cross_encoder.py
      evaluate.py
    app/
      streamlit_app.py
    similarity/
      document_loader.py
      sentence_splitter.py
      indexing.py
      matching.py
      scoring.py
  outputs/
    figures/
    metrics/
    predictions/
  reports/
    final_report.docx
    slides.pptx
```

## 13. Phan cong 5 nguoi

### Thanh vien 1: Data + EDA

- Tai SNLI bang `load_dataset`.
- Lam preprocessing.
- Ve thong ke va bieu do.
- Tao synthetic document evaluation.
- Viet muc dataset va preprocessing trong bao cao.

### Thanh vien 2: Baseline + metrics

- Lam TF-IDF + cosine baseline.
- Lam Logistic Regression/SVM NLI baseline neu can.
- Xay dung metric Precision@k, MRR, cosine threshold, confusion matrix.
- Viet muc baseline va danh gia.

### Thanh vien 3: SentenceTransformer

- Fine-tune MiniLM/SBERT tren SNLI/AllNLI.
- Thu MultipleNegativesRankingLoss va hard negatives.
- Luu model, ve loss chart.
- Viet muc mo hinh de xuat.

### Thanh vien 4: Cross-Encoder + hybrid

- Fine-tune hoac tich hop Cross-Encoder NLI.
- Lam reranking top-k.
- Ket hop score cosine + entailment probability.
- Phan tich loi va vi du dung/sai.

### Thanh vien 5: Web + deploy + demo

- Xay Streamlit/Gradio web.
- Xu ly upload PDF/DOCX/TXT.
- Lam 2 chuc nang chinh: semantic Ctrl+F va compare two documents.
- Docker hoa, quay video demo, chuan bi kich ban bao cao.

Tat ca thanh vien:

- Cung viet report, slide, va luyen van dap.

## 14. Timeline de xuat 4 tuan

### Tuan 1: chot bai toan va data

- Doc barem, chot scope tieng Anh.
- Tai SNLI, EDA, preprocessing.
- Lam baseline TF-IDF dau tien.
- Tao synthetic document test set.

Deliverable:

- Notebook EDA.
- Bang thong ke dataset.
- Baseline result ban dau.

### Tuan 2: train model

- Fine-tune SentenceTransformer.
- Train/fine-tune Cross-Encoder NLI.
- Log loss, metric validation.
- Chon threshold similarity tren validation.

Deliverable:

- Checkpoint model.
- Bang metric validation/test.
- Confusion matrix/loss chart.

### Tuan 3: web demo va danh gia

- Xay Streamlit/Gradio.
- Tich hop upload file va semantic search.
- Tich hop compare two documents.
- Chay benchmark tren synthetic docs.

Deliverable:

- App chay duoc local.
- Bang matched pairs va similarity percent.
- Ket qua so sanh model.

### Tuan 4: bao cao, slide, video

- Hoan thien report toi thieu 40 trang.
- Lam slide 12-18 trang.
- Quay video demo.
- Chuan bi cau hoi van dap.

Deliverable:

- Report final.
- Slide final.
- Video demo.
- Docker/app link.

## 15. Khung bao cao 40 trang

1. Gioi thieu de tai: 3-4 trang.
2. Tong quan NLP, similarity search, NLI, embedding: 6-8 trang.
3. Mo ta bai toan, input/output, vi du: 3-4 trang.
4. Dataset SNLI va EDA: 6-8 trang.
5. Preprocessing: 3-4 trang.
6. Mo hinh baseline va mo hinh de xuat: 8-10 trang.
7. Thuc nghiem va ket qua: 7-9 trang.
8. Demo ung dung: 3-4 trang.
9. Ket luan va huong phat trien: 2 trang.
10. Tai lieu tham khao IEEE.

Can co:

- Pipeline diagram.
- Pseudo-code semantic search.
- Pseudo-code compare two documents.
- Loss chart.
- Confusion matrix.
- Bang so sanh model.
- Anh giao dien demo.
- Vi du loi va phan tich.

## 16. Pseudo-code can dua vao bao cao

### Semantic search trong mot file

```text
Algorithm SemanticSearch(document, query, model, top_k, threshold)
1. sentences = SplitDocumentIntoSentences(document)
2. sentence_embeddings = model.encode(sentences)
3. query_embedding = model.encode(query)
4. scores = CosineSimilarity(query_embedding, sentence_embeddings)
5. candidates = TopK(scores, top_k)
6. candidates = Filter(candidates, score >= threshold)
7. if reranker is enabled:
       candidates = CrossEncoderRerank(query, candidates)
8. return ranked candidates
```

### So sanh hai file

```text
Algorithm CompareDocuments(docA, docB, model, threshold)
1. sentencesA = SplitDocumentIntoSentences(docA)
2. sentencesB = SplitDocumentIntoSentences(docB)
3. embA = model.encode(sentencesA)
4. embB = model.encode(sentencesB)
5. S = CosineSimilarityMatrix(embA, embB)
6. pairs = OneToOneMatching(S)
7. matched = Filter(pairs, similarity >= threshold)
8. percent = 2 * Count(matched) / (len(sentencesA) + len(sentencesB)) * 100
9. return percent, matched pairs, unmatched sentences
```

## 17. Bang ket qua mong doi

Bang nay de dien so lieu sau khi train:

| Model | Train data | Similarity metric | Precision@1 | Precision@5 | MRR | NLI Acc | Macro F1 | Toc do |
|---|---|---|---:|---:|---:|---:|---:|---|
| TF-IDF cosine | SNLI text | Cosine | ... | ... | ... | - | - | Rat nhanh |
| Pretrained MiniLM | Pretrained | Cosine | ... | ... | ... | - | - | Nhanh |
| Fine-tuned MiniLM | SNLI/AllNLI | Cosine | ... | ... | ... | - | - | Nhanh |
| Hybrid rerank | SNLI/AllNLI | Cosine + entailment prob | ... | ... | ... | ... | ... | Cham hon |

## 18. Rủi ro va cach xu ly

| Rui ro | Cach xu ly |
|---|---|
| SNLI la tieng Anh, demo tieng Viet kem | Chot scope tieng Anh; neu mo rong thi dung ViANLI/XNLI + multilingual model |
| Cross-Encoder cham | Chi rerank top-k tu bi-encoder |
| PDF extract loi layout | Demo voi PDF text-based; noi ro han che voi scan PDF |
| Threshold tuy tien | Chon threshold tren validation de toi uu F1 |
| Data leakage | Giu split goc, synthetic docs tao rieng tu validation/test |
| Train lau | Dung MiniLM/DistilBERT, 1-3 epoch tren Colab/Kaggle |

## 19. Ket luan phuong an

Phuong an nen bao ve:

- Dataset chinh: SNLI.
- Bieu dien van ban: TF-IDF baseline va Sentence Embedding.
- Mo hinh de xuat: Fine-tuned SentenceTransformer tren cap entailment/hard negative.
- Mo hinh cai tien: Hybrid bi-encoder retrieval + cross-encoder NLI reranking.
- Demo: web semantic Ctrl+F va compare two documents.
- Metric: Cosine similarity, Precision@k, MRR, Accuracy/F1 cho NLI, matching F1 cho document comparison.

Voi cau truc nay, de tai dap ung du cac muc trong barem: dataset, preprocessing, vectorization, baseline, mo hinh de xuat, so sanh, metric, demo ung dung, bao cao khoa hoc va kha nang van dap.

## 20. Tai lieu tham khao nen trich dan

- Stanford NLP, SNLI Corpus: https://nlp.stanford.edu/projects/snli/
- Hugging Face dataset card, `stanfordnlp/snli`: https://huggingface.co/datasets/stanfordnlp/snli
- Hugging Face dataset card, `sentence-transformers/all-nli`: https://huggingface.co/datasets/sentence-transformers/all-nli
- SentenceTransformers model card, `all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- SentenceTransformers NLI training examples: https://sbert.net/examples/sentence_transformer/training/nli/
- SentenceTransformers losses documentation: https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- Faiss documentation: https://faiss.ai/
