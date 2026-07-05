# Kịch Bản Slide Thuyết Trình Đồ Án NLP

Đề tài: **Hệ thống tìm kiếm ngữ nghĩa và so sánh độ tương đồng tài liệu bằng Sentence Embedding kết hợp Natural Language Inference**

Gợi ý cách dùng tài liệu này:

- Phần **[NỘI DUNG SLIDE]** là nội dung ngắn nên đưa lên slide.
- Phần **[HÌNH ẢNH / BẢNG]** là hình, bảng hoặc demo nên chèn.
- Phần **[LỜI NÓI]** là speaker notes để thuyết trình.
- Các slide đã cố tình bám theo barem: bài toán, dataset, tiền xử lý, vector hóa, mô hình, train, metric, so sánh, demo, hạn chế, hướng phát triển.

---

## SLIDE 1 - Trang bìa

**[NỘI DUNG SLIDE]**

- Môn học: Xử lý Ngôn ngữ Tự nhiên
- Đề tài: Semantic Document Similarity Search
- Hệ thống tìm kiếm ngữ nghĩa và so sánh độ tương đồng tài liệu
- Thành viên nhóm, lớp, giảng viên hướng dẫn

**[HÌNH ẢNH / BẢNG]**

- Ảnh minh họa pipeline ngắn: Document + Query -> Semantic Search -> Ranked Results.
- Có thể dùng screenshot giao diện Streamlit ở tab Semantic Search.

**[LỜI NÓI]**

"Kính chào thầy/cô và các bạn. Nhóm em xin trình bày đồ án cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên với đề tài hệ thống tìm kiếm ngữ nghĩa và so sánh độ tương đồng tài liệu. Ý tưởng chính của nhóm là xây dựng một công cụ giống như chức năng tìm kiếm trong tài liệu, nhưng thay vì chỉ tìm theo từ khóa giống Ctrl+F, hệ thống có thể tìm theo ý nghĩa. Ví dụ người dùng nhập câu 'a person is riding a bicycle', hệ thống vẫn có thể tìm được câu 'a man is cycling on the road' dù hai câu không trùng hoàn toàn từ vựng.

Ngoài chức năng tìm kiếm trong một file, nhóm còn xây dựng chức năng so sánh hai tài liệu ở mức câu. Hệ thống sẽ tách tài liệu thành các câu, tính độ tương đồng ngữ nghĩa giữa các câu, ghép các cặp câu giống nghĩa, sau đó đưa ra phần trăm tương đồng giữa hai tài liệu. Trong quá trình thực hiện, nhóm có sử dụng nhiều phương pháp từ baseline TF-IDF đến Sentence Embedding và Cross-Encoder NLI để vừa đáp ứng yêu cầu về mô hình baseline, vừa có mô hình cải tiến và phần demo chạy được."

---

## SLIDE 2 - Động cơ chọn đề tài

**[NỘI DUNG SLIDE]**

- Tìm kiếm từ khóa không hiểu được câu đồng nghĩa.
- So sánh tài liệu bằng trùng lặp chuỗi dễ bỏ sót paraphrase.
- Nhu cầu thực tế:
  - tìm thông tin trong tài liệu dài,
  - phát hiện nội dung tương tự,
  - hỗ trợ đọc, kiểm tra, đối chiếu văn bản.
- Mục tiêu: chuyển từ keyword matching sang semantic matching.

**[HÌNH ẢNH / BẢNG]**

- Ví dụ 2 câu:
  - Query: "A child is running over leaves."
  - Document: "A small child in blue shirt and blue jeans is running through a wooded path over dry leaves."
  - Kết quả mong muốn: match cao dù không trùng toàn bộ từ.

**[LỜI NÓI]**

"Lý do nhóm chọn đề tài này xuất phát từ hạn chế rất quen thuộc của tìm kiếm từ khóa. Nếu trong tài liệu có câu 'A small child is running through a wooded path over dry leaves', còn người dùng tìm 'A child is running over leaves', thì Ctrl+F truyền thống chỉ dựa vào chuỗi ký tự nên có thể không trả về kết quả tốt nếu cách diễn đạt khác nhau. Trong thực tế, văn bản thường có rất nhiều cách diễn đạt cùng một ý: đồng nghĩa, diễn giải lại, thay đổi trật tự câu hoặc dùng từ khác.

Vì vậy nhóm định nghĩa bài toán là tìm kiếm và so sánh theo ngữ nghĩa. Thay vì hỏi hai câu có giống ký tự hay không, hệ thống hỏi hai câu có gần nghĩa hay không. Đây là một bài toán phù hợp với NLP vì cần biểu diễn văn bản thành vector, cần tính similarity, và với mô hình Cross-Encoder còn cần dự đoán quan hệ ngữ nghĩa giữa hai câu theo hướng Natural Language Inference.

Từ động cơ đó, nhóm xây dựng một ứng dụng có thể nhận file văn bản, PDF hoặc DOCX, tách thành từng câu, sau đó cho phép người dùng nhập câu truy vấn và nhận lại các câu liên quan nhất kèm điểm số."

---

## SLIDE 3 - Xác định bài toán theo barem

**[NỘI DUNG SLIDE]**

- Loại bài toán:
  - Similarity
  - Retrieval
  - Classification phụ trợ với NLI
- Input:
  - File `.txt`, `.pdf`, `.docx`
  - Query hoặc hai tài liệu
  - Model, threshold, top-k
- Output:
  - Danh sách câu giống nghĩa
  - Điểm similarity / entailment probability
  - Label NLI nếu có
  - Tỉ lệ tương đồng hai tài liệu

**[HÌNH ẢNH / BẢNG]**

| Thành phần | Nội dung |
|---|---|
| Input | Document + query |
| Output | Top-k câu liên quan + score |
| Task | Semantic Search / Similarity Retrieval |
| Model | TF-IDF, MiniLM, SFT-BE, Cross-Encoder |

**[LỜI NÓI]**

"Theo yêu cầu đầu tiên trong barem, nhóm cần xác định chính xác bài toán. Đề tài của nhóm không chỉ thuộc một nhãn duy nhất, mà gồm ba thành phần liên quan với nhau.

Thứ nhất là Similarity, vì hệ thống cần đo độ giống nhau giữa hai câu hoặc hai đoạn văn bản. Với các mô hình embedding, độ giống nhau này được tính bằng cosine similarity giữa hai vector.

Thứ hai là Retrieval, vì trong chức năng tìm kiếm, hệ thống có một query và rất nhiều câu trong tài liệu. Nhiệm vụ là truy hồi top-k câu có khả năng liên quan nhất với query.

Thứ ba là Classification phụ trợ, vì nhóm dùng Cross-Encoder theo hướng Natural Language Inference. Với mỗi cặp câu, Cross-Encoder dự đoán một trong ba quan hệ: entailment, neutral hoặc contradiction. Trong ngữ cảnh semantic search, xác suất entailment được dùng như một tín hiệu mạnh cho việc hai câu có gần nghĩa hay không.

Input của hệ thống là tài liệu do người dùng upload và câu truy vấn, hoặc hai tài liệu cần so sánh. Output là danh sách câu match, điểm số, nhãn NLI nếu dùng Cross-Encoder, và với chức năng so sánh hai file là phần trăm tương đồng cùng bảng các cặp câu tương ứng."

---

## SLIDE 4 - Phạm vi và kịch bản sử dụng

**[NỘI DUNG SLIDE]**

- Phạm vi chính: văn bản tiếng Anh.
- Lý do: AllNLI là dataset tiếng Anh, phù hợp với semantic similarity và NLI.
- Chức năng 1: Semantic Search trong một file.
- Chức năng 2: Compare Documents.
- Chức năng 3: Benchmark model và xem kết quả thực nghiệm.

**[HÌNH ẢNH / BẢNG]**

- Screenshot 3 tab của app:
  - Semantic Search
  - Compare Documents
  - Benchmark

**[LỜI NÓI]**

"Nhóm xác định phạm vi chính của đồ án là văn bản tiếng Anh. Lý do là dataset AllNLI mà nhóm sử dụng được xây dựng cho tiếng Anh, có đầy đủ các cặp câu và nhãn quan hệ ngữ nghĩa. Nếu mở rộng sang tiếng Việt thì cần dataset như ViNLI, XNLI tiếng Việt hoặc dùng mô hình multilingual. Tuy nhiên để đảm bảo tính nhất quán giữa dữ liệu huấn luyện, mô hình và đánh giá, nhóm tập trung làm tốt phiên bản tiếng Anh.

Hệ thống có ba kịch bản sử dụng chính. Kịch bản thứ nhất là semantic search trong một file: người dùng upload tài liệu, nhập query, hệ thống trả về các câu gần nghĩa nhất. Kịch bản thứ hai là compare documents: người dùng upload hai tài liệu, hệ thống tìm các cặp câu tương đồng và tính phần trăm giống nhau. Kịch bản thứ ba là benchmark: trong app có phần hiển thị các bảng kết quả, loss curve, confusion matrix và so sánh mô hình.

Các kịch bản này giúp đồ án không dừng ở mức notebook huấn luyện, mà có một pipeline hoàn chỉnh từ dữ liệu, mô hình, đánh giá cho đến ứng dụng demo."

---

## SLIDE 5 - Dataset: AllNLI

**[NỘI DUNG SLIDE]**

- Nguồn: Hugging Face `sentence-transformers/all-nli`.
- Ngôn ngữ: English.
- Nguồn gốc: kết hợp SNLI và MultiNLI.
- Các subset dùng trong project:
  - `pair-class`: NLI classification.
  - `pair-score`: similarity scoring.
  - `pair`: positive pairs cho bi-encoder.
  - `triplet`: anchor-positive-negative.
- Nhãn:
  - entailment,
  - neutral,
  - contradiction.

**[HÌNH ẢNH / BẢNG]**

- Bảng mô tả subset:

| Subset | Cột chính | Mục đích |
|---|---|---|
| pair-class | premise, hypothesis, label | NLI / Cross-Encoder |
| pair-score | sentence1, sentence2, score | Similarity regression |
| pair | anchor, positive | Bi-Encoder training |
| triplet | anchor, positive, negative | Hard-negative training |

**[LỜI NÓI]**

"Dataset chính của nhóm là AllNLI trên Hugging Face, tên đầy đủ trong code là `sentence-transformers/all-nli`. Đây là dataset rất phù hợp cho bài toán semantic similarity vì nó được cộng đồng SentenceTransformers sử dụng rộng rãi để huấn luyện các mô hình sentence embedding.

AllNLI có nguồn gốc từ các bộ dữ liệu Natural Language Inference như SNLI và MultiNLI. Dữ liệu gồm các cặp câu và nhãn quan hệ giữa chúng. Nếu câu hypothesis có thể được suy ra từ premise thì nhãn là entailment. Nếu hai câu không đủ thông tin để kết luận thì là neutral. Nếu hai câu mâu thuẫn nhau thì là contradiction.

Trong project, nhóm dùng nhiều dạng subset khác nhau. `pair-class` dùng cho bài toán phân loại NLI và đánh giá entailment-as-similarity. `pair-score` chuyển nhãn thành điểm similarity, ví dụ entailment gần 1, neutral khoảng 0.5, contradiction gần 0. `pair` cung cấp cặp anchor-positive để train bi-encoder. `triplet` có thêm negative để huấn luyện mô hình phân biệt câu giống nghĩa và câu không giống nghĩa.

Việc dùng AllNLI giúp nhóm đáp ứng yêu cầu của barem về dataset thực tế, có nguồn rõ ràng, có nhãn, có split train-dev-test, và phù hợp trực tiếp với bài toán NLP đang giải quyết."

---

## SLIDE 6 - Thống kê dữ liệu và phân bố nhãn

**[NỘI DUNG SLIDE]**

- Bản processed local hiện tại:
  - train: 5,000 mẫu
  - dev: 5,000 mẫu
  - test: 5,000 mẫu
- Phân bố nhãn khá cân bằng:
  - train: entailment 33.7%, neutral 33.56%, contradiction 32.74%
  - dev: entailment 35.36%, neutral 32.66%, contradiction 31.98%
  - test: entailment 35.18%, neutral 32.54%, contradiction 32.28%
- Không trộn split để tránh data leakage.

**[HÌNH ẢNH / BẢNG]**

- `outputs/figures/allnli/pair-class/rows_by_split.png`
- `outputs/figures/allnli/pair-class/label_distribution.png`
- Bảng `outputs/tables/allnli/pair-class/label_distribution.csv`

**[LỜI NÓI]**

"Sau khi chuẩn bị dữ liệu, nhóm có một bản processed local gồm 5,000 mẫu cho mỗi split train, dev và test. Đây là bản sample để thuận tiện cho việc chạy local và demo. Trong các thí nghiệm lớn hơn như Cross-Encoder và Fine-tuned MiniLM, nhóm cũng có các kết quả huấn luyện trên số mẫu lớn hơn, ví dụ Cross-Encoder dùng 300,000 mẫu train và khoảng 19,600 mẫu dev/test, Fine-tuned MiniLM có metadata ghi nhận 500,000 mẫu train.

Điểm quan trọng trong phần dataset là phân bố nhãn khá cân bằng. Ở train, entailment chiếm 33.7%, neutral 33.56%, contradiction 32.74%. Dev và test cũng có tỉ lệ gần tương tự. Điều này giúp việc huấn luyện và đánh giá ổn định hơn, vì model không bị thiên lệch mạnh về một class.

Nhóm giữ nguyên nguyên tắc train-dev-test: train dùng để học mô hình, dev dùng để chọn threshold hoặc theo dõi quá trình huấn luyện, test chỉ dùng để báo cáo kết quả cuối. Việc này nhằm tránh data leakage, tức là tránh trường hợp thông tin từ test bị lộ vào quá trình huấn luyện hoặc chọn tham số."

---

## SLIDE 7 - Ví dụ dữ liệu và ý nghĩa nhãn

**[NỘI DUNG SLIDE]**

- Entailment:
  - Premise: "A small child ... is running through ... dry leaves."
  - Hypothesis: "A child is running over leaves."
- Neutral:
  - Premise: "Two men enjoying a beer together."
  - Hypothesis: "The two men are drunk."
- Contradiction:
  - Premise: "He was free to go."
  - Hypothesis: "He was not a free man."

**[HÌNH ẢNH / BẢNG]**

- Bảng ví dụ từ `outputs/tables/allnli/pair-class/examples_by_label.csv`.
- Có thể tô màu:
  - entailment: xanh,
  - neutral: vàng,
  - contradiction: đỏ.

**[LỜI NÓI]**

"Slide này minh họa trực tiếp ý nghĩa ba nhãn trong dataset. Với entailment, câu hypothesis là một hệ quả hoặc một cách diễn đạt cùng nghĩa với premise. Ví dụ premise nói một đứa trẻ đang chạy qua con đường có lá khô, còn hypothesis nói một đứa trẻ đang chạy trên lá. Hai câu không giống hệt nhau về từ, nhưng ý chính là tương thích, nên đây là entailment.

Với neutral, hypothesis có thể đúng nhưng không được đảm bảo bởi premise. Ví dụ premise chỉ nói hai người đàn ông đang uống bia với nhau, còn hypothesis nói họ say. Việc uống bia không đủ để kết luận chắc chắn họ say, nên quan hệ là neutral.

Với contradiction, hai câu mâu thuẫn trực tiếp. Ví dụ 'He was free to go' và 'He was not a free man' diễn đạt hai trạng thái trái ngược.

Ba nhãn này rất hữu ích cho bài toán của nhóm. Entailment được xem là tín hiệu của similarity cao, contradiction là similarity thấp, còn neutral nằm ở giữa. Đó là lý do AllNLI phù hợp để biến bài toán NLI thành bài toán semantic similarity và retrieval."

---

## SLIDE 8 - Tiền xử lý dữ liệu

**[NỘI DUNG SLIDE]**

- Dataset AllNLI:
  - normalize whitespace,
  - tạo cột clean text,
  - tính độ dài ký tự/token,
  - tính lexical overlap,
  - map label id -> label name.
- File người dùng upload:
  - đọc TXT/PDF/DOCX,
  - normalize whitespace,
  - sentence splitting,
  - bỏ câu quá ngắn,
  - giữ metadata: sentence_id, page.

**[HÌNH ẢNH / BẢNG]**

- Pipeline:

```text
Raw text -> Clean text -> Sentence split -> SentenceRecord -> Model input
```

**[LỜI NÓI]**

"Phần tiền xử lý được chia thành hai pipeline. Pipeline thứ nhất dành cho dataset AllNLI khi huấn luyện và đánh giá. Nhóm chuẩn hóa khoảng trắng, tạo các cột sạch như `premise_clean` và `hypothesis_clean`, map nhãn số thành tên nhãn, đồng thời tính thêm các đặc trưng thống kê như số ký tự, số token và lexical overlap. Các đặc trưng này phục vụ EDA và phân tích dữ liệu.

Pipeline thứ hai dành cho file người dùng upload trong ứng dụng. Với file TXT, hệ thống đọc trực tiếp nội dung. Với PDF, hệ thống dùng PyMuPDF để extract text theo trang. Với DOCX, hệ thống dùng python-docx để đọc các paragraph. Sau đó văn bản được chuẩn hóa khoảng trắng, tách câu, bỏ các đoạn quá ngắn và tạo `SentenceRecord`. Mỗi `SentenceRecord` gồm `sentence_id`, nội dung câu và số trang nếu trích xuất được từ PDF.

Một điểm nhóm vừa cải thiện là xử lý xuống dòng trong PDF. Ban đầu, nếu PDF xuống dòng giữa câu, hệ thống có thể tách nhầm thành hai câu. Nhóm đã sửa để newline trong giữa câu được xem là khoảng trắng, và chỉ tách câu sau dấu kết câu như dấu chấm, chấm hỏi, chấm than. Nhờ đó model nhận được câu đầy đủ hơn và phần highlight trong demo cũng bôi đúng cả câu hơn."

---

## SLIDE 9 - Trực quan hóa dữ liệu

**[NỘI DUNG SLIDE]**

- Các biểu đồ EDA:
  - số dòng theo split,
  - phân bố nhãn,
  - histogram độ dài token,
  - lexical overlap theo nhãn,
  - top words.
- Mục đích:
  - kiểm tra cân bằng nhãn,
  - hiểu độ dài câu,
  - xem mức trùng từ giữa các label,
  - hỗ trợ chọn preprocessing và model.

**[HÌNH ẢNH / BẢNG]**

- `outputs/figures/allnli/pair-class/token_length_histogram.png`
- `outputs/figures/allnli/pair-class/lexical_overlap_by_label.png`
- `outputs/figures/allnli/pair-class/top_words.png`

**[LỜI NÓI]**

"Theo barem, đồ án cần có thống kê và trực quan hóa dữ liệu. Nhóm đã tạo các biểu đồ EDA cho AllNLI pair-class. Biểu đồ số dòng theo split cho thấy bản local có 5,000 mẫu ở train, dev và test. Biểu đồ phân bố nhãn cho thấy ba class tương đối cân bằng.

Histogram độ dài token giúp nhóm biết các câu trong dataset thường không quá dài, do đó khi dùng Transformer, `max_length=128` là lựa chọn hợp lý cho phần lớn mẫu. Nếu max length quá ngắn thì model mất thông tin; nếu quá dài thì tốn bộ nhớ và thời gian train.

Biểu đồ lexical overlap cũng rất quan trọng. Nó cho thấy chỉ dựa vào trùng từ là chưa đủ. Có những cặp entailment có lexical overlap thấp vì chúng diễn đạt lại bằng từ khác. Ngược lại, cũng có contradiction có lexical overlap cao, ví dụ hai câu gần giống nhau nhưng khác một từ phủ định. Điều này giải thích vì sao TF-IDF baseline có giới hạn, và vì sao cần sentence embedding hoặc Cross-Encoder để hiểu nghĩa sâu hơn."

---

## SLIDE 10 - Biểu diễn văn bản

**[NỘI DUNG SLIDE]**

- TF-IDF:
  - sparse vector,
  - dựa vào tần suất từ và độ hiếm của từ,
  - cosine similarity.
- Sentence Embedding:
  - dense vector,
  - biểu diễn ý nghĩa toàn câu,
  - MiniLM/SFT-BE.
- Cross-Encoder:
  - nhận trực tiếp cặp câu,
  - dự đoán entailment/neutral/contradiction.

**[HÌNH ẢNH / BẢNG]**

| Phương pháp | Vector | Ưu điểm | Hạn chế |
|---|---|---|---|
| TF-IDF | sparse | nhanh, dễ giải thích | yếu với paraphrase |
| Bi-Encoder | dense | nhanh, semantic | có thể bỏ sót tương tác cặp câu |
| Cross-Encoder | pair input | chính xác hơn | chậm khi nhiều câu |

**[LỜI NÓI]**

"Để xử lý văn bản, bắt buộc phải chuyển text thành dạng mà máy tính có thể tính toán được. Nhóm dùng ba hướng biểu diễn.

Hướng thứ nhất là TF-IDF. Đây là baseline truyền thống, biểu diễn mỗi câu bằng sparse vector dựa trên các từ và n-gram xuất hiện trong câu. TF-IDF nhanh và dễ giải thích, nhưng hạn chế là phụ thuộc nhiều vào trùng lặp từ vựng.

Hướng thứ hai là sentence embedding. Với MiniLM hoặc SFT-BE, mỗi câu được encode thành một dense vector. Nếu hai câu gần nghĩa, vector của chúng có xu hướng gần nhau trong không gian embedding. Nhờ đó có thể dùng cosine similarity để đo độ giống nghĩa.

Hướng thứ ba là Cross-Encoder. Khác với bi-encoder, Cross-Encoder không encode hai câu riêng biệt. Nó nhận cả cặp câu cùng lúc, cho phép attention giữa token của câu A và token của câu B. Vì vậy nó thường chính xác hơn trong việc phân biệt entailment, neutral và contradiction. Đổi lại, nó chậm hơn vì phải chạy model cho từng cặp câu."

---

## SLIDE 11 - Các mô hình trong hệ thống

**[NỘI DUNG SLIDE]**

- Baseline:
  - TF-IDF + cosine similarity.
- Đối sánh:
  - Pretrained MiniLM.
- Mô hình cải tiến:
  - Fine-tuned MiniLM.
  - SFT-BE checkpoint.
  - Cross-Encoder NLI.
- Hệ thống đề xuất:
  - Bi-Encoder retrieval + Cross-Encoder reranking.

**[HÌNH ẢNH / BẢNG]**

```text
TF-IDF
Pretrained MiniLM
Fine-tuned MiniLM
SFT-BE
Cross-Encoder
Hybrid = Retriever + Reranker
```

**[LỜI NÓI]**

"Để đáp ứng yêu cầu của barem về baseline, mô hình cải tiến và mô hình đối sánh, nhóm không chỉ dùng một model. Nhóm bắt đầu với TF-IDF làm baseline. Đây là mô hình đơn giản, có thể chạy nhanh, dùng để chứng minh rằng tìm kiếm theo từ khóa có giới hạn.

Tiếp theo là Pretrained MiniLM, một SentenceTransformer đã được huấn luyện trước cho sentence similarity. Đây là mô hình đối sánh để xem nếu không fine-tune riêng trên dữ liệu của project thì kết quả như thế nào.

Sau đó nhóm dùng Fine-tuned MiniLM, được fine-tune trên AllNLI pair-score theo metadata kết quả. Mục tiêu là cải thiện embedding space để các câu entailment gần nhau hơn.

SFT-BE là một checkpoint custom bi-encoder của nhóm, được train theo hướng distillation từ teacher model. Cross-Encoder NLI là mô hình phân loại cặp câu thành entailment, neutral hoặc contradiction.

Cuối cùng, hệ thống đề xuất trong demo là hybrid: retriever như MiniLM hoặc SFT-BE tìm nhanh các ứng viên tốt nhất, sau đó Cross-Encoder rerank để tăng độ chính xác."

---

## SLIDE 12 - Pipeline semantic search

**[NỘI DUNG SLIDE]**

```text
Upload file
-> Extract text
-> Normalize + split sentence
-> Encode sentences
-> Encode query
-> Compute similarity
-> Filter threshold
-> Sort top-k
-> Highlight result
```

- Nếu hybrid:

```text
Retriever top candidates -> Cross-Encoder rerank -> final score
```

**[HÌNH ẢNH / BẢNG]**

- Sơ đồ pipeline từ app.
- Có thể lấy screenshot phần search result và câu được highlight.

**[LỜI NÓI]**

"Pipeline semantic search bắt đầu khi người dùng upload file. Hệ thống đọc nội dung theo định dạng: TXT, PDF hoặc DOCX. Sau đó văn bản được chuẩn hóa và tách thành danh sách câu. Mỗi câu có ID riêng và có thể có thông tin trang.

Với mô hình embedding, hệ thống encode toàn bộ câu trong document thành ma trận embedding. Query của người dùng cũng được encode thành một vector. Sau đó hệ thống tính cosine similarity giữa query vector và từng sentence vector. Các câu có điểm thấp hơn threshold bị loại, các câu còn lại được sắp xếp giảm dần và lấy top-k.

Với chế độ hybrid, pipeline có thêm bước reranking. Đầu tiên retriever như MiniLM hoặc SFT-BE tìm khoảng 30 candidate tốt nhất. Sau đó Cross-Encoder chỉ chạy trên các candidate này, không chạy trên toàn bộ tài liệu. Điểm cuối cùng được tính bằng công thức kết hợp giữa retrieval score và entailment probability. Cách này cân bằng giữa tốc độ và độ chính xác."

---

## SLIDE 13 - Pseudo-code semantic search

**[NỘI DUNG SLIDE]**

```text
Algorithm SemanticSearch(document, query, model, top_k, threshold)
1. sentences = SplitDocumentIntoSentences(document)
2. embeddings = model.encode(sentences)
3. q = model.encode(query)
4. scores = CosineSimilarity(q, embeddings)
5. candidates = TopK(scores)
6. candidates = Filter(score >= threshold)
7. if reranker:
       candidates = CrossEncoderRerank(query, candidates)
8. return ranked candidates
```

**[HÌNH ẢNH / BẢNG]**

- Đặt pseudo-code bên trái, ảnh demo bên phải.

**[LỜI NÓI]**

"Đây là pseudo-code của chức năng chính. Bước đầu tiên là tách tài liệu thành câu. Bước thứ hai là encode các câu bằng model. Với TF-IDF, encode nghĩa là transform thành sparse vector. Với MiniLM hoặc SFT-BE, encode nghĩa là chạy Transformer để lấy dense embedding.

Bước thứ ba encode query. Sau đó tính cosine similarity giữa query và từng câu. Bước thứ năm lấy top-k candidate, bước thứ sáu lọc theo threshold. Threshold không chọn tùy tiện, mà trong phần đánh giá nhóm chọn threshold trên dev set để tối ưu F1.

Nếu bật reranker, candidate sẽ được đưa vào Cross-Encoder. Cross-Encoder nhận từng cặp câu `(candidate, query)` và trả ra xác suất entailment, neutral, contradiction. Entailment probability được dùng để rerank kết quả.

Pseudo-code này cũng cho thấy pipeline hoàn chỉnh theo yêu cầu barem: text -> preprocessing -> vectorization -> model -> output."

---

## SLIDE 14 - Pipeline compare documents

**[NỘI DUNG SLIDE]**

```text
Document A -> sentences A
Document B -> sentences B
Encode A, Encode B
Compute similarity matrix S[i][j]
Select candidate pairs
Greedy one-to-one matching
Filter by threshold
Similarity percent
```

- Công thức:

```text
similarity_percent = 2 * matched_pairs / (num_A + num_B) * 100
```

**[HÌNH ẢNH / BẢNG]**

- Ma trận similarity giữa các câu A và B.
- Screenshot tab Compare Documents.

**[LỜI NÓI]**

"Chức năng thứ hai là so sánh hai tài liệu. Thay vì so sánh toàn bộ file như một chuỗi dài, nhóm tách mỗi file thành danh sách câu. Gọi tài liệu A có n câu và tài liệu B có m câu. Hệ thống tính ma trận similarity S kích thước n nhân m, trong đó S[i][j] là độ giống nghĩa giữa câu i của tài liệu A và câu j của tài liệu B.

Sau đó hệ thống chọn các cặp có điểm cao và dùng greedy matching một-một. Lý do cần matching một-một là để tránh một câu trong A bị ghép với quá nhiều câu trong B, làm phần trăm tương đồng bị phóng đại.

Các cặp có điểm dưới threshold bị loại. Số cặp còn lại được dùng để tính phần trăm tương đồng theo công thức 2 lần số matched pairs chia tổng số câu của hai tài liệu. Công thức này giống logic F1 ở mức đơn giản: nếu hai tài liệu có nhiều câu match với nhau thì phần trăm cao; nếu một tài liệu có nhiều câu riêng không match thì phần trăm giảm.

Kết quả cuối là phần trăm tương đồng và bảng các cặp câu match, bao gồm sentence id, page, score và label NLI nếu có."

---

## SLIDE 15 - TF-IDF baseline

**[NỘI DUNG SLIDE]**

- Mục tiêu: baseline lexical matching.
- Input: `premise_clean`, `hypothesis_clean`.
- Vectorizer:
  - n-gram 1-2,
  - max_features 50,000,
  - min_df 2,
  - stop_words = english,
  - sublinear_tf = True.
- Score: cosine similarity.
- Test:
  - F1 = 0.558
  - ROC-AUC = 0.667
  - Precision@1 = 0.842

**[HÌNH ẢNH / BẢNG]**

- Bảng kết quả TF-IDF từ `outputs/tfidf_baseline/metrics.json`.
- Có thể chèn ablation TF-IDF.

**[LỜI NÓI]**

"Baseline đầu tiên là TF-IDF. Đây là phương pháp vector hóa truyền thống, dựa trên tần suất từ trong câu và mức độ hiếm của từ trong corpus. Nhóm dùng TF-IDF với n-gram từ 1 đến 2, tối đa 50,000 feature, min_df bằng 2 và stopword tiếng Anh.

Mô hình này không cần train theo nghĩa deep learning, nhưng vectorizer được fit trên train split. Sau đó với mỗi cặp premise-hypothesis, nhóm transform thành vector và tính cosine similarity. Nếu similarity vượt threshold thì dự đoán hai câu là semantic similar, tức tương ứng với entailment.

Kết quả test của TF-IDF là F1 khoảng 0.558 và ROC-AUC khoảng 0.667. Đây là baseline hợp lý nhưng chưa cao, vì TF-IDF phụ thuộc nhiều vào từ trùng nhau. Với các cặp paraphrase dùng từ khác, TF-IDF có thể cho điểm thấp. Ngược lại, với contradiction có nhiều từ giống nhau nhưng khác phủ định, TF-IDF có thể cho điểm cao nhầm.

Baseline này rất quan trọng vì nó tạo mốc so sánh để chứng minh các mô hình embedding và NLI thật sự cải thiện khả năng hiểu ngữ nghĩa."

---

## SLIDE 16 - Preprocessing ablation cho TF-IDF

**[NỘI DUNG SLIDE]**

- So sánh các biến thể preprocessing:
  - original clean text unigram,
  - lowercase,
  - remove punctuation,
  - remove digits,
  - bigram.
- Kết quả gần nhau:
  - unigram F1 khoảng 0.565,
  - bigram F1 khoảng 0.558.
- Nhận xét: preprocessing đơn giản không giải quyết được semantic gap.

**[HÌNH ẢNH / BẢNG]**

- `outputs/tables/tfidf_preprocessing_ablation.csv`

**[LỜI NÓI]**

"Theo yêu cầu barem, nhóm cần có so sánh giữa các kỹ thuật preprocessing. Với TF-IDF, nhóm chạy ablation trên nhiều biến thể: giữ nguyên clean text, lowercase, bỏ punctuation, bỏ digits và dùng bigram.

Kết quả cho thấy các biến thể unigram cho F1 khoảng 0.565, còn bigram có F1 khoảng 0.558. Sự khác biệt không quá lớn. Điều này cho thấy vấn đề chính không nằm ở việc lowercase hay bỏ dấu câu, mà nằm ở giới hạn của biểu diễn lexical. TF-IDF chỉ biết từ nào xuất hiện, không hiểu sâu quan hệ ngữ nghĩa.

Ví dụ hai câu có ý giống nhau nhưng dùng từ khác thì TF-IDF không có đủ thông tin để kéo chúng lại gần. Ngược lại, hai câu có rất nhiều từ giống nhau nhưng một câu phủ định câu kia thì TF-IDF vẫn có thể cho điểm cao.

Kết luận của phần ablation là preprocessing có tác động, nhưng để cải thiện rõ rệt semantic similarity, cần mô hình biểu diễn câu tốt hơn, như SentenceTransformer hoặc Cross-Encoder."

---

## SLIDE 17 - Pretrained MiniLM

**[NỘI DUNG SLIDE]**

- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- Không fine-tune trong project.
- Embedding dimension: 384.
- Score: cosine similarity giữa sentence embeddings.
- Test:
  - Accuracy = 0.677
  - F1 = 0.634
  - ROC-AUC = 0.781
  - Precision@1 = 0.974
  - MRR = 0.984

**[HÌNH ẢNH / BẢNG]**

- Bảng so sánh TF-IDF vs Pretrained MiniLM.

**[LỜI NÓI]**

"Mô hình đối sánh tiếp theo là Pretrained MiniLM, cụ thể là `sentence-transformers/all-MiniLM-L6-v2`. Đây là một mô hình nhẹ, nhanh và phổ biến cho semantic search. Nó encode mỗi câu thành vector 384 chiều. Khi hai câu gần nghĩa, cosine similarity giữa hai vector sẽ cao.

Khác với TF-IDF, MiniLM có khả năng nắm bắt semantic similarity tốt hơn. Kết quả test cho thấy F1 tăng từ khoảng 0.558 của TF-IDF lên khoảng 0.634. ROC-AUC tăng từ 0.667 lên 0.781. Điều này chứng minh dense sentence embedding phù hợp hơn với bài toán similarity.

Ở retrieval metric, MiniLM đạt Precision@1 khoảng 0.974 và MRR khoảng 0.984 trong thiết lập candidate pool của nhóm. Đây là kết quả rất tốt cho việc tìm câu entailment trong một nhóm candidate có nhiễu.

Tuy nhiên, MiniLM pretrained chưa được fine-tune riêng theo pipeline của nhóm trên AllNLI pair-score trong artifact chính. Vì vậy nhóm tiếp tục fine-tune MiniLM để kiểm tra liệu việc học trên dữ liệu NLI/similarity có cải thiện thêm không."

---

## SLIDE 18 - Fine-tuned MiniLM

**[NỘI DUNG SLIDE]**

- Base model: `all-MiniLM-L6-v2`.
- Training data: AllNLI `pair-score`.
- Actual train samples: 500,000.
- Loss: CosineSimilarityLoss.
- Epoch: 1.
- Batch size: 64.
- GPU: Tesla T4.
- Test:
  - F1 = 0.717
  - ROC-AUC = 0.867
  - Accuracy = 0.761

**[HÌNH ẢNH / BẢNG]**

- `outputs/finetuned_minilm/training_metadata.json`
- `outputs/finetuned_minilm/model_comparison.csv`

**[LỜI NÓI]**

"Sau pretrained MiniLM, nhóm dùng Fine-tuned MiniLM. Theo metadata đã lưu, model này được fine-tune từ `all-MiniLM-L6-v2` trên AllNLI pair-score với 500,000 mẫu train. Loss được dùng là CosineSimilarityLoss, epoch bằng 1, batch size 64, learning rate 5e-6 và train trên GPU Tesla T4.

Ý tưởng của fine-tuning là làm cho embedding space phù hợp hơn với dữ liệu NLI/similarity. Các cặp entailment hoặc có score cao sẽ được kéo gần nhau hơn; các cặp contradiction hoặc score thấp sẽ được đẩy xa hơn.

Kết quả cho thấy Fine-tuned MiniLM cải thiện rõ rệt so với pretrained. F1 tăng lên khoảng 0.717, ROC-AUC đạt khoảng 0.867, accuracy khoảng 0.761. So với TF-IDF, mức tăng F1 là rất đáng kể. So với Pretrained MiniLM, kết quả cũng cao hơn, chứng minh fine-tune trên AllNLI giúp model học được quan hệ ngữ nghĩa phù hợp với task.

Điểm cần nói rõ khi bảo vệ là metadata Fine-tuned MiniLM đến từ quy trình train trước đó, trong khi một số baseline local dùng sample 5,000. Vì vậy khi so sánh khoa học, nhóm cần thống nhất benchmark dev/test chung. Trong báo cáo, nên ghi rõ protocol đánh giá và nếu cần thì chạy lại evaluation tất cả model trên cùng test set."

---

## SLIDE 19 - SFT-BE checkpoint

**[NỘI DUNG SLIDE]**

- SFT-BE: custom shallow factorized Transformer bi-encoder.
- Training:
  - data: Wikimedia Wikipedia 20231101.en distillation,
  - teacher: sentence-transformer model,
  - loss: Stage0TeacherDistillationLoss.
- Training process:
  - train loss: 0.8575 -> 0.0736,
  - steps: 2,712,000,
  - elapsed: 42.39 hours,
  - GPU: RTX 3090.
- Test:
  - F1 = 0.618
  - ROC-AUC = 0.756
  - MRR = 0.980

**[HÌNH ẢNH / BẢNG]**

- `outputs/figures/training/sftbe_stage0_loss.png`
- `outputs/figures/training/sftbe_stage0_learning_rate.png`

**[LỜI NÓI]**

"Ngoài MiniLM, nhóm có một mô hình custom là SFT-BE checkpoint. Đây là một bi-encoder Transformer tự xây dựng theo hướng shallow factorized Transformer. Model được train bằng teacher-student distillation trên dữ liệu Wikipedia. Teacher model encode câu thành embedding, còn student SFT-BE học để tạo embedding gần với teacher.

Trong quá trình train, loss giảm từ khoảng 0.8575 xuống 0.0736 qua hơn 2.7 triệu bước, thời gian khoảng 42.39 giờ trên RTX 3090. Điều này cho thấy student model học được biểu diễn gần teacher ở mức distillation.

Khi đánh giá trên AllNLI pair-class local, SFT-BE đạt F1 khoảng 0.618, ROC-AUC khoảng 0.756 và MRR khoảng 0.980. F1 thấp hơn Fine-tuned MiniLM, nhưng retrieval metric vẫn tốt. Lý do có thể là SFT-BE được train bằng Wikipedia distillation, không trực tiếp fine-tune trên AllNLI entailment/contradiction. Nó học biểu diễn câu tổng quát, nhưng chưa tối ưu riêng cho bài toán NLI similarity.

Trong hệ thống demo, SFT-BE vẫn hữu ích như một retriever nhanh, đặc biệt khi kết hợp với Cross-Encoder để rerank."

---

## SLIDE 20 - Cross-Encoder NLI

**[NỘI DUNG SLIDE]**

- Base model: `distilbert-base-uncased`.
- Task: 3-class NLI.
- Input: cặp câu `(text_a, text_b)`.
- Output:
  - entailment,
  - neutral,
  - contradiction.
- Training:
  - train rows: 300,000,
  - dev rows: 19,657,
  - test rows: 19,656,
  - epoch: 1,
  - batch size: 32.
- Test:
  - Accuracy = 80.37%
  - Macro F1 = 80.25%
  - Entailment F1 = 83.42%

**[HÌNH ẢNH / BẢNG]**

- `outputs/figures/training/cross_encoder_train_eval_loss.png`
- `outputs/cross_encoder_outputs/cross_encoder_confusion_matrix.csv`

**[LỜI NÓI]**

"Cross-Encoder là mô hình quan trọng nhất cho phần reranking. Nhóm dùng `distilbert-base-uncased` và fine-tune thành classifier 3 lớp cho NLI. Input của Cross-Encoder là một cặp câu, không phải từng câu riêng biệt. Nhờ đó, model có thể học tương tác trực tiếp giữa token của hai câu.

Cross-Encoder được train trên AllNLI pair-class với 300,000 mẫu train, khoảng 19,657 mẫu dev và 19,656 mẫu test. Số epoch là 1, batch size 32, learning rate 2e-5 và train trên Tesla T4.

Kết quả test cho thấy accuracy đạt khoảng 80.37%, macro F1 khoảng 80.25%, riêng entailment F1 khoảng 83.42%. Đây là kết quả khá tốt vì DistilBERT là model nhẹ, tốc độ nhanh hơn các model lớn như BERT-base hoặc DeBERTa.

Trong semantic search, nhóm không dùng Cross-Encoder để scan toàn bộ document nếu document quá dài, vì chi phí lớn. Thay vào đó, Cross-Encoder được dùng sau bước retriever. Retriever tìm candidate nhanh, Cross-Encoder chấm lại candidate để tăng độ chính xác."

---

## SLIDE 21 - Kết quả huấn luyện Cross-Encoder

**[NỘI DUNG SLIDE]**

- Train loss giảm ổn định.
- Dev loss bám sát train loss.
- Không thấy dấu hiệu overfitting rõ.
- Test accuracy: 80.37%.
- Test macro F1: 80.25%.
- Test entailment ROC-AUC: 94.96%.

**[HÌNH ẢNH / BẢNG]**

- `outputs/figures/training/cross_encoder_train_eval_loss.png`
- `outputs/figures/training/cross_encoder_eval_metrics.png`

**[LỜI NÓI]**

"Slide này tập trung vào quá trình train Cross-Encoder. Đường train loss và dev loss giảm ổn định trong một epoch. Nếu train loss giảm nhưng dev loss tăng, đó sẽ là dấu hiệu overfitting. Ở đây dev loss vẫn giảm và bám tương đối sát train loss, nên có thể nói model học được pattern tổng quát thay vì chỉ ghi nhớ dữ liệu train.

Test accuracy đạt 80.37%, macro F1 đạt 80.25%. Macro F1 quan trọng vì nó tính trung bình trên các class, tránh trường hợp model chỉ tốt ở class nhiều mẫu. Entailment ROC-AUC đạt 94.96%, cho thấy xác suất entailment có khả năng phân biệt khá tốt giữa cặp similar và non-similar.

Đây là cơ sở để nhóm dùng entailment probability như một điểm reranking trong semantic search. Khi Cross-Encoder dự đoán entailment cao, ta có thêm bằng chứng rằng candidate thực sự gần nghĩa với query."

---

## SLIDE 22 - Confusion matrix Cross-Encoder

**[NỘI DUNG SLIDE]**

- Confusion matrix trên test:
  - Entailment đúng: 5,835 / 6,831.
  - Neutral đúng: 4,768 / 6,348.
  - Contradiction đúng: 5,194 / 6,477.
- Lỗi thường gặp:
  - neutral bị nhầm với entailment,
  - contradiction bị nhầm với neutral,
  - câu có phủ định hoặc suy luận ngầm khó hơn.

**[HÌNH ẢNH / BẢNG]**

| True \ Pred | entailment | neutral | contradiction |
|---|---:|---:|---:|
| entailment | 5835 | 655 | 341 |
| neutral | 827 | 4768 | 753 |
| contradiction | 496 | 787 | 5194 |

**[LỜI NÓI]**

"Confusion matrix giúp nhóm phân tích lỗi, không chỉ nhìn một con số accuracy. Với class entailment, model dự đoán đúng 5,835 trên 6,831 mẫu. Với neutral, đúng 4,768 trên 6,348. Với contradiction, đúng 5,194 trên 6,477.

Ta thấy neutral là class khó nhất. Điều này hợp lý vì neutral thường nằm giữa entailment và contradiction. Một câu neutral có thể liên quan chủ đề với premise nhưng không đủ thông tin để kết luận. Vì vậy model đôi khi nhầm neutral thành entailment nếu hai câu có nhiều thông tin giống nhau, hoặc nhầm neutral thành contradiction nếu có dấu hiệu khác biệt.

Contradiction cũng có lỗi khi câu mâu thuẫn phụ thuộc vào phủ định hoặc thông tin ngầm. Ví dụ chỉ một từ 'not' hoặc một chi tiết nhỏ có thể đổi hoàn toàn nhãn. Đây là lý do Cross-Encoder tốt hơn bi-encoder ở nhiều trường hợp, vì nó nhìn trực tiếp từng token của cặp câu. Tuy nhiên nó vẫn không hoàn hảo, nhất là với những câu cần suy luận ngữ cảnh sâu."

---

## SLIDE 23 - Hybrid retrieval + reranking

**[NỘI DUNG SLIDE]**

- Vấn đề:
  - Bi-Encoder nhanh nhưng có thể chưa chính xác nhất.
  - Cross-Encoder chính xác hơn nhưng chậm nếu chạy toàn bộ.
- Giải pháp:

```text
Step 1: Retriever lấy top candidate
Step 2: Cross-Encoder rerank candidate
Step 3: final_score = alpha * entailment + (1-alpha) * retrieval_score
```

- Mặc định demo:
  - `SFT-BE + Cross-Encoder`
  - alpha = 0.55.

**[HÌNH ẢNH / BẢNG]**

- Sơ đồ hai tầng:
  - Fast retrieval
  - Accurate reranking

**[LỜI NÓI]**

"Hybrid là phần hệ thống đề xuất của nhóm. Ý tưởng là kết hợp ưu điểm của hai loại model. Bi-Encoder như MiniLM hoặc SFT-BE encode từng câu riêng biệt nên rất nhanh. Ta có thể encode toàn bộ document một lần, sau đó query chỉ cần encode một lần và tính dot product. Nhưng vì hai câu được encode độc lập, model có thể bỏ sót một số tương tác tinh tế.

Cross-Encoder thì ngược lại. Nó nhận cả cặp câu, chính xác hơn trong việc đánh giá quan hệ entailment, nhưng nếu document có hàng nghìn câu thì chạy Cross-Encoder với tất cả cặp `(sentence, query)` sẽ chậm.

Do đó nhóm dùng chiến lược hai tầng. Tầng 1 là retriever, lấy top candidate nhanh. Tầng 2 là Cross-Encoder, chỉ rerank các candidate này. Điểm cuối được tính bằng công thức `alpha * entailment + (1-alpha) * retrieval_score`. Trong app, alpha mặc định là 0.55, nghĩa là Cross-Encoder có trọng số 55% và retriever có trọng số 45%.

Cách này thực tế hơn cho demo, vì vừa giữ được tốc độ phản hồi, vừa tận dụng được độ chính xác của NLI."

---

## SLIDE 24 - Bảng so sánh mô hình

**[NỘI DUNG SLIDE]**

| Model | F1 | ROC-AUC | P@1 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 0.558 | 0.667 | 0.842 | 0.877 |
| Pretrained MiniLM | 0.634 | 0.781 | 0.974 | 0.984 |
| Fine-tuned MiniLM | 0.717 | 0.867 | 0.925 | 0.950 |
| SFT-BE checkpoint | 0.618 | 0.756 | 0.966 | 0.980 |
| Cross-Encoder NLI | 0.834 | 0.950 | - | - |
| Fine-tuned MiniLM + Cross-Encoder | 0.834 | 0.950 | 0.928 | 0.953 |

**[HÌNH ẢNH / BẢNG]**

- `outputs/tables/final_model_summary.csv`
- Nên thêm chú thích: cần thống nhất benchmark nếu báo cáo chính thức.

**[LỜI NÓI]**

"Bảng này tổng hợp kết quả chính. TF-IDF baseline đạt F1 khoảng 0.558, ROC-AUC 0.667. Pretrained MiniLM cải thiện lên F1 0.634 và ROC-AUC 0.781. Fine-tuned MiniLM tiếp tục tăng lên F1 0.717 và ROC-AUC 0.867. Điều này cho thấy sentence embedding tốt hơn TF-IDF, và fine-tuning trên AllNLI giúp embedding phù hợp hơn với task.

SFT-BE checkpoint có F1 khoảng 0.618, thấp hơn Fine-tuned MiniLM nhưng retrieval metric vẫn mạnh, MRR khoảng 0.980. Cross-Encoder NLI đạt F1 binary similarity khoảng 0.834 và ROC-AUC 0.950, cao nhất trong bảng về phân biệt entailment và non-entailment.

Hybrid Fine-tuned MiniLM + Cross-Encoder dùng retrieval metric từ bước candidate retrieval và binary metric từ Cross-Encoder, nên vừa có khả năng search vừa có khả năng rerank.

Một điểm nhóm cần trình bày trung thực là các kết quả hiện có có thể đến từ các quy mô dữ liệu khác nhau: một số baseline local dùng sample 5,000, Cross-Encoder dùng tập lớn hơn. Vì vậy trong báo cáo chính thức, nhóm nên nói rõ protocol và nếu có thời gian thì chạy lại tất cả evaluation trên cùng dev/test để so sánh tuyệt đối công bằng. Dù vậy, xu hướng kết quả vẫn hợp lý: lexical baseline thấp nhất, embedding tốt hơn, Cross-Encoder mạnh nhất về phân loại cặp câu."

---

## SLIDE 25 - Metric đánh giá

**[NỘI DUNG SLIDE]**

- Similarity:
  - cosine similarity,
  - threshold chọn trên dev.
- Classification:
  - accuracy,
  - precision,
  - recall,
  - F1,
  - ROC-AUC.
- Retrieval:
  - Precision@1,
  - Recall@5,
  - MRR,
  - mean rank.

**[HÌNH ẢNH / BẢNG]**

- Bảng metric theo task:

| Task | Metric |
|---|---|
| Similarity | cosine, ROC-AUC, F1 |
| Retrieval | P@1, Recall@5, MRR |
| NLI Classification | Accuracy, Macro F1, Confusion matrix |

**[LỜI NÓI]**

"Vì đề tài gồm Similarity, Retrieval và Classification phụ trợ, nhóm dùng nhiều metric thay vì chỉ một metric. Với similarity, điểm cơ bản là cosine similarity giữa hai embedding. Tuy nhiên để đánh giá binary similar hay not similar, nhóm chọn threshold trên dev set, sau đó báo cáo F1 và ROC-AUC trên test.

Với NLI classification, nhóm báo cáo accuracy, precision, recall, F1 và macro F1. Macro F1 quan trọng vì nó thể hiện chất lượng trung bình trên ba class entailment, neutral, contradiction. Nhóm cũng dùng confusion matrix để phân tích lỗi.

Với retrieval, nhóm dùng Precision@1, Recall@5, MRR và mean rank. Precision@1 cho biết model có đưa câu đúng lên vị trí đầu tiên hay không. Recall@5 cho biết câu đúng có nằm trong top 5 không. MRR đo vị trí trung bình của câu đúng theo nghịch đảo rank, càng gần 1 càng tốt.

Các metric này phù hợp với yêu cầu barem: Similarity cần cosine similarity, Retrieval cần Precision@k, Classification cần accuracy và F1."

---

## SLIDE 26 - Demo ứng dụng Streamlit

**[NỘI DUNG SLIDE]**

- Framework: Streamlit.
- Input:
  - TXT, PDF, DOCX,
  - query,
  - model, threshold, top-k.
- Output:
  - câu match được highlight,
  - score,
  - page / sentence id,
  - label NLI nếu có,
  - CSV download.
- Tabs:
  - Semantic Search,
  - Compare Documents,
  - Benchmark.

**[HÌNH ẢNH / BẢNG]**

- Screenshot giao diện app.
- Demo trực tiếp:
  1. Upload PDF.
  2. Nhập query.
  3. Chọn SFT-BE + Cross-Encoder.
  4. Bấm Search.
  5. Click result để scroll đến câu.

**[LỜI NÓI]**

"Phần demo được xây bằng Streamlit. Nhóm chọn Streamlit vì framework này đơn giản, phù hợp với upload file, hiển thị bảng, biểu đồ và chạy nhanh local. App có ba tab chính.

Tab Semantic Search cho phép upload một file, nhập query, chọn model, threshold, số lượng kết quả top-k và số candidate cho reranking. Khi bấm Search, app gọi toàn bộ pipeline: đọc file, tách câu, encode, tính similarity, lọc threshold và highlight kết quả trong tài liệu.

Tab Compare Documents cho phép upload hai file. Hệ thống tách câu của cả hai file, tính ma trận similarity, ghép cặp câu tương đồng và hiển thị phần trăm giống nhau cùng bảng matched pairs.

Tab Benchmark hiển thị các bảng kết quả thực nghiệm, giúp người xem không chỉ thấy demo mà còn thấy bằng chứng định lượng.

Trong giao diện, khi người dùng click vào một kết quả bên trái, app lưu `active_sentence_id` và tự scroll đến câu đó trong phần document. Các câu match được highlight để người dùng dễ kiểm tra."

---

## SLIDE 27 - Luồng code trong demo

**[NỘI DUNG SLIDE]**

- `streamlit_app.py`:
  - nhận input,
  - render UI,
  - gọi search/compare.
- `document_utils.py`:
  - extract TXT/PDF/DOCX,
  - split sentences,
  - tạo `SentenceRecord`.
- `similarity_engine.py`:
  - load model,
  - tính TF-IDF / embedding / Cross-Encoder,
  - semantic search,
  - compare documents.

**[HÌNH ẢNH / BẢNG]**

```text
UI -> document_utils -> similarity_engine -> results -> UI highlight
```

**[LỜI NÓI]**

"Về mặt code, app được chia thành ba phần rõ ràng. `streamlit_app.py` phụ trách giao diện. File này nhận file upload, query, model choice, threshold và top-k từ người dùng, sau đó gọi các hàm xử lý.

`document_utils.py` phụ trách đọc tài liệu. Nó có các hàm `extract_txt`, `extract_pdf`, `extract_docx` và `extract_uploaded_file`. Sau khi đọc text, nó gọi `split_sentences` để tạo danh sách `SentenceRecord`. Mỗi record gồm sentence_id, text và page. Như nhóm đã nói, phần tách câu đã được chỉnh để xuống dòng giữa câu trong PDF không làm câu bị cắt sai.

`similarity_engine.py` là lõi xử lý model. File này load TF-IDF vectorizer, MiniLM, SFT-BE hoặc Cross-Encoder. Hàm `semantic_search` là trung tâm của chức năng tìm kiếm. Nếu chọn TF-IDF thì tính TF-IDF score. Nếu chọn embedding thì encode câu và query rồi dot product. Nếu chọn Cross-Encoder thì predict entailment probability. Nếu chọn hybrid thì kết hợp retrieval score và entailment probability.

Cách chia module này giúp pipeline dễ bảo trì và giải thích khi vấn đáp."

---

## SLIDE 28 - Phân tích hạn chế

**[NỘI DUNG SLIDE]**

- Hạn chế dataset:
  - chủ yếu tiếng Anh,
  - NLI sentence-pair chưa bao phủ mọi domain tài liệu.
- Hạn chế kỹ thuật:
  - PDF extract có thể lỗi với file scan/layout phức tạp,
  - Cross-Encoder chậm nếu chạy toàn bộ câu,
  - threshold phụ thuộc dữ liệu dev,
  - so sánh cần benchmark chung để tuyệt đối công bằng.
- Hạn chế mô hình:
  - khó với câu dài,
  - khó với phủ định tinh tế,
  - neutral là class dễ nhầm.

**[HÌNH ẢNH / BẢNG]**

- Bảng lỗi thường gặp:

| Lỗi | Nguyên nhân | Cách cải thiện |
|---|---|---|
| Cắt câu sai | PDF layout | sentence segmentation tốt hơn |
| Neutral nhầm entailment | ngữ nghĩa gần nhưng chưa đủ suy ra | train thêm hard examples |
| Cross-Encoder chậm | pairwise inference | rerank top-k |

**[LỜI NÓI]**

"Phần hạn chế rất quan trọng vì barem yêu cầu phân tích vì sao mô hình tốt/xấu và điều kiện hạn chế. Dataset AllNLI chủ yếu là tiếng Anh, nên hệ thống hiện tại phù hợp nhất với tài liệu tiếng Anh. Nếu dùng cho tiếng Việt, kết quả có thể không tốt nếu không thay model và dataset.

Về kỹ thuật, PDF extraction có thể lỗi nếu file là scan ảnh hoặc có layout nhiều cột. Nhóm xử lý được PDF text-based, nhưng với scan PDF cần OCR, đây là hướng mở rộng. Cross-Encoder chính xác nhưng chậm nếu chạy trên toàn bộ các cặp câu, nên nhóm dùng nó ở bước reranking top-k.

Threshold cũng là một yếu tố cần chú ý. Threshold được chọn trên dev set, nên nếu chuyển sang domain khác như y tế, pháp lý hoặc văn bản kỹ thuật, threshold có thể cần hiệu chỉnh lại.

Về mô hình, neutral là class khó vì nó nằm giữa entailment và contradiction. Một số câu cần suy luận ngữ cảnh sâu hoặc phụ thuộc vào phủ định tinh tế cũng dễ gây lỗi. Đây là lý do nhóm cần confusion matrix và ví dụ lỗi để phân tích."

---

## SLIDE 29 - Hướng phát triển

**[NỘI DUNG SLIDE]**

- Đánh giá công bằng hơn:
  - chạy lại tất cả model trên cùng dev/test full.
- Cải thiện mô hình:
  - DeBERTa/RoBERTa Cross-Encoder,
  - hard negative mining,
  - domain adaptation.
- Cải thiện tốc độ:
  - FAISS index,
  - caching embedding,
  - batch reranking.
- Mở rộng ứng dụng:
  - OCR cho scan PDF,
  - hỗ trợ tiếng Việt,
  - export report,
  - deploy online.

**[HÌNH ẢNH / BẢNG]**

- Roadmap 3 giai đoạn:
  - short-term,
  - mid-term,
  - long-term.

**[LỜI NÓI]**

"Trong hướng phát triển, việc đầu tiên là chuẩn hóa benchmark. Nhóm nên chạy lại tất cả model trên cùng một dev/test set, có thể là full AllNLI pair-class, để bảng so sánh hoàn toàn công bằng. Điều này đặc biệt quan trọng nếu dùng kết quả trong báo cáo chính thức.

Về mô hình, có thể thử Cross-Encoder mạnh hơn như RoBERTa hoặc DeBERTa, đồng thời dùng hard negative mining để model học tốt hơn các trường hợp khó. Với bi-encoder, có thể fine-tune bằng triplet loss hoặc MultipleNegativesRankingLoss trên nhiều hard negatives.

Về tốc độ, khi tài liệu lớn, nên dùng FAISS để index embedding thay vì tính tuyến tính qua toàn bộ câu. Ngoài ra có thể cache embedding của document, để khi người dùng search nhiều query trên cùng một file thì không phải encode lại.

Về ứng dụng, có thể thêm OCR cho scan PDF, hỗ trợ tiếng Việt bằng multilingual model, thêm chức năng export báo cáo so sánh tài liệu, và deploy lên Hugging Face Spaces hoặc server riêng."

---

## SLIDE 30 - Kết luận

**[NỘI DUNG SLIDE]**

- Đã xác định bài toán Semantic Similarity + Retrieval + NLI.
- Đã dùng dataset thực tế AllNLI.
- Đã có pipeline:

```text
text -> preprocessing -> vectorization -> model -> output
```

- Đã triển khai baseline và mô hình cải tiến:
  - TF-IDF,
  - MiniLM,
  - SFT-BE,
  - Cross-Encoder,
  - Hybrid.
- Đã có demo Streamlit chạy được.
- Kết quả tốt nhất: Cross-Encoder/Hybrid đạt F1 khoảng 0.834, ROC-AUC khoảng 0.950.

**[HÌNH ẢNH / BẢNG]**

- Một slide tổng kết gồm 4 ô:
  - Dataset,
  - Model,
  - Evaluation,
  - Demo.

**[LỜI NÓI]**

"Tổng kết lại, nhóm đã xây dựng một hệ thống tìm kiếm ngữ nghĩa và so sánh độ tương đồng tài liệu. Về mặt bài toán, đề tài thuộc nhóm Similarity và Retrieval, có Classification phụ trợ bằng NLI.

Về dữ liệu, nhóm sử dụng AllNLI từ Hugging Face, có nguồn rõ ràng, có nhãn entailment, neutral, contradiction và có các subset phù hợp với từng mô hình. Nhóm cũng thực hiện EDA, thống kê phân bố nhãn, độ dài câu, lexical overlap và ví dụ dữ liệu.

Về mô hình, nhóm triển khai baseline TF-IDF, đối sánh bằng Pretrained MiniLM, mô hình cải tiến Fine-tuned MiniLM và SFT-BE, cùng Cross-Encoder NLI. Hệ thống đề xuất là hybrid retrieval + reranking, giúp kết hợp tốc độ của bi-encoder và độ chính xác của Cross-Encoder.

Về đánh giá, nhóm dùng các metric phù hợp với task: cosine similarity, F1, ROC-AUC, Precision@1, Recall@5, MRR, accuracy và macro F1. Kết quả tốt nhất thuộc về Cross-Encoder/Hybrid với F1 khoảng 0.834 và ROC-AUC khoảng 0.950.

Cuối cùng, nhóm đã triển khai demo Streamlit với upload file, semantic search, compare documents và benchmark. Đây là pipeline hoàn chỉnh từ dữ liệu đến ứng dụng."

---

# Kịch Bản Demo Trực Tiếp

## Demo 1 - Semantic Search

**Các bước thao tác**

1. Mở app Streamlit.
2. Vào tab `Semantic Search`.
3. Upload một file PDF/DOCX/TXT tiếng Anh.
4. Chọn model `SFT-BE + Cross-Encoder` hoặc `Fine-tuned MiniLM + Cross-Encoder`.
5. Nhập query, ví dụ: "a woman earns money from her talent" hoặc một câu gần nghĩa với nội dung tài liệu.
6. Chỉnh `Results = 10`, `Threshold = 0.62`, `Candidates = 30`.
7. Bấm `Search`.
8. Click vào một kết quả bên trái để app scroll đến câu được highlight.

**Lời nói khi demo**

"Ở đây em upload một tài liệu PDF. Hệ thống sẽ đọc text trong PDF, tách thành các câu và hiển thị như một trang tài liệu. Em nhập một query không nhất thiết trùng từ với câu trong tài liệu. Khi bấm Search, hệ thống encode query và các câu trong document, sau đó tìm các câu có ý nghĩa gần nhất.

Kết quả bên trái có score. Nếu dùng hybrid, score là kết hợp giữa retrieval score và xác suất entailment từ Cross-Encoder. Khi em click vào kết quả, câu tương ứng được highlight trong tài liệu. Điểm khác với Ctrl+F là query không cần trùng từ tuyệt đối, vì model tìm theo embedding và quan hệ ngữ nghĩa."

## Demo 2 - Compare Documents

**Các bước thao tác**

1. Vào tab `Compare Documents`.
2. Upload Document A và Document B.
3. Chọn model hybrid.
4. Chọn threshold.
5. Bấm `Compare`.
6. Chỉ vào:
   - Similarity percent,
   - matched pairs,
   - bảng câu A/câu B/score/label.

**Lời nói khi demo**

"Ở chức năng thứ hai, hệ thống so sánh hai tài liệu ở mức câu. Mỗi tài liệu được tách thành danh sách câu. Sau đó hệ thống tính similarity matrix giữa các câu của hai tài liệu, chọn các cặp có điểm cao, rồi matching một-một. Kết quả là phần trăm tương đồng và bảng các cặp câu giống nghĩa.

Chức năng này hữu ích khi hai tài liệu không copy nguyên văn nhưng có nhiều câu diễn giải lại. Nếu chỉ so sánh chuỗi, ta dễ bỏ sót. Còn ở đây hệ thống so sánh theo nghĩa."

## Demo 3 - Benchmark

**Các bước thao tác**

1. Vào tab `Benchmark`.
2. Mở bảng final model summary.
3. Chỉ vào:
   - TF-IDF baseline,
   - Pretrained MiniLM,
   - Fine-tuned MiniLM,
   - Cross-Encoder,
   - Hybrid.
4. Mở loss curve và confusion matrix nếu có.

**Lời nói khi demo**

"Tab Benchmark cho thấy đồ án không chỉ có giao diện mà có đánh giá định lượng. Ở đây ta thấy TF-IDF là baseline thấp nhất, MiniLM cải thiện nhờ embedding, Fine-tuned MiniLM tốt hơn pretrained, và Cross-Encoder/Hybrid đạt kết quả cao nhất về F1 và ROC-AUC. Các biểu đồ loss và confusion matrix giúp phân tích quá trình huấn luyện và lỗi của model."

---

# Câu Hỏi Vấn Đáp Dự Kiến

## Câu 1: Vì sao không chỉ dùng TF-IDF?

**Trả lời**

"TF-IDF nhanh và dễ giải thích nhưng chỉ dựa vào từ vựng. Nếu hai câu gần nghĩa nhưng dùng từ khác, TF-IDF có thể cho điểm thấp. Ngược lại, hai câu nhiều từ giống nhau nhưng mâu thuẫn vì một từ phủ định, TF-IDF có thể cho điểm cao nhầm. Vì vậy TF-IDF phù hợp làm baseline, còn mô hình chính nên dùng sentence embedding hoặc Cross-Encoder."

## Câu 2: Vì sao dùng AllNLI?

**Trả lời**

"AllNLI phù hợp vì dữ liệu gồm các cặp câu với nhãn entailment, neutral, contradiction. Các nhãn này có thể chuyển thành bài toán similarity: entailment là similar cao, contradiction là thấp, neutral ở giữa. Ngoài ra AllNLI có các subset thuận tiện cho nhiều cách train như pair-class, pair-score, pair và triplet."

## Câu 3: Cross-Encoder khác Bi-Encoder thế nào?

**Trả lời**

"Bi-Encoder encode hai câu riêng biệt thành hai vector rồi tính cosine similarity. Nó nhanh vì có thể encode document trước. Cross-Encoder nhận cả cặp câu cùng lúc, nên attention có thể nhìn tương tác token giữa hai câu. Cross-Encoder thường chính xác hơn nhưng chậm hơn, vì phải chạy model cho từng cặp."

## Câu 4: Vì sao cần hybrid?

**Trả lời**

"Hybrid giúp cân bằng tốc độ và độ chính xác. Bi-Encoder tìm candidate nhanh trên toàn tài liệu. Cross-Encoder chỉ rerank top candidate, nên không quá chậm nhưng vẫn cải thiện chất lượng kết quả."

## Câu 5: Threshold được chọn như thế nào?

**Trả lời**

"Threshold được chọn trên dev set. Nhóm tính score cho các cặp dev, thử các threshold và chọn threshold cho F1 tốt nhất. Sau đó threshold này được giữ cố định khi báo cáo kết quả trên test set."

## Câu 6: Có data leakage không?

**Trả lời**

"Pipeline giữ tách biệt train, dev và test. Train dùng để học mô hình hoặc fit vectorizer. Dev dùng để chọn threshold và theo dõi quá trình train. Test chỉ dùng để báo cáo kết quả cuối. Không trộn test vào train."

## Câu 7: Vì sao kết quả giữa các model cần benchmark chung?

**Trả lời**

"Nếu các model đánh giá trên tập test khác nhau thì so sánh không công bằng. Vì vậy khi báo cáo chính thức, cần nói rõ tập dev/test dùng chung. Các artifact hiện có có thể đến từ các lần chạy khác nhau, nên nhóm cần thống nhất protocol hoặc chạy lại evaluation trên cùng một benchmark."

## Câu 8: Vì sao PDF đôi khi highlight không hết câu?

**Trả lời**

"Nguyên nhân là PDF extraction có thể chèn xuống dòng giữa câu. Nếu sentence splitter coi newline là ranh giới câu thì câu bị cắt thành nhiều đoạn. Nhóm đã sửa để newline trong giữa câu được normalize thành khoảng trắng, và chỉ tách câu sau dấu kết câu. Vì vậy highlight sẽ bám sát câu đầy đủ hơn."

## Câu 9: Nếu dùng tiếng Việt thì cần làm gì?

**Trả lời**

"Cần thay dataset hoặc bổ sung dataset tiếng Việt như ViNLI/XNLI tiếng Việt, dùng multilingual sentence transformer hoặc PhoBERT/XLM-R, và thêm word segmentation cho tiếng Việt nếu cần. Phiên bản hiện tại tập trung tiếng Anh vì AllNLI là tiếng Anh."

## Câu 10: Hạn chế lớn nhất của hệ thống là gì?

**Trả lời**

"Hạn chế lớn nhất là domain và tốc độ. Dataset chủ yếu là tiếng Anh và dạng sentence-pair, nên nếu đưa vào domain chuyên ngành có thể cần fine-tune thêm. Cross-Encoder chính xác nhưng chậm nếu chạy trên toàn bộ câu, nên hệ thống phải dùng reranking top-k."

---

# Checklist Bám Barem

- Xác định bài toán: Slide 3.
- Input/Output: Slide 3, 4.
- Dataset thực tế: Slide 5, 6, 7.
- Số mẫu, ngôn ngữ, nhãn, split: Slide 5, 6.
- Thống kê dữ liệu và biểu đồ: Slide 6, 9.
- Tiền xử lý: Slide 8, 16.
- Vector hóa: Slide 10.
- Baseline: Slide 15.
- Mô hình cải tiến: Slide 18, 19, 20, 23.
- Pipeline hoàn chỉnh: Slide 12, 13, 14, 27.
- Train/val/test, tránh leakage: Slide 6, Q&A.
- Loss/training process: Slide 19, 21.
- Hyperparameters: Slide 15, 18, 20.
- Metric phù hợp: Slide 25.
- So sánh model: Slide 24.
- So sánh preprocessing: Slide 16.
- Confusion matrix/phân tích lỗi: Slide 22, 28.
- Demo ứng dụng: Slide 26 và phần kịch bản demo.
- Hạn chế/hướng phát triển: Slide 28, 29.
- Kết luận: Slide 30.
