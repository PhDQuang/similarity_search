# Kaggle notebooks for clean full AllNLI training

Moi notebook co the chay doc lap tren Kaggle:

1. `01_train_tfidf_clean_full_eval5k_kaggle.ipynb`
2. `02_train_minilm_clean_full_earlystop_eval5k_kaggle.ipynb`
3. `03_train_cross_encoder_clean_full_earlystop_eval5k_kaggle.ipynb`
4. `04_train_sftbe_clean_full_earlystop_eval5k_kaggle.ipynb`

Moi notebook tu:

- clone repo vao `/kaggle/working/similarity_search` neu chua co;
- cai `fix/requirements-kaggle.txt` va `fix` package;
- tim clean dataset da upload trong `/kaggle/input/allnli-70-15-15-clean/...`;
- neu khong co clean dataset upload san, tu tao lai clean dataset bang
  `similarity_search_fix.data.prepare_allnli_70_15_15_clean`;
- train model tren clean train split day du;
- dung early stopping voi MiniLM, Cross-Encoder, SFT-BE;
- danh gia pair classification tren test sample 5k va do thoi gian scoring;
- nen model + outputs thanh file zip trong `/kaggle/working` de tai ve.

TF-IDF la baseline non-iterative nen khong co early stopping.

SFT-BE can upload checkpoint `stage0_final.pt` vao Kaggle Dataset va sua bien
`SFTBE_CHECKPOINT_PATH` trong notebook neu duong dan khac.

