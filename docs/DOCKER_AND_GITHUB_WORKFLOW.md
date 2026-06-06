# Docker and GitHub workflow

This workflow is designed for a 5-person NLP project team.

## 1. Local setup without Docker

Run once after cloning the repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For model training:

```powershell
python -m pip install -r requirements-train.txt
```

For the web demo:

```powershell
python -m pip install -r requirements-app.txt
```

## 2. Prepare data and run EDA locally

Quick sample for testing:

```powershell
python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
python -m similarity_search.data.eda_allnli --subset pair-class
```

Full recommended data preparation:

```powershell
python -m similarity_search.data.prepare_allnli --subsets pair-class pair-score triplet
python -m similarity_search.data.eda_allnli --subset pair-class
python -m similarity_search.data.eda_allnli --subset pair-score
python -m similarity_search.data.eda_allnli --subset triplet
```

## 3. Docker setup

Before running Docker commands, start Docker Desktop and wait until it says the
engine is running.

Build the data/EDA image:

```powershell
docker compose build dev
```

Run a quick help check:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --help
```

Prepare a small AllNLI sample in Docker:

```powershell
docker compose run --rm dev python -m similarity_search.data.prepare_allnli --subsets pair-class --max-rows-per-split 5000
```

Run EDA in Docker:

```powershell
docker compose run --rm dev python -m similarity_search.data.eda_allnli --subset pair-class
```

Build the training image:

```powershell
docker compose build train
```

Build and run the web image:

```powershell
docker compose build web
docker compose up web
```

Open:

```text
http://localhost:8501
```

Stop the web service:

```powershell
docker compose down
```

Common Windows issue:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

Fix: open Docker Desktop, wait for the Linux engine to start, then rerun the
same command.

## 4. First GitHub push

Initialize Git:

```powershell
git init
git branch -M main
git status
```

Add and commit project files:

```powershell
git add .
git commit -m "Initial NLP similarity search project"
```

Create an empty repository on GitHub, then connect it:

```powershell
git remote add origin https://github.com/<your-org-or-username>/<repo-name>.git
git push -u origin main
```

Alternative with GitHub CLI:

```powershell
gh auth login
gh repo create <repo-name> --public --source . --remote origin --push
```

Use `--private` instead of `--public` if the project should not be public yet.

## 5. Daily team workflow

Start from the latest `main`:

```powershell
git switch main
git pull origin main
```

Create a feature branch:

```powershell
git switch -c feature/data-eda
```

Work, then commit:

```powershell
git status
git add .
git commit -m "Add AllNLI EDA outputs"
```

Push the branch:

```powershell
git push -u origin feature/data-eda
```

Open a pull request on GitHub. Another member should review it before merging.

After merge, update local `main`:

```powershell
git switch main
git pull origin main
```

## 6. Recommended branches for this project

```text
feature/data-eda
feature/baseline-tfidf
feature/train-sbert
feature/cross-encoder
feature/web-demo
feature/report
```

## 7. What belongs where

GitHub:

- Code.
- Config files.
- Notebook source.
- Small figures and metric CSVs.
- Report source and slides if not too large.

Hugging Face Hub:

- Final fine-tuned SentenceTransformer model.
- Final Cross-Encoder model.
- Model card and training notes.

Google Drive:

- Demo video.
- Large PowerPoint/PDF files.
- Temporary checkpoints.
- Backup of final report.

Do not commit:

- Downloaded raw datasets.
- Full processed dataset files.
- Model checkpoints.
- `.venv`.
- Tokens or passwords.
