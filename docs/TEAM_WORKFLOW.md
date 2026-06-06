# Team workflow

## Recommended tools

Use **GitHub** as the central place for project work:

- Source code.
- Notebooks.
- Report source files.
- Issues and task tracking.
- Pull requests and code review.

Use **Hugging Face Hub** for:

- Fine-tuned SentenceTransformer model.
- Cross-Encoder model.
- Public model cards and experiment notes.

Use **Google Drive** for files that are large or presentation-focused:

- Demo video.
- Slides.
- Temporary raw exports.
- Large model checkpoints before pushing the final model to Hugging Face.

Avoid using Google Drive as the main code workspace because concurrent edits,
version history, dependency changes, and merge conflicts are much harder to
manage than with Git.

## Branching convention

Recommended branches:

```text
main
feature/data-eda
feature/baseline-tfidf
feature/train-sbert
feature/cross-encoder
feature/web-demo
feature/report
```

Each member works on one feature branch and opens a pull request to `main`.

## Issue/task convention

Create GitHub issues with labels:

```text
data
eda
model
evaluation
web
report
bug
```

Each issue should have:

- Goal.
- Expected output.
- Owner.
- Deadline.
- Link to related notebook/script.

## What should not be committed

Do not commit:

- Full downloaded datasets.
- Model checkpoints.
- Large output folders.
- `.venv`.
- Secret tokens.

Commit:

- Code.
- Small config files.
- Small example files.
- Final metric tables.
- Important figures for the report.

