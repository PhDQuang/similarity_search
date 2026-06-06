# AllNLI dataset notes

## Dataset

Use:

```python
from datasets import load_dataset

ds = load_dataset("sentence-transformers/all-nli", "pair-class")
```

AllNLI is a combination of SNLI and MultiNLI prepared by SentenceTransformers.
It is suitable for semantic textual similarity and embedding model fine-tuning.

## Subsets

| Subset | Main columns | Suggested use |
|---|---|---|
| `pair-class` | `premise`, `hypothesis`, `label` | EDA, baseline classification, Cross-Encoder NLI |
| `pair-score` | `sentence1`, `sentence2`, `score` | Similarity regression and threshold selection |
| `pair` | `anchor`, `positive` | Bi-encoder training with positive pairs |
| `triplet` | `anchor`, `positive`, `negative` | Triplet training with hard negatives |

## Label meaning

| Label id | Label name | Meaning |
|---:|---|---|
| 0 | entailment | Sentence B can be inferred from sentence A |
| 1 | neutral | The relation is not determined |
| 2 | contradiction | The two sentences conflict |

For similarity scoring, a common mapping is:

```text
entailment -> 1.0
neutral -> 0.5
contradiction -> 0.0
```

## Project usage

- `pair-class`: analyze label distribution, text length, lexical overlap, and
  train/evaluate a 3-class NLI model.
- `pair-score`: evaluate cosine similarity thresholds.
- `triplet`: train the proposed SentenceTransformer embedding model.
- Synthetic document sets can be created later by grouping anchors and
  positives, then shuffling sentence order to test document comparison.

