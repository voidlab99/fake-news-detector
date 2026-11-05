# Fake News Detector (NLP)

Fine-tunes **DistilBERT** to classify news articles as real or fake, benchmarks it against a
**TF-IDF + Logistic Regression** baseline, and serves it through a **FastAPI** endpoint with a
small web UI for real-time predictions.

**Stack:** Python · Hugging Face Transformers · PyTorch · scikit-learn · FastAPI · Docker · GitHub Actions

## Results

Held-out test set (10% stratified split, fixed seed, identical for both models):

| Model                        | F1  | Precision | Recall |
|------------------------------|-----|-----------|--------|
| TF-IDF + Logistic Regression | _fill in from `baseline_metrics.json`_ | | |
| DistilBERT (fine-tuned)      | _fill in from `distilbert_metrics.json`_ | | |

Out-of-distribution check (`python evaluate.py --external other.csv`):

| Model      | In-distribution F1 | External-dataset F1 | Drop |
|------------|--------------------|---------------------|------|
| DistilBERT | _fill in_          | _fill in_           |      |

## Project layout

```
.
├── data/                  # Fake.csv and True.csv go here (git-ignored)
├── data_utils.py          # loading, de-leaking, dedup, shared split
├── baseline.py            # TF-IDF + LogReg baseline
├── train.py               # DistilBERT fine-tuning -> ./model
├── evaluate.py            # metrics, confusion matrix, error analysis, external dataset
├── app/
│   ├── main.py            # FastAPI: POST /predict, GET /health, serves the UI
│   └── static/index.html  # web front end
├── tests/                 # pytest: data pipeline + API
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put the ISOT Fake.csv and True.csv in data/
python baseline.py
python train.py                                            # GPU recommended
python train.py --epochs 2 --max_len 128 --batch_size 32   # faster, CPU-friendly
python evaluate.py

uvicorn app.main:app --reload                              # http://127.0.0.1:8000
```

### API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste the article headline and body here ..."}'
```

```json
{"label": "FAKE", "confidence": 0.97, "probabilities": {"REAL": 0.03, "FAKE": 0.97}}
```

### Docker

```bash
python train.py                      # creates ./model
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

### Tests

```bash
pip install pytest
pytest -q     # data-pipeline tests always run; API tests need torch + a trained ./model
```

## Design decisions

- **Leakage control.** News datasets like ISOT let models score highly by spotting the publisher
  (e.g. a "(Reuters) -" dateline) instead of judging the content. `data_utils.py` strips those
  tells and URLs and drops duplicate articles so they can't straddle train and test.
- **One shared split** for both models, so the comparison is apples to apples.
- **Train/serve consistency.** The training truncation length is saved with the tokenizer and
  reused by `evaluate.py` and the API.
- **Error analysis.** `evaluate.py` writes the misclassified articles, most confident mistakes
  first, to `results/test_errors.csv`.

## Limitations

- The model learns **writing patterns in its training data**. It does not fact-check claims.
- ISOT labels reflect the *source* (known outlets vs. known unreliable sites). A real article
  from an unreliable outlet, or a false one from a reputable outlet, is not what this measures.
- Expect lower performance on other sources, topics, or time periods than the held-out score
  suggests. Use the external-dataset check to quantify this.
- Articles are truncated to the training length (default 256 tokens).

## Ideas for future work

- Evaluate on additional datasets (LIAR, FakeNewsNet) and report the generalization gap for both models.
- Add token-level explanations (e.g. SHAP or attention attribution) to the UI.
- Calibrate the probabilities (temperature scaling) and report expected calibration error.
- Publish the trained model on the Hugging Face Hub instead of shipping weights in the repo.
