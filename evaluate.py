"""Evaluate the trained DistilBERT model beyond a single F1 number.

    python evaluate.py                          # held-out test split + error analysis
    python evaluate.py --external other.csv     # out-of-distribution check

The external CSV needs `text` and `label` columns (1 = fake, 0 = real), ideally from a
different source than the training data (e.g. LIAR, FakeNewsNet, or news you collect yourself).
A large drop from the in-distribution score shows how much the model learned publisher style
rather than anything that transfers.

Outputs go to ./results/:
    test_report.json / external_report.json   metrics + confusion matrix
    test_errors.csv                           misclassified articles, most confident mistakes first
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import clean, load_dataset, split

MODEL_DIR = "model"
OUT = Path("results")


def predict_fake_prob(texts, tok, model, device, batch_size=32):
    """P(FAKE) for each text, batched."""
    max_len = min(tok.model_max_length, 512)
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size], truncation=True, max_length=max_len, padding=True, return_tensors="pt"
            ).to(device)
            probs.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist())
    return probs


def summarize(name, df, p_fake):
    preds = [int(p >= 0.5) for p in p_fake]
    y = df["label"].tolist()
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    report = {
        "n": len(df),
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "confusion_matrix": {"true_real_pred_real": int(tn), "true_real_pred_fake": int(fp),
                             "true_fake_pred_real": int(fn), "true_fake_pred_fake": int(tp)},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n== {name} ({len(df):,} articles) ==")
    print(json.dumps(report, indent=2))
    return report, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", help="CSV with `text` and `label` columns from a different source")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # Same split as training (fixed seed), so this is the untouched test set.
    _, _, test = split(load_dataset())
    test = test.reset_index(drop=True)
    p_fake = predict_fake_prob(test["text"].tolist(), tok, model, device)
    in_dist, preds = summarize("test", test, p_fake)

    # Error analysis: which articles does the model get wrong, and how confidently?
    errors = test.assign(p_fake=p_fake, pred=preds)
    errors = errors[errors["pred"] != errors["label"]].copy()
    errors["confidence_in_wrong_answer"] = errors.apply(
        lambda r: r["p_fake"] if r["pred"] == 1 else 1 - r["p_fake"], axis=1
    )
    errors["text"] = errors["text"].str.slice(0, 400)
    errors = errors.sort_values("confidence_in_wrong_answer", ascending=False)
    errors.to_csv(OUT / "test_errors.csv", index=False)
    print(f"\nSaved {len(errors):,} misclassified articles to {OUT / 'test_errors.csv'}")

    if args.external:
        ext = pd.read_csv(args.external)
        ext = ext.dropna(subset=["text", "label"]).copy()
        ext["text"] = ext["text"].map(clean)
        ext["label"] = ext["label"].astype(int)
        ext_p = predict_fake_prob(ext["text"].tolist(), tok, model, device)
        ext_report, _ = summarize("external", ext, ext_p)
        drop = in_dist["f1"] - ext_report["f1"]
        print(f"\nF1 drop out-of-distribution: {drop:+.3f}  ({in_dist['f1']:.3f} -> {ext_report['f1']:.3f})")


if __name__ == "__main__":
    main()
