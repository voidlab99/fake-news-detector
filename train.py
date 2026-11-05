"""Fine-tune DistilBERT for fake-news classification.

    python train.py                 # defaults: 3 epochs, max_len 256
    python train.py --epochs 2 --max_len 128 --batch_size 32
"""
import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data_utils import SEED, load_dataset, split

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "model"
ID2LABEL = {0: "REAL", 1: "FAKE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    set_seed(SEED)
    df = load_dataset()
    train_df, val_df, test_df = split(df)
    print(f"Articles: {len(df):,}  (train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,})")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    def to_ds(d):
        return Dataset.from_pandas(d, preserve_index=False).map(tokenize, batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = map(to_ds, (train_df, val_df, test_df))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    targs = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    trainer.save_model(OUT_DIR)
    tok.model_max_length = args.max_len  # saved with the tokenizer so evaluate.py / the API truncate identically
    tok.save_pretrained(OUT_DIR)
    with open("distilbert_metrics.json", "w") as f:
        json.dump({**test_metrics, "max_len": args.max_len}, f, indent=2)
    print(f"Saved model to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
