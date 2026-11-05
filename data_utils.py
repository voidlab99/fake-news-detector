"""Shared data loading / cleaning for the baseline and the DistilBERT model.

Expects the ISOT-style layout (two CSVs with `title` and `text` columns):
    data/Fake.csv
    data/True.csv
Any dataset works as long as you end up with columns: text, label (1 = fake, 0 = real).
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
SEED = 42

# Publisher "tells" that let a model cheat instead of learning style/content.
LEAK_PATTERNS = [
    r"^.{0,60}\(Reuters\)\s*-\s*",  # "WASHINGTON (Reuters) - "
    r"\b(Reuters|21st Century Wire|Featured image via[^.]*)\b",
    r"https?://\S+",
]


def clean(text: str) -> str:
    text = str(text)
    for pat in LEAK_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset() -> pd.DataFrame:
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    real = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"], real["label"] = 1, 0
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean)
    df = df[df["text"].str.len() > 50]
    df = df.drop_duplicates(subset="text")  # duplicates across splits inflate scores
    return df[["text", "label"]].reset_index(drop=True)


def split(df: pd.DataFrame):
    """80 / 10 / 10 stratified split, same for every model so results are comparable."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    return train, val, test
