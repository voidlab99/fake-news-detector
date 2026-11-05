"""TF-IDF + Logistic Regression baseline."""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

from data_utils import load_dataset, split


def main():
    df = load_dataset()
    train, _, test = split(df)
    print(f"Articles: {len(df):,}  (train {len(train):,} / test {len(test):,})")

    model = make_pipeline(
        TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
        LogisticRegression(max_iter=1000, C=4.0),
    )
    model.fit(train["text"], train["label"])
    preds = model.predict(test["text"])

    f1 = f1_score(test["label"], preds)
    print(classification_report(test["label"], preds, target_names=["real", "fake"]))
    print(f"TF-IDF baseline F1: {f1:.4f}")
    with open("baseline_metrics.json", "w") as f:
        json.dump({"f1": f1}, f, indent=2)


if __name__ == "__main__":
    main()
