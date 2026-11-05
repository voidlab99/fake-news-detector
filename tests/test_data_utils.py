import pandas as pd
import pytest

import data_utils
from data_utils import clean, load_dataset, split


def test_clean_removes_publisher_tells_and_urls():
    raw = "WASHINGTON (Reuters) - The senate met today. Read more at https://example.com/story now."
    out = clean(raw)
    assert "Reuters" not in out
    assert "http" not in out
    assert "The senate met today." in out


def test_clean_collapses_whitespace():
    assert clean("a   b\n\n c\t d") == "a b c d"


@pytest.fixture
def csv_dir(tmp_path, monkeypatch):
    body = "word " * 40
    fake = pd.DataFrame({"title": [f"fake {i}" for i in range(60)], "text": [f"{body}{i}" for i in range(60)]})
    real = pd.DataFrame({"title": [f"real {i}" for i in range(60)], "text": [f"{body}x{i}" for i in range(60)]})
    # a duplicate and a too-short row that load_dataset must drop
    fake = pd.concat([fake, fake.iloc[[0]], pd.DataFrame({"title": ["t"], "text": ["short"]})], ignore_index=True)
    fake.to_csv(tmp_path / "Fake.csv", index=False)
    real.to_csv(tmp_path / "True.csv", index=False)
    monkeypatch.setattr(data_utils, "DATA_DIR", tmp_path)
    return tmp_path


def test_load_dataset_labels_dedups_and_filters(csv_dir):
    df = load_dataset()
    assert set(df.columns) == {"text", "label"}
    assert len(df) == 120  # 60 fake + 60 real, duplicate and short row removed
    assert df["label"].value_counts().to_dict() == {1: 60, 0: 60}


def test_split_is_stratified_disjoint_and_deterministic(csv_dir):
    df = load_dataset()
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert not (set(train["text"]) & set(test["text"]))
    assert not (set(train["text"]) & set(val["text"]))
    assert not (set(val["text"]) & set(test["text"]))
    for part in (train, val, test):
        assert part["label"].mean() == pytest.approx(0.5, abs=0.1)
    train2, _, _ = split(df)
    assert train["text"].tolist() == train2["text"].tolist()
