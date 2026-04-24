"""FinBERT training data pipeline — temporal-document split, tokenization."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


LABEL_MAP = {"dovish": 0, "neutral": 1, "hawkish": 2}


def load_and_split(
    labels_path: Path = Path("labeling/labels.csv"),
    corpus_path: Path = Path("labeling/corpus_v1.csv"),
    train_pct: float = 0.7,
    val_pct: float = 0.15,
) -> DatasetDict:
    if not labels_path.exists():
        return DatasetDict({"train": Dataset.from_list([]), "validation": Dataset.from_list([]), "test": Dataset.from_list([])})

    labels_df = pd.read_csv(labels_path)
    corpus_df = pd.read_csv(corpus_path)
    df = labels_df.merge(corpus_df, on="sentence_id")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    df = df[df["sentence"].astype(str).str.len() > 20]

    # Temporal-document split
    df = df.sort_values("date") if "date" in df.columns else df.sample(frac=1, random_state=42)
    unique_docs = df["doc_id"].unique() if "doc_id" in df.columns else df.index.unique()
    n_docs = len(unique_docs)
    train_docs = set(unique_docs[: int(n_docs * train_pct)])
    val_docs = set(unique_docs[int(n_docs * train_pct) : int(n_docs * (train_pct + val_pct))])
    test_docs = set(unique_docs[int(n_docs * (train_pct + val_pct)) :])

    def to_ds(subset):
        return Dataset.from_pandas(subset[["sentence", "label"]].reset_index(drop=True))

    return DatasetDict({
        "train": to_ds(df[df["doc_id"].isin(train_docs)] if "doc_id" in df.columns else df.iloc[: int(n_docs * train_pct)]),
        "validation": to_ds(df[df["doc_id"].isin(val_docs)] if "doc_id" in df.columns else df.iloc[int(n_docs * train_pct) : int(n_docs * (train_pct + val_pct))]),
        "test": to_ds(df[df["doc_id"].isin(test_docs)] if "doc_id" in df.columns else df.iloc[int(n_docs * (train_pct + val_pct)) :]),
    })
