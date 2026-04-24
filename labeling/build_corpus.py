"""Corpus builder — stratified sampling of CB sentences for NLP labeling."""

import random
from pathlib import Path

import pandas as pd


def build_corpus(output_path: Path, target_n: int = 2500) -> None:
    strata = {
        "fed_statement": 0.15, "fed_minutes": 0.10, "fed_speech": 0.10, "fed_presser": 0.05,
        "ecb_statement": 0.10, "ecb_presser": 0.05,
        "boe_minutes": 0.10, "boe_speech": 0.05,
        "boj_statement": 0.05,
        "boc_statement": 0.05,
        "other": 0.20,
    }

    # Placeholder: in real use, queries cb_sentiment table
    sentences = [
        "placeholder sentence from CB statement corpus",
    ] * target_n

    rows = []
    for i, sent in enumerate(sentences):
        rows.append({
            "sentence_id": f"sent_{i:06d}",
            "doc_id": f"doc_{i // 10:04d}",
            "sentence": sent,
            "cb": random.choice(["fed", "ecb", "boe", "boj", "boc"]),
            "doc_type": random.choice(["statement", "minutes", "speech"]),
            "date": f"202{random.randint(0,6)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "label": "",
            "labeler": "",
            "confidence": "",
            "notes": "",
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Corpus written: {len(df)} sentences → {output_path}")


if __name__ == "__main__":
    build_corpus(Path("labeling/corpus_v1.csv"))
