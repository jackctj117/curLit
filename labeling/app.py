"""Streamlit labeling app for CB sentence classification."""

import streamlit as st
import pandas as pd
from pathlib import Path

CORPUS_PATH = Path("labeling/corpus_v1.csv")
LABELS_PATH = Path("labeling/labels.csv")

st.set_page_config(page_title="CB Statement Labeler", layout="wide")


@st.cache_data
def load_corpus() -> pd.DataFrame:
    if CORPUS_PATH.exists():
        return pd.read_csv(CORPUS_PATH)
    return pd.DataFrame()


def load_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=["sentence_id", "label", "confidence", "notes"])


def save_label(sentence_id: str, label: int, confidence: str, notes: str) -> None:
    labels = load_labels()
    labels = labels[labels["sentence_id"] != sentence_id]
    new = pd.DataFrame([{"sentence_id": sentence_id, "label": label,
                          "confidence": confidence, "notes": notes}])
    labels = pd.concat([labels, new], ignore_index=True)
    labels.to_csv(LABELS_PATH, index=False)


def main() -> None:
    st.title("Central Bank Statement Labeler")

    corpus = load_corpus()
    labels = load_labels()

    if corpus.empty:
        st.warning("No corpus found. Run build_corpus.py first.")
        return

    labeled_ids = set(labels["sentence_id"])
    unlabeled = corpus[~corpus["sentence_id"].isin(labeled_ids)]
    st.metric("Progress", f"{len(labels)} / {len(corpus)}")

    if unlabeled.empty:
        st.success("All sentences labeled!")
        return

    idx = st.session_state.get("idx", 0) % len(unlabeled)
    row = unlabeled.iloc[idx]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{idx + 1} / {len(unlabeled)}**")
        st.markdown(f"*{row.get('cb', '').upper()} — {row.get('doc_type', '')} — {row.get('date', '')}*")
        st.markdown("---")
        st.markdown(f"## {row['sentence']}")
        st.markdown("---")

    with col2:
        st.markdown("**Guidelines**")
        st.caption("- Hawkish → tighter policy")
        st.caption("- Dovish → easier policy")
        st.caption("- Neutral → descriptive/balanced")

    confidence = st.select_slider("Confidence", ["low", "medium", "high"], value="medium")
    notes = st.text_input("Notes", "")

    c1, c2, c3, c4 = st.columns(4)

    def record(label: int) -> None:
        save_label(str(row["sentence_id"]), label, confidence, notes)
        st.session_state["idx"] = idx + 1
        st.rerun()

    with c1:
        st.button("Dovish (0)", on_click=record, args=(0,), use_container_width=True)
    with c2:
        st.button("Neutral (1)", on_click=record, args=(1,), use_container_width=True)
    with c3:
        st.button("Hawkish (2)", on_click=record, args=(2,), use_container_width=True)
    with c4:
        st.button("Skip", on_click=lambda: st.session_state.update({"idx": idx + 1}) or st.rerun(),
                  use_container_width=True)


if __name__ == "__main__":
    main()
