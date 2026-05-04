"""Streamlit research dashboard for paper triage (CL-28j).

Operator-facing UI sitting in front of the ``research_papers`` table.
The pipeline upstream:

  PaperIngester (CL-5b6)  → research_papers rows
  RelevanceScorer (CL-366) → relevance_score column populated
  this dashboard           → operator triages: skim / read / discard / implement
  MetaLearner (CL-fsj)     → reads operator decisions back into
                             RelevanceScorer weight overlays

Design choices:

- One row = one paper. Queue view shows a sortable table; clicking a
  row opens the single-paper detail pane in the right column.
- Status buttons commit immediately (no save button) — research-flow
  velocity matters more than transactional atomicity for triage.
- Filters in the sidebar so they persist across reruns.
- The notes field is debounced (commits on blur, not on each keystroke)
  to avoid spamming Postgres on every character.

Run locally:
  streamlit run research/dashboard.py --server.port 8501

Run in compose: see deploy/Dockerfile + docker-compose.app.yml; this
file is mounted at /opt/curlit/research/dashboard.py and a `streamlit`
service can be added later (filed as a follow-up).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import bindparam, create_engine, text

logger = logging.getLogger(__name__)


# Status vocabulary — kept as constants here so the buttons + filters
# can't drift out of sync with the values written to the DB.
_STATUS_UNREAD: str = "unread"
_STATUS_SKIM_LATER: str = "skim_later"
_STATUS_READ: str = "read"
_STATUS_DISCARDED: str = "discarded"
_STATUS_FOR_IMPLEMENTATION: str = "for_implementation"

_ALL_STATUSES: tuple[str, ...] = (
    _STATUS_UNREAD, _STATUS_SKIM_LATER, _STATUS_READ,
    _STATUS_DISCARDED, _STATUS_FOR_IMPLEMENTATION,
)

# Implementation priorities. 0 = unranked. 1-5 are user-set on
# accept (1=top of queue, 5=backlog). Above 5 → no slot in our
# implementation pipeline so we cap at 5.
_PRIORITIES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


# --------------------------------------------------------------------- #
# DB helpers
# --------------------------------------------------------------------- #


@st.cache_resource
def _engine() -> Any:
    """Single engine reused across reruns. Streamlit's @cache_resource
    survives reruns within a session."""
    db_url = os.environ.get("DATABASE_URL") or (
        f"postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}"
    )
    return create_engine(db_url, pool_pre_ping=True)


def load_papers(
    statuses: list[str] | None = None,
    min_score: float = -10.0,
    sources: list[str] | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Pull paper rows for the queue view.

    Filters are pushed to SQL to avoid pulling the whole table client-
    side as the corpus grows. The triage index makes
    ``WHERE read_status IN (...) ORDER BY relevance_score DESC``
    O(rows-matched) — fast even with 50k papers.
    """
    statuses = statuses or list(_ALL_STATUSES)
    # Use bindparam(expanding=True) so the same SQL works across
    # Postgres (which has ANY) and sqlite (which doesn't). SQLAlchemy
    # expands the list into IN (:p1, :p2, …) at execute time.
    where = [
        "read_status IN :statuses",
        "relevance_score >= :min_score",
    ]
    params: dict[str, Any] = {
        "statuses": statuses, "min_score": min_score, "limit": limit,
    }
    if sources:
        where.append("source IN :sources")
        params["sources"] = sources
    sql = f"""
        SELECT paper_id, title, source, authors, abstract, url, pdf_url,
               published_date, ingested_at, relevance_score, read_status,
               my_notes, implementation_priority
        FROM research_papers
        WHERE {' AND '.join(where)}
        ORDER BY relevance_score DESC, COALESCE(published_date, ingested_at) DESC
        LIMIT :limit
    """
    stmt = text(sql).bindparams(
        bindparam("statuses", expanding=True),
        *([bindparam("sources", expanding=True)] if sources else []),
    )
    return pd.read_sql(stmt, _engine(), params=params)


def update_paper_status(paper_id: str, new_status: str) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE research_papers
                SET read_status = :s
                WHERE paper_id = :pid
            """),
            {"s": new_status, "pid": paper_id},
        )


def update_paper_priority(paper_id: str, priority: int) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE research_papers
                SET implementation_priority = :p
                WHERE paper_id = :pid
            """),
            {"p": priority, "pid": paper_id},
        )


def update_paper_notes(paper_id: str, notes: str) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE research_papers
                SET my_notes = :n
                WHERE paper_id = :pid
            """),
            {"n": notes, "pid": paper_id},
        )


def list_sources() -> list[str]:
    """Distinct paper sources for the sidebar source filter."""
    with _engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT source FROM research_papers "
            "WHERE source IS NOT NULL ORDER BY source",
        )).fetchall()
    return [r[0] for r in rows]


def _coerce_authors(raw: Any) -> str:
    """authors column is JSONB ([list of strings]) — render as comma-list."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            # Not JSON — treat as a single author name verbatim.
            return str(raw)
    if isinstance(raw, list):
        return ", ".join(str(a) for a in raw[:6]) + ("…" if len(raw) > 6 else "")
    return str(raw)


# --------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------- #


def _render_sidebar() -> dict[str, Any]:
    """Sidebar filters. Returns the filter dict the queue uses."""
    st.sidebar.header("Filters")

    statuses_selected = st.sidebar.multiselect(
        "Status",
        options=list(_ALL_STATUSES),
        default=[_STATUS_UNREAD],
        help="Triage queue starts with unread; toggle to revisit older decisions.",
    )

    min_score = st.sidebar.slider(
        "Min relevance score",
        min_value=-10.0, max_value=25.0, value=0.0, step=0.5,
        help=(
            "RelevanceScorer total = keyword + author + category. "
            "Score < 0 typically means red-flagged or off-topic; "
            "12-17 = discuss; >=18 = propose paper-mode strategy."
        ),
    )

    available_sources = list_sources()
    sources_selected = st.sidebar.multiselect(
        "Source", options=available_sources, default=available_sources or None,
    )

    limit = st.sidebar.number_input(
        "Max rows", min_value=10, max_value=2000, value=200, step=10,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Status changes commit on click. Notes commit on blur. "
        "Pipeline: PaperIngester → RelevanceScorer → here → MetaLearner.",
    )

    return {
        "statuses": statuses_selected,
        "min_score": min_score,
        "sources": sources_selected,
        "limit": int(limit),
    }


def _render_queue(df: pd.DataFrame) -> str | None:
    """Left column — the queue table. Returns selected paper_id or None."""
    if df.empty:
        st.info(
            "No papers match these filters. Try widening status or lowering "
            "the min-score slider.",
        )
        return None

    # Compact view columns. Authors come from JSONB; render as string.
    view = df.copy()
    view["authors_str"] = view["authors"].map(_coerce_authors)
    display = view[[
        "paper_id", "title", "source", "authors_str",
        "relevance_score", "read_status", "implementation_priority",
        "published_date",
    ]].rename(columns={
        "authors_str": "authors",
        "relevance_score": "score",
        "read_status": "status",
        "implementation_priority": "prio",
    })

    # Use st.dataframe with selection enabled (Streamlit ≥1.30).
    selection = st.dataframe(
        display,
        use_container_width=True,
        height=520,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "paper_id": st.column_config.TextColumn(width="small"),
            "title": st.column_config.TextColumn(width="large"),
            "score": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    rows = getattr(selection, "selection", {}).get("rows", [])
    if not rows:
        return None
    return str(display.iloc[rows[0]]["paper_id"])


def _render_detail(paper: pd.Series) -> None:
    """Right column — single-paper detail pane with action buttons."""
    st.subheader(paper["title"])
    st.caption(
        f"**{paper.get('source') or 'unknown'}** · "
        f"{_coerce_authors(paper.get('authors'))} · "
        f"score {float(paper.get('relevance_score') or 0):.2f}",
    )

    if paper.get("url"):
        col_url, col_pdf = st.columns(2)
        col_url.link_button("Open page", str(paper["url"]))
        if paper.get("pdf_url"):
            col_pdf.link_button("Open PDF", str(paper["pdf_url"]))

    st.markdown("**Abstract**")
    st.write(paper.get("abstract") or "_(no abstract)_")

    # Action buttons — five outcomes covering the queue workflow.
    st.markdown("---")
    st.markdown("**Decision**")
    cols = st.columns(5)
    actions = [
        ("Skim later", _STATUS_SKIM_LATER, cols[0]),
        ("Read now", _STATUS_READ, cols[1]),
        ("Discard", _STATUS_DISCARDED, cols[2]),
        ("For impl", _STATUS_FOR_IMPLEMENTATION, cols[3]),
        ("Reset", _STATUS_UNREAD, cols[4]),
    ]
    for label, target_status, col in actions:
        if col.button(label, key=f"act_{target_status}_{paper['paper_id']}"):
            update_paper_status(str(paper["paper_id"]), target_status)
            st.toast(f"Status → {target_status}")
            st.rerun()

    # Priority
    current_prio = int(paper.get("implementation_priority") or 0)
    new_prio = st.selectbox(
        "Implementation priority",
        options=list(_PRIORITIES),
        index=current_prio if current_prio in _PRIORITIES else 0,
        key=f"prio_{paper['paper_id']}",
        help="0=unranked, 1=top of queue, 5=backlog.",
    )
    if new_prio != current_prio:
        update_paper_priority(str(paper["paper_id"]), int(new_prio))
        st.toast(f"Priority → {new_prio}")

    # Notes — debounced via st.text_area; commits on blur (Streamlit reruns).
    notes = st.text_area(
        "Notes",
        value=paper.get("my_notes") or "",
        height=140,
        key=f"notes_{paper['paper_id']}",
        help=(
            "Why this paper got the status it got. The MetaLearner reads "
            "these into evaluation_data over time so don't be cryptic."
        ),
    )
    if notes != (paper.get("my_notes") or ""):
        update_paper_notes(str(paper["paper_id"]), notes)
        st.toast("Notes saved")


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="curLit research triage",
        page_icon=":books:",
        layout="wide",
    )
    st.title("curLit — research triage")
    st.caption(
        "Papers from arXiv / NBER / FRBSF / BIS / blogs, scored by "
        "RelevanceScorer (CL-366). Triage flow feeds MetaLearner (CL-fsj) "
        "back into source weights.",
    )

    filters = _render_sidebar()
    df = load_papers(**filters)

    # Two-column split: 0.55 queue / 0.45 detail. Wider detail when
    # a paper is selected (most read time goes there).
    col_queue, col_detail = st.columns([0.55, 0.45], gap="large")

    with col_queue:
        st.markdown(f"**{len(df)} papers** (newest + highest-score first)")
        selected_id = _render_queue(df)

    with col_detail:
        if selected_id is not None and not df.empty:
            paper = df.set_index("paper_id").loc[selected_id]
            _render_detail(paper)
        else:
            st.info("Select a paper from the queue to view + triage.")


if __name__ == "__main__":
    main()
