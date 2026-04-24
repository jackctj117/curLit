# 13 — Research Workflow

Paper ingestion, relevance scoring, evaluation rubric, labeling tools.

## Paper Ingester

**Location:** `src/research/paper_ingester.py`
**Purpose:** Pull new academic papers from SSRN, NBER, arxiv into research database.

```python
import httpx
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import text
from feedparser import parse as parse_feed


logger = logging.getLogger(__name__)


class PaperIngester:
    NBER_FEED = 'https://www.nber.org/rss/new.xml'
    ARXIV_QF_FEED = 'http://arxiv.org/rss/q-fin'
    ARXIV_ECON_FEED = 'http://arxiv.org/rss/econ'
    SSRN_RECENT = 'https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=203'
    
    def __init__(self, engine):
        self.engine = engine
        self._ensure_schema()
    
    def _ensure_schema(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS research_papers (
                    paper_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors JSONB,
                    abstract TEXT,
                    url TEXT,
                    pdf_url TEXT,
                    published_date TIMESTAMPTZ,
                    ingested_at TIMESTAMPTZ NOT NULL,
                    keywords JSONB,
                    categories JSONB,
                    relevance_score NUMERIC DEFAULT 0,
                    read_status TEXT DEFAULT 'unread',
                    my_notes TEXT,
                    implementation_priority INT DEFAULT 0,
                    evaluation_data JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_papers_status 
                    ON research_papers(read_status, relevance_score DESC);
            """))
    
    def ingest_arxiv(self, days_back: int = 7) -> int:
        feeds = [
            ('arxiv-qf', self.ARXIV_QF_FEED),
            ('arxiv-econ', self.ARXIV_ECON_FEED),
        ]
        
        total = 0
        for source, url in feeds:
            try:
                feed = parse_feed(url)
                cutoff = datetime.utcnow() - timedelta(days=days_back)
                
                for entry in feed.entries:
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date < cutoff:
                        continue
                    
                    paper_id = entry.id.split('/abs/')[-1]
                    
                    with self.engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO research_papers 
                                (paper_id, source, title, authors, abstract, url, pdf_url,
                                 published_date, ingested_at)
                            VALUES (:pid, :src, :title, :authors, :abstract, :url, :pdf,
                                    :pub, :ing)
                            ON CONFLICT (paper_id) DO NOTHING
                        """), {
                            'pid': paper_id,
                            'src': source,
                            'title': entry.title,
                            'authors': json.dumps([a.get('name', '') 
                                                   for a in entry.get('authors', [])]),
                            'abstract': entry.summary,
                            'url': entry.link,
                            'pdf': entry.link.replace('/abs/', '/pdf/') + '.pdf',
                            'pub': pub_date,
                            'ing': datetime.utcnow(),
                        })
                    total += 1
                    
                logger.info(f"Ingested {len(feed.entries)} from {source}")
            except Exception as e:
                logger.error(f"Failed to ingest {source}: {e}")
        
        return total
    
    def ingest_nber(self, days_back: int = 14) -> int:
        try:
            feed = parse_feed(self.NBER_FEED)
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            count = 0
            
            for entry in feed.entries:
                if hasattr(entry, 'published_parsed'):
                    pub = datetime(*entry.published_parsed[:6])
                    if pub < cutoff:
                        continue
                else:
                    pub = datetime.utcnow()
                
                paper_id = f"nber-{entry.link.split('/')[-1].replace('.pdf', '')}"
                
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO research_papers 
                            (paper_id, source, title, abstract, url, pdf_url,
                             published_date, ingested_at)
                        VALUES (:pid, 'nber', :title, :abstract, :url, :pdf, :pub, :ing)
                        ON CONFLICT (paper_id) DO NOTHING
                    """), {
                        'pid': paper_id,
                        'title': entry.title,
                        'abstract': entry.get('summary', ''),
                        'url': entry.link,
                        'pdf': entry.link,
                        'pub': pub,
                        'ing': datetime.utcnow(),
                    })
                count += 1
            
            return count
        except Exception as e:
            logger.error(f"NBER ingest failed: {e}")
            return 0
```

## Relevance Scorer

**Location:** `src/research/relevance.py`
**Purpose:** Score new papers against personal relevance criteria. Surfaces high-priority papers first.

```python
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class RelevanceCriteria:
    keywords_high: dict[str, float] = field(default_factory=dict)
    keywords_low: dict[str, float] = field(default_factory=dict)
    authors_high: dict[str, float] = field(default_factory=dict)
    categories_boost: dict[str, float] = field(default_factory=dict)
    sources_weight: dict[str, float] = field(default_factory=dict)


# Default criteria for FX systematic trading
DEFAULT_FX_CRITERIA = RelevanceCriteria(
    keywords_high={
        # Strategy-specific
        'currency carry': 3.0,
        'fx momentum': 3.0,
        'rate differential': 3.0,
        'central bank communication': 2.5,
        'fomc statement': 2.5,
        'monetary policy surprise': 2.5,
        'cot positioning': 2.0,
        'speculative positioning': 2.0,
        # Methodological
        'walk-forward': 2.0,
        'regime detection': 2.0,
        'risk parity': 2.0,
        'factor model': 1.5,
        # FX general
        'foreign exchange': 1.5,
        'currency risk': 1.5,
        'g10 currencies': 2.0,
        'exchange rate': 1.0,
    },
    keywords_low={
        'high frequency': -1.0,
        'cryptocurrency': -2.0,
        'bitcoin': -2.0,
        'theoretical': -0.5,
        'pure theory': -1.5,
    },
    authors_high={
        'cliff asness': 3.0,
        'campbell harvey': 3.0,
        'lasse pedersen': 3.0,
        'tobias moskowitz': 2.5,
        'mark taylor': 2.5,
        'lukas menkhoff': 3.0,
        'lucio sarno': 3.0,
    },
    categories_boost={
        'q-fin.PM': 2.0,    # Portfolio management
        'q-fin.TR': 1.5,    # Trading and microstructure
        'q-fin.ST': 1.0,    # Statistical finance
        'q-fin.RM': 1.5,    # Risk management
        'econ.GN': 0.5,
    },
    sources_weight={
        'nber': 1.5,
        'ssrn': 1.2,
        'arxiv-qf': 1.0,
        'arxiv-econ': 0.8,
    },
)


class RelevanceScorer:
    def __init__(self, criteria: RelevanceCriteria = None):
        self.criteria = criteria or DEFAULT_FX_CRITERIA
    
    def score_paper(self, paper: dict) -> float:
        score = 0.0
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Keyword matching
        for keyword, weight in self.criteria.keywords_high.items():
            count = len(re.findall(re.escape(keyword), text))
            score += count * weight
        
        for keyword, weight in self.criteria.keywords_low.items():
            count = len(re.findall(re.escape(keyword), text))
            score += count * weight
        
        # Author matching
        authors_str = ' '.join(paper.get('authors', [])).lower() \
            if isinstance(paper.get('authors'), list) else str(paper.get('authors', '')).lower()
        for author, weight in self.criteria.authors_high.items():
            if author in authors_str:
                score += weight
        
        # Category matching
        categories = paper.get('categories', []) or []
        for cat in categories:
            score += self.criteria.categories_boost.get(cat, 0)
        
        # Source weight
        source = paper.get('source', '')
        score *= self.criteria.sources_weight.get(source, 1.0)
        
        return score
    
    def score_and_update_database(self, engine):
        from sqlalchemy import text
        with engine.connect() as conn:
            unscored = conn.execute(text("""
                SELECT paper_id, title, abstract, authors, categories, source
                FROM research_papers WHERE relevance_score = 0
            """)).fetchall()
        
        for row in unscored:
            paper = {
                'title': row.title,
                'abstract': row.abstract or '',
                'authors': row.authors or [],
                'categories': row.categories or [],
                'source': row.source,
            }
            score = self.score_paper(paper)
            
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE research_papers SET relevance_score = :s
                    WHERE paper_id = :pid
                """), {'s': score, 'pid': row.paper_id})
        
        logger.info(f"Scored {len(unscored)} papers")
```

## Paper Evaluation Rubric

**Location:** `src/research/evaluation.py`
**Purpose:** Structured 0-25 scale for evaluating whether to implement a paper's strategy.

```python
from dataclasses import dataclass


@dataclass
class PaperEvaluation:
    """
    Score each criterion 0-5; total max 25.
    Use a paper's evaluation_data JSONB column to store.
    """
    paper_id: str
    
    # Core criteria (each 0-5)
    economic_rationale: int = 0          # Does the why make sense?
    data_quality: int = 0                # Quality of data used
    implementation_feasibility: int = 0  # Can I actually do this?
    edge_persistence: int = 0            # Will this still work?
    diversification_value: int = 0       # Adds to portfolio?
    
    # Modifiers (negative)
    red_flags: int = 0                   # Suspicious things observed
    
    notes: str = ''
    decision: str = 'pending'  # 'implement', 'paper_track', 'reject'
    
    @property
    def total_score(self) -> int:
        return (self.economic_rationale + self.data_quality + 
                self.implementation_feasibility + self.edge_persistence +
                self.diversification_value - self.red_flags)


# Evaluation criteria details:
EVALUATION_GUIDE = """
Economic Rationale (0-5):
  5 - Clear economic mechanism, well-tested across multiple regimes
  3 - Plausible mechanism, some empirical support
  1 - Weak or unclear rationale; pattern matching without theory
  0 - No coherent rationale

Data Quality (0-5):
  5 - Free, accessible, vintage-aware data sources
  3 - Available data, some adjustments needed
  1 - Difficult or expensive data
  0 - Proprietary data unavailable to retail

Implementation Feasibility (0-5):
  5 - Can implement in days with existing stack
  3 - Requires moderate new infrastructure
  1 - Requires significant new infrastructure
  0 - Requires institutional infrastructure

Edge Persistence (0-5):
  5 - Effect documented for 20+ years, structural reason it persists
  3 - Documented for 10+ years, some evidence of persistence
  1 - Recent or unclear persistence
  0 - Likely arbitraged away

Diversification Value (0-5):
  5 - Strongly negative correlation to existing strategies
  3 - Low positive correlation
  1 - Moderate positive correlation
  0 - High positive correlation; redundant

Red Flags (subtract):
  - In-sample optimization without out-of-sample validation
  - Excessive parameter tuning
  - Survivorship bias in dataset
  - Look-ahead in backtest
  - Cherry-picked time periods
  - Reliance on hard-to-replicate broker behavior

Decisions:
  20+ → Implement
  15-19 → Paper-track for 90 days
  10-14 → Document but don't pursue
  <10 → Reject
"""
```

## Streamlit Labeling App

**Location:** `labeling/app.py`
**Purpose:** UI for labeling CB sentences for FinBERT training.

```python
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path


CORPUS_PATH = Path('labeling/corpus_v1.csv')
LABELS_PATH = Path('labeling/labels.csv')

LABEL_MAP = {0: 'dovish', 1: 'neutral', 2: 'hawkish'}
LABEL_COLORS = {0: '#3498db', 1: '#95a5a6', 2: '#e74c3c'}


def load_corpus():
    return pd.read_csv(CORPUS_PATH)


def load_labels():
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=[
        'sentence_id', 'label', 'confidence', 'notes', 'labeled_by', 'labeled_at'
    ])


def save_label(sentence_id, label, confidence, notes, labeler):
    labels_df = load_labels()
    new_row = pd.DataFrame([{
        'sentence_id': sentence_id,
        'label': label,
        'confidence': confidence,
        'notes': notes,
        'labeled_by': labeler,
        'labeled_at': datetime.utcnow().isoformat(),
    }])
    
    labels_df = labels_df[labels_df['sentence_id'] != sentence_id]
    labels_df = pd.concat([labels_df, new_row], ignore_index=True)
    labels_df.to_csv(LABELS_PATH, index=False)


def get_next_unlabeled(corpus_df, labels_df, labeler):
    labeled_ids = labels_df[labels_df['labeled_by'] == labeler]['sentence_id'].values
    unlabeled = corpus_df[~corpus_df['sentence_id'].isin(labeled_ids)]
    if len(unlabeled) == 0:
        return None
    return unlabeled.iloc[0]


def main():
    st.set_page_config(page_title='CB Sentiment Labeler', layout='wide')
    st.title('Central Bank Sentiment Labeling')
    
    if 'labeler' not in st.session_state:
        st.session_state.labeler = ''
    
    if not st.session_state.labeler:
        labeler = st.text_input('Your name (for tracking):')
        if labeler:
            st.session_state.labeler = labeler
            st.rerun()
        return
    
    corpus_df = load_corpus()
    labels_df = load_labels()
    
    my_labels = labels_df[labels_df['labeled_by'] == st.session_state.labeler]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Sentences', len(corpus_df))
    col2.metric('You Labeled', len(my_labels))
    col3.metric('Total Labeled', len(labels_df['sentence_id'].unique()))
    
    if len(my_labels) > 0:
        progress = len(my_labels) / len(corpus_df)
        col4.metric('Your Progress', f'{progress:.0%}')
    
    st.divider()
    
    sentence_row = get_next_unlabeled(corpus_df, labels_df, st.session_state.labeler)
    
    if sentence_row is None:
        st.success('All sentences labeled. Thank you!')
        return
    
    st.subheader(f"Sentence #{sentence_row['sentence_id']}")
    
    meta = st.columns(3)
    meta[0].caption(f"Source: **{sentence_row['cb'].upper()}**")
    meta[1].caption(f"Type: **{sentence_row['doc_type']}**")
    meta[2].caption(f"Date: **{sentence_row['date']}**")
    
    st.markdown(f"### {sentence_row['sentence']}")
    
    st.divider()
    st.markdown('**Label:**')
    
    cols = st.columns(3)
    with cols[0]:
        if st.button('🔵 Dovish', use_container_width=True, type='secondary'):
            st.session_state.pending_label = 0
    with cols[1]:
        if st.button('⚪ Neutral', use_container_width=True):
            st.session_state.pending_label = 1
    with cols[2]:
        if st.button('🔴 Hawkish', use_container_width=True, type='secondary'):
            st.session_state.pending_label = 2
    
    if 'pending_label' in st.session_state:
        confidence = st.select_slider(
            'Confidence', options=['low', 'medium', 'high'], value='medium'
        )
        notes = st.text_area('Notes (optional)', height=80)
        
        if st.button('Submit Label', type='primary'):
            save_label(
                sentence_id=sentence_row['sentence_id'],
                label=st.session_state.pending_label,
                confidence=confidence,
                notes=notes,
                labeler=st.session_state.labeler,
            )
            del st.session_state.pending_label
            st.rerun()


if __name__ == '__main__':
    main()
```

## Active Learning Selector

**Location:** `src/research/active_learning.py`
**Purpose:** Select the most informative sentences for labeling next, given current model.

```python
import numpy as np
from typing import Literal


def select_for_labeling(
    unlabeled_sentences: list[str],
    model,
    n: int = 50,
    strategy: Literal['uncertainty', 'margin', 'diverse'] = 'margin',
) -> list[int]:
    """
    Select the most informative sentences for labeling.
    """
    predictions = model.predict(unlabeled_sentences)
    
    if strategy == 'uncertainty':
        # Highest entropy
        entropies = []
        for p in predictions:
            probs = np.array([p['probs']['dovish'], p['probs']['neutral'], 
                             p['probs']['hawkish']])
            ent = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(ent)
        scores = entropies
    
    elif strategy == 'margin':
        # Smallest gap between top-2 predictions
        margins = []
        for p in predictions:
            probs = sorted([p['probs']['dovish'], p['probs']['neutral'], 
                           p['probs']['hawkish']], reverse=True)
            margins.append(probs[0] - probs[1])
        scores = [-m for m in margins]
    
    elif strategy == 'diverse':
        # Stratified by predicted class
        by_class = {0: [], 1: [], 2: []}
        for i, p in enumerate(predictions):
            class_idx = max(p['probs'].items(), key=lambda x: x[1])[0]
            class_id = {'dovish': 0, 'neutral': 1, 'hawkish': 2}[class_idx]
            by_class[class_id].append((i, p))
        
        selected = []
        per_class = n // 3
        for class_id, items in by_class.items():
            items.sort(key=lambda x: -x[1]['confidence'])
            uncertain = items[per_class:]
            np.random.shuffle(uncertain)
            selected.extend([i for i, _ in uncertain[:per_class]])
        return selected
    
    top_indices = np.argsort(scores)[-n:].tolist()
    return top_indices
```

## Research Dashboard

**Location:** `scripts/research_dashboard.py`
**Purpose:** Streamlit app showing pending paper queue, scored by relevance.

```python
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os


def main():
    st.set_page_config(page_title='Research Queue', layout='wide')
    st.title('FX Research Pipeline')
    
    db_url = os.environ.get('DATABASE_URL', 'postgresql://localhost/fx')
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT paper_id, source, title, authors, abstract, url,
                   relevance_score, read_status, my_notes, 
                   implementation_priority, ingested_at
            FROM research_papers
            ORDER BY relevance_score DESC, ingested_at DESC
        """), conn)
    
    st.metric('Total Papers', len(df))
    st.metric('Unread', (df['read_status'] == 'unread').sum())
    st.metric('Implementing', (df['read_status'] == 'implementing').sum())
    
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            'Status',
            options=df['read_status'].unique().tolist(),
            default=['unread']
        )
    with col2:
        min_score = st.slider('Min Relevance Score', 0.0, 20.0, 5.0)
    
    filtered = df[
        (df['read_status'].isin(status_filter)) & 
        (df['relevance_score'] >= min_score)
    ]
    
    st.write(f"Showing {len(filtered)} papers")
    
    for _, paper in filtered.head(20).iterrows():
        with st.expander(f"[{paper['relevance_score']:.1f}] {paper['title']}"):
            st.caption(f"Source: {paper['source']} | "
                       f"Authors: {paper['authors']} | "
                       f"Ingested: {paper['ingested_at']}")
            st.write(paper['abstract'])
            st.markdown(f"[Read full paper]({paper['url']})")
            
            new_status = st.selectbox(
                'Status', 
                ['unread', 'reading', 'evaluating', 'implementing', 'rejected'],
                index=['unread', 'reading', 'evaluating', 'implementing', 'rejected'].index(paper['read_status']),
                key=f"status_{paper['paper_id']}"
            )
            new_notes = st.text_area('Notes', value=paper['my_notes'] or '',
                                     key=f"notes_{paper['paper_id']}")
            
            if st.button('Update', key=f"update_{paper['paper_id']}"):
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE research_papers 
                        SET read_status = :s, my_notes = :n
                        WHERE paper_id = :pid
                    """), {'s': new_status, 'n': new_notes, 'pid': paper['paper_id']})
                st.success('Updated')
                st.rerun()


if __name__ == '__main__':
    main()
```

## Trading Plan Document Template

**Location:** `docs/trading_plan.md`
**Purpose:** Personal trading plan; this is the anchor document referenced during drawdowns.

```markdown
# FX Trading System — Trading Plan

## Strategies

For each strategy, document:
- Thesis (in your own words, not copied from a paper)
- Why it should work economically
- Conditions under which it would fail
- Expected Sharpe, drawdown, hit rate (from backtest)
- Position size and risk per trade
- Entry and exit rules
- When to retire

## Capital Allocation

- Total trading capital: $X
- Source of capital: [where this money came from]
- What happens if all is lost: [the honest answer]
- Position scaling rules: [when to scale up/down]

## Behavioral Commitments

- I will not modify strategy parameters during drawdowns
- I will paper trade for minimum 90 days before going live
- I will go live with 25% of intended size initially
- I will not check positions more than twice per day except during major events
- If I find myself wanting to override the system, I will wait 24 hours
- I will not add capital after a drawdown
- I will scale up only after sustained positive performance

## Performance Gates

- 6 months of paper trading: must hit X Sharpe to go live
- 3 months live tiny: must hit Y to scale to small
- 6 months live small: must hit Z to scale to intended

## Exit Plan

If at year 2 the system has not produced [specific outcome], 
I will:
- Wind down all positions
- Document lessons learned
- Reallocate capital to passive investments
- Not start a new trading project for at least 12 months

## Review Schedule

- Daily: 10-minute morning check
- Weekly: 1-hour deep review
- Monthly: 2-hour performance attribution
- Quarterly: 4-hour strategy review
- Annually: full system audit
```
