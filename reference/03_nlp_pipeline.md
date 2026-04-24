# 03 — NLP Pipeline

Central bank statement scraping, preprocessing, sentiment scoring, and diff analysis.

## Document Dataclass

**Location:** `src/nlp/scrapers/base.py`
**Purpose:** Common document representation across all CB sources.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
import json
import logging

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class Document:
    cb: str                    # 'fed', 'ecb', 'boe', etc.
    doc_type: str              # 'statement', 'minutes', 'speech', 'presser'
    title: str
    date: datetime
    url: str
    speaker: str | None = None
    raw_html: str = ""
    raw_text: str = ""
    metadata: dict = field(default_factory=dict)
    
    @property
    def doc_id(self) -> str:
        h = hashlib.sha256(f"{self.cb}:{self.url}".encode()).hexdigest()[:16]
        return f"{self.cb}_{self.doc_type}_{self.date.strftime('%Y%m%d')}_{h}"


class CBScraper(ABC):
    def __init__(self, raw_dir: Path, timeout: int = 30):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (research scraper)'},
            follow_redirects=True,
        )
    
    @abstractmethod
    def list_documents(self, since: datetime) -> list[dict]:
        """Return list of {url, date, doc_type, title, speaker?} dicts."""
        pass
    
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(min=2, max=30))
    def fetch_url(self, url: str) -> str:
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.text
    
    @abstractmethod
    def parse_document(self, html: str, meta: dict) -> Document:
        pass
    
    def save_document(self, doc: Document) -> Path:
        path = self.raw_dir / f"{doc.doc_id}.json"
        path.write_text(json.dumps({
            'cb': doc.cb,
            'doc_type': doc.doc_type,
            'title': doc.title,
            'date': doc.date.isoformat(),
            'url': doc.url,
            'speaker': doc.speaker,
            'raw_text': doc.raw_text,
            'metadata': doc.metadata,
        }, indent=2))
        return path
    
    def run(self, since: datetime) -> list[Document]:
        docs = []
        for meta in self.list_documents(since):
            doc_id_stem = f"{self.cb_name}_{meta['doc_type']}_{meta['date'].strftime('%Y%m%d')}"
            existing = list(self.raw_dir.glob(f"{doc_id_stem}_*.json"))
            if existing:
                logger.debug(f"Skipping already-fetched {doc_id_stem}")
                continue
            try:
                html = self.fetch_url(meta['url'])
                doc = self.parse_document(html, meta)
                self.save_document(doc)
                docs.append(doc)
                logger.info(f"Saved {doc.doc_id}")
            except Exception as e:
                logger.error(f"Failed {meta['url']}: {e}")
        return docs
```

## Fed Statement Scraper

**Location:** `src/nlp/scrapers/fed.py`
**Purpose:** Concrete scraper for FOMC statements.

```python
from datetime import datetime
import re
from bs4 import BeautifulSoup

from src.nlp.scrapers.base import CBScraper, Document


class FedStatementScraper(CBScraper):
    cb_name = 'fed'
    CALENDAR_URL = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
    
    def list_documents(self, since: datetime) -> list[dict]:
        html = self.fetch_url(self.CALENDAR_URL)
        soup = BeautifulSoup(html, 'html.parser')
        docs = []
        
        for link in soup.find_all('a', href=re.compile(r'monetary\d+a\.htm')):
            href = link.get('href', '')
            m = re.search(r'monetary(\d{8})a\.htm', href)
            if not m:
                continue
            date = datetime.strptime(m.group(1), '%Y%m%d')
            if date < since:
                continue
            docs.append({
                'url': f'https://www.federalreserve.gov{href}' 
                       if href.startswith('/') else href,
                'date': date,
                'doc_type': 'statement',
                'title': f'FOMC Statement {date.strftime("%Y-%m-%d")}',
            })
        return docs
    
    def parse_document(self, html: str, meta: dict) -> Document:
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('div', id='article') or \
                  soup.find('div', class_='col-md-8')
        if content is None:
            content = soup.find('body')
        
        for tag in content.find_all(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        paragraphs = [p.get_text(strip=True) for p in content.find_all('p') 
                      if p.get_text(strip=True)]
        paragraphs = [p for p in paragraphs 
                      if not p.startswith('For release') 
                      and 'Implementation Note' not in p]
        text = '\n\n'.join(paragraphs)
        
        return Document(
            cb='fed',
            doc_type='statement',
            title=meta['title'],
            date=meta['date'],
            url=meta['url'],
            raw_html=html,
            raw_text=text,
        )
```

## Text Preprocessor

**Location:** `src/nlp/preprocessing.py`
**Purpose:** Clean raw CB text and segment into sentences/paragraphs.

```python
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessedDocument:
    doc_id: str
    cb: str
    doc_type: str
    date: datetime
    sentences: list[str]
    paragraphs: list[str]
    word_count: int
    metadata: dict


class TextPreprocessor:
    BOILERPLATE_PATTERNS = [
        r'For release at.*?ET',
        r'Implementation Note issued.*?$',
        r'Last Update:.*?$',
        r'^\s*\d+\s*$',
        r'[A-Z][A-Z\s]{20,}',
    ]
    
    def __init__(self):
        import spacy
        self.nlp = spacy.load('en_core_web_sm', 
                              disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe('sentencizer')
    
    def clean(self, text: str) -> str:
        for pat in self.BOILERPLATE_PATTERNS:
            text = re.sub(pat, '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def segment(self, text: str) -> tuple[list[str], list[str]]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = []
        for para in paragraphs:
            doc = self.nlp(para)
            sentences.extend([s.text.strip() for s in doc.sents 
                            if len(s.text.strip()) > 10])
        return sentences, paragraphs
    
    def process(self, doc) -> ProcessedDocument:
        cleaned = self.clean(doc.raw_text)
        sentences, paragraphs = self.segment(cleaned)
        return ProcessedDocument(
            doc_id=doc.doc_id,
            cb=doc.cb,
            doc_type=doc.doc_type,
            date=doc.date,
            sentences=sentences,
            paragraphs=paragraphs,
            word_count=sum(len(s.split()) for s in sentences),
            metadata=doc.metadata,
        )
```

## Hawkish/Dovish Lexicons

**Location:** `src/nlp/lexicons.py`
**Purpose:** Domain-specific word lists for CB sentiment scoring.

```python
HAWKISH_TERMS = {
    # Strong hawkish
    'additional firming', 'further tightening', 'more restrictive', 
    'higher for longer', 'committed to returning', 'combat inflation',
    'firmly anchored', 'determined', 'vigilant', 'forceful',
    
    # Moderate hawkish  
    'inflation remains elevated', 'persistent', 'stubborn', 'sticky',
    'upside risks', 'tight labor', 'wage pressures', 'robust',
    'strong demand', 'overheating',
    
    # Policy direction
    'raise', 'increase', 'hike', 'tightening', 'restrictive stance',
    'above neutral', 'sufficiently restrictive',
}

DOVISH_TERMS = {
    # Strong dovish
    'rate cut', 'easing', 'accommodate', 'support growth', 
    'downside risks', 'economic slack', 'disinflation',
    'inflation has moderated', 'labor market has cooled',
    
    # Moderate dovish
    'patient', 'gradual', 'careful', 'monitor', 'data-dependent',
    'balanced', 'cumulative', 'lags of monetary policy', 
    'transmission', 'softened', 'eased',
    
    # Policy direction
    'lower', 'reduce', 'cut', 'normalize', 'toward neutral',
    'less restrictive',
}

UNCERTAINTY_TERMS = {
    'uncertain', 'uncertainty', 'risks', 'elevated uncertainty',
    'difficult to assess', 'highly uncertain', 'could', 'might',
    'perhaps', 'possibly',
}

INTENSIFIERS = {
    'strongly', 'firmly', 'significantly', 'substantially',
    'materially', 'notably', 'considerably',
}

HEDGES = {
    'somewhat', 'modestly', 'slightly', 'marginally', 'to some degree',
    'appears', 'seems',
}
```

## Lexicon Scorer

**Location:** `src/nlp/lexicon_scorer.py`
**Purpose:** Fast, interpretable keyword-based hawkish/dovish scoring.

```python
import re
from dataclasses import dataclass

from src.nlp.lexicons import (HAWKISH_TERMS, DOVISH_TERMS, UNCERTAINTY_TERMS,
                                 INTENSIFIERS, HEDGES)


@dataclass
class LexiconScores:
    hawkish_count: int
    dovish_count: int
    uncertainty_count: int
    intensifier_count: int
    hedge_count: int
    net_score: float      # (hawkish - dovish) / (hawkish + dovish + 1)
    intensity: float      # (intensifiers - hedges) / word_count


class LexiconScorer:
    def __init__(self, hawkish=None, dovish=None, uncertainty=None,
                 intensifiers=None, hedges=None):
        self.hawkish = hawkish or HAWKISH_TERMS
        self.dovish = dovish or DOVISH_TERMS
        self.uncertainty = uncertainty or UNCERTAINTY_TERMS
        self.intensifiers = intensifiers or INTENSIFIERS
        self.hedges = hedges or HEDGES
        self._hawkish_re = self._compile(self.hawkish)
        self._dovish_re = self._compile(self.dovish)
        self._uncertainty_re = self._compile(self.uncertainty)
        self._intensifier_re = self._compile(self.intensifiers)
        self._hedge_re = self._compile(self.hedges)
    
    def _compile(self, terms):
        pattern = r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def score_text(self, text: str) -> LexiconScores:
        words = text.split()
        wc = max(len(words), 1)
        h = len(self._hawkish_re.findall(text))
        d = len(self._dovish_re.findall(text))
        u = len(self._uncertainty_re.findall(text))
        i = len(self._intensifier_re.findall(text))
        hg = len(self._hedge_re.findall(text))
        
        net = (h - d) / (h + d + 1)
        intensity = (i - hg) / wc
        
        return LexiconScores(h, d, u, i, hg, net, intensity)
    
    def score_sentences(self, sentences: list[str]) -> list[LexiconScores]:
        return [self.score_text(s) for s in sentences]
```

## Transformer Scorer (FinBERT)

**Location:** `src/nlp/transformer_scorer.py`
**Purpose:** ML-based sentence classification, fine-tuned from FinBERT.

```python
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)


class CBSentenceDataset(Dataset):
    def __init__(self, sentences: list[str], labels: list[int], 
                 tokenizer, max_length: int = 256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerScorer:
    LABELS = {0: 'dovish', 1: 'neutral', 2: 'hawkish'}
    
    def __init__(self, model_path: str = 'ProsusAI/finbert', 
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
        ).to(self.device)
        self.model.eval()
    
    def fine_tune(self, train_sentences, train_labels, val_sentences, val_labels,
                   output_dir: str, epochs: int = 4):
        train_ds = CBSentenceDataset(train_sentences, train_labels, 
                                      self.tokenizer)
        val_ds = CBSentenceDataset(val_sentences, val_labels, self.tokenizer)
        
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
        )
        
        def compute_metrics(pred):
            from sklearn.metrics import f1_score, accuracy_score
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                'f1_macro': f1_score(labels, preds, average='macro'),
                'accuracy': accuracy_score(labels, preds),
            }
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
    
    @torch.no_grad()
    def score_sentences(self, sentences: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=256, return_tensors='pt').to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                results.append({
                    'dovish': float(p[0]),
                    'neutral': float(p[1]),
                    'hawkish': float(p[2]),
                    'hawkish_score': float(p[2] - p[0]),  # -1 to +1
                })
        return results
```

## Statement Differ

**Location:** `src/nlp/diff.py`
**Purpose:** Diff consecutive statements to extract hawkish/dovish shifts — the key trading signal.

```python
from dataclasses import dataclass
import difflib


@dataclass
class StatementDiff:
    added_sentences: list[str]
    removed_sentences: list[str]
    modified_pairs: list[tuple[str, str]]
    added_hawkish_score: float
    removed_hawkish_score: float
    net_shift: float
    raw_diff_ratio: float


class StatementDiffer:
    def __init__(self, lexicon_scorer,
                 transformer_scorer=None,
                 similarity_threshold: float = 0.6):
        self.lex = lexicon_scorer
        self.tfm = transformer_scorer
        self.sim_threshold = similarity_threshold
    
    def diff(self, current: list[str], previous: list[str]) -> StatementDiff:
        matcher = difflib.SequenceMatcher(None, previous, current)
        added, removed, modified = [], [], []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                added.extend(current[j1:j2])
            elif tag == 'delete':
                removed.extend(previous[i1:i2])
            elif tag == 'replace':
                for old in previous[i1:i2]:
                    best_match, best_ratio = None, 0
                    for new in current[j1:j2]:
                        r = difflib.SequenceMatcher(None, old, new).ratio()
                        if r > best_ratio and r > self.sim_threshold:
                            best_match, best_ratio = new, r
                    if best_match:
                        modified.append((old, best_match))
                    else:
                        removed.append(old)
                matched_new = {m[1] for m in modified}
                for new in current[j1:j2]:
                    if new not in matched_new:
                        added.append(new)
        
        added_text = ' '.join(added)
        removed_text = ' '.join(removed)
        added_score = self.lex.score_text(added_text).net_score if added else 0
        removed_score = self.lex.score_text(removed_text).net_score if removed else 0
        
        net_shift = added_score - removed_score
        
        if self.tfm and modified:
            old_scores = self.tfm.score_sentences([p[0] for p in modified])
            new_scores = self.tfm.score_sentences([p[1] for p in modified])
            pair_shifts = [n['hawkish_score'] - o['hawkish_score']
                          for o, n in zip(old_scores, new_scores)]
            if pair_shifts:
                net_shift += sum(pair_shifts) / len(pair_shifts)
        
        raw_ratio = matcher.ratio()
        return StatementDiff(
            added, removed, modified,
            added_score, removed_score, net_shift,
            raw_ratio,
        )
```

## Inference Service

**Location:** `src/nlp/inference.py`
**Purpose:** Production inference wrapper around the fine-tuned model.

```python
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CBSentimentModel:
    def __init__(self, model_path: Path, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(self.device)
        self.model.eval()
        self.id2label = {0: 'dovish', 1: 'neutral', 2: 'hawkish'}
    
    @torch.no_grad()
    def predict(self, sentences: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors='pt'
            ).to(self.device)
            
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            
            for p, pred in zip(probs, preds):
                results.append({
                    'prediction': self.id2label[int(pred)],
                    'confidence': float(p[pred]),
                    'probs': {
                        'dovish': float(p[0]),
                        'neutral': float(p[1]),
                        'hawkish': float(p[2]),
                    },
                    'hawkish_score': float(p[2] - p[0]),
                })
        return results
    
    @torch.no_grad()
    def predict_document(self, sentences: list[str]) -> dict:
        """Aggregate sentence-level predictions to document-level."""
        results = self.predict(sentences)
        
        scores = [r['hawkish_score'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        total_weight = sum(confidences)
        weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight
        
        pred_counts = {'dovish': 0, 'neutral': 0, 'hawkish': 0}
        for r in results:
            pred_counts[r['prediction']] += 1
        
        non_neutral = [(s, r) for s, r in zip(sentences, results) 
                        if r['prediction'] != 'neutral']
        non_neutral.sort(key=lambda x: x[1]['confidence'], reverse=True)
        top_signals = [{'sentence': s, **r} for s, r in non_neutral[:5]]
        
        return {
            'hawkish_score': weighted_score,
            'sentence_count': len(sentences),
            'pred_distribution': pred_counts,
            'top_signals': top_signals,
            'sentence_predictions': results,
        }
```

## Temperature Calibration

**Location:** `training/calibrate.py`
**Purpose:** Fit temperature scaling to make confidence scores honest.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(model, tokenizer, val_dataset, device):
    """Fit temperature parameter on validation set."""
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_dataset:
            inputs = {k: v.unsqueeze(0).to(device) 
                      for k, v in batch.items() if k != 'labels'}
            logits = model(**inputs).logits
            all_logits.append(logits.cpu())
            all_labels.append(batch['labels'].unsqueeze(0))
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(all_logits), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    return scaler.temperature.item()


def compute_ece(labels, confs, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / len(confs)) * abs(bin_acc - bin_conf)
    return ece
```

## Training Data Loader

**Location:** `training/data.py`
**Purpose:** Load and split labeled data for FinBERT fine-tuning.

```python
import json
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


LABEL_MAP = {'dovish': 0, 'neutral': 1, 'hawkish': 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class DataConfig:
    labels_path: Path = Path('labeling/labels.csv')
    corpus_path: Path = Path('labeling/corpus_v1.csv')
    output_dir: Path = Path('training/data')
    model_name: str = 'ProsusAI/finbert'
    max_length: int = 256
    train_pct: float = 0.7
    val_pct: float = 0.15
    min_confidence: str = 'low'


def load_and_split(config: DataConfig) -> DatasetDict:
    labels_df = pd.read_csv(config.labels_path)
    corpus_df = pd.read_csv(config.corpus_path)
    
    df = labels_df.merge(corpus_df, on='sentence_id')
    
    conf_order = {'low': 0, 'medium': 1, 'high': 2}
    min_conf = conf_order[config.min_confidence]
    df = df[df['confidence'].map(conf_order) >= min_conf]
    
    df['label'] = df['label'].astype(int)
    df['sentence'] = df['sentence'].astype(str)
    df = df[df['sentence'].str.len() > 20]
    
    # Document-level split (no sentence leakage) AND temporal
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    unique_docs = df[['doc_id', 'date']].drop_duplicates().sort_values('date')
    n_docs = len(unique_docs)
    train_cutoff = int(n_docs * config.train_pct)
    val_cutoff = int(n_docs * (config.train_pct + config.val_pct))
    
    train_docs = set(unique_docs['doc_id'].iloc[:train_cutoff])
    val_docs = set(unique_docs['doc_id'].iloc[train_cutoff:val_cutoff])
    test_docs = set(unique_docs['doc_id'].iloc[val_cutoff:])
    
    train_df = df[df['doc_id'].isin(train_docs)]
    val_df = df[df['doc_id'].isin(val_docs)]
    test_df = df[df['doc_id'].isin(test_docs)]
    
    def to_ds(d):
        return Dataset.from_pandas(
            d[['sentence', 'label', 'cb', 'doc_type']].reset_index(drop=True)
        )
    
    return DatasetDict({
        'train': to_ds(train_df),
        'validation': to_ds(val_df),
        'test': to_ds(test_df),
    })


def tokenize_dataset(dataset: DatasetDict, config: DataConfig) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    def tokenize(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized, tokenizer
```

## Training Script

**Location:** `training/train.py`
**Purpose:** Full FinBERT fine-tuning pipeline with weighted loss for class imbalance.

```python
import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from training.data import load_and_split, tokenize_dataset, DataConfig


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'f1_dovish': f1_score(labels, preds, labels=[0], average='macro'),
        'f1_neutral': f1_score(labels, preds, labels=[1], average='macro'),
        'f1_hawkish': f1_score(labels, preds, labels=[2], average='macro'),
        'precision_macro': precision_score(labels, preds, average='macro', 
                                             zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro',
                                       zero_division=0),
    }


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced data."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, 
                      num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, model.config.num_labels), 
                         labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    data_config = DataConfig()
    datasets = load_and_split(data_config)
    tokenized, tokenizer = tokenize_dataset(datasets, data_config)
    
    train_labels = np.array(tokenized['train']['labels'])
    class_weights_np = compute_class_weight(
        'balanced', classes=np.array([0, 1, 2]), y=train_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        data_config.model_name,
        num_labels=3,
        id2label={0: 'dovish', 1: 'neutral', 2: 'hawkish'},
        label2id={'dovish': 0, 'neutral': 1, 'hawkish': 2},
        ignore_mismatched_sizes=True,
    )
    
    output_dir = Path('models/cb-sentiment-v1')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to=['tensorboard'],
        run_name='cb-sentiment-finbert-v1',
        seed=42,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    train_result = trainer.train()
    val_metrics = trainer.evaluate(tokenized['validation'])
    
    test_results = trainer.predict(tokenized['test'])
    test_preds = test_results.predictions.argmax(axis=-1)
    test_labels = test_results.label_ids
    
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))


if __name__ == '__main__':
    main()
```

## NLP Data Provider

**Location:** `src/nlp/provider.py`
**Purpose:** Query interface for strategies to access NLP data.

```python
from datetime import datetime
import pandas as pd
from sqlalchemy import text


class NLPDataProvider:
    def __init__(self, engine):
        self.engine = engine
    
    def get_recent_diff_events(self, since: datetime, 
                                 cbs: list[str]) -> list[dict]:
        query = text("""
            SELECT ts, cb, doc_id, prev_doc_id, net_shift, 
                   added_hawkish, removed_hawkish, change_ratio
            FROM cb_diff_events
            WHERE ts >= :since AND cb = ANY(:cbs)
            ORDER BY ts DESC
        """)
        result = pd.read_sql(query, self.engine, 
                              params={'since': since, 'cbs': cbs})
        return result.to_dict('records')
    
    def get_historical_diff_scores(self, cbs: list[str], 
                                     lookback_years: int = 5) -> pd.DataFrame:
        since = datetime.utcnow() - pd.Timedelta(days=lookback_years * 365)
        query = text("""
            SELECT ts, cb, net_shift FROM cb_diff_events
            WHERE ts >= :since AND cb = ANY(:cbs)
        """)
        return pd.read_sql(query, self.engine, 
                            params={'since': since, 'cbs': cbs})
```

## NLP Service API

**Location:** `src/nlp/service.py`
**Purpose:** FastAPI service to serve model inference locally.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.nlp.inference import CBSentimentModel

app = FastAPI()

model = None


class ScoreRequest(BaseModel):
    sentences: list[str]


@app.on_event('startup')
def load_model():
    global model
    model = CBSentimentModel(model_path='/opt/fx-system/models/cb-sentiment-v1/final')


@app.post('/score')
def score(req: ScoreRequest):
    return model.predict(req.sentences)


@app.post('/score_document')
def score_document(req: ScoreRequest):
    return model.predict_document(req.sentences)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}
```
