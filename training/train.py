"""FinBERT fine-tuning — WeightedLossTrainer, per-class F1, early stopping."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments,
)
from sklearn.utils.class_weight import compute_class_weight

from .data import load_and_split

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_dovish": f1_score(labels, preds, labels=[0], average="macro"),
        "f1_neutral": f1_score(labels, preds, labels=[1], average="macro"),
        "f1_hawkish": f1_score(labels, preds, labels=[2], average="macro"),
    }


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main(model_name: str = "ProsusAI/finbert", output_dir: Path = Path("models/cb-sentiment-v1")) -> None:
    datasets = load_and_split()
    if len(datasets["train"]) == 0:
        logger.warning("No training data — skipping fine-tuning")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=256)

    tokenized = datasets.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_labels = np.array(tokenized["train"]["labels"])
    class_weights = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_labels), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=4,
        per_device_train_batch_size=8, per_device_eval_batch_size=16,
        learning_rate=2e-5, weight_decay=0.01, lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(), seed=42,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights, model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    if len(tokenized["test"]) > 0:
        test_result = trainer.predict(tokenized["test"])
        metrics = compute_metrics((test_result.predictions, test_result.label_ids))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info("Test F1 macro: %.3f", metrics["f1_macro"])

    logger.info("Model saved to %s/final", output_dir)
