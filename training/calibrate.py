"""Temperature calibration — fit temperature parameter, reduce ECE."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float)
    labels_t = torch.tensor(labels, dtype=torch.long)

    scaler = TemperatureScaler()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = float(scaler.temperature.item())
    print(f"Optimal temperature: {temp:.3f}")

    before = torch.softmax(logits_t, dim=-1).numpy()
    after = torch.softmax(scaler(logits_t), dim=-1).detach().numpy()
    ece_before = _compute_ece(labels, before)
    ece_after = _compute_ece(labels, after)
    print(f"ECE before: {ece_before:.4f}, after: {ece_after:.4f}")

    torch.save(torch.tensor(temp), "models/cb-sentiment-v1/final/temperature.pt")
    return temp


def _compute_ece(labels, probs, n_bins: int = 10) -> float:
    confs = np.max(probs, axis=-1)
    preds = np.argmax(probs, axis=-1)
    accs = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(labels)) * abs(accs[mask].mean() - confs[mask].mean())
    return float(ece)
