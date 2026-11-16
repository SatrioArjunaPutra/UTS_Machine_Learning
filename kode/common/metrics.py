"""Metrik yang dipakai grader + metrik bantu.

- Task 1: pinball loss (q=0.5, q=0.9) dan coverage P90.
- Task 2: average precision (PR-AUC) dan total cost.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

import config


def pinball(y, pred, q: float) -> float:
    """Rata-rata pinball loss: max(q*(y-pred), (q-1)*(y-pred))."""
    d = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, upper) -> float:
    """Proporsi y <= batas atas P90 (target 0.90)."""
    return float(np.mean(np.asarray(y) <= np.asarray(upper)))


def pr_auc(y, prob) -> float:
    return float(average_precision_score(y, prob))


def brier(y, prob) -> float:
    return float(brier_score_loss(y, prob))


def total_cost(y, flag, nominal, review_cost: int = config.REVIEW_COST) -> float:
    """cost = sum(nominal fraud yang lolos) + review_cost x jumlah flag."""
    y = np.asarray(y).astype(bool)
    flag = np.asarray(flag).astype(bool)
    nominal = np.asarray(nominal, dtype=float)
    return float(nominal[y & ~flag].sum() + review_cost * flag.sum())


def cost_rule_flag(prob, nominal, review_cost: int = config.REVIEW_COST, multiplier: float = 1.0):
    """Aturan Bayes: flag jika biaya harapan bila lolos (p x nominal) > biaya review.

    Flag   -> biaya pasti review_cost (fraud tertahan, kamus_data.md)
    Lolos  -> biaya harapan p x nominal
    """
    return (np.asarray(prob) * np.asarray(nominal, dtype=float) > multiplier * review_cost).astype(int)
