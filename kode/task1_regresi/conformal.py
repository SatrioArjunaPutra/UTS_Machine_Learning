"""Task 1: kalibrasi interval P90 dengan split conformal (skala log).

Skor non-konformitas s = log1p(y) - log1p(p90). Koreksi = kuantil ke
ceil((n+1)*0.9)/n dari s pada calibration set (jendela waktu terbaru,
terpisah dari data latih). P90 terkalibrasi = expm1(log1p(p90) + koreksi).
Skala log membuat koreksi bersifat multiplikatif, cocok untuk durasi yang miring.
Varian Mondrian menghitung koreksi per `layanan`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * level)) - 1
    return float(s[min(max(k, 0), n - 1)])


class ConformalP90:
    def __init__(self, level: float = config.TARGET_COVERAGE, group_col: str | None = None,
                 min_group: int = 500):
        self.level, self.group_col, self.min_group = level, group_col, min_group

    def fit(self, y_cal, p90_cal, groups: pd.Series | None = None):
        s = np.log1p(np.asarray(y_cal)) - np.log1p(np.asarray(p90_cal))
        self.global_ = _conformal_quantile(s, self.level)
        self.group_ = {}
        if self.group_col is not None and groups is not None:
            g = np.asarray(groups).astype(str)
            for lv in np.unique(g):
                m = g == lv
                if m.sum() >= self.min_group:
                    self.group_[lv] = _conformal_quantile(s[m], self.level)
        return self

    def corrections(self, groups: pd.Series | None = None) -> np.ndarray:
        if not self.group_ or groups is None:
            return None
        g = pd.Series(np.asarray(groups).astype(str))
        return g.map(self.group_).fillna(self.global_).to_numpy()

    def transform(self, p90, groups: pd.Series | None = None) -> np.ndarray:
        corr = self.corrections(groups)
        if corr is None:
            corr = self.global_
        return np.expm1(np.log1p(np.asarray(p90)) + corr)
