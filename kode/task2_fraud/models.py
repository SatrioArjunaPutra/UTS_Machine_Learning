"""Task 2: model fraud (LightGBM, logistic regression) dan kalibrator probabilitas."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from task2_fraud.features import CAT_COLS


def lgb_clf_params(num_leaves: int = 31, n_estimators: int = 3000, learning_rate: float = 0.03,
                   scale_pos_weight: float = 1.0, min_child_samples: int = 200) -> dict:
    return dict(objective="binary", n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                scale_pos_weight=scale_pos_weight, max_cat_to_onehot=8, cat_smooth=20,
                random_state=config.SEED, n_jobs=config.N_JOBS, deterministic=True,
                force_row_wise=True, verbose=-1)


class FraudLGB:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None, early_stopping=None):
        self.model_ = lgb.LGBMClassifier(**lgb_clf_params(**self.kw))
        fit_kw = {}
        if X_val is not None and early_stopping:
            fit_kw = dict(eval_X=X_val, eval_y=y_val, eval_metric="average_precision",
                          callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        self.model_.fit(X, y, sample_weight=sample_weight, **fit_kw)
        self.best_iter_ = self.model_.best_iteration_ or self.model_.n_estimators
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


class FraudLogReg:
    """Baseline linear: imputasi median + scaling + one-hot."""

    def fit(self, X: pd.DataFrame, y):
        nums = [c for c in X.columns if c not in CAT_COLS]
        # kolom miring non-negatif di-log1p; daftar kolom ditetapkan saat fit
        self.log_cols_ = [c for c in nums if X[c].min() >= 0 and X[c].max() > 100]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])
        self.model_ = make_pipeline(pre, LogisticRegression(max_iter=1000, C=0.5))
        self.model_.fit(self._prep(X), y)
        return self

    def _prep(self, X):
        X = X.copy()
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        num = [c for c in X.columns if c not in CAT_COLS]
        X[num] = X[num].astype(float).replace([np.inf, -np.inf], np.nan)
        for c in self.log_cols_:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(self._prep(X))[:, 1]


def undersample_idx(y: np.ndarray, ratio: int, seed: int = config.SEED) -> np.ndarray:
    """Semua positif + negatif acak sebanyak ratio x jumlah positif."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = rng.choice(neg, size=min(len(neg), ratio * len(pos)), replace=False)
    return np.sort(np.r_[pos, keep])


# ---------------------------------------------------------------- kalibrator
def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Regresi logistik pada logit(skor): monoton, urutan (dan PR-AUC) terjaga."""

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic + tie-break kecil dari skor mentah.

    Isotonic menghasilkan plateau (nilai sama) yang menghapus urutan di dalam
    plateau dan bisa menurunkan PR-AUC. Tie-break 1e-6 x skor mentah mengembalikan
    urutan tanpa mengubah kalibrasi secara berarti.
    """

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p, y)
        return self

    def transform(self, p, tie_break: bool = True):
        out = self.iso_.predict(p)
        if tie_break:
            out = np.clip(out + 1e-6 * np.asarray(p), 0, 1)
        return out


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p)
