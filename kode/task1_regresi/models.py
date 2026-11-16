"""Task 1: baseline, LightGBM quantile, dan model linear dengan target log."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from task1_regresi.features import CAT_LOW


# ---------------------------------------------------------------- baseline
class GroupQuantileBaseline:
    """Kuantil empiris per layanan x pulau_tujuan (fallback: layanan, lalu global)."""

    def __init__(self, q: float):
        self.q = q

    def fit(self, X: pd.DataFrame, y: pd.Series):
        d = pd.DataFrame({"l": X["layanan"].astype(str).values,
                          "p": X["pulau_tujuan"].astype(str).values, "y": y.values})
        self.g2_ = d.groupby(["l", "p"])["y"].quantile(self.q)
        self.g1_ = d.groupby("l")["y"].quantile(self.q)
        self.g0_ = float(d["y"].quantile(self.q))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        l = X["layanan"].astype(str).values
        p = X["pulau_tujuan"].astype(str).values
        key = pd.MultiIndex.from_arrays([l, p])
        out = pd.Series(self.g2_.reindex(key).to_numpy())
        out = out.fillna(pd.Series(l).map(self.g1_)).fillna(self.g0_)
        return out.to_numpy()


# ---------------------------------------------------------------- LightGBM
def lgb_params(alpha: float, num_leaves: int = 63, min_child_samples: int = 100,
               n_estimators: int = 3000, learning_rate: float = 0.05) -> dict:
    return dict(objective="quantile", alpha=alpha, n_estimators=n_estimators,
                learning_rate=learning_rate, num_leaves=num_leaves,
                min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, max_cat_to_onehot=8, cat_smooth=20,
                random_state=config.SEED, n_jobs=config.N_JOBS, deterministic=True,
                force_row_wise=True, verbose=-1)


class LGBQuantile:
    """LightGBM quantile; `log_target=True` melatih pada log1p(y).

    Kuantil invarian terhadap transformasi monoton, jadi expm1(kuantil log)
    tetap kuantil y, tetapi skala log membuat pohon lebih stabil.
    """

    def __init__(self, alpha: float, log_target: bool = False, **kw):
        self.alpha, self.log_target, self.kw = alpha, log_target, kw

    def _t(self, y):
        return np.log1p(y) if self.log_target else y

    def fit(self, X, y, X_val=None, y_val=None, early_stopping: int | None = None):
        self.model_ = lgb.LGBMRegressor(**lgb_params(self.alpha, **self.kw))
        fit_kw = {}
        if X_val is not None and early_stopping:
            fit_kw = dict(eval_X=X_val, eval_y=self._t(y_val), eval_metric="quantile",
                          callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        self.model_.fit(X, self._t(y), **fit_kw)
        self.best_iter_ = self.model_.best_iteration_ or self.model_.n_estimators
        return self

    def predict(self, X) -> np.ndarray:
        p = self.model_.predict(X)
        return np.expm1(p) if self.log_target else p


# ---------------------------------------------------------------- linear
class LogLinearQuantile:
    """Ridge pada log1p(y); P50 = expm1(prediksi), P90 = expm1(prediksi + kuantil residual).

    Residual log yang kira-kira simetris membuat prediksi mean-log mendekati median.
    """

    def __init__(self, alpha_ridge: float = 1.0):
        self.alpha_ridge = alpha_ridge

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cats = [c for c in CAT_LOW if c in X.columns]
        nums = [c for c in X.columns if c not in cats]
        self.nums_, self.cats_ = nums, cats
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
        ])
        self.model_ = make_pipeline(pre, Ridge(alpha=self.alpha_ridge))
        Xn = self._prep(X)
        ly = np.log1p(y.to_numpy())
        self.model_.fit(Xn, ly)
        self.resid_ = ly - self.model_.predict(Xn)
        return self

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cats_:
            X[c] = X[c].astype(str)
        # kolom miring positif -> log1p agar hubungan lebih linear
        for c in ["jarak_km", "berat_kg", "volume_liter", "nilai_barang", "volume_penjual_30h",
                  "antrean_hub_asal", "umur_akun_penjual_hari", "berat_volumetrik", "berat_tagih"]:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        base = self.model_.predict(self._prep(X))
        shift = float(np.quantile(self.resid_, q))     # kuantil residual log di data latih
        return np.expm1(base + shift)
