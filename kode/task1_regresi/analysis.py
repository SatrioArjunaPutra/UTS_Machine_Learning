"""Task 1f: analisis error per grup + interpretasi (SHAP via LightGBM, PDP)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common.metrics import coverage, pinball
from common.utils import save_fig, save_table


def error_by_group(df: pd.DataFrame, y, p50, p90, p90_raw, col: str) -> pd.DataFrame:
    d = pd.DataFrame({"g": df[col].astype(str).values, "y": np.asarray(y), "p50": p50,
                      "p90": p90, "p90_raw": p90_raw})
    rows = []
    for g, s in d.groupby("g"):
        rows.append({col: g, "n": len(s), "median_aktual": round(s.y.median(), 1),
                     "median_p50": round(s.p50.median(), 1),
                     "pinball_q50": round(pinball(s.y, s.p50, 0.5), 2),
                     "pinball_q90": round(pinball(s.y, s.p90, 0.9), 2),
                     "coverage_p90_mentah": round(coverage(s.y, s.p90_raw), 3),
                     "coverage_p90_kalibrasi": round(coverage(s.y, s.p90), 3)})
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    save_table(out, "task1", f"error_per_{col}")
    return out


def shap_importance(lgb_model, X: pd.DataFrame, name: str, n: int = 5000, top: int = 15,
                    unit: str = "jam") -> pd.DataFrame:
    """Mean |SHAP| memakai TreeSHAP bawaan LightGBM (pred_contrib=True), dalam skala target model."""
    Xs = X.sample(min(n, len(X)), random_state=config.SEED)
    contrib = lgb_model.booster_.predict(Xs, pred_contrib=True)[:, :-1]
    imp = pd.DataFrame({"fitur": X.columns, "mean_abs_shap": np.abs(contrib).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    save_table(imp, "task1", f"shap_{name}")
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh(t.fitur, t.mean_abs_shap, color="#2b6cb0")
    ax.set_xlabel(f"rata-rata |SHAP| ({unit})")
    ax.set_title(f"Kontribusi fitur, model {name}")
    save_fig(fig, "task1", f"shap_{name}")
    return imp


def partial_dependence(models: dict, X: pd.DataFrame, features: list[str], n: int = 3000) -> None:
    """PDP manual: ganti satu kolom dengan nilai grid, rata-ratakan prediksi."""
    Xs = X.sample(min(n, len(X)), random_state=config.SEED)
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.4))
    for ax, f in zip(np.atleast_1d(axes), features):
        grid = np.nanquantile(Xs[f].astype(float), np.linspace(0.02, 0.98, 25))
        for label, m in models.items():
            vals = []
            for g in grid:
                Xg = Xs.copy()
                Xg[f] = g
                vals.append(np.mean(m.predict(Xg)))
            ax.plot(grid, vals, label=label)
        ax.set_xlabel(f)
        ax.set_ylabel("prediksi rata-rata (jam)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Partial dependence")
    save_fig(fig, "task1", "pdp")
