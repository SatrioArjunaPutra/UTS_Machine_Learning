"""Task 3e: uji stabilitas dengan subsampling (20 run, 80% data)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import config
from common.utils import record_metric, save_fig, save_table


def bootstrap_stability(P: np.ndarray, k: int, final_labels: np.ndarray, runs: int = 20,
                        frac: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    labs = []
    for b in range(runs):
        idx = rng.choice(len(P), int(frac * len(P)), replace=False)
        g = GaussianMixture(k, covariance_type="full", n_init=2, random_state=config.SEED + 100 + b).fit(P[idx])
        labs.append(g.predict(P))
    pair = [adjusted_rand_score(labs[a], labs[b]) for a in range(runs) for b in range(a + 1, runs)]
    vs_final = [adjusted_rand_score(final_labels, l) for l in labs]
    out = pd.DataFrame({"jenis": ["antar pasangan run"] * len(pair) + ["run vs model final"] * len(vs_final),
                        "ARI": np.round(pair + vs_final, 4)})
    save_table(out, "task3", "stabilitas_bootstrap")
    summ = out.groupby("jenis")["ARI"].agg(["mean", "median", "std", "min", "max"]).round(4)
    summ["n_di_bawah_0,9"] = out.assign(low=out.ARI < 0.9).groupby("jenis")["low"].sum()
    save_table(summ, "task3", "stabilitas_ringkas", index=True)
    record_metric("task3", "stabilitas", summ.to_dict(orient="index"))
    record_metric("task3", "stabilitas_runs", runs)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.boxplot([pair, vs_final], tick_labels=["antar pasangan run\n(190 pasang)", "run vs model final\n(20 run)"])
    ax.set_ylabel("ARI")
    ax.set_title(f"Stabilitas GMM k={k}, subsampel 80%")
    save_fig(fig, "task3", "stabilitas_bootstrap")
    return summ
