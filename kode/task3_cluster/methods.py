"""Task 3c-3d: perbandingan metode clustering dan pemilihan jumlah cluster."""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

import config
from common.utils import record_metric, save_fig, save_table

SUB_N = 10000     # metode O(n^2) (Ward: ~0,4 GB jarak) pada subsampel, label dipropagasi kNN


def _sub_then_propagate(model, P: np.ndarray, rng) -> np.ndarray:
    idx = rng.choice(len(P), min(SUB_N, len(P)), replace=False)
    lab_sub = model.fit_predict(P[idx])
    keep = lab_sub >= 0                         # HDBSCAN: -1 = noise, tidak dipakai untuk propagasi
    if keep.sum() == 0 or len(np.unique(lab_sub[keep])) < 2:
        return np.zeros(len(P), dtype=int)      # tidak ada struktur kepadatan: satu cluster
    knn = KNeighborsClassifier(15, n_jobs=config.N_JOBS).fit(P[idx][keep], lab_sub[keep])
    return knn.predict(P)


def fit_method(name: str, P: np.ndarray, k: int, seed: int = config.SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "K-Means":
        return KMeans(k, n_init=10, random_state=seed).fit_predict(P)
    if name == "Gaussian Mixture (full)":
        return GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(P).predict(P)
    if name == "Agglomerative (Ward)":
        return _sub_then_propagate(AgglomerativeClustering(k, linkage="ward"), P, rng)
    if name == "HDBSCAN":
        mcs = max(50, int(0.01 * min(SUB_N, len(P))))          # cluster minimal 1% subsampel
        return _sub_then_propagate(HDBSCAN(min_cluster_size=mcs, min_samples=10), P, rng)
    raise ValueError(name)


def _stability(name, P, k, runs=5, frac=0.8) -> list[float]:
    """Latih pada subsampel 80%, prediksi ke seluruh data, ARI antar pasangan run."""
    rng = np.random.default_rng(config.SEED)
    labs = []
    for b in range(runs):
        idx = rng.choice(len(P), int(frac * len(P)), replace=False)
        if name == "K-Means":
            m = KMeans(k, n_init=3, random_state=config.SEED + b).fit(P[idx])
            labs.append(m.predict(P))
        elif name == "Gaussian Mixture (full)":
            m = GaussianMixture(k, covariance_type="full", n_init=2, random_state=config.SEED + b).fit(P[idx])
            labs.append(m.predict(P))
        else:
            lab_idx = fit_method(name, P[idx], k, seed=config.SEED + b)
            if len(np.unique(lab_idx)) < 2:
                labs.append(np.zeros(len(P), dtype=int))
            else:
                labs.append(KNeighborsClassifier(15, n_jobs=config.N_JOBS).fit(P[idx], lab_idx).predict(P))
    return [adjusted_rand_score(labs[a], labs[b]) for a in range(runs) for b in range(a + 1, runs)]


def compare_methods(P: np.ndarray, k: int, ref_name: str = "Gaussian Mixture (full)") -> tuple[pd.DataFrame, dict]:
    labels, rows = {}, []
    for name in ["K-Means", "Gaussian Mixture (full)", "Agglomerative (Ward)", "HDBSCAN"]:
        t0 = time.perf_counter()
        lab = fit_method(name, P, k)
        dt = time.perf_counter() - t0
        labels[name] = lab
        st = _stability(name, P, k, runs=4)
        n_cl = len(np.unique(lab))
        ok = n_cl > 1
        rows.append({"metode": name, "n_cluster": n_cl,
                     "silhouette": round(silhouette_score(P, lab, sample_size=10000, random_state=config.SEED), 3) if ok else np.nan,
                     "davies_bouldin": round(davies_bouldin_score(P, lab), 3) if ok else np.nan,
                     "calinski_harabasz": round(calinski_harabasz_score(P, lab), 0) if ok else np.nan,
                     "stabilitas_ARI": round(float(np.mean(st)), 3), "waktu_detik": round(dt, 1)})
    out = pd.DataFrame(rows)
    out["ARI_vs_GMM"] = [round(adjusted_rand_score(labels[ref_name], labels[m]), 3) for m in out["metode"]]
    save_table(out, "task3", "perbandingan_metode")
    record_metric("task3", "perbandingan_metode", out.set_index("metode").to_dict(orient="index"))
    return out, labels


def choose_k(P: np.ndarray, ks=range(3, 13), runs: int = 4) -> tuple[int, pd.DataFrame]:
    rng = np.random.default_rng(config.SEED)
    rows = []
    for k in ks:
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=config.SEED).fit(P)
        lab = g.predict(P)
        labs = [GaussianMixture(k, covariance_type="full", n_init=2, random_state=config.SEED + b)
                .fit(P[rng.choice(len(P), int(0.8 * len(P)), replace=False)]).predict(P) for b in range(runs)]
        st = [adjusted_rand_score(labs[a], labs[b]) for a in range(runs) for b in range(a + 1, runs)]
        rows.append({"k": k, "BIC": round(g.bic(P), 0),
                     "silhouette": round(silhouette_score(P, lab, sample_size=10000, random_state=config.SEED), 4),
                     "stabilitas_ARI": round(float(np.mean(st)), 4),
                     "ukuran_cluster_terkecil_%": round(100 * np.bincount(lab).min() / len(lab), 2)})
    t = pd.DataFrame(rows)
    # Aturan: kandidat = k dengan stabilitas >= 0.9; di antaranya pilih BIC terkecil.
    cand = t[t["stabilitas_ARI"] >= 0.9]
    k_best = int((cand if len(cand) else t).sort_values("BIC").iloc[0]["k"])
    t["dipilih"] = t["k"] == k_best
    save_table(t, "task3", "pemilihan_k")
    record_metric("task3", "k_dipilih", k_best)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(t.k, t.BIC / 1e3, marker="o")
    ax[0].set_title("BIC (x1000), lebih kecil lebih baik")
    ax[1].plot(t.k, t.silhouette, marker="o", color="#dd6b20")
    ax[1].set_title("Silhouette")
    ax[2].plot(t.k, t.stabilitas_ARI, marker="o", color="#2f855a")
    ax[2].axhline(0.9, color="k", ls="--", lw=1)
    ax[2].set_title("Stabilitas ARI antar-run")
    for a in ax:
        a.axvline(k_best, color="red", ls=":", lw=1.5)
        a.set_xlabel("k")
    save_fig(fig, "task3", "pemilihan_k")
    return k_best, t
