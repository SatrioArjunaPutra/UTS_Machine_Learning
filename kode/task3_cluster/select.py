"""Task 3b: seleksi/pembobotan fitur dengan bukti.

Bukti yang dihasilkan:
1. Kolom n01-n15: korelasi Spearman maksimum dengan semua kolom lain (~0).
2. eta^2 tiap fitur terhadap clustering awal: fitur tanpa hubungan dengan
   struktur cluster (eta^2 ~ 0) dibuang.
3. Perbandingan varian matriks fitur (silhouette + stabilitas ARI antar-run).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config
from common.utils import record_metric, save_fig, save_table
from task3_cluster.preprocess import (CAT_SHARES, LIN_COLS, LOG_COLS, NOISE_COLS, PROFILE_IRRELEVANT,
                                      TIME_SHARES, Preprocessor)


def noise_evidence(df: pd.DataFrame) -> pd.DataFrame:
    num = df.drop(columns=["id_pelanggan", "perangkat"]).select_dtypes("number")
    other = [c for c in num.columns if c not in NOISE_COLS]
    rho = num.sample(30000, random_state=config.SEED).corr(method="spearman")
    rows = []
    for c in NOISE_COLS:
        rows.append({"kolom": c, "max_abs_spearman_dengan_kolom_lain": round(rho.loc[c, other].abs().max(), 4),
                     "min": round(df[c].min(), 2), "median": round(df[c].median(), 2), "max": round(df[c].max(), 2),
                     "skew": round(df[c].skew(), 2)})
    out = pd.DataFrame(rows)
    save_table(out, "task3", "bukti_noise_n01_n15")
    record_metric("task3", "max_abs_rho_noise", float(out["max_abs_spearman_dengan_kolom_lain"].max()))
    return out


def eta_squared(df_model: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    cand = pd.concat([np.log1p(df_model[LOG_COLS].fillna(0)), df_model[LIN_COLS],
                      np.sqrt(df_model[CAT_SHARES]), np.sqrt(df_model[TIME_SHARES]),
                      df_model[PROFILE_IRRELEVANT].astype(float), df_model[NOISE_COLS],
                      pd.get_dummies(df_model["perangkat"], prefix="perangkat").astype(float)], axis=1)
    Z = pd.DataFrame(StandardScaler().fit_transform(cand), columns=cand.columns)
    g = Z.groupby(labels)
    eta = (g.mean().pow(2).mul(g.size(), axis=0).sum() / len(Z)) / Z.var(ddof=0)
    out = eta.sort_values(ascending=False).rename("eta2").reset_index().rename(columns={"index": "fitur"})
    out["eta2"] = out["eta2"].round(4)
    out["dipakai"] = ~out["fitur"].isin(NOISE_COLS + PROFILE_IRRELEVANT) & ~out["fitur"].str.startswith("perangkat")
    save_table(out, "task3", "eta2_fitur")
    used = out[out["dipakai"]].head(20).iloc[::-1]
    drop = out[~out["dipakai"]].iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6.2), sharex=True)
    ax[0].barh(used["fitur"], used["eta2"], color="#2b6cb0")
    ax[0].set_title("20 fitur dipakai dengan η² tertinggi")
    ax[1].barh(drop["fitur"], drop["eta2"], color="#999")
    ax[1].set_title("Fitur tidak dipakai (noise, profil, perangkat)")
    for a in ax:
        a.set_xlabel("η² terhadap clustering awal")
        a.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    save_fig(fig, "task3", "eta2_fitur")
    return out


def variant_comparison(df_model: pd.DataFrame, k: int, n: int = 30000, runs: int = 5) -> pd.DataFrame:
    sample = df_model.sample(min(n, len(df_model)), random_state=config.SEED)
    variants = {
        "final: Hellinger + amalgamasi, tanpa noise/profil": dict(comp="hellinger_amalg"),
        "CLR + amalgamasi": dict(comp="clr_amalg"),
        "CLR 16 kategori": dict(comp="clr_16"),
        "Hellinger 16 kategori": dict(comp="hellinger_16"),
        "porsi mentah (tanpa transformasi)": dict(comp="porsi_mentah"),
        "+ n01-n15 dan profil tak relevan": dict(comp="hellinger_amalg", extra=NOISE_COLS + PROFILE_IRRELEVANT),
    }
    rng = np.random.default_rng(config.SEED)
    rows = []
    for name, kw in variants.items():
        P = Preprocessor(**kw).fit_transform(sample)
        lab = GaussianMixture(k, covariance_type="full", random_state=config.SEED, n_init=3).fit(P).predict(P)
        labs = [GaussianMixture(k, covariance_type="full", n_init=2, random_state=config.SEED + b)
                .fit(P[rng.choice(len(P), int(0.8 * len(P)), replace=False)]).predict(P) for b in range(runs)]
        aris = [adjusted_rand_score(labs[a], labs[b]) for a in range(runs) for b in range(a + 1, runs)]
        rows.append({"varian": name, "silhouette": round(silhouette_score(P, lab, sample_size=10000,
                                                                          random_state=config.SEED), 3),
                     "stabilitas_ARI_rata2": round(float(np.mean(aris)), 3),
                     "stabilitas_ARI_std": round(float(np.std(aris)), 3)})
    out = pd.DataFrame(rows)
    save_table(out, "task3", "perbandingan_varian_fitur")
    record_metric("task3", "varian_fitur", out.set_index("varian").to_dict(orient="index"))
    return out
