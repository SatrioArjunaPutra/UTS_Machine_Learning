"""Task 3g: pelabelan anomali (-1).

Anomali referensi DIKELUARKAN dari perhitungan ARI, jadi label -1 tidak bisa
menaikkan ARI, tetapi pelanggan normal yang salah diberi -1 bisa menurunkannya.
Karena itu anomali ditentukan secara konservatif:
1. Aturan domain: bot (sesi_per_minggu > 100, durasi sesi sangat pendek,
   konversi ~0, klik iklan & wishlist ekstrem) -> terpisah jelas di PCA.
2. Konsensus dua detektor pada sisa pelanggan: IsolationForest DAN
   log-likelihood GMM sama-sama berada di 0,5% teratas skor anomali.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from common.utils import record_metric, save_table

TOP_FRAC = 0.005
PROFILE_COLS = ["sesi_per_minggu", "durasi_sesi_menit", "rasio_konversi", "klik_iklan_30h", "wishlist",
                "frekuensi_90h", "belanja_90h", "recency_hari", "rata2_keranjang", "item_per_pesanan"]


def consensus_outliers(Z: np.ndarray, P: np.ndarray, gmm) -> tuple[np.ndarray, dict]:
    iso = IsolationForest(n_estimators=300, random_state=config.SEED, n_jobs=config.N_JOBS).fit(Z)
    s_iso = -iso.score_samples(Z)
    s_gmm = -gmm.score_samples(P)
    a = s_iso >= np.quantile(s_iso, 1 - TOP_FRAC)
    b = s_gmm >= np.quantile(s_gmm, 1 - TOP_FRAC)
    info = {"isolation_forest_top": int(a.sum()), "gmm_loglik_top": int(b.sum()), "konsensus": int((a & b).sum())}
    return a & b, info


def summarize(df: pd.DataFrame, labels: np.ndarray, groups: dict, info: dict) -> None:
    n = len(df)
    rows = [{"sumber": "bot (sesi_per_minggu > 100)", "n": int(groups["bot"].sum())},
            {"sumber": "IsolationForest 0,5% teratas", "n": info["isolation_forest_top"]},
            {"sumber": "log-likelihood GMM 0,5% terendah", "n": info["gmm_loglik_top"]},
            {"sumber": "konsensus kedua detektor", "n": info["konsensus"]},
            {"sumber": "TOTAL label -1", "n": int((labels == -1).sum())}]
    t = pd.DataFrame(rows)
    t["persen"] = (100 * t["n"] / n).round(2)
    save_table(t, "task3", "anomali_sumber")
    prof = pd.DataFrame({
        "bot": df.loc[groups["bot"], PROFILE_COLS].median(),
        "anomali konsensus": df.loc[(labels == -1) & ~groups["bot"], PROFILE_COLS].median(),
        "pelanggan normal": df.loc[labels >= 0, PROFILE_COLS].median(),
    }).round(2)
    save_table(prof, "task3", "anomali_profil", index=True)
    record_metric("task3", "n_anomali", int((labels == -1).sum()))
    record_metric("task3", "persen_anomali", round(100 * float((labels == -1).mean()), 2))
    record_metric("task3", "n_bot", int(groups["bot"].sum()))
    record_metric("task3", "n_anomali_konsensus", info["konsensus"])
