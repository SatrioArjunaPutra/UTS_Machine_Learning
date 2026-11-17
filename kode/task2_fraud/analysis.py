"""Task 2e-2g: pergeseran pola, reliability, dan analisis error."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common.metrics import total_cost
from common.utils import save_fig, save_table


def group_errors(df: pd.DataFrame, y, prob, flag, col: str) -> pd.DataFrame:
    d = pd.DataFrame({"g": df[col].astype(str).to_numpy(), "y": np.asarray(y).astype(int),
                      "p": prob, "f": np.asarray(flag).astype(int), "nom": df["nominal"].to_numpy()})
    rows = []
    for g, s in d.groupby("g"):
        tp = int(((s.y == 1) & (s.f == 1)).sum())
        rows.append({col: g, "n": len(s), "n_fraud": int(s.y.sum()),
                     "fraud_rate_%": round(100 * s.y.mean(), 3),
                     "n_flag": int(s.f.sum()),
                     "recall": round(tp / max(s.y.sum(), 1), 3),
                     "presisi": round(tp / max(s.f.sum(), 1), 3),
                     "kerugian_lolos_juta": round(s.nom[(s.y == 1) & (s.f == 0)].sum() / 1e6, 2),
                     "total_cost_juta": round(total_cost(s.y, s.f, s.nom) / 1e6, 2)})
    out = pd.DataFrame(rows).sort_values("total_cost_juta", ascending=False)
    save_table(out, "task2", f"error_per_{col}")
    return out


def fraud_patterns(df: pd.DataFrame, y, flag) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(int)
    base = y.mean()
    pats = {
        "perangkat baru + IP luar negeri": (df.perangkat_baru == 1) & (df.negara_ip_beda == 1),
        "kartu tidak hadir": df.kartu_tidak_hadir == 1,
        "lonjakan transaksi (trx_1j >= 3)": df.trx_1j >= 3,
        "akun baru (< 30 hari)": df.umur_akun_hari < 30,
        "jauh dari lokasi biasa (> 300 km)": df.jarak_lokasi_km > 300,
        "email sekali pakai": df.email_domain == "sekali_pakai",
        "gagal login >= 3 dalam 24 jam": df.gagal_login_24j >= 3,
        "ganti password 7 hari terakhir": df.ganti_password_7h > 0,
        "dini hari (00.00-04.59)": df.jam < 5,
        "nominal >= 1 juta": df.nominal >= 1_000_000,
    }
    rows = []
    for name, m in pats.items():
        m = np.asarray(m)
        if m.sum() == 0:
            continue
        yf = y[m]
        rows.append({"pola": name, "n": int(m.sum()), "porsi_transaksi_%": round(100 * m.mean(), 2),
                     "porsi_fraud_%": round(100 * yf.sum() / max(y.sum(), 1), 1),
                     "fraud_rate_%": round(100 * yf.mean(), 2), "lift": round(yf.mean() / base, 1),
                     "recall_model": round(((yf == 1) & (flag[m] == 1)).sum() / max(yf.sum(), 1), 3)})
    out = pd.DataFrame(rows).sort_values("lift", ascending=False)
    save_table(out, "task2", "pola_fraud")
    return out


def top_false_negatives(df: pd.DataFrame, y, prob, flag, n: int = 10) -> pd.DataFrame:
    m = (np.asarray(y) == 1) & (np.asarray(flag) == 0)
    d = df.loc[m, ["id_transaksi", "nominal", "kanal", "kategori_merchant", "perangkat_baru",
                   "kartu_tidak_hadir", "umur_akun_hari"]].copy()
    d["prob"] = np.round(np.asarray(prob)[m], 4)
    d["ambang_prob_aturan_biaya"] = np.round(config.REVIEW_COST / d["nominal"], 4)
    out = d.sort_values("nominal", ascending=False).head(n)
    save_table(out, "task2", "false_negative_termahal")
    return out


def reliability_plot(y, probs: dict, name: str = "reliability", n_bins: int = 20) -> None:
    """Reliability diagram skala log, bin dengan jumlah baris sama (probabilitas fraud sangat miring)."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    y = np.asarray(y)
    floor = 1e-5
    for label, p in probs.items():
        p = np.asarray(p)
        q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
        mp = pd.Series(p).groupby(b).mean()
        my = pd.Series(y).groupby(b).mean()
        ax.plot(np.maximum(mp, floor), np.maximum(my, floor), marker="o", ms=3.5, label=label)
    ax.plot([floor, 1], [floor, 1], "k--", lw=1, label="kalibrasi sempurna")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor, 1)
    ax.set_ylim(floor, 1)
    ax.set_xlabel("rata-rata probabilitas prediksi per bin")
    ax.set_ylabel("fraud rate aktual per bin")
    ax.set_title(f"Reliability diagram F3 ({n_bins} bin sama banyak, skala log)")
    ax.legend(fontsize=8)
    save_fig(fig, "task2", name)


def shift_plots(tr: pd.DataFrame) -> pd.DataFrame:
    m = tr["waktu"].dt.to_period("M").astype(str)
    t = tr.groupby([m, "kanal"])["is_fraud"].mean().unstack() * 100
    save_table(t.round(3).reset_index(names="bulan"), "task2", "fraud_rate_bulan_kanal")
    k = tr.groupby([m, "kategori_merchant"])["is_fraud"].mean().unstack() * 100
    save_table(k.round(3).reset_index(names="bulan"), "task2", "fraud_rate_bulan_kategori")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    t.plot(ax=ax, marker="o")
    ax.set_ylabel("fraud rate (%)")
    ax.set_xlabel("bulan (Agustus belum matang)")
    ax.set_title("Fraud rate per kanal per bulan")
    save_fig(fig, "task2", "fraud_rate_bulan_kanal")
    return t
