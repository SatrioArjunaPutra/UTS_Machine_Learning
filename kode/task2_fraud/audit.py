"""Task 2a-2b: audit pembentukan label dan ketersediaan fitur."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from common.utils import record_metric, save_fig, save_table

AVAIL = {
    "id_transaksi": ("ID", "Identitas"), "id_pengguna": ("ID", "Dipakai hanya untuk fitur histori"),
    "id_merchant": ("ID", "Dipakai hanya untuk fitur histori"),
    "waktu": ("Ya", "Dipakai untuk split dan fitur histori"),
    "flag_review_manual": ("Tidak", "'Pernah ditinjau' bersifat retrospektif: review terjadi setelah "
                                    "transaksi (sering karena kecurigaan/komplain)"),
    "is_fraud": ("Label", "Terkonfirmasi saat data ditarik; belum matang untuk transaksi terbaru"),
}
DEFAULT = ("Ya", "Kamus: profil/riwayat sebelum transaksi")


def maturity_curve(tr: pd.DataFrame) -> int:
    """Fraud rate vs umur transaksi saat data ditarik -> estimasi delay label D."""
    age = (pd.Timestamp(config.DATA_PULL_DATE) - tr["waktu"]).dt.days
    wk = (age // 7) * 7
    g = tr.groupby(wk)["is_fraud"].agg(["mean", "size"]).rename(columns={"mean": "fraud_rate", "size": "n"})
    g.index.name = "umur_hari_awal_minggu"
    plateau = float(g.loc[g.index >= 56, "fraud_rate"].median())
    reach = g.index[(g["fraud_rate"] >= 0.9 * plateau)]
    d_est = int(reach.min() + 7) if len(reach) else config.LABEL_DELAY_DAYS
    g = g.reset_index()
    save_table(g, "task2", "kurva_kematangan_label")
    record_metric("task2", "fraud_rate_plateau", round(plateau, 5))
    record_metric("task2", "delay_label_estimasi_hari", d_est)
    record_metric("task2", "fraud_rate_minggu_terakhir", round(float(g.iloc[0]["fraud_rate"]), 5))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    gg = g[g.umur_hari_awal_minggu <= 120]
    ax.bar(gg.umur_hari_awal_minggu, 100 * gg.fraud_rate, width=6, color="#2b6cb0")
    ax.axhline(100 * plateau, color="k", ls="--", lw=1, label=f"plateau {100 * plateau:.2f}%")
    ax.axvline(d_est, color="red", ls=":", lw=1.5, label=f"D = {d_est} hari")
    ax.set_xlabel("umur transaksi saat data ditarik (hari)")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("Label fraud transaksi terbaru belum lengkap")
    ax.legend()
    save_fig(fig, "task2", "kurva_kematangan_label")
    return d_est


def review_flag_analysis(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    age = (pd.Timestamp(config.DATA_PULL_DATE) - tr["waktu"]).dt.days
    b = pd.cut(age, [-1, 7, 14, 28, 56, 120, 250], labels=["0-7", "8-14", "15-28", "29-56", "57-120", ">120"])
    g = tr.groupby(b, observed=True).agg(flag_rate=("flag_review_manual", "mean"),
                                         fraud_rate=("is_fraud", "mean"))
    g["P(fraud | flag=1)"] = tr[tr.flag_review_manual == 1].groupby(b[tr.flag_review_manual == 1],
                                                                     observed=True)["is_fraud"].mean()
    g.index.name = "umur_hari"
    save_table(g.round(4).reset_index(), "task2", "flag_review_per_umur")
    record_metric("task2", "flag_review", {
        "fraud_rate_flag1": round(float(tr.loc[tr.flag_review_manual == 1, "is_fraud"].mean()), 4),
        "fraud_rate_flag0": round(float(tr.loc[tr.flag_review_manual == 0, "is_fraud"].mean()), 4),
        "share_fraud_flagged": round(float(tr.loc[tr.is_fraud == 1, "flag_review_manual"].mean()), 4),
        "flag_rate_train": round(float(tr.flag_review_manual.mean()), 4),
        "flag_rate_test": round(float(te.flag_review_manual.mean()), 4),
        "auc_univariat": round(float(roc_auc_score(tr.is_fraud, tr.flag_review_manual)), 4),
    })


def drift_v15(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """v15: rata-rata per bulan (train + test) dan AUC univariat per bulan."""
    rows = []
    for name, d in [("train", tr), ("test", te)]:
        for mth, s in d.groupby(d["waktu"].dt.to_period("M").astype(str)):
            r = {"bulan": mth, "data": name, "rata2_v15": round(float(s["v15"].mean()), 3)}
            if name == "train":
                a = roc_auc_score(s["is_fraud"], s["v15"])
                r["auc_v15_dalam_bulan"] = round(max(a, 1 - a), 3)
            rows.append(r)
    out = pd.DataFrame(rows)
    save_table(out, "task2", "drift_v15")
    ts = (tr["waktu"] - tr["waktu"].min()).dt.total_seconds()
    record_metric("task2", "v15_spearman_waktu", round(float(tr["v15"].corr(ts, method="spearman")), 3))
    return out


def audit_table(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    y = tr["is_fraud"]
    rows = []
    for c in tr.columns:
        s = tr[c]
        num = pd.api.types.is_numeric_dtype(s) and c != "is_fraud"
        auc = np.nan
        if num:
            a = roc_auc_score(y, s.fillna(s.median()))
            auc = max(a, 1 - a)
        av, why = AVAIL.get(c, DEFAULT)
        rows.append({"kolom": c, "tipe": str(s.dtype), "missing_train_%": round(100 * s.isna().mean(), 2),
                     "missing_test_%": round(100 * te[c].isna().mean(), 2) if c in te else np.nan,
                     "n_unik": s.nunique(), "auc_univariat": round(auc, 3) if pd.notna(auc) else "",
                     "tersedia_saat_transaksi": av, "alasan": why})
    df = pd.DataFrame(rows)
    save_table(df, "task2", "audit_kolom")
    miss = tr["menit_sejak_trx_terakhir"].isna()
    record_metric("task2", "missing_menit_%", round(100 * float(miss.mean()), 2))
    record_metric("task2", "missing_menit_dan_trx7h_nol_%", round(100 * float((miss & (tr.trx_7h == 0)).mean()), 2))
    record_metric("task2", "fraud_rate_train", round(float(y.mean()), 5))
