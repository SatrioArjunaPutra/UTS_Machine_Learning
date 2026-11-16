"""Task 2c: fitur perilaku pengguna/merchant TANPA informasi masa depan.

Dua jenis fitur:
1. Fitur histori tanpa label (frekuensi, kecepatan transaksi merchant, merchant
   baru bagi pengguna, dst.). Dihitung pada gabungan train+test yang diurutkan
   menurut waktu, dan setiap baris hanya melihat baris SEBELUMNYA
   (cumcount / shift / rolling closed='left'). Transaksi test boleh menjadi
   histori bagi transaksi test berikutnya karena sudah terjadi.
2. Fitur histori BERLABEL (fraud rate merchant/pengguna). Label baru diketahui
   ~D hari setelah transaksi, jadi baris pada waktu t hanya memakai label dari
   transaksi dengan waktu < min(t - D, cutoff), dengan cutoff = batas data latih
   model. Label test tidak pernah dipakai.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLS = ["id_transaksi", "id_pengguna", "id_merchant"]
TIME = "waktu"
TARGET = "is_fraud"
LEAK_COLS = ["flag_review_manual"]
CAT_COLS = ["kategori_merchant", "kanal", "tipe_perangkat", "email_domain"]
# v15 naik linear terhadap waktu (rata-rata +0,3/bulan) tanpa sinyal fraud (AUC ~0,51
# per bulan) -> nilai test di luar rentang latih. Kolom hitungan kumulatif tumbuh
# dengan waktu sehingga distribusi test bergeser secara sistematis.
DRIFT_COLS = ["v15"]
TREND_COLS = ["u_n_sebelum", "m_n_sebelum", "u_hari_sejak_pertama", "u_n_merchant_sebelum",
              "m_n_berlabel", "u_n_berlabel", "m_n_fraud", "u_n_fraud"]


def behavioral_features(all_df: pd.DataFrame) -> pd.DataFrame:
    """all_df: gabungan train+test (kolom asli). Kembalikan fitur baru, index sama."""
    d = all_df[[TIME, "id_pengguna", "id_merchant", "nominal"]].copy()
    d["_ord"] = np.arange(len(d))
    d = d.sort_values([TIME, "_ord"], kind="stable")
    out = pd.DataFrame(index=d.index)

    gu = d.groupby("id_pengguna", sort=False)
    out["u_n_sebelum"] = gu.cumcount()
    out["u_hari_sejak_pertama"] = (d[TIME] - gu[TIME].transform("min")).dt.total_seconds() / 86400
    pair_first = ~d.duplicated(["id_pengguna", "id_merchant"])
    out["u_merchant_baru"] = pair_first.astype(int)
    out["u_n_merchant_sebelum"] = pair_first.groupby(d["id_pengguna"]).cumsum() - pair_first.astype(int)
    # rata-rata nominal SEBELUMNYA = (cumsum - nilai sekarang) / jumlah sebelumnya
    u_prev_sum = gu["nominal"].cumsum() - d["nominal"]
    out["u_rasio_nominal_hist"] = d["nominal"] / (u_prev_sum / out["u_n_sebelum"].replace(0, np.nan))

    gm = d.groupby("id_merchant", sort=False)
    out["m_n_sebelum"] = gm.cumcount()
    m_prev_sum = gm["nominal"].cumsum() - d["nominal"]
    out["m_rasio_nominal_hist"] = d["nominal"] / (m_prev_sum / out["m_n_sebelum"].replace(0, np.nan))
    # Kecepatan transaksi merchant: rolling closed='left' (baris itu sendiri tidak ikut)
    ds = d.sort_values(["id_merchant", TIME, "_ord"], kind="stable")
    r = ds.set_index(TIME).groupby("id_merchant", sort=False)["nominal"]
    for win in ["1h", "24h"]:
        cnt = r.rolling(win, closed="left").count().to_numpy()
        out.loc[ds.index, f"m_trx_{win}"] = np.nan_to_num(cnt)
    return out.loc[all_df.index]


def label_history(all_df: pd.DataFrame, key: str, cutoff: pd.Timestamp, delay_days: int,
                  smoothing: float, prefix: str) -> pd.DataFrame:
    """Fraud rate historis per `key` memakai label yang sudah matang saja.

    Untuk baris pada waktu t: label dari transaksi berlabel dengan
    waktu < min(t - delay, cutoff). Diimplementasi dengan merge_asof
    (allow_exact_matches=False => ketat lebih kecil).
    """
    delay = pd.Timedelta(days=delay_days)
    lab = all_df[all_df[TARGET].notna() & (all_df[TIME] < cutoff)][[TIME, key, TARGET]].copy()
    lab = lab.sort_values(TIME, kind="stable")
    lab["cum_n"] = lab.groupby(key).cumcount() + 1
    lab["cum_f"] = lab.groupby(key)[TARGET].cumsum()
    prior = float(lab[TARGET].mean())

    q = pd.DataFrame({"_i": np.arange(len(all_df)), key: all_df[key].to_numpy(),
                      "_b": np.minimum(all_df[TIME] - delay, cutoff)})
    q = q.sort_values("_b", kind="stable")
    m = pd.merge_asof(q, lab[[TIME, key, "cum_n", "cum_f"]].rename(columns={TIME: "_t"}),
                      left_on="_b", right_on="_t", by=key, direction="backward",
                      allow_exact_matches=False)
    m = m.sort_values("_i")
    n = m["cum_n"].fillna(0).to_numpy()
    f = m["cum_f"].fillna(0).to_numpy()
    out = pd.DataFrame(index=all_df.index)
    out[f"{prefix}_n_berlabel"] = n
    out[f"{prefix}_n_fraud"] = f
    out[f"{prefix}_pernah_fraud"] = (f > 0).astype(int)
    out[f"{prefix}_fraud_rate"] = (f + smoothing * prior) / (n + smoothing)
    return out


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["trx_pertama"] = X["menit_sejak_trx_terakhir"].isna().astype(int)
    X["log_nominal"] = np.log1p(X["nominal"])
    X["nominal_per_saldo"] = X["nominal"] / (X["saldo_sebelum"] + 1)
    X["hari_minggu"] = X[TIME].dt.dayofweek
    return X


def to_model_matrix(df: pd.DataFrame, cat_levels: dict | None = None, drop_leak: bool = True,
                    drop_drift: bool = True):
    """Pilih kolom fitur; kategori -> pandas.Categorical dengan level dari data latih."""
    excl = set(ID_COLS) | {TIME, TARGET}
    if drop_leak:
        excl |= set(LEAK_COLS)
    if drop_drift:
        excl |= set(DRIFT_COLS) | set(TREND_COLS)
    cols = [c for c in df.columns if c not in excl]
    X = df[cols].copy()
    if cat_levels is None:
        cat_levels = {c: sorted(X[c].astype(str).unique()) for c in CAT_COLS}
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
    return X, cat_levels
