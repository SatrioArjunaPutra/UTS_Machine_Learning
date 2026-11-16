"""Task 1: pembersihan data, fitur, dan encoding kategori.

Keputusan berbasis audit (lihat audit.py):
- durasi_jam >= 1000 yang bernilai bulat tercatat dalam MENIT -> dibagi 60.
- jarak_km = 9999 (sentinel) dan 0 -> NaN.
- umur_akun_penjual_hari > 9125 (25 tahun) mustahil -> NaN.
- jumlah_scan dibuang: scan tercatat sepanjang perjalanan, tidak tersedia saat pickup.
- kota_tujuan (405 kategori) -> frequency + target encoding out-of-fold.
- kategori yang tidak ada di data latih (mis. mitra_kurir K13) -> NaN (cabang "missing").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config

ID_COL = "id_paket"
DATE_COL = "tgl_pickup"
TARGET = "durasi_jam"
LEAK_COLS = ["jumlah_scan"]
CAT_LOW = ["layanan", "mitra_kurir", "hub_asal", "kota_asal", "pulau_tujuan", "rute"]
HIGH_CARD = "kota_tujuan"

UNIT_ERROR_MIN = 1000          # jam; nilai >= ini dan bulat = menit
MAX_UMUR_AKUN_HARI = 9125      # 25 tahun


def fix_target(y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Kembalikan target terkoreksi + mask baris yang dikoreksi."""
    mask = (y >= UNIT_ERROR_MIN) & (y % 1 == 0)
    y2 = y.where(~mask, y / 60.0)
    return y2, mask


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    stats = {
        "jarak_9999": int((df["jarak_km"] == 9999).sum()),
        "jarak_0": int((df["jarak_km"] == 0).sum()),
        "umur_akun_mustahil": int((df["umur_akun_penjual_hari"] > MAX_UMUR_AKUN_HARI).sum()),
    }
    df["jarak_km"] = df["jarak_km"].where(~df["jarak_km"].isin([9999, 0]))
    df["umur_akun_penjual_hari"] = df["umur_akun_penjual_hari"].where(
        df["umur_akun_penjual_hari"] <= MAX_UMUR_AKUN_HARI)
    return df, stats


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[DATE_COL])
    df["hari_minggu"] = d.dt.dayofweek
    df["akhir_pekan"] = (df["hari_minggu"] >= 5).astype(int)
    df["rute"] = df["hub_asal"].astype(str) + "_" + df["pulau_tujuan"].astype(str)
    df["berat_hilang"] = df["berat_kg"].isna().astype(int)
    df["berat_volumetrik"] = df["volume_liter"] / 5.0            # L x 1000 cm3 / 5000
    df["berat_tagih"] = np.fmax(df["berat_kg"], df["berat_volumetrik"])
    return df


class HighCardEncoder:
    """Frequency + target encoding (mean log1p target) dengan smoothing.

    Untuk baris latih, target encoding dihitung out-of-fold (5 fold) supaya
    label baris itu sendiri tidak ikut menghitung encodingnya.
    """

    def __init__(self, col: str = HIGH_CARD, smoothing: float = 50.0, seed: int = config.SEED):
        self.col, self.m, self.seed = col, smoothing, seed

    def _te_map(self, x: pd.Series, ly: pd.Series):
        g = pd.DataFrame({"x": x.values, "y": ly.values}).groupby("x")["y"].agg(["sum", "count"])
        prior = ly.mean()
        return (g["sum"] + self.m * prior) / (g["count"] + self.m), prior

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        x = df[self.col].astype(str)
        ly = np.log1p(y)
        self.freq_ = x.value_counts(normalize=True)
        self.te_, self.prior_ = self._te_map(x, ly)
        te = np.empty(len(df))
        for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(df):
            mp, pr = self._te_map(x.iloc[tr], ly.iloc[tr])
            te[va] = x.iloc[va].map(mp).fillna(pr).to_numpy()
        out = pd.DataFrame(index=df.index)
        out[f"{self.col}_freq"] = x.map(self.freq_).fillna(0).to_numpy()
        out[f"{self.col}_te"] = te
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.col].astype(str)
        out = pd.DataFrame(index=df.index)
        out[f"{self.col}_freq"] = x.map(self.freq_).fillna(0).to_numpy()
        out[f"{self.col}_te"] = x.map(self.te_).fillna(self.prior_).to_numpy()
        return out


def feature_columns(df: pd.DataFrame, drop_leak: bool = True, high_card: str = "freq_te") -> list[str]:
    excl = {ID_COL, DATE_COL, TARGET, HIGH_CARD}
    if drop_leak:
        excl |= set(LEAK_COLS)
    cols = [c for c in df.columns if c not in excl and not c.startswith(f"{HIGH_CARD}_")]
    if high_card == "freq_te":
        cols += [f"{HIGH_CARD}_freq", f"{HIGH_CARD}_te"]
    elif high_card == "freq":
        cols += [f"{HIGH_CARD}_freq"]
    elif high_card == "te":
        cols += [f"{HIGH_CARD}_te"]
    elif high_card == "native":
        cols += [HIGH_CARD]
    return cols


def build_matrices(train: pd.DataFrame, y_train: pd.Series, others: list[pd.DataFrame],
                   drop_leak: bool = True, high_card: str = "freq_te"):
    """Fit encoder pada `train` saja, terapkan ke `others` (validasi/test).

    Kategori diubah ke pandas.Categorical dengan kategori dari data latih,
    sehingga kategori baru menjadi NaN.
    """
    enc = HighCardEncoder()
    tr = pd.concat([train.reset_index(drop=True),
                    enc.fit_transform(train.reset_index(drop=True), y_train.reset_index(drop=True))], axis=1)
    outs = []
    for o in others:
        o = o.reset_index(drop=True)
        outs.append(pd.concat([o, enc.transform(o)], axis=1))
    cats = CAT_LOW + ([HIGH_CARD] if high_card == "native" else [])
    for c in cats:
        levels = sorted(tr[c].astype(str).unique())
        tr[c] = pd.Categorical(tr[c].astype(str), categories=levels)
        for o in outs:
            o[c] = pd.Categorical(o[c].astype(str), categories=levels)   # unseen -> NaN
    cols = feature_columns(tr, drop_leak=drop_leak, high_card=high_card)
    return tr[cols], [o[cols] for o in outs], cols
