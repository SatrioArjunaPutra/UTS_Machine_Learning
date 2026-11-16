"""Skema validasi berbasis waktu (expanding window) dengan assert anti-leakage.

Test berisi masa depan (Sep-Okt 2025, horizon 2 bulan), jadi setiap fold
melatih pada data sebelum `val_start` dan memvalidasi pada 2 bulan berikutnya.
Untuk fraud, data latih juga dipotong `label_delay_days` sebelum `val_start`
karena label transaksi terbaru belum matang pada saat model dibuat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_end: pd.Timestamp     # eksklusif
    val_start: pd.Timestamp
    val_end: pd.Timestamp       # eksklusif


def time_folds(dates: pd.Series, val_windows: list[tuple[str, str]],
               label_delay_days: int = 0) -> list[Fold]:
    dates = pd.to_datetime(dates)
    folds = []
    for i, (vs, ve) in enumerate(val_windows, start=1):
        vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
        train_end = vs - pd.Timedelta(days=label_delay_days)
        tr = np.flatnonzero((dates < train_end).to_numpy())
        va = np.flatnonzero(((dates >= vs) & (dates < ve)).to_numpy())
        # --- assert anti-leakage ---
        assert len(tr) > 0 and len(va) > 0, f"fold {i} kosong"
        assert dates.iloc[tr].max() < vs, "data latih menyentuh periode validasi"
        assert dates.iloc[tr].max() < train_end, "data latih melewati batas delay label"
        folds.append(Fold(f"F{i}", tr, va, train_end, vs, ve))
    return folds


def describe_folds(folds: list[Fold], dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    rows = []
    for f in folds:
        rows.append({
            "fold": f.name,
            "latih_dari": dates.iloc[f.train_idx].min().date(),
            "latih_sampai": dates.iloc[f.train_idx].max().date(),
            "validasi_dari": f.val_start.date(),
            "validasi_sampai": (f.val_end - pd.Timedelta(days=1)).date(),
            "n_latih": len(f.train_idx),
            "n_validasi": len(f.val_idx),
        })
    return pd.DataFrame(rows)
