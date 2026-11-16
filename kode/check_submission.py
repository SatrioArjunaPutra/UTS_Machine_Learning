"""Validasi format ketiga file prediksi sebelum submit.

    python kode/check_submission.py

Keluar dengan kode 1 jika ada pengecekan yang gagal.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config

RESULTS: list[tuple[bool, str]] = []


def req(cond: bool, msg: str) -> None:
    RESULTS.append((bool(cond), msg))
    print(("OK    " if cond else "GAGAL ") + msg)


def _ids(name: str, col: str) -> set:
    return set(pd.read_csv(config.DATA_DIR / name, usecols=[col])[col])


def check_regresi() -> None:
    p = config.PRED_REGRESI
    req(p.exists(), f"{p.name} ada di root repo")
    if not p.exists():
        return
    r = pd.read_csv(p)
    req(list(r.columns) == ["id_paket", "p50", "p90"], "regresi: header persis id_paket,p50,p90")
    ids = _ids("regresi_test.csv.gz", "id_paket")
    req(len(r) == len(ids), f"regresi: {len(r):,} baris = {len(ids):,} paket test")
    req(r["id_paket"].is_unique, "regresi: id_paket unik")
    req(set(r["id_paket"]) == ids, "regresi: himpunan id_paket sama dengan test")
    req(r[["p50", "p90"]].notna().all().all() and np.isfinite(r[["p50", "p90"]]).all().all(),
        "regresi: tanpa NaN/inf")
    req((r["p90"] >= r["p50"]).all(), "regresi: p90 >= p50 untuk semua baris")
    req((r["p50"] > 0).all(), "regresi: p50 > 0")


def check_klasifikasi() -> None:
    p = config.PRED_KLASIFIKASI
    req(p.exists(), f"{p.name} ada di root repo")
    if not p.exists():
        return
    c = pd.read_csv(p)
    req(list(c.columns) == ["id_transaksi", "prob", "tandai"], "klasifikasi: header persis id_transaksi,prob,tandai")
    ids = _ids("klasifikasi_test.csv.gz", "id_transaksi")
    req(len(c) == len(ids), f"klasifikasi: {len(c):,} baris = {len(ids):,} transaksi test")
    req(c["id_transaksi"].is_unique, "klasifikasi: id_transaksi unik")
    req(set(c["id_transaksi"]) == ids, "klasifikasi: himpunan id_transaksi sama dengan test")
    req(c["prob"].notna().all() and c["prob"].between(0, 1).all(), "klasifikasi: prob di [0, 1] tanpa NaN")
    req(pd.api.types.is_integer_dtype(c["tandai"]) and set(c["tandai"].unique()) <= {0, 1},
        "klasifikasi: tandai integer 0/1")


def check_cluster() -> None:
    p = config.PRED_CLUSTER
    req(p.exists(), f"{p.name} ada di root repo")
    if not p.exists():
        return
    k = pd.read_csv(p)
    req(list(k.columns) == ["id_pelanggan", "cluster"], "cluster: header persis id_pelanggan,cluster")
    ids = _ids("clustering.csv.gz", "id_pelanggan")
    req(len(k) == len(ids), f"cluster: {len(k):,} baris = {len(ids):,} pelanggan")
    req(k["id_pelanggan"].is_unique, "cluster: id_pelanggan unik")
    req(set(k["id_pelanggan"]) == ids, "cluster: himpunan id_pelanggan sama dengan data")
    req(pd.api.types.is_integer_dtype(k["cluster"]), "cluster: bertipe integer")
    req((k["cluster"] >= -1).all(), "cluster: hanya -1 (anomali) atau >= 0")


def main() -> int:
    RESULTS.clear()
    check_regresi()
    check_klasifikasi()
    check_cluster()
    n_fail = sum(not ok for ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} pengecekan lulus")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
