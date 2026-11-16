"""Utilitas bersama: seed, logging waktu, penyimpanan tabel/grafik/metrik.

Setiap angka yang muncul di laporan disimpan lewat `save_table` atau
`record_metric`, sehingga laporan tidak pernah memuat angka ketik-tangan.
"""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import config

_METRICS_PATH = config.OUT_DIR / "metrics.json"


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for sub in ["", "task1", "task2", "task3"]:
        (config.OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    print(f"[mulai] {label}", flush=True)
    yield
    dt = time.perf_counter() - t0
    print(f"[selesai] {label}: {dt:,.1f} detik", flush=True)
    record_metric("waktu_detik", label, round(dt, 1))


def save_table(df: pd.DataFrame, task: str, name: str, index: bool = False) -> None:
    path = config.OUT_DIR / task / f"{name}.csv"
    df.to_csv(path, index=index)


def save_fig(fig, task: str, name: str) -> None:
    path = config.OUT_DIR / task / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _load_metrics() -> dict:
    if _METRICS_PATH.exists():
        return json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def record_metric(section: str, key: str, value) -> None:
    """Simpan satu angka ke outputs/metrics.json (dibaca oleh pembuat laporan)."""
    m = _load_metrics()
    m.setdefault(section, {})[key] = _to_jsonable(value)
    _METRICS_PATH.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v
