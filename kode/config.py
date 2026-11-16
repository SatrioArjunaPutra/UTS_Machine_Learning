"""Konfigurasi global: seed, path, dan konstanta bisnis.

Semua modul mengambil nilai dari sini supaya hasil bisa direproduksi
dengan satu perintah (python kode/run_all.py).
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]          # root repo
DATA_DIR = ROOT / "data_uts_ml"
KODE_DIR = ROOT / "kode"
OUT_DIR = KODE_DIR / "outputs"                       # tabel, grafik, metrics.json

# File prediksi wajib di root repo (nama persis sesuai soal)
PRED_REGRESI = ROOT / "prediksi_regresi.csv"
PRED_KLASIFIKASI = ROOT / "prediksi_klasifikasi.csv"
PRED_CLUSTER = ROOT / "cluster.csv"

# Data latih ditarik 31 Agustus 2025 (kamus_data.md)
DATA_PULL_DATE = "2025-09-01"   # batas eksklusif: semua waktu train < tanggal ini

# Task 2: parameter biaya (kamus_data.md)
REVIEW_COST = 25_000
# Delay kematangan label fraud (hari). Diestimasi di task2_fraud/audit.py dari
# kurva fraud rate vs umur transaksi; nilai ini dipakai bila estimasi dimatikan.
LABEL_DELAY_DAYS = 28

# Task 1: target kuantil
QUANTILES = (0.5, 0.9)
TARGET_COVERAGE = 0.9

# Jumlah thread; LightGBM deterministic=True tetap reproducible dengan nilai ini
N_JOBS = 8
