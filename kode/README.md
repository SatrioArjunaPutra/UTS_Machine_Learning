# Kode UTS Machine Learning

Pipeline untuk tiga task UTS: regresi kuantil durasi pengiriman, deteksi fraud berbasis biaya, dan segmentasi pelanggan. Hanya memakai ML klasik (scikit-learn dan LightGBM).

## Reproduksi dengan satu perintah

```bash
pip install -r kode/requirements.txt
python kode/run_all.py
```

Perintah ini dijalankan dari **root repo** dan menghasilkan:

- `prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, dan `cluster.csv` di root repo.
- Semua tabel (`.csv`) dan grafik (`.png`) yang dikutip laporan, di `kode/outputs/task1|task2|task3/`.
- Semua angka laporan di `kode/outputs/metrics.json`.
- Validasi format ketiga CSV di akhir run (`check_submission.py`). Proses keluar dengan kode 1 jika ada yang gagal.

Laporan dibangun ulang dari hasil di atas dengan:

```bash
python kode/report/build_report.py
```

| Perintah | Kegunaan | Waktu terukur (laptop 12 thread, RAM 8 GB) |
| --- | --- | --- |
| `python kode/run_all.py` | Semua task + validasi | 16–26 menit |
| `python kode/run_all.py --task 1` | Task 1 saja (reproduksi live) | ±7–9 menit |
| `python kode/run_all.py --task 2` | Task 2 saja | ±5–8 menit |
| `python kode/run_all.py --task 3` | Task 3 saja | ±5–8 menit |
| `python kode/run_all.py --quick` | Subsampel, hanya untuk debugging | 2–4 menit per task |
| `python kode/check_submission.py` | Cek format 3 CSV | detik |

Determinisme sudah diuji: tiga run penuh menghasilkan ketiga CSV dengan hash MD5 identik (`kode/outputs/md5_run1–3.txt`). Setelah perubahan penugasan pelanggan frekuensi 0 di Task 3, `cluster.csv` berubah; run penuh ulang di Python 3.12.10 menghasilkan `prediksi_regresi.csv` dan `prediksi_klasifikasi.csv` yang identik dengan run 1–3, dan dua run Task 3 menghasilkan `cluster.csv` identik (`kode/outputs/md5_run4.txt`).

## Lingkungan

- Diuji pada Python 3.14.6 dan 3.12.10 (hasil identik). Versi library dikunci di `requirements.txt` dan dicetak di awal setiap run.
- Seed tunggal `SEED = 42` di `kode/config.py`. LightGBM memakai `deterministic=True` dan `force_row_wise=True`.
- Data dibaca dari `data_uts_ml/` di root repo (5 file `.csv.gz` + `kamus_data.md`), memakai path relatif.

## Struktur

```
kode/
├── config.py              # seed, path, biaya review, delay label
├── run_all.py             # entry point tunggal
├── check_submission.py    # validasi format 3 CSV
├── common/                # io (cek skema), validation (fold waktu + assert), metrics, utils
├── task1_regresi/         # audit, features, models, conformal, analysis, pipeline
├── task2_fraud/           # audit, features, models, analysis, pipeline
├── task3_cluster/         # preprocess, select, methods, stability, anomaly, profile, pipeline
├── report/build_report.py # laporan.pdf dari outputs/
└── outputs/               # hasil run (di-generate)
```

## Keputusan penting (detail di laporan)

| Task | Keputusan | Bukti |
| --- | --- | --- |
| 1 | `jumlah_scan` dibuang (tidak tersedia saat pickup) | `outputs/task1/ablasi_leakage.csv` |
| 1 | `durasi_jam` ≥ 1000 yang bulat dibagi 60 (tercatat dalam menit) | `outputs/task1/temuan_kualitas.csv` |
| 1 | P90 dikalibrasi dengan conformal prediction (skala log) | `outputs/task1/kalibrasi_konformal.csv` |
| 2 | `flag_review_manual` dibuang (retrospektif) | `outputs/task2/ablasi_leakage.csv` |
| 2 | `v15` dibuang (proxy waktu tanpa sinyal fraud) | `outputs/task2/adversarial_top_fitur_semua.csv` |
| 2 | Data latih dipotong D hari karena label belum matang | `outputs/task2/kurva_kematangan_label.png` |
| 2 | Flag jika `p × nominal > 25.000` | `outputs/task2/strategi_keputusan.csv` |
| 3 | `n01–n15`, `rating_beri_rata2`, `jumlah_alamat`, `kota_tier` tidak dipakai | `outputs/task3/eta2_fitur.csv`, `bukti_noise_n01_n15.csv` |
| 3 | Komposisi ditransformasi Hellinger | `outputs/task3/perbandingan_varian_fitur.csv` |
| 3 | Pelanggan `frekuensi_90h` = 0 bukan segmen sendiri; dimasukkan ke segmen aktif (QDA pada fitur yang terisi) | `outputs/task3/frekuensi0_penugasan.csv` |
| 3 | Bot dan outlier konsensus diberi label -1 | `outputs/task3/anomali_sumber.csv` |
