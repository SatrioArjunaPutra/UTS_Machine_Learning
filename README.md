# UTS Machine Learning 2026

Pipeline machine learning klasik untuk tiga masalah bisnis: estimasi waktu tiba paket (regresi kuantil),
deteksi fraud transaksi dengan keputusan berbasis biaya, dan segmentasi pelanggan.
Semua model memakai scikit-learn dan LightGBM. Tidak ada neural network dan tidak ada data eksternal.

**Kelompok:** M Marcheliano Taji Suryana dan Satrio Arjuna Putra

| Anggota | Bagian |
|---|---|
| M Marcheliano Taji Suryana | Task 1, Task 3e (stabilitas) dan perakitan pipeline Task 3, pemuatan data dan validasi waktu, `kode/run_all.py`, pembangun laporan |
| Satrio Arjuna Putra | Task 2, Task 3a–3d, 3f, 3g (preprocessing, seleksi fitur, metode, profil, anomali), metrik dan `kode/check_submission.py`, helper PDF |

Rincian peran, link pull request, dan penggunaan AI ada di [`kontribusi.md`](kontribusi.md).

## Ringkasan task

| Task | Masalah | Data | Model utama | File prediksi |
|---|---|---|---|---|
| 1 | Estimasi durasi pengiriman saat pickup (P50 dan P90) | 250.000 paket latih (Jan–Agu 2025), 60.000 paket uji (Sep–Okt 2025) | LightGBM quantile; pembanding: kuantil per grup dan Ridge pada log target | `prediksi_regresi.csv` |
| 2 | Deteksi fraud dan keputusan transaksi yang ditandai | 403.077 transaksi latih, 106.674 transaksi uji | LightGBM + fitur perilaku tanpa informasi masa depan, kalibrasi Platt, flag jika p × nominal > Rp25.000 | `prediksi_klasifikasi.csv` |
| 3 | Segmentasi pelanggan dan anomali | 120.000 pelanggan (ringkasan 90 hari) | log1p + Hellinger + PCA 8, Gaussian Mixture k = 7, anomali konsensus = −1 | `cluster.csv` |

Hasil utama pada validasi berbasis waktu:

| Task | Hasil |
|---|---|
| 1 | Pinball q50 6,93 jam dan q90 4,37 jam (fold Jul–Agu), 34% dan 32% lebih baik dari baseline |
| 2 | PR-AUC 0,771 (prevalensi fraud 0,64%); total cost Rp75,6 juta, dibanding Rp790,1 juta tanpa flag dan Rp2,56 miliar bila semua ditandai |
| 3 | 7 segmen dengan stabilitas ARI antar-run median 0,978; 1,47% pelanggan berlabel anomali |

**Penjelasan tiap task** (masalah, alur model, alasan setiap pilihan, hasil, cara menjalankan):

- Task 1 → [`kode/task1_regresi/README.md`](kode/task1_regresi/README.md)
- Task 2 → [`kode/task2_fraud/README.md`](kode/task2_fraud/README.md)
- Task 3 → [`kode/task3_cluster/README.md`](kode/task3_cluster/README.md)

Hasil, analisis, dan semua angka ada di `laporan.pdf`. Setiap angka berasal dari
`kode/outputs/metrics.json` dan tabel di `kode/outputs/` yang dihasilkan kode.

## Struktur repo

```
.
├── laporan.pdf                  # laporan Task 1–3 (maks. 18 halaman)
├── kontribusi.md                # peran anggota, link PR, penggunaan AI
├── prediksi_regresi.csv         # id_paket,p50,p90
├── prediksi_klasifikasi.csv     # id_transaksi,prob,tandai
├── cluster.csv                  # id_pelanggan,cluster
└── kode/                        # seluruh kode + README (lihat kode/README.md)
    ├── run_all.py               # satu perintah untuk semua angka laporan dan 3 file prediksi
    ├── check_submission.py      # validasi format 3 file prediksi (22 cek)
    ├── config.py                # seed, path, biaya review, delay label
    ├── requirements.txt         # versi library dikunci
    ├── common/                  # pemuatan data, validasi waktu, metrik, utilitas
    ├── task1_regresi/
    ├── task2_fraud/
    ├── task3_cluster/
    ├── report/                  # pembangun laporan.pdf dari kode/outputs/
    └── outputs/                 # tabel, grafik, metrics.json, log run, hash MD5
```

## Cara menjalankan

1. Instal (diuji dengan Python 3.12.10 dan 3.14.6):

   ```bash
   pip install -r kode/requirements.txt
   ```

2. Unduh `data_uts_ml.zip` dari dosen dan ekstrak ke `data_uts_ml/` di root repo.
   Data tidak disertakan di repo.

3. Reproduksi semua angka laporan dan ketiga file prediksi (dari root repo):

   ```bash
   python kode/run_all.py                  # semua task, 16–26 menit di laptop 12 thread, RAM 8 GB
   python kode/check_submission.py         # cek format ketiga file prediksi
   python kode/report/build_report.py      # bangun ulang laporan.pdf dari kode/outputs/
   ```

   Satu task saja: `--task 1`, `--task 2`, atau `--task 3`.

Detail lingkungan, seed, waktu per task, dan keputusan penting ada di [`kode/README.md`](kode/README.md).

## Catatan

- Hanya ML klasik (scikit-learn, LightGBM). Tidak ada data eksternal.
- Validasi meniru pemakaian nyata: expanding window per bulan dengan horizon 2 bulan seperti data uji.
  Tuning hanya di fold awal; fold terakhir tidak dipakai untuk tuning.
- Kolom yang tidak tersedia saat prediksi dibuang setelah dibuktikan lewat ablasi:
  `jumlah_scan` (Task 1) dan `flag_review_manual` (Task 2).
- Encoder, scaler, dan kalibrator di-fit pada data latih. Data uji hanya dipakai untuk prediksi akhir;
  fitur histori Task 2 memakai transaksi sebelumnya tanpa label.
- Seed 42 dan LightGBM deterministik. Dua laptop dengan Python berbeda menghasilkan ketiga file prediksi yang identik.
