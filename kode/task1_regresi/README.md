# Task 1 – Regresi: estimasi waktu tiba (P50 dan P90)

## Masalah

Perusahaan logistik ingin menjanjikan perkiraan waktu tiba saat paket diambil kurir. Model memprediksi
`durasi_jam` paket September–Oktober 2025 sebagai median (P50) dan batas atas 90% (P90).
Hanya fitur yang tersedia saat pickup yang boleh dipakai.
Penilaian: pinball loss q = 0,5 dan q = 0,9, serta coverage P90.

## Alur

1. **Audit dan pembersihan** (`audit.py`, `features.py`)
   - 370 nilai `durasi_jam` ≥ 1.000 yang bulat ternyata tercatat dalam menit, jadi dibagi 60.
   - `jarak_km` = 9999 (sentinel) dan 0 → NaN; `umur_akun_penjual_hari` > 25 tahun → NaN.
   - `jumlah_scan` dibuang (leakage, lihat tabel alasan).
2. **Fitur** (`features.py`): hari dalam minggu, akhir pekan, rute hub × pulau tujuan, berat volumetrik,
   indikator berat kosong. `kota_tujuan` (405 kategori) memakai frequency encoding + target encoding out-of-fold.
   Kategori yang tidak ada di data latih (mis. mitra kurir K13 di data uji) dipetakan ke NaN.
3. **Validasi** (`../common/validation.py`): expanding window dengan horizon 2 bulan seperti data uji.
   F1: latih < 1 Mei → validasi Mei–Jun; F2: < 1 Jun → Jun–Jul; F3: < 1 Jul → Jul–Agu.
   Tuning hanya di F1–F2, jadi F3 menjadi estimasi yang jujur.
4. **Model** (`models.py`): baseline kuantil per layanan × pulau tujuan, Ridge pada log1p(durasi) dengan kuantil
   residual, dan LightGBM quantile (α = 0,5 dan 0,9).
5. **Ketidakpastian** (`conformal.py`): split conformal di skala log diuji, global dan per layanan.
6. **Analisis** (`analysis.py`): error per layanan, pulau tujuan, dan mitra kurir; SHAP (TreeSHAP LightGBM) dan
   partial dependence.
7. **Model final** (`pipeline.py`): dilatih ulang pada seluruh data Jan–Agu, lalu memprediksi data uji.
   Setiap baris dijamin p90 ≥ p50.

## Alasan pilihan utama

| Keputusan | Alasan dan bukti |
|---|---|
| `jumlah_scan` dibuang | Scan tercatat sepanjang perjalanan sampai data ditarik, jadi belum ada saat pickup. Korelasi Spearman dengan log durasi 0,77, jauh di atas `jarak_km` (0,50). Ablasi F3: pinball q50 6,93 → 5,46 dan q90 4,37 → 2,90, perbaikan yang tidak mungkin terulang saat dipakai. |
| Validasi berbasis waktu | Data uji adalah bulan-bulan berikutnya. Random 5-fold CV memberi pinball q50 6,69 vs 6,77 dan q90 4,23 vs 4,28 di validasi waktu, sedikit lebih optimistis. |
| LightGBM quantile | Di F3, pinball q50/q90 6,93/4,37 jam, dibanding Ridge 7,33/4,53 dan baseline 10,49/6,46. |
| Kalibrasi konformal tidak dipakai | Model dilatih < 1 Mei, dikalibrasi pada Mei–Jun, diuji pada Jul–Agu. Coverage P90 turun dari 0,890 ke 0,877 dan pinball q90 memburuk 4,438 → 4,482. Coverage per fold 0,911 → 0,896 → 0,877: miskalibrasi berubah antar periode, sehingga asumsi exchangeability conformal tidak terpenuhi. Aturan pemilihan (coverage 0,9 ± 0,015, lalu pinball terkecil) ditetapkan sebelum melihat hasil. |
| Kategori baru → NaN | Mitra kurir K13 hanya ada di data uji (2.342 paket). Simulasi dengan menahan K12 dari data latih: pinball q50 6,71 → 6,81 dan coverage P90 0,894 → 0,853, tetap jauh di atas baseline. |

## Hasil (fold F3: latih < 1 Jul, validasi Jul–Agu)

| Model | Pinball q50 (jam) | Pinball q90 (jam) | Coverage P90 |
|---|---|---|---|
| Baseline kuantil grup | 10,49 | 6,46 | 0,889 |
| Ridge, log target | 7,33 | 4,53 | 0,898 |
| **LightGBM quantile** | **6,93** | **4,37** | 0,877 |

## Cara menjalankan

Dari root repo, dengan `data_uts_ml/` sudah ada:

```bash
python kode/run_all.py --task 1   # ±7–9 menit
```

Hasil: `prediksi_regresi.csv` di root repo, tabel dan grafik di `kode/outputs/task1/`
(antara lain `audit_kolom.csv`, `ablasi_leakage.csv`, `perbandingan_model.csv`, `kalibrasi_konformal.csv`,
`error_per_*.csv`, `shap_P50.png`, `pdp.png`).
