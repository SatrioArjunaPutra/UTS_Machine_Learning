# Task 2 – Klasifikasi: deteksi fraud berbasis biaya

## Masalah

Divisi pembayaran ingin menandai transaksi berisiko untuk ditinjau sebelum diproses. Model memprediksi
probabilitas `is_fraud` transaksi September–Oktober 2025, lalu memutuskan transaksi mana yang ditandai.
Fraud hanya ±0,6% dari transaksi, dan setiap transaksi yang ditandai berbiaya tinjau Rp25.000.
Penilaian: PR-AUC dari kolom `prob`, dan total cost dari kolom `tandai`:
cost = Σ nominal fraud yang lolos + 25.000 × jumlah transaksi yang ditandai.

## Alur

1. **Audit label dan fitur** (`audit.py`)
   - Kurva kematangan label: fraud rate menurut umur transaksi saat data ditarik → delay label D = 28 hari.
   - `flag_review_manual` dibuang (leakage); `v15` dan kolom hitungan kumulatif dibuang (drift terhadap waktu).
2. **Fitur perilaku** (`features.py`): merchant baru bagi pengguna, jumlah merchant berbeda sebelumnya, rasio nominal
   terhadap histori pengguna dan merchant, transaksi merchant 1 jam/24 jam terakhir (rolling `closed='left'`).
   Fitur berlabel (fraud rate merchant/pengguna, pengguna pernah fraud) hanya memakai label transaksi dengan
   waktu < min(t − D, batas latih). Label data uji tidak pernah dipakai. Ada uji future-invariance otomatis.
3. **Validasi** (`../common/validation.py`): tiga fold waktu, horizon 2 bulan. Data latih setiap fold dipotong
   D hari sebelum awal validasi (hanya label yang sudah matang), dan validasi terakhir berakhir 31 Agu − D.
4. **Model** (`models.py`): regresi logistik sebagai pembanding, LightGBM sebagai model utama. Tuning hanya di F1–F2.
5. **Imbalance**: tanpa penanganan vs class weight vs undersampling 1:20, ditambah ambang global sebagai pembanding.
6. **Kalibrasi dua tahap**: model bantu dilatih sampai 28 hari sebelum batas latih, kalibrator (Platt atau isotonic)
   dilatih pada 28 hari terakhir data latih, lalu dipakai untuk model utama.
7. **Keputusan**: tandai jika p × nominal > 25.000.
8. **Analisis** (`analysis.py`): error per kanal, kategori merchant, dan pola fraud; adversarial validation train vs uji;
   adaptasi bobot recency.
9. **Model final** (`pipeline.py`): dilatih pada transaksi sampai 3 Agu 2025 (label sudah matang).

## Alasan pilihan utama

| Keputusan | Alasan dan bukti |
|---|---|
| `flag_review_manual` dibuang | "Pernah ditinjau" bersifat retrospektif: review terjadi setelah transaksi. Fraud rate 30,4% bila 1 vs 0,13% bila 0, dan rate flag di data uji (1,23%) lebih rendah dari data latih (1,71%). Ablasi F3: PR-AUC 0,771 → 0,908, kenaikan semu. |
| `v15` dan hitungan kumulatif dibuang | `v15` berkorelasi Spearman 0,578 dengan waktu dan naik ±0,3 per bulan sampai di luar rentang data latih, tetapi AUC-nya ±0,5 di dalam setiap bulan. Membuangnya tidak menurunkan PR-AUC (0,7706 vs 0,7711), dan AUC adversarial train vs uji turun dari 0,991 ke 0,956. |
| Delay label 28 hari | Fraud rate minggu terakhir data latih 0,09%, dibanding plateau 0,62%: label transaksi terbaru belum terkonfirmasi. Tanpa pemotongan, model akan belajar bahwa transaksi baru "aman". |
| Fitur perilaku | Menaikkan PR-AUC F3 dari 0,749 ke 0,771. |
| Tanpa penanganan imbalance | Cost F3 Rp75,6 juta, dibanding Rp85,5 juta (class weight) dan Rp78,2 juta (undersampling). Weighting dan resampling merusak kalibrasi. |
| Aturan p × nominal > 25.000 | Bila ditandai, biayanya pasti Rp25.000 (fraud tertahan); bila tidak, biaya harapannya p × nominal. Ambangnya berbeda per transaksi. Pengali 0,5–2× memberi cost Rp73,8–82,5 juta, jadi pengali 1 sudah dekat optimum. |
| Kalibrasi Platt | Brier score terkecil (0,00232 vs isotonic 0,00235 dan tanpa kalibrasi 0,00241), dan urutan probabilitas (PR-AUC) tidak berubah. |

## Hasil (fold F3: latih < 7 Mei, validasi 4 Jun – 3 Agu)

| Model / strategi | PR-AUC | Total cost |
|---|---|---|
| Regresi logistik | 0,658 | – |
| LightGBM tanpa fitur perilaku | 0,749 | – |
| **LightGBM + fitur perilaku, aturan biaya** | **0,771** | **Rp75,6 juta** |
| Ambang probabilitas global | 0,771 | Rp88,8 juta |
| Tidak ada yang ditandai | – | Rp790,1 juta |
| Semua ditandai | – | Rp2.564,8 juta |

Prevalensi fraud di F3: 0,64%. Pada data uji, 1.541 transaksi (1,44%) ditandai.

## Cara menjalankan

Dari root repo, dengan `data_uts_ml/` sudah ada:

```bash
python kode/run_all.py --task 2   # ±5–8 menit
```

Hasil: `prediksi_klasifikasi.csv` di root repo, tabel dan grafik di `kode/outputs/task2/`
(antara lain `kurva_kematangan_label.png`, `ablasi_leakage.csv`, `imbalance.csv`, `kalibrasi.csv`,
`strategi_keputusan.csv`, `pola_fraud.csv`).
