# Task 3 – Clustering: segmentasi pelanggan

## Masalah

Tim marketing ingin membagi ±120.000 pelanggan menjadi segmen yang bermakna dan bisa ditindaklanjuti.
Data tidak berlabel. Pelanggan yang tidak cocok dengan segmen mana pun diberi label −1.
Penilaian: Adjusted Rand Index (ARI) terhadap segmen referensi dosen; anomali referensi dikeluarkan dari perhitungan.

## Alur

1. **Pemisahan awal** (`preprocess.py`)
   - Bot (`sesi_per_minggu` > 100, 1.672 pelanggan) → kandidat anomali −1.
   - Pelanggan dengan `frekuensi_90h` = 0 (6.655, bukan bot) tidak ikut melatih model, karena fitur transaksinya
     tidak terdefinisi (`rata2_keranjang` kosong tepat di baris ini). Mereka dimasukkan ke segmen di langkah 8.
2. **Preprocessing** (`preprocess.py`): log1p untuk 15 kolom miring; transformasi Hellinger (akar kuadrat) untuk porsi
   kategori (7 kategori minor diamalgamasi) dan porsi jam; winsorize persentil 0,1–99,9; standardisasi;
   PCA 8 komponen (49,2% varians).
3. **Seleksi fitur** (`select.py`): bukti noise untuk `n01–n15`, η² setiap fitur terhadap clustering awal,
   dan perbandingan varian matriks fitur.
4. **Jumlah cluster dan metode** (`methods.py`): k = 3–12 dinilai dengan BIC, silhouette, dan stabilitas;
   K-Means, Gaussian Mixture, Agglomerative (Ward), dan HDBSCAN dibandingkan pada k terpilih.
5. **Model final** (`pipeline.py`): Gaussian Mixture full covariance, k = 7, `n_init` = 5.
6. **Stabilitas** (`stability.py`): 20 run pada subsampel acak 80%, ARI antar pasangan run dan terhadap model final.
7. **Anomali** (`anomaly.py`): bot, ditambah pelanggan yang masuk 0,5% teratas skor anomali pada **kedua** detektor
   (IsolationForest dan log-likelihood GMM terendah, 97 pelanggan).
8. **Pelanggan frekuensi 0** (`preprocess.assign_freq0`): dimasukkan ke 7 segmen dengan QDA pada fitur yang terisi.
9. **Profil** (`profile.py`): nama segmen dari ciri paling menonjol, profil dalam satuan asli, dan satu aksi marketing.

## Alasan pilihan utama

| Keputusan | Alasan dan bukti |
|---|---|
| `n01–n15` dibuang | Korelasi Spearman maksimum dengan kolom lain hanya 0,021, distribusinya generik (uniform, lognormal, Poisson). Kamus hanya menyebut `n01–n10`. |
| `rating_beri_rata2`, `jumlah_alamat`, `kota_tier` dibuang | η² ≈ 0 terhadap struktur cluster. `perangkat` tidak dijadikan input (biner merusak asumsi Gaussian), tetapi berasosiasi dengan segmen akhir (Cramér's V 0,26), jadi dipakai sebagai validasi eksternal. |
| Hellinger, bukan CLR | Nol eksak mencapai 18,6% (`porsi_pertanian`), sehingga CLR bergantung pada nilai pengganti nol. Silhouette 0,165 vs 0,144 dan stabilitas 0,963 vs 0,931. |
| k = 7 | BIC terkecil di antara k yang stabilitasnya ≥ 0,9 (k = 7: 0,964). |
| Gaussian Mixture | Stabilitas tertinggi (0,981) dan sejalan dengan K-Means (ARI 0,774). HDBSCAN hanya menemukan 2 cluster dengan stabilitas 0,384, karena segmen saling tumpang tindih tanpa celah kepadatan. |
| Frekuensi 0 bukan segmen tersendiri | 80,1% dari mereka punya recency < 90 hari dan 41,5% punya retur, jadi bukan kelompok "tidak belanja". QDA konsisten dengan GMM (marginal Gaussian tetap Gaussian); akurasi holdout pada pelanggan aktif 0,868, dan 74,9% masuk segmen 6. |
| Anomali konservatif | Anomali referensi dikeluarkan dari ARI, jadi label −1 tidak bisa menaikkan ARI, sedangkan −1 yang salah bisa menurunkannya. |

## Hasil

Stabilitas 20 run: ARI antar pasangan run median 0,978; ARI run vs model final median 0,985.

| Cluster | Segmen | % pelanggan | Aksi marketing |
|---|---|---|---|
| 0 | Konversi tinggi & sering belanja, fokus makanan minuman | 8,45 | Upsell produk pelengkap di halaman checkout |
| 1 | Keranjang besar & belanja besar, fokus elektronik | 12,28 | Cicilan 0% dan rekomendasi produk bernilai tinggi |
| 2 | Usia matang & sering belanja, fokus ibu bayi | 16,92 | Komunikasi lewat email/WA dengan rekomendasi kebutuhan keluarga |
| 3 | Pengguna paylater & sering retur, fokus fashion | 9,53 | Promo khusus paylater dengan kontrol limit dan risiko |
| 4 | Pemburu voucher & penggemar flash sale, fokus voucher digital | 20,98 | Voucher personal bernilai kecil dengan masa berlaku pendek |
| 5 | Member baru & belanja kecil | 10,96 | Onboarding: voucher pembelian kedua |
| 6 | Mulai jarang & belanja kecil | 19,40 | Kampanye win-back: voucher kembali belanja 14 hari |
| −1 | Anomali (bot & profil ekstrem) | 1,47 | Dikecualikan dari kampanye; audit akun bot |

## Cara menjalankan

Dari root repo, dengan `data_uts_ml/` sudah ada:

```bash
python kode/run_all.py --task 3   # ±5–8 menit
```

Hasil: `cluster.csv` di root repo, tabel dan grafik di `kode/outputs/task3/`
(antara lain `pemilihan_k.csv`, `perbandingan_metode.csv`, `perbandingan_varian_fitur.csv`,
`stabilitas_ringkas.csv`, `frekuensi0_penugasan.csv`, `profil_segmen.csv`, `profil_heatmap.png`).
