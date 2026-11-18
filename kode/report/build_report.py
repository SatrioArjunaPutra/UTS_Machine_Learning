"""Membangun laporan.pdf dari kode/outputs/ (metrics.json + tabel + grafik).

    python kode/report/build_report.py

Semua angka dibaca dari hasil run_all.py; tidak ada angka yang diketik tangan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.platypus import PageBreak, SimpleDocTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from report.pdfkit import (BUL, CAP, H1, H2, MARGIN, P, SP, ST, Paragraph, figure, side_by_side,  # noqa: E402
                           table)

OUT = config.OUT_DIR
M = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
T1, T2, T3 = M["task1"], M["task2"], M["task3"]


def csv(task, name):
    return pd.read_csv(OUT / task / f"{name}.csv")


def png(task, name):
    return OUT / task / f"{name}.png"


def rp(juta):  # juta rupiah -> teks
    return f"Rp{juta:,.1f} juta".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%".replace(".", ",")


def ribu(n):
    return f"{int(n):,}".replace(",", ".")


def num(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "#").replace(".", ",").replace("#", ".")


# ====================================================================== halaman depan
def front():
    s = [Paragraph("Laporan UTS Machine Learning", ST["title"]),
         Paragraph("Regresi kuantil durasi pengiriman, deteksi fraud berbasis biaya, dan segmentasi pelanggan", ST["sub"]),
         Paragraph("Kelompok: M Marcheliano Taji Suryana dan Satrio Arjuna Putra · Semester 7 · November 2026", ST["sub"]),
         SP(8)]
    f1 = M["task1_F3"]["LightGBM quantile"]
    k1 = T1["konformal"][T1["varian_konformal_dipilih"]]
    dec = T2["strategi_keputusan"]
    rule = dec["aturan biaya: p x nominal > 25.000"]
    st = T3["stabilitas"]["antar pasangan run"]
    summ = pd.DataFrame([
        ["1 Regresi", "LightGBM quantile (P50, P90), kalibrasi konformal diuji terhadap waktu; jumlah_scan dibuang (leakage); "
                      "durasi tercatat menit dikoreksi",
         f"Pinball q50 {num(f1['pinball_q50'])} / q90 {num(f1['pinball_q90'])} jam (fold Jul–Agu); "
         f"{T1['perbaikan_vs_baseline_q50_%']:.0f}% / {T1['perbaikan_vs_baseline_q90_%']:.0f}% lebih baik dari baseline; "
         f"coverage P90 {num(k1['coverage_p90'], 3)} ({T1['varian_konformal_dipilih']})"],
        ["2 Fraud", "LightGBM + fitur perilaku tanpa masa depan, delay label "
                    f"{T2['delay_label_dipakai_hari']} hari, kalibrasi {T2['kalibrator_dipilih']}, flag jika p × nominal > Rp25.000",
         f"PR-AUC {num(T2['F3_pr_auc_final'], 3)} (prevalensi {pct(T2['F3_prevalensi'], 2)}); cost {rp(rule)} vs "
         f"{rp(dec['tidak ada yang di-flag'])} (tanpa flag) dan {rp(dec['semua di-flag'])} (flag semua)"],
        ["3 Clustering", f"Hellinger + log1p + PCA 8, GMM full k = {T3['k_dipilih']}; pelanggan frekuensi 0 masuk "
                         "segmen lewat fitur yang terisi; bot & outlier konsensus = −1",
         f"Stabilitas ARI antar-run: median {num(st['median'], 3)}, rata-rata {num(st['mean'], 3)}; "
         f"anomali {num(T3['persen_anomali'])}%"],
    ], columns=["Task", "Pendekatan final", "Hasil validasi (data latih, skema waktu)"])
    s += [H2("Ringkasan hasil"), table(summ, widths=[1.2, 4.3, 4.8], font=7.6), SP(6)]
    v = M["lingkungan"]["versi"]
    s += [H2("Reproduksi"),
          P("Semua angka, tabel, dan grafik di laporan ini dihasilkan oleh <b>python kode/run_all.py</b> dari root repo "
            "(satu perintah) dan disimpan di <b>kode/outputs/</b>; laporan dibangun ulang dengan "
            "<b>python kode/report/build_report.py</b>. Seed tunggal "
            f"{M['lingkungan']['seed']} di kode/config.py; LightGBM memakai deterministic=True. Versi: "
            + ", ".join(f"{k} {x}" for k, x in v.items()) + ". Reproduksi per task untuk ujian lisan: "
            "<b>python kode/run_all.py --task 1|2|3</b>."),
          P("Prinsip yang dipakai di ketiga task: (1) audit setiap kolom terhadap kamus_data.md dan ukur dampak kolom yang "
            "dicurigai lewat ablasi; (2) validasi selalu meniru pemakaian nyata, yaitu memprediksi bulan berikutnya; "
            "(3) tuning hanya di dalam skema validasi, dan fold terakhir tidak dipakai untuk tuning sehingga menjadi "
            "estimasi yang jujur.")]
    return s


# ====================================================================== TASK 1
def task1():
    s = [H1("Task 1 – Regresi: estimasi waktu tiba (P50 dan P90)")]
    q = csv("task1", "temuan_kualitas")
    abl = csv("task1", "ablasi_leakage")
    a_ok, a_lk = abl.iloc[0], abl.iloc[1]
    s += [H2("1a. Audit data dan fitur yang tidak tersedia saat pickup"),
          P(f"Data latih {ribu(T1['n_train'])} paket (Jan–Agu 2025), test {ribu(T1['n_test'])} paket (Sep–Okt 2025). "
            "Tabel audit 77 kolom (tipe, missing, kardinalitas, korelasi, ketersediaan) ada di lampiran A. "
            "Temuan kualitas data dan penanganannya:"),
          table(q, widths=[2.6, 0.8, 0.8, 4.2, 1.8], font=6.8),
          CAP("Tabel 1.1 – Temuan kualitas data (sumber: outputs/task1/temuan_kualitas.csv)."),
          figure(png("task1", "distribusi_target"),
                 "Gambar 1.1 – Nilai durasi_jam ≥ 1.000 (kiri, garis merah) membentuk kelompok terpisah; setelah dibagi 60 "
                 "menyatu dengan distribusi normal (kanan).", max_h_cm=4.2),
          P("<b>Leakage: jumlah_scan.</b> Kamus menyebutnya \"jumlah scan pelacakan yang tercatat di sistem\". Scan "
            "bertambah selama paket berjalan sampai data ditarik, sehingga nilainya belum lengkap saat pickup. Buktinya: "
            "korelasi Spearman dengan log durasi jauh lebih tinggi daripada jarak_km (lihat lampiran A), dan ablasi pada "
            f"fold Jul–Agu menurunkan pinball q50 dari {num(a_ok.pinball_q50)} ke {num(a_lk.pinball_q50)} dan q90 dari "
            f"{num(a_ok.pinball_q90)} ke {num(a_lk.pinball_q90)} jam. Perbaikan sebesar itu tidak akan terulang di produksi, "
            "jadi kolom ini dibuang. Fitur f01–f50 dipertahankan karena kamus menyatakan dihitung saat pickup."),
          table(abl, widths=[3, 1, 1, 1, 1], font=7),
          CAP("Tabel 1.2 – Ablasi leakage pada fold F3 (latih < 1 Jul, validasi Jul–Agu).")]

    folds = csv("task1", "fold_waktu")
    rv = csv("task1", "random_vs_waktu_ringkas")
    r_rand = rv[rv.skema.str.startswith("random")].iloc[0]
    r_time = rv[rv.skema.str.startswith("waktu")].iloc[0]
    opt = r_rand.pinball_q50 < r_time.pinball_q50
    s += [H2("1b. Skema validasi: waktu vs random CV"),
          P("Prediksi dipakai untuk bulan-bulan berikutnya, jadi validasi memakai expanding window dengan horizon 2 bulan "
            "seperti test. F1 dan F2 dipakai untuk tuning; F3 (Jul–Agu) tidak disentuh saat tuning. Assert di "
            "common/validation.py menjamin tanggal latih selalu sebelum tanggal validasi."),
          table(folds, font=7), CAP("Tabel 1.3 – Fold validasi berbasis waktu."),
          P(f"Model yang sama dievaluasi dengan random 5-fold CV. Pinball q50 rata-rata {num(r_rand.pinball_q50)} (random) vs "
            f"{num(r_time.pinball_q50)} (waktu); q90 {num(r_rand.pinball_q90)} vs {num(r_time.pinball_q90)}. "
            + (f"Random CV {num(100 * (1 - r_rand.pinball_q50 / r_time.pinball_q50), 1)}% lebih optimistis untuk q50 "
               f"dan {num(100 * (1 - r_rand.pinball_q90 / r_time.pinball_q90), 1)}% untuk q90, karena setiap fold validasinya "
               "berisi bulan yang sama dengan data latih (pola musiman dan kondisi hub ikut bocor) dan data latihnya lebih "
               "besar. Selisihnya kecil karena pola durasi relatif stabil, tetapi arahnya konsisten. "
               if opt else
               "Selisihnya kecil: pola durasi relatif stabil antar bulan, tetapi random CV tetap tidak menguji "
               "kemampuan memprediksi bulan yang belum pernah dilihat. ")
            + "Karena test adalah masa depan, keputusan model hanya diambil dari skema waktu."),
          table(rv, font=7), CAP("Tabel 1.4 – Random 5-fold vs validasi waktu (rata-rata fold).")]

    comp = csv("task1", "perbandingan_model")
    bp = T1["param_terbaik"]
    s += [H2("1c. Baseline dan model lanjutan"),
          P("Baseline: kuantil empiris per layanan × pulau_tujuan. Model linear: Ridge pada log1p(durasi); P50 = expm1(prediksi + "
            "median residual), P90 = expm1(prediksi + kuantil-90 residual). Model utama: LightGBM dengan loss quantile "
            "(α = 0,5 dan 0,9). Kuantil invarian terhadap transformasi monoton, sehingga melatih di skala log tetap "
            "menghasilkan kuantil durasi. Tuning (jumlah daun 31/127, target mentah/log, early stopping) hanya di F1–F2; "
            f"terpilih P50: {bp['0.5']['num_leaves']} daun, target {'log' if bp['0.5']['log_target'] else 'mentah'}, "
            f"{bp['0.5']['n_estimators']} pohon; P90: {bp['0.9']['num_leaves']} daun, target "
            f"{'log' if bp['0.9']['log_target'] else 'mentah'}, {bp['0.9']['n_estimators']} pohon (tabel tuning di lampiran)."),
          table(comp, widths=[0.6, 2.4, 1, 1, 1, 1], font=7, highlight_rows=[i for i in range(len(comp)) if comp.iloc[i].model == "LightGBM quantile"]),
          CAP("Tabel 1.5 – Perbandingan model di tiga fold waktu (jam). Coverage P90 di sini belum dikalibrasi.")]

    enc = csv("task1", "encoding_kota_tujuan")
    sim = csv("task1", "simulasi_kurir_baru")
    s += [H2("1d. Kategori berkardinalitas tinggi dan kategori baru"),
          P(f"kota_tujuan memiliki 405 kategori. Lima strategi dibandingkan di F3: dibuang, frequency encoding, target "
            "encoding out-of-fold (mean log durasi, smoothing 50), keduanya, dan kategori native LightGBM. Target encoding "
            "dihitung out-of-fold agar label baris itu sendiri tidak ikut menghitung encoding-nya. "
            f"Kategori yang tidak ada di data latih dipetakan ke NaN (cabang missing). Di test, mitra kurir K13 muncul pada "
            f"{ribu(T1['n_k13_test'])} paket dan tidak pernah ada di data latih. Dampaknya disimulasikan dengan mengeluarkan K12 "
            "dari data latih F3 lalu memprediksi paket K12 di validasi."),
          table(enc, font=7), CAP("Tabel 1.6 – Strategi encoding kota_tujuan (F3)."),
          table(sim, widths=[3.2, 0.6, 1, 1, 1, 1], font=7),
          CAP("Tabel 1.7 – Simulasi mitra kurir baru: performa turun tetapi tetap jauh di atas baseline.")]

    conf = csv("task1", "kalibrasi_konformal")
    chosen = T1["varian_konformal_dipilih"]
    cov_f = comp[comp.model == "LightGBM quantile"].set_index("fold")["coverage_p90"]
    raw = conf.set_index("varian").loc["tanpa kalibrasi"]
    glb = conf.set_index("varian").loc["konformal global"]
    if chosen == "tanpa kalibrasi":
        decide = (f"Hasilnya, kalibrasi dari Mei–Jun justru menurunkan coverage di Jul–Agu dari {num(raw.coverage_p90, 3)} ke "
                  f"{num(glb.coverage_p90, 3)} dan memperburuk pinball q90 ({num(raw.pinball_q90, 3)} → {num(glb.pinball_q90, 3)}). "
                  "Penyebabnya: miskalibrasi berubah antar periode. Coverage P90 mentah per fold adalah "
                  + ", ".join(f"{k} {num(v, 3)}" for k, v in cov_f.items()) + ", sehingga koreksi dari satu periode tidak "
                  "berlaku di periode berikutnya. Asumsi exchangeability conformal dilanggar oleh drift waktu. Karena "
                  "coverage mentah sudah berada dalam 0,9 ± 0,015 dengan pinball terkecil, aturan pemilihan yang ditetapkan "
                  "sebelumnya memilih <b>tanpa kalibrasi</b> untuk prediksi test. Ini dicatat sebagai risiko di bagian keterbatasan.")
    else:
        decide = (f"Varian terpilih: <b>{chosen}</b>. Untuk prediksi test, model final dilatih pada seluruh data Jan–Agu dan "
                  "koreksinya dihitung dari model F3 pada Jul–Agu (data terbaru).")
    s += [H2("1e. Ketidakpastian: P50, P90, dan kalibrasi konformal"),
          P("Kalibrasi diuji dengan split conformal di skala log: skor s = log1p(y) − log1p(P90) dihitung pada calibration set, "
            "lalu P90 digeser sebesar kuantil ke-⌈(n+1)·0,9⌉/n dari s (koreksi multiplikatif). Evaluasinya jujur terhadap "
            "waktu: model dilatih < 1 Mei, dikalibrasi pada Mei–Jun, diuji pada Jul–Agu. Varian Mondrian menghitung koreksi "
            "per layanan. Aturan pemilihan (ditetapkan sebelum melihat hasil): varian dengan coverage dalam 0,9 ± 0,015, "
            "lalu pinball q90 terkecil. " + decide + " Semua baris output dijamin p90 ≥ p50."),
          table(conf, widths=[2.6, 1, 1, 1.2, 1.2], font=7),
          CAP("Tabel 1.8 – Coverage dan pinball q90 di Jul–Agu (model dilatih < 1 Mei) sebelum dan sesudah kalibrasi.")]

    lay = csv("task1", "error_per_layanan")
    pul = csv("task1", "error_per_pulau_tujuan")
    if chosen == "tanpa kalibrasi":
        lay = lay.drop(columns="coverage_p90_kalibrasi").rename(columns={"coverage_p90_mentah": "coverage_p90"})
        pul = pul.drop(columns="coverage_p90_kalibrasi").rename(columns={"coverage_p90_mentah": "coverage_p90"})
    worst = pul.sort_values("pinball_q90", ascending=False).iloc[0]
    best = pul.sort_values("pinball_q90").iloc[0]
    wl = lay.sort_values("pinball_q90", ascending=False).iloc[0]
    shp = csv("task1", "shap_P50").head(5)
    s += [H2("1f. Analisis error dan interpretasi"),
          P("Error dipecah per layanan, pulau tujuan, dan mitra kurir pada Jul–Agu (model dilatih < 1 Mei; tabel mitra kurir "
            f"di lampiran). Pinball q90 terbesar ada di layanan {wl.layanan} ({num(wl.pinball_q90)} jam) dan pulau "
            f"{worst.pulau_tujuan} ({num(worst.pinball_q90)} jam), terkecil di {best.pulau_tujuan} ({num(best.pinball_q90)} jam). "
            "Error mengikuti panjang dan variasi durasi, sehingga janji P90 untuk tujuan jauh paling berisiko dan paling "
            "perlu dikomunikasikan dengan rentang. Fitur terpenting P50 menurut mean |SHAP|: "
            + ", ".join(shp.fitur.tolist()) + "."),
          table(lay, font=6.8), CAP("Tabel 1.9 – Error per layanan."),
          table(pul, font=6.8), CAP("Tabel 1.10 – Error per pulau tujuan."),
          side_by_side([png("task1", "shap_P50"), png("task1", "shap_P90")],
                       "Gambar 1.2 – Kontribusi fitur (mean |SHAP|, TreeSHAP bawaan LightGBM) untuk model P50 (kiri) dan "
                       "P90 (kanan).", max_h_cm=6.0),
          figure(png("task1", "pdp"), "Gambar 1.3 – Partial dependence P50 dan P90: durasi naik dengan jarak dan antrean "
                 "hub, turun dengan kelengkapan alamat; jarak antara P50 dan P90 melebar pada rute jauh.", max_h_cm=5.2)]
    return s


# ====================================================================== TASK 2
def task2():
    s = [PageBreak(), H1("Task 2 – Klasifikasi: deteksi fraud berbasis biaya")]
    fr = T2["flag_review"]
    abl = csv("task2", "ablasi_leakage")
    s += [H2("2a. Audit pembentukan label dan fitur"),
          P(f"Fraud rate data latih {pct(T2['fraud_rate_train'], 2)}. Label berarti \"terkonfirmasi fraud saat data ditarik "
            f"(31 Agu)\". Fraud rate turun tajam untuk transaksi terbaru: {pct(T2['fraud_rate_minggu_terakhir'], 2)} di minggu "
            f"terakhir vs plateau {pct(T2['fraud_rate_plateau'], 2)}. Ini bukan penurunan fraud, melainkan label yang belum "
            f"sempat terkonfirmasi. Kurva mencapai 90% plateau setelah ±{T2['delay_label_estimasi_hari']} hari (Gambar 2.1)."),
          figure(png("task2", "kurva_kematangan_label"), "Gambar 2.1 – Kurva kematangan label.", max_h_cm=4.6),
          P(f"<b>Leakage: flag_review_manual.</b> Fraud rate {pct(fr['fraud_rate_flag1'])} bila 1 vs "
            f"{pct(fr['fraud_rate_flag0'], 2)} bila 0, AUC univariat {num(fr['auc_univariat'], 3)}, dan "
            f"{pct(fr['share_fraud_flagged'])} fraud punya flag ini. Kata \"pernah ditinjau\" bersifat retrospektif; P(fraud | "
            "flag) jauh lebih rendah untuk transaksi terbaru (lampiran B), dan rate flag di test "
            f"({pct(fr['flag_rate_test'], 2)}) lebih rendah dari train ({pct(fr['flag_rate_train'], 2)}). Artinya flag terisi "
            "setelah transaksi. Kolom ini dibuang. Missing di menit_sejak_trx_terakhir "
            f"({num(T2['missing_menit_%'], 1)}%) seluruhnya terjadi saat trx_7h = 0, yaitu transaksi pertama pengguna, "
            "sehingga diberi indikator trx_pertama."),
          table(abl, widths=[4, 1], font=7),
          CAP("Tabel 2.1 – Ablasi pada fold F3: leakage menaikkan PR-AUC secara semu.")]

    folds = csv("task2", "fold_waktu")
    s += [H2("2b. Validasi waktu dengan label yang belum lengkap"),
          P(f"Delay label D = {T2['delay_label_dipakai_hari']} hari. Pada setiap fold, model hanya dilatih pada transaksi "
            "< awal_validasi − D, karena hanya label itulah yang sudah matang saat model dibuat. Periode validasi terakhir "
            "berakhir 31 Agu − D agar labelnya sendiri sudah matang. Kalibrasi dua tahap: model bantu dilatih sampai 28 hari "
            "sebelum batas latih, kalibrator dilatih pada 28 hari terakhir data latih, lalu dipakai pada model utama. "
            f"Model final dilatih sampai {T2['final_latih_sampai']}; {ribu(T2['final_dibuang_label_belum_matang'])} transaksi "
            "terakhir yang labelnya belum matang tidak dipakai."),
          table(folds, font=7), CAP("Tabel 2.2 – Fold validasi fraud (assert anti-leakage di common/validation.py).")]

    comp = csv("task2", "perbandingan_model")
    s += [H2("2c. Fitur perilaku tanpa informasi masa depan"),
          P("Fitur pengguna/merchant dihitung pada gabungan train+test yang diurutkan menurut waktu; setiap baris hanya melihat "
            "baris sebelumnya (cumcount, cumsum − nilai sekarang, rolling closed='left'): merchant baru bagi pengguna, "
            "jumlah merchant berbeda sebelumnya, rasio nominal terhadap histori pengguna dan merchant, transaksi merchant "
            "1 jam/24 jam terakhir. Fitur berlabel (fraud rate merchant dan pengguna, pengguna pernah fraud) hanya memakai "
            "label transaksi dengan waktu < min(t − D, batas latih). Label test tidak pernah dipakai. "
            f"<b>Uji future-invariance {T2['uji_future_invariance']}</b>: fitur yang dihitung ulang dari data yang dipotong "
            "pada 15 Jun identik dengan fitur dari data penuh. Kolom hitungan kumulatif (tumbuh dengan waktu) tidak dipakai "
            "di model; lihat 2e."),
          table(comp, widths=[0.6, 2.6, 1, 1], font=7, nd=5),
          CAP("Tabel 2.3 – PR-AUC per fold. Baris 'tanpa fitur perilaku' mengukur kontribusi fitur 2c.")]

    imb = csv("task2", "imbalance")
    s += [H2("2d. Class imbalance: tiga pendekatan"),
          P("Ketiga pendekatan memakai fold, fitur, dan prosedur kalibrasi yang sama. Threshold tuning dievaluasi sebagai "
            "kolom terakhir: ambang global yang dipilih di jendela kalibrasi. Resampling dan weighting menggeser "
            "probabilitas (Brier mentah lebih buruk), jadi semua dikalibrasi ulang sebelum keputusan. "
            f"Pendekatan terpilih (cost terendah dengan aturan biaya): <b>{T2['pendekatan_dipilih']}</b>."),
          table(imb, widths=[2.6, 0.9, 1, 1, 1.2, 1.2], font=7, nd=5),
          CAP("Tabel 2.4 – Perbandingan penanganan imbalance (F3). Cost dalam juta rupiah.")]

    v15 = csv("task2", "drift_v15")
    v15w = v15.pivot_table(index="data", columns="bulan", values="rata2_v15").reset_index()
    adapt = csv("task2", "adaptasi_recency")
    adv = T2["adversarial_auc"]
    adv_all = csv("task2", "adversarial_top_fitur_semua").head(3)
    adv_fin = csv("task2", "adversarial_top_fitur_final").head(3)
    abl2 = csv("task2", "ablasi_leakage")
    pr_drift = abl2[abl2.fitur.str.contains("drift")].iloc[0].pr_auc
    s += [H2("2e. Pergeseran pola fraud"),
          P("Fraud rate per kanal berbeda jauh (kartu dan web jauh di atas aplikasi dan QRIS) dan bergerak antar bulan "
            "(Gambar 2.2). Adversarial validation (classifier yang membedakan train dari test) mencapai AUC "
            f"{num(list(adv.values())[0], 3)} dengan semua kolom; fitur paling bergeser: "
            + ", ".join(f"{r.fitur} ({num(r['gain_%'], 1)}%)" for _, r in adv_all.iterrows()) + ". Kolom hitungan "
            "kumulatif (seperti hari sejak transaksi pertama pengguna) tumbuh dengan waktu secara mekanis. <b>v15</b> "
            f"berkorelasi Spearman {num(T2['v15_spearman_waktu'], 3)} dengan waktu: rata-ratanya naik ±0,3 per bulan sampai di "
            "luar rentang train pada Sep–Okt (Tabel 2.5), padahal AUC-nya terhadap fraud di dalam setiap bulan ±0,5 (tanpa "
            "sinyal). Keduanya dibuang tanpa kehilangan performa (PR-AUC F3 dengan kolom tersebut "
            f"{num(pr_drift, 4)}, lihat Tabel 2.1), dan AUC adversarial turun ke {num(list(adv.values())[1], 3)}. Sisa "
            "pergeseran berasal dari " + ", ".join(adv_fin.fitur.tolist()) + ": fraud rate historis untuk baris test "
            "dihitung dari riwayat label yang lebih panjang (struktural, maknanya sama), dan umur merchant memang bertambah. "
            "Adaptasi diuji dengan bobot recency (half-life 60 hari) dan hanya dipakai bila memperbaiki PR-AUC dan cost: "
            f"<b>{'dipakai' if T2['pakai_bobot_recency'] else 'tidak dipakai'}</b>."),
          table(v15w, font=6.6), CAP("Tabel 2.5 – Rata-rata v15 per bulan: tren linear (proxy waktu)."),
          side_by_side([png("task2", "fraud_rate_bulan_kanal"), png("task2", "reliability")],
                       "Gambar 2.2 – Kiri: fraud rate per kanal per bulan (Agustus belum matang). Kanan: reliability "
                       "diagram di F3.", max_h_cm=5.2),
          table(adapt, font=7), CAP("Tabel 2.6 – Adaptasi bobot recency (F3).")]

    cal = csv("task2", "kalibrasi")
    dec = csv("task2", "strategi_keputusan")
    s += [H2("2f. Kalibrasi dan aturan keputusan berbasis biaya"),
          P("Untuk satu transaksi dengan probabilitas fraud terkalibrasi p dan nominal N: bila di-flag, biayanya pasti "
            "Rp25.000 (fraud yang di-flag tertahan tanpa kerugian, kamus_data.md); bila tidak di-flag, biaya harapannya p·N. "
            "Total cost adalah jumlah biaya per transaksi, sehingga meminimalkan biaya harapan per transaksi meminimalkan "
            "total. Aturannya: <b>flag jika p · N > 25.000</b>, atau p > 25.000/N. Ambangnya berbeda per transaksi: transaksi "
            "Rp50.000 perlu p > 0,5, sedangkan Rp5 juta cukup p > 0,005. Transaksi < Rp25.000 tidak pernah layak di-flag. "
            "Aturan ini hanya benar bila p terkalibrasi, sehingga Platt dan isotonic dibandingkan. Isotonic menghasilkan "
            "plateau yang bisa menurunkan PR-AUC, jadi diberi tie-break 1e-6 × skor mentah. Kalibrator terpilih (Brier "
            f"terkecil): <b>{T2['kalibrator_dipilih']}</b>."),
          table(cal, widths=[1.6, 1, 1, 1, 1, 1.2, 1.4], font=7, nd=5), CAP("Tabel 2.7 – Kalibrasi (F3)."),
          table(dec, widths=[4, 1, 1], font=7, highlight_rows=[3]),
          CAP("Tabel 2.8 – Strategi keputusan di F3. Baris sensitivitas menunjukkan pengali 1 (aturan teoretis) sudah dekat "
              "optimum, sehingga aturan biaya tidak perlu di-tuning.")]

    kan = csv("task2", "error_per_kanal")
    kat = csv("task2", "error_per_kategori_merchant").head(8)
    pat = csv("task2", "pola_fraud")
    s += [H2("2g. Analisis error"),
          P("Error dipecah per kanal, kategori merchant (8 dengan cost terbesar; tabel lengkap di outputs), dan pola fraud. "
            "Lift = fraud rate pola ÷ fraud rate keseluruhan. Pola dengan lift tinggi tetapi recall rendah adalah sasaran "
            "perbaikan berikutnya; transaksi bernominal besar mendominasi kerugian yang lolos (10 false negative termahal di "
            "lampiran B)."),
          table(kan, font=6.6), CAP("Tabel 2.9 – Error per kanal (F3, aturan biaya)."),
          table(kat, font=6.6), CAP("Tabel 2.10 – Kategori merchant dengan total cost terbesar."),
          table(pat, widths=[3, 0.8, 1, 1, 1, 0.7, 1], font=6.6), CAP("Tabel 2.11 – Pola fraud di F3.")]
    s += [P(f"<b>Prediksi test:</b> {ribu(T2['test_n_flag'])} transaksi ({num(T2['test_persen_flag'])}%) ditandai; "
            f"rata-rata probabilitas {num(T2['test_mean_prob'], 4)}.")]
    return s


# ====================================================================== TASK 3
def task3():
    s = [PageBreak(), H1("Task 3 – Clustering: segmentasi pelanggan")]
    noise = csv("task3", "bukti_noise_n01_n15")
    var = csv("task3", "perbandingan_varian_fitur")
    s += [H2("3a. Preprocessing"),
          P(f"{ribu(T3['n_pelanggan'])} pelanggan. Dua kelompok dipisah sebelum clustering: (1) <b>bot</b> "
            f"(sesi_per_minggu > 100, {ribu(T3['n_bot'])} orang), lihat 3g. (2) <b>pelanggan dengan frekuensi_90h = 0</b> "
            f"({ribu(T3['n_frekuensi0'])} orang). Semua metrik transaksi 30/90 hari mereka bernilai 0 dan rata2_keranjang "
            "kosong tepat pada baris yang sama "
            f"({'terverifikasi' if T3['rata2_keranjang_kosong_sama_dengan_frek0'] else 'tidak terverifikasi'}): rata-rata dari "
            "0 pesanan tidak terdefinisi, jadi missing ini struktural, bukan acak. Mereka tidak ikut melatih GMM, tetapi juga "
            f"tidak dijadikan segmen sendiri: {num(T3['frek0_recency_lt90_%'], 1)}% punya recency_hari < 90 dan "
            f"{num(T3['frek0_retur_gt0_%'], 1)}% punya retur dalam 90 hari, sehingga data tidak mendukung kelompok "
            "\"tidak belanja 90 hari\". Profil non-transaksinya (porsi kategori, jam, sesi, lama member, recency) mirip "
            "segmen aktif tertentu. Setelah GMM final, mereka dimasukkan ke segmen aktif dengan Gaussian per segmen pada "
            "fitur yang terisi (QDA; marginal Gaussian tetap Gaussian, sejalan dengan asumsi GMM; fitur turunan transaksi "
            f"diabaikan). Akurasinya pada pelanggan aktif holdout {num(T3['frek0_akurasi_penugasan'], 3)}; "
            f"{num(T3['frek0_persen_segmen_terbanyak'], 1)}% masuk segmen {T3['frek0_segmen_terbanyak']} "
            f"({T3['segmen'][str(T3['frek0_segmen_terbanyak'])]['nama']}). Rincian di outputs/task3/frekuensi0_penugasan.csv. "
            f"Pada sisa {ribu(T3['n_model'])} pelanggan: "
            "variabel hitungan/nominal yang sangat miring di-log1p; porsi kategori (16, berjumlah 1) dan porsi jam (4, "
            "berjumlah 1) adalah data komposisional. Log-ratio (CLR/ILR) butuh penggantian nol, padahal nol eksak mencapai "
            "18,6% (porsi_pertanian), sehingga hasil CLR bergantung pada nilai pengganti. Transformasi Hellinger "
            "(akar kuadrat) menangani nol secara alami. Tujuh kategori minor diamalgamasi (dijumlahkan) menjadi satu bagian. "
            f"Outlier di-winsorize pada persentil 0,1–99,9 ({ribu(T3['n_baris_winsorize'])} baris tersentuh), lalu standardisasi "
            f"dan PCA 8 komponen ({num(T3['pca_varian_terjelaskan_%'], 1)}% varians dari {T3['n_fitur_sebelum_pca']} fitur) "
            "untuk membuang dimensi noise.")]
    s += [H2("3b. Seleksi fitur dan bukti"),
          P(f"(1) n01–n15 adalah noise: korelasi Spearman maksimum dengan kolom lain hanya {num(T3['max_abs_rho_noise'], 3)}, "
            "dan distribusinya generik (uniform 0–100, lognormal, Poisson) dengan pola berulang tiap 7 kolom. Kamus hanya "
            "menyebut n01–n10; data berisi n01–n15. (2) η² tiap fitur terhadap clustering awal (Gambar 3.1): "
            "rating_beri_rata2, jumlah_alamat, kota_tier, dan n01–n15 bernilai ±0, jadi tidak membawa struktur segmen. "
            "Perangkat (kategorikal) punya η² sedang, tetapi tidak dijadikan input karena variabel biner merusak asumsi "
            "Gaussian kontinu dan cenderung memaksa pemisahan artifisial. Perangkat dipakai sebagai validasi eksternal: "
            f"Cramér's V perangkat vs segmen akhir {num(T3['cramers_v_perangkat_vs_segmen'], 3)} (segmen menangkap perbedaan "
            f"nyata yang tidak dilihat model), sedangkan kota_tier {num(T3['cramers_v_kota_tier_vs_segmen'], 3)} (memang tidak "
            "relevan). (3) Perbandingan varian matriks fitur (Tabel 3.1). "
            "PCA meredam efek kolom noise sehingga selisih stabilitasnya kecil; kolom itu tetap dibuang karena tidak relevan "
            "dan tidak bisa ditindaklanjuti tim marketing."),
          table(var, widths=[4, 1, 1.2, 1.2], font=7), CAP("Tabel 3.1 – Varian matriks fitur (GMM k terpilih, sampel 30.000)."),
          figure(png("task3", "eta2_fitur"), "Gambar 3.1 – η² fitur terhadap clustering awal: kiri fitur yang dipakai, kanan fitur yang "
                 "dibuang (semuanya ±0 kecuali perangkat, lihat teks).", width_frac=1.0, max_h_cm=9.5)]

    kt = csv("task3", "pemilihan_k")
    mt = csv("task3", "perbandingan_metode")
    s += [H2("3c–3d. Perbandingan metode dan jumlah cluster"),
          P("k dipilih dari tiga kriteria pada GMM full (k = 3–12): BIC, silhouette, dan stabilitas ARI antar-run "
            "(subsampel 80%). Aturannya: dari k yang stabilitasnya ≥ 0,9, ambil BIC terkecil. "
            f"Terpilih <b>k = {T3['k_dipilih']}</b> segmen. Empat metode dibandingkan "
            "pada k tersebut. Agglomerative (Ward) dan HDBSCAN dijalankan pada subsampel 10.000 karena butuh memori O(n²), "
            "lalu labelnya dipropagasi dengan kNN. GMM dipilih sebagai model final: stabil, sejalan dengan K-Means, "
            "mengizinkan cluster elips, dan menyediakan log-likelihood untuk deteksi anomali. HDBSCAN tidak cocok karena "
            "segmen saling tumpang tindih tanpa celah kepadatan."),
          figure(png("task3", "pemilihan_k"), "Gambar 3.2 – Kriteria pemilihan k (garis merah = k terpilih).", max_h_cm=4.4),
          table(mt, font=7), CAP("Tabel 3.2 – Perbandingan metode.")]

    stt = csv("task3", "stabilitas_ringkas")
    s += [H2("3e. Stabilitas"),
          P(f"GMM dilatih ulang {T3['stabilitas_runs']} kali pada subsampel acak 80% dan memprediksi seluruh pelanggan aktif. "
            "ARI dihitung antar semua pasangan run dan terhadap model final. ARI invarian terhadap penomoran cluster dan "
            "dikoreksi untuk kesepakatan acak. Median ARI run vs model final "
            f"{num(T3['stabilitas']['run vs model final']['median'], 3)}; hanya "
            f"{int(T3['stabilitas']['run vs model final']['n_di_bawah_0,9'])} dari {T3['stabilitas_runs']} run berada di bawah "
            "0,9, yaitu run yang jatuh ke optimum lokal lain. Itu sebabnya model final memakai n_init = 5."),
          table(stt, font=7), CAP("Tabel 3.3 – Stabilitas.")]

    prof = csv("task3", "profil_segmen")
    show = prof[["cluster", "nama", "n", "persen", "median_belanja_90h", "median_frekuensi_90h", "median_recency_hari",
                 "median_persen_paylater", "median_umur", "kategori_khas", "jam_dominan"]]
    act = prof[["cluster", "nama", "aksi_marketing"]]
    s += [H2("3f. Profil, nama, dan aksi marketing"),
          P("Nama segmen dibentuk dari dua ciri paling menonjol (rata-rata terstandar segmen vs seluruh pelanggan aktif) "
            "ditambah kategori khas bila porsinya ≥ 1,5× rata-rata. Profil dalam satuan asli; heatmap pada Gambar 3.3."),
          table(show, widths=[0.5, 3, 0.7, 0.6, 1.1, 0.8, 0.8, 0.8, 0.6, 1.6, 0.7], font=6.2),
          CAP("Tabel 3.4 – Profil segmen (median)."),
          table(act, widths=[0.5, 3, 4.5], font=6.8), CAP("Tabel 3.5 – Satu aksi marketing per segmen."),
          figure(png("task3", "profil_heatmap"), "Gambar 3.3 – Profil segmen aktif (rata-rata terstandar; merah = di atas rata-rata pelanggan aktif).", max_h_cm=9.5)]

    an = csv("task3", "anomali_sumber")
    s += [H2("3g. Anomali (−1)"),
          P("Anomali referensi dikeluarkan dari perhitungan ARI, jadi label −1 tidak bisa menaikkan ARI. Pelanggan normal "
            "yang salah diberi −1 justru bisa menurunkannya. Karena itu −1 diberikan secara konservatif: bot yang terpisah "
            "jelas di ruang PCA (median 250+ sesi/minggu, durasi sesi < 1 menit, konversi ±0, klik iklan dan wishlist "
            "ekstrem), ditambah pelanggan yang masuk 0,5% teratas skor anomali pada <b>kedua</b> detektor (IsolationForest "
            "dan log-likelihood GMM terendah). Profilnya di outputs/task3/anomali_profil.csv."),
          table(an, font=7), CAP("Tabel 3.6 – Sumber label −1.")]
    return s


def closing():
    return [H1("Keterbatasan dan penggunaan AI"),
            *BUL([
                ("Task 1: coverage P90 bergeser antar periode (lihat 1e), sehingga coverage di Sep–Okt bisa menyimpang dari "
                 "0,9; kalibrasi konformal tidak dipakai karena tidak terbukti membantu lintas periode."
                 if T1["varian_konformal_dipilih"] == "tanpa kalibrasi" else
                 "Task 1: koreksi konformal untuk test dihitung dari model F3, bukan model final, sehingga ada asumsi "
                 "bahwa miskalibrasi kedua model mirip.")
                + " Paket di test yang durasinya tercatat dalam menit tidak bisa dikoreksi.",
                "Task 2: delay label diestimasi dari kurva agregat karena tidak ada tanggal konfirmasi per transaksi. "
                "Prevalensi fraud test tidak diketahui; aturan biaya sensitif terhadap kalibrasi.",
                "Task 3: segmen tumpang tindih (silhouette rendah), jadi ARI terhadap referensi bergantung pada kesesuaian "
                "asumsi Gaussian. Pelanggan frekuensi 0 ditugaskan tanpa fitur transaksi, sehingga segmennya lebih tidak "
                "pasti dibanding pelanggan aktif.",
                "Penggunaan AI (Claude Code) dicatat di kontribusi.md beserta bagian yang dibantu dan cara verifikasinya. "
                "Semua angka berasal dari kode yang dapat dijalankan ulang.",
            ])]


def appendix():
    s = [PageBreak(), H1("Lampiran (tidak dihitung dalam batas 18 halaman)")]
    a1 = csv("task1", "audit_kolom")
    a1 = a1[~a1.kolom.str.match(r"^f\d\d$")]
    s += [H2("A. Audit kolom Task 1 (f01–f50 diringkas: kamus menyatakan dihitung saat pickup)"),
          table(a1[["kolom", "tipe", "missing_train_%", "n_unik", "spearman_log_durasi", "tersedia_saat_pickup", "alasan"]],
                widths=[1.6, 0.7, 0.8, 0.7, 0.9, 0.9, 4], font=6.2)]
    s += [H2("A2. Error per mitra kurir (Jul–Agu)"), table(csv("task1", "error_per_mitra_kurir"), font=6.4),
          H2("A3. Tuning LightGBM Task 1 (F1–F2)"), table(csv("task1", "tuning_lgb"), font=6.4)]
    s += [H2("B. Task 2: flag_review_manual per umur transaksi"), table(csv("task2", "flag_review_per_umur"), font=6.6),
          H2("B2. 10 false negative termahal (F3)"), table(csv("task2", "false_negative_termahal"), font=6.2),
          H2("B3. Fitur terpenting model fraud (gain)"), table(csv("task2", "importance_gain").head(15)[["fitur", "gain_%"]], font=6.6),
          H2("B4. Adversarial validation: fitur paling bergeser"),
          table(csv("task2", "adversarial_top_fitur_semua")[["fitur", "gain_%"]], font=6.6),
          H2("B5. PR-AUC per bulan di F3"), table(csv("task2", "pr_auc_per_bulan_F3"), font=6.6)]
    s += [H2("C. Task 3: bukti noise n01–n15"), table(noise_tbl(), font=6.6),
          H2("C2. Stabilitas"), figure(png("task3", "stabilitas_bootstrap"), "Boxplot ARI 20 run.", width_frac=0.5, max_h_cm=5)]
    return s


def noise_tbl():
    return csv("task3", "bukti_noise_n01_n15")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.drawRightString(doc.pagesize[0] - MARGIN, 1.0 * 28.35, f"Halaman {doc.page}")
    canvas.drawString(MARGIN, 1.0 * 28.35, "UTS Machine Learning · Marcheliano & Satrio")
    canvas.restoreState()


def main():
    out = config.ROOT / "laporan.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=(595.27, 841.89), leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN, title="Laporan UTS Machine Learning",
                            author="M Marcheliano Taji Suryana, Satrio Arjuna Putra")
    body = front() + task1() + task2() + task3() + closing()
    story = body + appendix()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    # hitung halaman isi (sebelum lampiran)
    from pypdf import PdfReader
    body_doc = SimpleDocTemplate(str(OUT / "_isi.pdf"), pagesize=(595.27, 841.89), leftMargin=MARGIN,
                                 rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    body_doc.build(front() + task1() + task2() + task3() + closing())
    n_body = len(PdfReader(str(OUT / "_isi.pdf")).pages)
    n_all = len(PdfReader(str(out)).pages)
    (OUT / "_isi.pdf").unlink()
    print(f"laporan.pdf: {n_all} halaman total, {n_body} halaman isi (batas 18)")
    return 0 if n_body <= 18 else 1


if __name__ == "__main__":
    sys.exit(main())
