# Kontribusi Anggota dan Penggunaan AI

Kelompok: M Marcheliano Taji Suryana dan Satrio Arjuna Putra. Pembagian kerja 50/50: masing-masing memegang satu task 30 poin, separuh Task 3, dan sebagian modul bersama.

## Peran dan pull request

| Anggota | Tanggung jawab | Modul/file | Pull request |
| --- | --- | --- | --- |
| M Marcheliano Taji Suryana | Task 1 (1a–1f); Task 3e (stabilitas) dan perakitan pipeline Task 3; modul data dan validasi waktu; entry point; pembuat laporan | `kode/config.py`, `kode/common/io.py`, `kode/common/validation.py`, `kode/common/utils.py`, `kode/run_all.py`, `kode/task1_regresi/*`, `kode/task3_cluster/stability.py`, `kode/task3_cluster/pipeline.py`, `kode/report/build_report.py` | [#1](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/1), [#2](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/2), [#3](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/3), [#4](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/4), [#5](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/5), [#6](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/6) |
| Satrio Arjuna Putra | Task 2 (2a–2g); Task 3a–3d (preprocessing, seleksi fitur, metode, jumlah cluster), 3f, 3g (profil, anomali); metrik dan validasi submission; helper PDF | `kode/common/metrics.py`, `kode/check_submission.py`, `kode/task2_fraud/*`, `kode/task3_cluster/preprocess.py`, `kode/task3_cluster/select.py`, `kode/task3_cluster/methods.py`, `kode/task3_cluster/anomaly.py`, `kode/task3_cluster/profile.py`, `kode/report/pdfkit.py` | [#7](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/7), [#8](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/8), [#9](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/9), [#10](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/10), [#11](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/11) |
| Bersama | README, kontribusi.md, laporan.pdf akhir | `kode/README.md`, `kontribusi.md`, `laporan.pdf` | [#12](https://github.com/marcheliano-satrio/UTS_Machine_Learning/pull/12) |

Setiap PR direview dan dijalankan ulang oleh anggota lain sebelum di-merge.

## Penggunaan alat AI

| Alat | Bagian yang dibantu | Cara verifikasi oleh kelompok |
| --- | --- | --- |
| Claude Code (model Claude Opus 5.5, Anthropic) | Analisis soal dan PRD; EDA awal (pola nilai menit di `durasi_jam`, kurva kematangan label fraud, `v15` sebagai proxy waktu, struktur cluster di ruang PCA); penulisan kode semua modul di `kode/`; draf teks laporan dan README | Satrio menjalankan ulang `python kode/run_all.py` di laptop sendiri (Python 3.12, versi paket sesuai `kode/requirements.txt`): MD5 ketiga CSV sama dengan `kode/outputs/md5_run3.txt`, angka di `metrics.json` identik, dan `check_submission.py` lulus 22/22 |

Keputusan akhir (fitur yang dibuang, skema validasi, aturan keputusan, jumlah cluster) telah diperiksa oleh kelompok dan dapat dijelaskan oleh kedua anggota saat ujian lisan.

## Persetujuan

- [ ] M Marcheliano Taji Suryana menyetujui isi dokumen ini
- [ ] Satrio Arjuna Putra menyetujui isi dokumen ini
