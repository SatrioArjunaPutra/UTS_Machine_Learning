"""Task 3f: profil, nama, dan aksi marketing per segmen.

Nama dibentuk dari ciri yang paling menonjol (rata-rata terstandar segmen vs
seluruh pelanggan aktif), sehingga tetap konsisten bila kode dijalankan ulang.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import record_metric, save_fig, save_table
from task3_cluster.preprocess import CAT_SHARES, TIME_SHARES

# (kolom terstandar, arah) -> (frasa nama, aksi marketing)
TRAITS = {
    ("log_belanja_90h", +1): ("belanja besar", "Program loyalitas tier premium dan akses awal produk baru"),
    ("log_belanja_90h", -1): ("belanja kecil", "Bundling hemat dan gratis ongkir minimum belanja rendah"),
    ("log_frekuensi_90h", +1): ("sering belanja", "Langganan bulanan (subscribe & save) untuk kebutuhan rutin"),
    ("log_rata2_keranjang", +1): ("keranjang besar", "Cicilan 0% dan rekomendasi produk bernilai tinggi"),
    ("persen_voucher", +1): ("pemburu voucher", "Voucher personal bernilai kecil dengan masa berlaku pendek"),
    ("persen_flash_sale", +1): ("penggemar flash sale", "Notifikasi flash sale terjadwal sesuai jam aktif"),
    ("persen_paylater", +1): ("pengguna paylater", "Promo khusus paylater dengan kontrol limit dan risiko"),
    ("persen_cod", +1): ("andalkan COD", "Insentif pindah ke pembayaran digital (cashback e-wallet)"),
    ("umur", -1): ("usia muda", "Kampanye media sosial dan kolaborasi kreator"),
    ("umur", +1): ("usia matang", "Komunikasi lewat email/WA dengan rekomendasi kebutuhan keluarga"),
    ("log_recency_hari", +1): ("mulai jarang", "Kampanye win-back: voucher kembali belanja 14 hari"),
    ("log_retur_90h", +1): ("sering retur", "Panduan ukuran/ulasan foto untuk menekan retur"),
    ("log_sesi_per_minggu", +1): ("aktif di aplikasi", "Rekomendasi personal di beranda dan push notification"),
    ("log_rasio_konversi", +1): ("konversi tinggi", "Upsell produk pelengkap di halaman checkout"),
    ("log_lama_member_bulan", +1): ("member lama", "Hadiah ulang tahun keanggotaan dan poin ganda"),
    ("log_lama_member_bulan", -1): ("member baru", "Onboarding: voucher pembelian kedua"),
}
KEY_COLS = ["belanja_90h", "frekuensi_90h", "recency_hari", "rata2_keranjang", "persen_voucher",
            "persen_flash_sale", "persen_paylater", "persen_cod", "umur", "lama_member_bulan",
            "sesi_per_minggu", "rasio_konversi", "retur_90h"]


def _traits(zmeans: pd.Series, top: int = 2) -> list[tuple[str, int]]:
    cands = []
    for (col, sign), _ in TRAITS.items():
        if col in zmeans.index:
            cands.append((col, sign, sign * zmeans[col]))
    cands = sorted([c for c in cands if c[2] > 0], key=lambda c: -c[2])
    picked, used = [], set()
    for col, sign, val in cands:
        if col in used:
            continue
        picked.append((col, sign))
        used.add(col)
        if len(picked) == top:
            break
    return picked


def build_profiles(df: pd.DataFrame, labels: np.ndarray, Zdf: pd.DataFrame, model_mask: np.ndarray) -> pd.DataFrame:
    act = labels >= 0
    glob_share = df.loc[model_mask, CAT_SHARES].mean()
    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        r = {"cluster": int(c), "n": int(m.sum()), "persen": round(100 * m.mean(), 2)}
        for k in KEY_COLS:
            r[f"median_{k}"] = round(float(df.loc[m, k].median()), 2)
        share = df.loc[m, CAT_SHARES].mean()
        idx = (share / glob_share).sort_values(ascending=False)
        r["kategori_khas"] = idx.index[0][6:] + f" ({share[idx.index[0]] * 100:.0f}%)"
        r["jam_dominan"] = df.loc[m, TIME_SHARES].mean().idxmax()[10:]
        dev = df.loc[m, "perangkat"].value_counts(normalize=True)
        r["perangkat_dominan"] = f"{dev.index[0]} ({dev.iloc[0] * 100:.0f}%)"
        if c == -1:
            r["nama"], r["aksi_marketing"] = "Anomali (bot & profil ekstrem)", "Dikecualikan dari kampanye; audit akun bot"
        else:
            mm = m & model_mask
            z = Zdf.loc[np.flatnonzero(mm)].mean()        # Zdf diindeks posisi baris df
            tr = _traits(z)
            name = " & ".join(TRAITS[t][0] for t in tr).capitalize() if tr else "Rata-rata"
            if (share[idx.index[0]] / glob_share[idx.index[0]]) > 1.5:
                name += f", fokus {idx.index[0][6:].replace('_', ' ')}"
            r["nama"] = name
            r["aksi_marketing"] = TRAITS[tr[0]][1] if tr else "Kampanye umum"
        rows.append(r)
    prof = pd.DataFrame(rows)
    front = ["cluster", "nama", "n", "persen", "aksi_marketing", "kategori_khas", "jam_dominan", "perangkat_dominan"]
    prof = prof[front + [c for c in prof.columns if c not in front]]
    save_table(prof, "task3", "profil_segmen")
    record_metric("task3", "segmen", prof.set_index("cluster")[["nama", "n", "persen"]].to_dict(orient="index"))
    return prof


def profile_heatmap(Zdf: pd.DataFrame, labels_model: np.ndarray, names: dict) -> None:
    cols = ["log_belanja_90h", "log_frekuensi_90h", "log_rata2_keranjang", "log_recency_hari", "persen_voucher",
            "persen_flash_sale", "persen_paylater", "persen_cod", "umur", "log_lama_member_bulan",
            "log_sesi_per_minggu", "log_rasio_konversi", "log_retur_90h", "h_malam", "h_pagi"]
    cols = [c for c in cols if c in Zdf.columns]
    M = Zdf[cols].groupby(labels_model).mean()
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(M) + 2.2))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("log_", "") for c in cols], rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{i}: {str(names.get(i, i)).split(',')[0]}" for i in M.index], fontsize=10.5)
    fig.colorbar(im, ax=ax, label="rata-rata terstandar")
    ax.set_title("Profil segmen aktif (merah = di atas rata-rata)")
    save_fig(fig, "task3", "profil_heatmap")
