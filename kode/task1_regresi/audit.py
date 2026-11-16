"""Task 1a: audit kualitas data dan ketersediaan fitur saat pickup."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import record_metric, save_fig, save_table

# Ketersediaan per kolom berdasarkan kamus_data.md + bukti EDA
AVAILABILITY = {
    "id_paket": ("ID", "Identitas, bukan fitur"),
    "tgl_pickup": ("Ya", "Waktu pickup; dipakai untuk split dan hari dalam minggu"),
    "jumlah_scan": ("Tidak", "Scan pelacakan tercatat sepanjang perjalanan sampai data ditarik; "
                             "korelasi Spearman dengan log durasi jauh di atas jarak"),
    "durasi_jam": ("Label", "Target, hanya di data latih"),
}
DEFAULT_AVAIL = ("Ya", "Kamus: diketahui saat pickup (profil, rute, prakiraan, fitur internal f01-f50)")


def audit_table(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    y = np.log1p(tr["durasi_jam"].clip(upper=999))
    rows = []
    for c in tr.columns:
        s = tr[c]
        num = pd.api.types.is_numeric_dtype(s)
        avail, reason = AVAILABILITY.get(c, DEFAULT_AVAIL)
        rho = s.corr(y, method="spearman") if num and c != "durasi_jam" else np.nan
        rows.append({
            "kolom": c, "tipe": str(s.dtype), "missing_train_%": round(100 * s.isna().mean(), 2),
            "missing_test_%": round(100 * te[c].isna().mean(), 2) if c in te else np.nan,
            "n_unik": s.nunique(), "min": s.min() if num else "", "max": s.max() if num else "",
            "spearman_log_durasi": round(rho, 3) if pd.notna(rho) else "",
            "tersedia_saat_pickup": avail, "alasan": reason,
        })
    df = pd.DataFrame(rows)
    save_table(df, "task1", "audit_kolom")
    return df


def _id(x, nd: int = 1) -> str:
    """Format angka gaya Indonesia (ribuan '.', desimal ',')."""
    return f"{x:,.{nd}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def quality_findings(tr_raw: pd.DataFrame, te_raw: pd.DataFrame, unit_mask: pd.Series,
                     clean_stats: dict, clean_stats_te: dict) -> pd.DataFrame:
    y = tr_raw["durasi_jam"]
    big = y[unit_mask]
    normal = y[y < 300]
    rows = [
        ("durasi_jam ≥ 1.000 dan bulat (tercatat dalam menit)", int(unit_mask.sum()), "",
         f"100% bulat vs {100 * (normal % 1 == 0).mean():.0f}% pada data normal; "
         f"median/60 = {_id(np.median(big / 60))} jam vs median normal {_id(np.median(normal))} jam",
         "Dibagi 60"),
        ("jarak_km = 9999 (sentinel)", clean_stats["jarak_9999"], clean_stats_te["jarak_9999"],
         "Nilai tepat 9999 berulang", "Diganti NaN"),
        ("jarak_km = 0", clean_stats["jarak_0"], clean_stats_te["jarak_0"],
         "Jarak rute nol tidak masuk akal untuk antar kota", "Diganti NaN"),
        ("umur_akun_penjual_hari > 25 tahun", clean_stats["umur_akun_mustahil"],
         clean_stats_te["umur_akun_mustahil"], f"Maks {_id(tr_raw['umur_akun_penjual_hari'].max(), 0)} hari",
         "Diganti NaN"),
        ("berat_kg kosong", int(tr_raw["berat_kg"].isna().sum()), int(te_raw["berat_kg"].isna().sum()),
         f"Median durasi {_id(tr_raw.loc[tr_raw.berat_kg.isna(), 'durasi_jam'].median())} jam bila kosong vs "
         f"{_id(tr_raw.loc[tr_raw.berat_kg.notna(), 'durasi_jam'].median())} jam bila terisi",
         "Dibiarkan NaN + indikator berat_hilang"),
        ("mitra_kurir hanya di test (K13)", 0,
         int((~te_raw["mitra_kurir"].isin(tr_raw["mitra_kurir"].unique())).sum()),
         "Kategori baru", "Dipetakan ke NaN (cabang missing)"),
        ("Duplikat id_paket", int(tr_raw["id_paket"].duplicated().sum()),
         int(te_raw["id_paket"].duplicated().sum()), "", "-"),
    ]
    df = pd.DataFrame(rows, columns=["temuan", "n_train", "n_test", "bukti", "penanganan"])
    save_table(df, "task1", "temuan_kualitas")
    record_metric("task1", "n_unit_menit", int(unit_mask.sum()))
    record_metric("task1", "n_jarak_9999_train", clean_stats["jarak_9999"])
    record_metric("task1", "n_k13_test", int((te_raw["mitra_kurir"] == "K13").sum()))
    return df


def plot_target(y_raw: pd.Series, y_fixed: pd.Series) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(np.log10(y_raw), bins=120, color="#555")
    ax[0].set_title("durasi_jam mentah (log10)")
    ax[0].axvline(np.log10(1000), color="red", ls="--", lw=1)
    ax[0].set_xlabel("log10 jam")
    ax[1].hist(np.log10(y_fixed), bins=120, color="#2b6cb0")
    ax[1].set_title("Setelah nilai menit dibagi 60")
    ax[1].set_xlabel("log10 jam")
    save_fig(fig, "task1", "distribusi_target")
