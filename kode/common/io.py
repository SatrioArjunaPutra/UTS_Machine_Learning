"""Pemuatan data + pengecekan skema terhadap kamus_data.md."""
from __future__ import annotations

import pandas as pd

import config

REG_EXPECTED = (
    ["id_paket", "tgl_pickup", "jam_pickup", "layanan", "mitra_kurir", "hub_asal", "kota_asal",
     "pulau_tujuan", "kota_tujuan", "jarak_km", "berat_kg", "volume_liter", "jumlah_item",
     "nilai_barang", "asuransi", "cod", "alamat_skor", "pengalaman_kurir_bulan", "rating_penjual",
     "umur_akun_penjual_hari", "volume_penjual_30h", "antrean_hub_asal", "indeks_kemacetan",
     "hujan_asal_mm", "hujan_tujuan_mm", "jumlah_scan"]
    + [f"f{i:02d}" for i in range(1, 51)]
)
CLF_EXPECTED = (
    ["id_transaksi", "waktu", "id_pengguna", "id_merchant", "kategori_merchant", "kanal",
     "tipe_perangkat", "nominal", "jam", "hari_libur", "umur_akun_hari", "level_kyc",
     "umur_pengguna_tahun", "email_domain", "umur_perangkat_hari", "perangkat_baru",
     "jumlah_akun_per_perangkat", "ganti_password_7h", "gagal_login_24j", "jarak_lokasi_km",
     "negara_ip_beda", "kartu_tidak_hadir", "menit_sejak_trx_terakhir", "trx_1j", "trx_24j",
     "trx_7h", "nominal_24j", "rasio_nominal_rata2", "saldo_sebelum", "persen_saldo",
     "umur_merchant_hari", "refund_merchant_30h", "volume_merchant_30h", "flag_review_manual"]
    + [f"v{i:02d}" for i in range(1, 51)]
)


def _check(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"{name}: kolom hilang {missing}"


def load_regresi() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(config.DATA_DIR / "regresi_train.csv.gz", parse_dates=["tgl_pickup"])
    te = pd.read_csv(config.DATA_DIR / "regresi_test.csv.gz", parse_dates=["tgl_pickup"])
    _check(tr, REG_EXPECTED + ["durasi_jam"], "regresi_train")
    _check(te, REG_EXPECTED, "regresi_test")
    assert tr["id_paket"].is_unique and te["id_paket"].is_unique
    return tr, te


def load_klasifikasi() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(config.DATA_DIR / "klasifikasi_train.csv.gz", parse_dates=["waktu"])
    te = pd.read_csv(config.DATA_DIR / "klasifikasi_test.csv.gz", parse_dates=["waktu"])
    _check(tr, CLF_EXPECTED + ["is_fraud"], "klasifikasi_train")
    _check(te, CLF_EXPECTED, "klasifikasi_test")
    assert tr["id_transaksi"].is_unique and te["id_transaksi"].is_unique
    return tr, te


def load_clustering() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "clustering.csv.gz")
    assert df["id_pelanggan"].is_unique
    return df
