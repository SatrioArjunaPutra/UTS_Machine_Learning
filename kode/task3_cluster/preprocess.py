"""Task 3a: preprocessing.

- Pelanggan tidak aktif (frekuensi_90h = 0) dipisah menjadi segmen sendiri:
  rata2_keranjang kosong tepat pada baris ini (rata-rata dari 0 pesanan tidak
  terdefinisi), dan semua metrik transaksi 30/90 hari bernilai 0.
- Bot (sesi_per_minggu > 100) dipisah sebagai kandidat anomali (anomaly.py).
- Variabel hitungan/nominal miring -> log1p.
- Komposisi (porsi kategori, porsi jam): transformasi Hellinger (akar kuadrat).
  Banyak nol eksak (hingga 18,6% di porsi_pertanian) membuat log-ratio (CLR/ILR)
  bergantung pada nilai pengganti nol; perbandingan di select.py menunjukkan
  Hellinger memberi silhouette dan stabilitas lebih baik. 7 kategori minor
  diamalgamasi menjadi satu bagian "lainnya" (operasi sah pada data komposisi).
- Outlier: winsorize persentil 0,1-99,9 setelah transformasi.
- Standardisasi, lalu PCA 8 komponen untuk membuang dimensi noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

ID = "id_pelanggan"
CAT_SHARES = ["porsi_fashion", "porsi_elektronik", "porsi_kecantikan", "porsi_rumah_tangga",
              "porsi_makanan_minuman", "porsi_ibu_bayi", "porsi_olahraga", "porsi_hobi", "porsi_otomotif",
              "porsi_kesehatan", "porsi_buku_alat_tulis", "porsi_voucher_digital", "porsi_gadget_aksesoris",
              "porsi_perabot", "porsi_pertanian", "porsi_lainnya"]
MINOR_CATS = ["porsi_olahraga", "porsi_hobi", "porsi_otomotif", "porsi_kesehatan",
              "porsi_buku_alat_tulis", "porsi_pertanian", "porsi_lainnya"]
MAJOR_CATS = [c for c in CAT_SHARES if c not in MINOR_CATS]
TIME_SHARES = ["porsi_jam_pagi", "porsi_jam_siang", "porsi_jam_sore", "porsi_jam_malam"]
LOG_COLS = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan",
            "retur_90h", "ulasan_90h", "sesi_per_minggu", "durasi_sesi_menit", "rasio_konversi",
            "lama_member_bulan", "wishlist", "klik_iklan_30h", "belanja_30h", "frekuensi_30h"]
LIN_COLS = ["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod", "umur"]
NOISE_COLS = [f"n{i:02d}" for i in range(1, 16)]
PROFILE_IRRELEVANT = ["rating_beri_rata2", "jumlah_alamat", "kota_tier"]
CATEGORICAL = ["kota_tier", "perangkat"]
BOT_SESI = 100
N_PCA = 8


def split_groups(df: pd.DataFrame) -> dict:
    dormant = (df["frekuensi_90h"] == 0).to_numpy()
    bot = (df["sesi_per_minggu"] > BOT_SESI).to_numpy() & ~dormant
    model = ~dormant & ~bot
    return dict(dormant=dormant, bot=bot, model=model)


def hellinger_block(df: pd.DataFrame) -> pd.DataFrame:
    comp = pd.concat([df[MAJOR_CATS], df[MINOR_CATS].sum(axis=1).rename("porsi_kategori_minor")], axis=1)
    out = np.sqrt(comp.clip(lower=0))
    out.columns = [f"h_{c[6:]}" for c in comp.columns]
    t = np.sqrt(df[TIME_SHARES].clip(lower=0))
    t.columns = [f"h_{c[6:]}" for c in TIME_SHARES]
    return pd.concat([out, t], axis=1)


def clr(X: np.ndarray, delta: float = 0.005) -> np.ndarray:
    X = np.asarray(X, float).copy()
    X[X <= 0] = delta                       # penggantian nol sederhana
    X = X / X.sum(1, keepdims=True)
    L = np.log(X)
    return L - L.mean(1, keepdims=True)


def feature_frame(df: pd.DataFrame, comp: str = "hellinger_amalg", extra: list[str] | None = None) -> pd.DataFrame:
    """Matriks fitur sebelum scaling. `comp` memilih transformasi komposisi."""
    X = np.log1p(df[LOG_COLS].fillna(0)).add_prefix("log_")
    X = pd.concat([X, df[LIN_COLS]], axis=1)
    if comp == "hellinger_amalg":
        X = pd.concat([X, hellinger_block(df)], axis=1)
    elif comp == "clr_amalg":
        cc = np.c_[df[MAJOR_CATS].to_numpy(), df[MINOR_CATS].sum(1).to_numpy()]
        X = pd.concat([X, pd.DataFrame(clr(cc), index=df.index).add_prefix("clrk_"),
                       pd.DataFrame(clr(df[TIME_SHARES].to_numpy()), index=df.index).add_prefix("clrj_")], axis=1)
    elif comp == "clr_16":
        X = pd.concat([X, pd.DataFrame(clr(df[CAT_SHARES].to_numpy()), index=df.index).add_prefix("clrk_"),
                       pd.DataFrame(clr(df[TIME_SHARES].to_numpy()), index=df.index).add_prefix("clrj_")], axis=1)
    elif comp == "hellinger_16":
        X = pd.concat([X, np.sqrt(df[CAT_SHARES]).add_prefix("h_"), np.sqrt(df[TIME_SHARES]).add_prefix("h_")], axis=1)
    elif comp == "porsi_mentah":
        X = pd.concat([X, df[CAT_SHARES], df[TIME_SHARES]], axis=1)
    if extra:
        X = pd.concat([X, df[extra]], axis=1)
    return X


class Preprocessor:
    def __init__(self, comp: str = "hellinger_amalg", extra: list[str] | None = None,
                 n_pca: int = N_PCA, winsor: tuple = (0.001, 0.999)):
        self.comp, self.extra, self.n_pca, self.winsor = comp, extra, n_pca, winsor

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = feature_frame(df, self.comp, self.extra)
        self.columns_ = list(X.columns)
        self.lo_ = X.quantile(self.winsor[0])
        self.hi_ = X.quantile(self.winsor[1])
        self.n_winsor_ = int(((X < self.lo_) | (X > self.hi_)).any(axis=1).sum())
        Xc = X.clip(self.lo_, self.hi_, axis=1)
        self.scaler_ = StandardScaler().fit(Xc)
        Z = self.scaler_.transform(Xc)
        self.pca_ = PCA(self.n_pca, random_state=config.SEED).fit(Z)
        self.Z_ = Z
        return self.pca_.transform(Z)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = feature_frame(df, self.comp, self.extra).clip(self.lo_, self.hi_, axis=1)
        return self.pca_.transform(self.scaler_.transform(X))
