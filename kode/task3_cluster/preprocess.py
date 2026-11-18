"""Task 3a: preprocessing.

- Pelanggan dengan frekuensi_90h = 0 tidak ikut melatih GMM: rata2_keranjang
  kosong tepat pada baris ini dan semua metrik transaksi 30/90 hari bernilai 0.
  Mereka bukan segmen tersendiri (80% recency < 90 hari, 41% punya retur), jadi
  setelah GMM final mereka dimasukkan ke segmen aktif lewat fitur yang terisi
  (assign_freq0, lihat pipeline.py).
- Bot (sesi_per_minggu > 100) dipisah sebagai kandidat anomali (anomaly.py),
  termasuk bot dengan frekuensi_90h = 0.
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


# kolom yang turunan langsung dari jumlah transaksi: kosong/0 bila frekuensi_90h = 0
TX_DERIVED = ["log_frekuensi_90h", "log_belanja_90h", "log_rata2_keranjang", "log_item_per_pesanan",
              "log_belanja_30h", "log_frekuensi_30h", "log_retur_90h", "log_ulasan_90h"]


def split_groups(df: pd.DataFrame) -> dict:
    bot = (df["sesi_per_minggu"] > BOT_SESI).to_numpy()
    dormant = (df["frekuensi_90h"] == 0).to_numpy() & ~bot
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
        return self.pca_.transform(self.standardize(df))

    def standardize(self, df: pd.DataFrame) -> np.ndarray:
        """Matriks terstandar (sebelum PCA) dengan batas winsorize dan scaler hasil fit."""
        X = feature_frame(df, self.comp, self.extra).clip(self.lo_, self.hi_, axis=1)
        return self.scaler_.transform(X)


def assign_freq0(pre: Preprocessor, Z_model: np.ndarray, lab_model: np.ndarray,
                 df_freq0: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Masukkan pelanggan frekuensi_90h = 0 ke segmen aktif.

    Fitur turunan transaksi tidak terdefinisi untuk mereka, jadi tiap segmen
    dimodelkan sebagai Gaussian pada fitur yang terisi (QDA, prior = ukuran
    segmen). Ini sejalan dengan asumsi GMM: marginal Gaussian tetap Gaussian.
    Akurasi diukur pada 30% pelanggan aktif yang tidak dipakai untuk fit.
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.model_selection import train_test_split

    obs = [i for i, c in enumerate(pre.columns_) if c not in TX_DERIVED]
    X = Z_model[:, obs]
    a, b, ya, yb = train_test_split(X, lab_model, test_size=0.3, random_state=config.SEED, stratify=lab_model)
    acc = float((QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(a, ya).predict(b) == yb).mean())
    qda = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(X, lab_model)
    return qda.predict(pre.standardize(df_freq0)[:, obs]), acc
