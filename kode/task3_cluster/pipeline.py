"""Task 3 (clustering pelanggan): orkestrasi 3a-3g + cluster.csv.

Alur: pisahkan bot (anomali) dan pelanggan frekuensi_90h = 0 (fitur transaksi
tidak terdefinisi) -> preprocessing (log1p, Hellinger, winsorize, standardisasi,
PCA 8) -> pilih k (BIC, silhouette, stabilitas) -> bandingkan 4 metode -> GMM full
final -> stabilitas 20 run -> anomali konsensus -> pelanggan frekuensi 0 masuk
segmen lewat fitur yang terisi -> profil & aksi marketing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture

import config
from common import io
from common.utils import record_metric, save_table, timer
from task3_cluster import anomaly, methods, profile, select, stability
from task3_cluster.preprocess import Preprocessor, assign_freq0, split_groups


def run(quick: bool = False) -> None:
    with timer("task3: muat data + preprocessing"):
        df = io.load_clustering()
        if quick:
            df = df.sample(30000, random_state=config.SEED).reset_index(drop=True)
        groups = split_groups(df)
        record_metric("task3", "n_pelanggan", len(df))
        record_metric("task3", "n_frekuensi0", int(groups["dormant"].sum()))
        record_metric("task3", "n_model", int(groups["model"].sum()))
        record_metric("task3", "rata2_keranjang_kosong_sama_dengan_frek0",
                      bool((df["rata2_keranjang"].isna() == (df["frekuensi_90h"] == 0)).all()))
        record_metric("task3", "porsi_kategori_jumlah_1",
                      bool(np.allclose(df.filter(like="porsi_").drop(columns=df.filter(like="porsi_jam").columns).sum(1), 1)))
        select.noise_evidence(df)
        model_idx = np.flatnonzero(groups["model"])
        dm = df.iloc[model_idx].reset_index(drop=True)
        pre = Preprocessor()
        P = pre.fit_transform(dm)
        record_metric("task3", "n_baris_winsorize", pre.n_winsor_)
        record_metric("task3", "pca_varian_terjelaskan_%", round(100 * float(pre.pca_.explained_variance_ratio_.sum()), 1))
        record_metric("task3", "n_fitur_sebelum_pca", len(pre.columns_))

    rng = np.random.default_rng(config.SEED)
    sub = rng.choice(len(P), min(40000, len(P)), replace=False)
    with timer("task3: pemilihan jumlah cluster"):
        k, _ = methods.choose_k(P[sub])
        print("  k terpilih:", k)

    with timer("task3: bukti seleksi fitur"):
        prelim = GaussianMixture(k, covariance_type="full", random_state=config.SEED).fit(P[sub]).predict(P)
        select.eta_squared(dm, prelim)
        select.variant_comparison(dm, k)

    with timer("task3: perbandingan metode"):
        methods.compare_methods(P, k)

    with timer("task3: model final + stabilitas"):
        gmm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=config.SEED).fit(P)
        lab = gmm.predict(P)
        stability.bootstrap_stability(P, k, lab, runs=20)

    with timer("task3: anomali + label final"):
        cons, info = anomaly.consensus_outliers(pre.Z_, P, gmm)
        order = pd.Series(dm["belanja_90h"].to_numpy()).groupby(lab).median().sort_values(ascending=False).index
        remap = {old: new for new, old in enumerate(order)}
        lab2 = np.array([remap[v] for v in lab])
        labels = np.full(len(df), -99, dtype=int)
        labels[model_idx] = np.where(cons, -1, lab2)
        labels[groups["bot"]] = -1
        anomaly.summarize(df, labels, groups, info)

    with timer("task3: pelanggan frekuensi 0 -> segmen aktif"):
        d0 = df.loc[groups["dormant"]]
        lab0, acc = assign_freq0(pre, pre.Z_[~cons], lab2[~cons], d0)
        labels[groups["dormant"]] = lab0
        assert (labels != -99).all()
        dist = pd.Series(lab0).value_counts(normalize=True)
        save_table(pd.DataFrame([
            {"bukti": "n pelanggan frekuensi_90h = 0 (bukan bot)", "nilai": len(d0)},
            {"bukti": "% dengan recency_hari < 90", "nilai": round(100 * float((d0["recency_hari"] < 90).mean()), 1)},
            {"bukti": "% dengan recency_hari < 30", "nilai": round(100 * float((d0["recency_hari"] < 30).mean()), 1)},
            {"bukti": "% dengan retur_90h > 0", "nilai": round(100 * float((d0["retur_90h"] > 0).mean()), 1)},
            {"bukti": "akurasi penugasan (QDA, holdout pelanggan aktif)", "nilai": round(acc, 3)},
            *[{"bukti": f"% masuk segmen {c}", "nilai": round(100 * float(v), 1)} for c, v in dist.sort_index().items()],
        ]), "task3", "frekuensi0_penugasan")
        record_metric("task3", "frek0_recency_lt90_%", round(100 * float((d0["recency_hari"] < 90).mean()), 1))
        record_metric("task3", "frek0_retur_gt0_%", round(100 * float((d0["retur_90h"] > 0).mean()), 1))
        record_metric("task3", "frek0_akurasi_penugasan", round(acc, 3))
        record_metric("task3", "frek0_segmen_terbanyak", int(dist.idxmax()))
        record_metric("task3", "frek0_persen_segmen_terbanyak", round(100 * float(dist.max()), 1))

    with timer("task3: profil segmen"):
        Zfull = pd.DataFrame(pre.Z_, columns=pre.columns_, index=model_idx)
        prof = profile.build_profiles(df, labels, Zfull, groups["model"])
        names = dict(zip(prof["cluster"], prof["nama"]))
        lm = labels[model_idx]
        keep = lm >= 0
        profile.profile_heatmap(Zfull.iloc[keep].reset_index(drop=True), lm[keep], names)
        act = labels >= 0
        ct = pd.crosstab(labels[act], df.loc[act, "perangkat"])
        v = np.sqrt(chi2_contingency(ct)[0] / (act.sum() * (min(ct.shape) - 1)))
        record_metric("task3", "cramers_v_perangkat_vs_segmen", round(float(v), 3))
        ct2 = pd.crosstab(labels[act], df.loc[act, "kota_tier"])
        v2 = np.sqrt(chi2_contingency(ct2)[0] / (act.sum() * (min(ct2.shape) - 1)))
        record_metric("task3", "cramers_v_kota_tier_vs_segmen", round(float(v2), 3))

    out = pd.DataFrame({"id_pelanggan": df["id_pelanggan"].to_numpy(), "cluster": labels.astype(int)})
    if not quick:
        out.to_csv(config.PRED_CLUSTER, index=False)
    record_metric("task3", "n_cluster_total", int(k))
