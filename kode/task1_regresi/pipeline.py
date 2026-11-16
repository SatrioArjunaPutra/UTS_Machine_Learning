"""Task 1 (regresi durasi): orkestrasi 1a-1f + prediksi test.

Skema validasi: expanding window, horizon 2 bulan seperti test (Sep-Okt).
  F1: latih < 1 Mei  -> validasi Mei-Jun
  F2: latih < 1 Jun  -> validasi Jun-Jul
  F3: latih < 1 Jul  -> validasi Jul-Agu   (tidak dipakai untuk tuning)
Tuning hanya di F1-F2; F3 menjadi estimasi jujur performa.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config
from common import io
from common.metrics import coverage, pinball
from common.utils import record_metric, save_fig, save_table, timer
from common.validation import describe_folds, time_folds
from task1_regresi import analysis, audit
from task1_regresi.conformal import ConformalP90
from task1_regresi.features import DATE_COL, TARGET, add_features, build_matrices, clean, fix_target
from task1_regresi.models import GroupQuantileBaseline, LGBQuantile, LogLinearQuantile

VAL_WINDOWS = [("2025-05-01", "2025-07-01"), ("2025-06-01", "2025-08-01"), ("2025-07-01", "2025-09-01")]
GRID = [dict(num_leaves=31, log_target=False), dict(num_leaves=127, log_target=False),
        dict(num_leaves=31, log_target=True), dict(num_leaves=127, log_target=True)]
HOLDOUT_COURIER = "K12"


def _eval(y, p50, p90) -> dict:
    return dict(pinball_q50=round(pinball(y, p50, 0.5), 3), pinball_q90=round(pinball(y, p90, 0.9), 3),
                coverage_p90=round(coverage(y, p90), 4), coverage_p50=round(coverage(y, p50), 4))


def _fit_pair(best: dict, Xtr, ytr):
    m50 = LGBQuantile(0.5, **best[0.5]).fit(Xtr, ytr)
    m90 = LGBQuantile(0.9, **best[0.9]).fit(Xtr, ytr)
    return m50, m90


def _predict_pair(m50, m90, X):
    p50 = m50.predict(X)
    p90 = np.maximum(m90.predict(X), p50)
    return p50, p90


def run(quick: bool = False) -> None:
    # ------------------------------------------------------------ 1a audit
    with timer("task1: muat data + audit"):
        tr_raw, te_raw = io.load_regresi()
        y_fixed, unit_mask = fix_target(tr_raw[TARGET])
        audit.audit_table(tr_raw, te_raw)
        audit.plot_target(tr_raw[TARGET], y_fixed)
        tr, st = clean(tr_raw)
        te, st_te = clean(te_raw)
        audit.quality_findings(tr_raw, te_raw, unit_mask, st, st_te)
        tr[TARGET] = y_fixed.to_numpy()
        tr = add_features(tr).sort_values(DATE_COL, kind="stable").reset_index(drop=True)
        te = add_features(te)
        if quick:
            tr = tr.sample(60_000, random_state=config.SEED).sort_values(DATE_COL, kind="stable").reset_index(drop=True)
        y = tr[TARGET]
        record_metric("task1", "n_train", len(tr))
        record_metric("task1", "n_test", len(te))

    def mats(idx_tr, idx_others, **kw):
        return build_matrices(tr.iloc[idx_tr], y.iloc[idx_tr], [tr.iloc[i] for i in idx_others], **kw)

    # ------------------------------------------------------------ 1b folds
    folds = time_folds(tr[DATE_COL], VAL_WINDOWS)
    save_table(describe_folds(folds, tr[DATE_COL]), "task1", "fold_waktu")

    # ------------------------------------------------------------ 1c tuning (F1, F2)
    with timer("task1: tuning LightGBM di F1-F2"):
        rows = []
        for f in folds[:2]:
            Xtr, (Xva,), _ = mats(f.train_idx, [f.val_idx])
            ytr, yva = y.iloc[f.train_idx], y.iloc[f.val_idx]
            for ci, cfg in enumerate(GRID):
                for q in config.QUANTILES:
                    m = LGBQuantile(q, **cfg).fit(Xtr, ytr, Xva, yva, early_stopping=100)
                    rows.append(dict(config=ci, num_leaves=cfg["num_leaves"], target_log=cfg["log_target"],
                                     fold=f.name, q=q, pinball=round(pinball(yva, m.predict(Xva), q), 3),
                                     best_iter=m.best_iter_))
        tune = pd.DataFrame(rows)
        save_table(tune, "task1", "tuning_lgb")
        best = {}
        for q in config.QUANTILES:
            agg = tune[tune.q == q].groupby("config").agg(pinball=("pinball", "mean"), it=("best_iter", "mean"))
            ci = int(agg["pinball"].idxmin())
            best[q] = dict(GRID[ci], n_estimators=int(agg.loc[ci, "it"] * 1.1) + 1)
        record_metric("task1", "param_terbaik", {str(k): v for k, v in best.items()})
        print("  parameter terbaik:", best)

    # ------------------------------------------------------------ 1c perbandingan model (F1-F3)
    with timer("task1: perbandingan model di F1-F3"):
        rows, keep = [], {}
        for f in folds:
            Xtr, (Xva,), cols = mats(f.train_idx, [f.val_idx])
            ytr, yva = y.iloc[f.train_idx], y.iloc[f.val_idx]
            b50 = GroupQuantileBaseline(0.5).fit(Xtr, ytr).predict(Xva)
            b90 = GroupQuantileBaseline(0.9).fit(Xtr, ytr).predict(Xva)
            rows.append(dict(fold=f.name, model="baseline kuantil grup", **_eval(yva, b50, np.maximum(b90, b50))))
            lin = LogLinearQuantile().fit(Xtr, ytr)
            l50, l90 = lin.predict_quantile(Xva, 0.5), lin.predict_quantile(Xva, 0.9)
            rows.append(dict(fold=f.name, model="linear (Ridge, log target)", **_eval(yva, l50, np.maximum(l90, l50))))
            m50, m90 = _fit_pair(best, Xtr, ytr)
            p50, p90 = _predict_pair(m50, m90, Xva)
            rows.append(dict(fold=f.name, model="LightGBM quantile", **_eval(yva, p50, p90)))
            if f.name == "F3":
                keep = dict(Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva, p50=p50, p90=p90, m50=m50, m90=m90, f=f)
        comp = pd.DataFrame(rows)
        save_table(comp, "task1", "perbandingan_model")
        f3 = comp[comp.fold == "F3"].set_index("model")
        for mdl in f3.index:
            record_metric("task1_F3", mdl, f3.loc[mdl].to_dict())
        base_f3 = f3.loc["baseline kuantil grup"]
        lgb_f3 = f3.loc["LightGBM quantile"]
        record_metric("task1", "perbaikan_vs_baseline_q50_%",
                      round(100 * (1 - lgb_f3.pinball_q50 / base_f3.pinball_q50), 1))
        record_metric("task1", "perbaikan_vs_baseline_q90_%",
                      round(100 * (1 - lgb_f3.pinball_q90 / base_f3.pinball_q90), 1))

    # ------------------------------------------------------------ 1b random CV vs waktu
    with timer("task1: random 5-fold CV vs validasi waktu"):
        rows = []
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=config.SEED).split(tr)):
            Xa, (Xb,), _ = mats(a, [b])
            m50, m90 = _fit_pair(best, Xa, y.iloc[a])
            p50, p90 = _predict_pair(m50, m90, Xb)
            rows.append(dict(skema="random 5-fold", fold=k + 1, **_eval(y.iloc[b], p50, p90)))
        rnd = pd.DataFrame(rows)
        tim = comp[comp.model == "LightGBM quantile"].assign(skema="waktu (F1-F3)")
        cv = pd.concat([rnd, tim.drop(columns="model")], ignore_index=True)
        save_table(cv, "task1", "random_vs_waktu")
        summ = cv.groupby("skema")[["pinball_q50", "pinball_q90", "coverage_p90"]].mean().round(3)
        save_table(summ, "task1", "random_vs_waktu_ringkas", index=True)
        record_metric("task1", "random_vs_waktu", summ.to_dict(orient="index"))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        summ[["pinball_q50", "pinball_q90"]].plot.bar(ax=ax, rot=0, color=["#2b6cb0", "#dd6b20"])
        ax.set_ylabel("pinball loss (jam)")
        ax.set_title("Random CV lebih optimistis daripada validasi waktu")
        save_fig(fig, "task1", "random_vs_waktu")

    f = keep["f"]
    # ------------------------------------------------------------ 1a ablasi leakage
    with timer("task1: ablasi jumlah_scan di F3"):
        Xtr_l, (Xva_l,), _ = mats(f.train_idx, [f.val_idx], drop_leak=False)
        m50l, m90l = _fit_pair(best, Xtr_l, keep["ytr"])
        p50l, p90l = _predict_pair(m50l, m90l, Xva_l)
        abl = pd.DataFrame([dict(fitur="tanpa jumlah_scan (dipakai)", **_eval(keep["yva"], keep["p50"], keep["p90"])),
                            dict(fitur="dengan jumlah_scan (leakage)", **_eval(keep["yva"], p50l, p90l))])
        save_table(abl, "task1", "ablasi_leakage")
        record_metric("task1", "ablasi_leakage", abl.set_index("fitur").to_dict(orient="index"))

    # ------------------------------------------------------------ 1d encoding kategori
    with timer("task1: perbandingan encoding kota_tujuan + simulasi kategori baru"):
        rows = []
        unseen = (~tr["kota_tujuan"].iloc[f.val_idx].isin(tr["kota_tujuan"].iloc[f.train_idx].unique())).mean()
        record_metric("task1", "persen_kota_tujuan_baru_F3", round(100 * unseen, 2))
        for hc in ["drop", "freq", "te", "freq_te", "native"]:
            Xa, (Xb,), _ = mats(f.train_idx, [f.val_idx], high_card=hc)
            m50, m90 = _fit_pair(best, Xa, keep["ytr"])
            p50, p90 = _predict_pair(m50, m90, Xb)
            rows.append(dict(encoding_kota_tujuan=hc, **_eval(keep["yva"], p50, p90)))
        enc = pd.DataFrame(rows)
        save_table(enc, "task1", "encoding_kota_tujuan")
        # Simulasi mitra kurir baru: K12 dikeluarkan dari data latih F3
        tr_idx = f.train_idx[(tr["mitra_kurir"].iloc[f.train_idx] != HOLDOUT_COURIER).to_numpy()]
        va_k = f.val_idx[(tr["mitra_kurir"].iloc[f.val_idx] == HOLDOUT_COURIER).to_numpy()]
        Xa, (Xb,), _ = mats(tr_idx, [va_k])
        m50, m90 = _fit_pair(best, Xa, y.iloc[tr_idx])
        p50u, p90u = _predict_pair(m50, m90, Xb)
        known_mask = (tr["mitra_kurir"].iloc[f.val_idx] == HOLDOUT_COURIER).to_numpy()
        sim = pd.DataFrame([
            dict(skenario=f"{HOLDOUT_COURIER} dikenal saat latih", n=int(known_mask.sum()),
                 **_eval(y.iloc[va_k], keep["p50"][known_mask], keep["p90"][known_mask])),
            dict(skenario=f"{HOLDOUT_COURIER} tidak pernah dilihat (seperti K13 di test)", n=len(va_k),
                 **_eval(y.iloc[va_k], p50u, p90u)),
        ])
        save_table(sim, "task1", "simulasi_kurir_baru")
        record_metric("task1", "simulasi_kurir_baru", sim.set_index("skenario").to_dict(orient="index"))

    # ------------------------------------------------------------ 1e kalibrasi konformal
    with timer("task1: kalibrasi konformal P90"):
        d = tr[DATE_COL]
        iA = np.flatnonzero((d < "2025-05-01").to_numpy())
        iC = np.flatnonzero(((d >= "2025-05-01") & (d < "2025-07-01")).to_numpy())
        iT = np.flatnonzero((d >= "2025-07-01").to_numpy())
        XA, (XC, XT), _ = mats(iA, [iC, iT])
        a50, a90 = _fit_pair(best, XA, y.iloc[iA])
        _, pC90 = _predict_pair(a50, a90, XC)
        pT50, pT90 = _predict_pair(a50, a90, XT)
        yT, gC, gT = y.iloc[iT], tr["layanan"].iloc[iC], tr["layanan"].iloc[iT]
        cg = ConformalP90().fit(y.iloc[iC], pC90)
        cm = ConformalP90(group_col="layanan").fit(y.iloc[iC], pC90, groups=gC)
        variants = {
            "tanpa kalibrasi": pT90,
            "konformal global": np.maximum(cg.transform(pT90), pT50),
            "konformal per layanan (Mondrian)": np.maximum(cm.transform(pT90, groups=gT), pT50),
        }
        rows = []
        for name, p in variants.items():
            per = pd.Series(np.asarray(yT) <= p).groupby(gT.to_numpy()).mean()
            rows.append(dict(varian=name, coverage_p90=round(coverage(yT, p), 4),
                             pinball_q90=round(pinball(yT, p, 0.9), 3),
                             coverage_min_layanan=round(per.min(), 4), coverage_max_layanan=round(per.max(), 4)))
        conf = pd.DataFrame(rows)
        save_table(conf, "task1", "kalibrasi_konformal")
        record_metric("task1", "konformal", conf.set_index("varian").to_dict(orient="index"))
        record_metric("task1", "koreksi_log_global", round(cg.global_, 4))
        record_metric("task1", "koreksi_log_layanan", {k: round(v, 4) for k, v in cm.group_.items()})
        # Pilih varian: coverage dalam 0.9 +/- 0.015, lalu pinball q90 terkecil
        ok = conf[(conf.coverage_p90 - 0.9).abs() <= 0.015]
        chosen = (ok if len(ok) else conf.assign(dev=(conf.coverage_p90 - 0.9).abs()).nsmallest(1, "dev"))
        chosen = chosen.sort_values("pinball_q90").iloc[0]["varian"]
        record_metric("task1", "varian_konformal_dipilih", chosen)
        print("  varian konformal:", chosen)

    # ------------------------------------------------------------ 1f analisis error (uji Jul-Agu)
    with timer("task1: analisis error per grup"):
        pT90c = variants[chosen]
        dfT = tr.iloc[iT]
        for col in ["layanan", "pulau_tujuan", "mitra_kurir"]:
            analysis.error_by_group(dfT, yT, pT50, pT90c, pT90, col)
        record_metric("task1", "uji_jul_agu_terkalibrasi", _eval(yT, pT50, pT90c))

    # ------------------------------------------------------------ final + test
    with timer("task1: model final + prediksi test"):
        Xall, (Xte,), cols = build_matrices(tr, y, [te])
        f50, f90 = _fit_pair(best, Xall, y)
        p50, p90 = _predict_pair(f50, f90, Xte)
        # Koreksi konformal dari data terbaru: model F3 (latih < 1 Jul) dikalibrasi pada Jul-Agu
        g3 = tr["layanan"].iloc[f.val_idx]
        if chosen.startswith("konformal per"):
            cf = ConformalP90(group_col="layanan").fit(keep["yva"], keep["p90"], groups=g3)
            p90 = cf.transform(p90, groups=te["layanan"])
        elif chosen == "konformal global":
            cf = ConformalP90().fit(keep["yva"], keep["p90"])
            p90 = cf.transform(p90)
        p50 = np.clip(p50, 0.1, None)
        p90 = np.maximum(p90, p50)
        out = pd.DataFrame({"id_paket": te["id_paket"].to_numpy(), "p50": np.round(p50, 3),
                            "p90": np.round(p90, 3)})
        out.to_csv(config.PRED_REGRESI, index=False)
        record_metric("task1", "test_median_p50", round(float(np.median(p50)), 2))
        record_metric("task1", "test_median_p90", round(float(np.median(p90)), 2))
        unit = lambda m: "skala log1p durasi" if m.log_target else "jam"
        analysis.shap_importance(f50.model_, Xall, "P50", unit=unit(f50))
        analysis.shap_importance(f90.model_, Xall, "P90", unit=unit(f90))
        lbl90 = "P90" if chosen == "tanpa kalibrasi" else "P90 (sebelum kalibrasi)"
        analysis.partial_dependence({"P50": f50, lbl90: f90}, Xall,
                                    ["jarak_km", "alamat_skor", "antrean_hub_asal"])
