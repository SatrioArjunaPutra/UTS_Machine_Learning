"""Task 2 (fraud): orkestrasi 2a-2g + prediksi test.

Skema validasi waktu dengan delay label D (diestimasi dari data):
  setiap fold melatih pada transaksi < val_start - D (label sudah matang saat
  model dibuat), dan validasi hanya pada periode yang labelnya sudah matang
  (berakhir paling lambat 31 Agu - D).
Kalibrasi dua tahap di setiap fold: model M_cal dilatih sampai 28 hari sebelum
batas latih, kalibrator dilatih pada 28 hari terakhir data latih, lalu dipakai
untuk model utama. Prosedur yang sama dipakai untuk prediksi test.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from common import io
from common.metrics import brier, cost_rule_flag, pr_auc, total_cost
from common.utils import record_metric, save_table, timer
from common.validation import describe_folds, time_folds
from task2_fraud import analysis, audit
from task2_fraud.features import (TARGET, TIME, base_features, behavioral_features, label_history,
                                  to_model_matrix)
from task2_fraud.models import (FraudLGB, FraudLogReg, IdentityCalibrator, IsotonicCalibrator,
                                PlattCalibrator, undersample_idx)

GRID = [dict(num_leaves=15), dict(num_leaves=63)]
CAL_DAYS = 28


def _best_threshold(p, y, nominal) -> float:
    """Ambang global pada probabilitas yang meminimalkan cost (dipilih di jendela kalibrasi)."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400)))
    costs = [total_cost(y, p >= t, nominal) for t in grid]
    return float(grid[int(np.argmin(costs))])


def run(quick: bool = False) -> None:
    pull = pd.Timestamp(config.DATA_PULL_DATE)
    # ------------------------------------------------------------ 2a-2b audit
    with timer("task2: muat data + audit label"):
        tr, te = io.load_klasifikasi()
        D = audit.maturity_curve(tr)
        audit.review_flag_analysis(tr, te)
        audit.audit_table(tr, te)
        audit.drift_v15(tr, te)
        analysis.shift_plots(tr)
        record_metric("task2", "delay_label_dipakai_hari", D)
        raw = pd.concat([tr, te.assign(is_fraud=np.nan)], ignore_index=True)
        is_test = np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)]
        del tr, te

    # ------------------------------------------------------------ 2c fitur perilaku
    with timer("task2: fitur perilaku + uji future-invariance"):
        beh = behavioral_features(raw)
        cut = pd.Timestamp("2025-06-15")
        sub = raw[raw[TIME] < cut]
        beh_sub = behavioral_features(sub)
        a, b = beh.loc[sub.index].to_numpy(float), beh_sub.to_numpy(float)
        ok_beh = bool(np.allclose(a, b, equal_nan=True))
        lh_full = label_history(raw, "id_merchant", pd.Timestamp("2025-05-01"), D, 20, "m")
        lh_sub = label_history(sub, "id_merchant", pd.Timestamp("2025-05-01"), D, 20, "m")
        ok_lh = bool(np.allclose(lh_full.loc[sub.index].to_numpy(float), lh_sub.to_numpy(float)))
        assert ok_beh and ok_lh, "fitur memakai informasi masa depan!"
        record_metric("task2", "uji_future_invariance", "lulus")
        all_df = base_features(pd.concat([raw, beh], axis=1))
        beh_cols = list(beh.columns)
        del raw, sub, beh_sub, beh, a, b, lh_full, lh_sub
        if quick:
            keep = is_test | (all_df[TARGET] == 1).to_numpy() | (np.random.default_rng(0).random(len(all_df)) < 0.25)
            all_df, is_test = all_df[keep].reset_index(drop=True), is_test[keep]
        y_all = all_df[TARGET].to_numpy()
        t_all = all_df[TIME]
        ntr = int((~is_test).sum())

    def mask(idx):
        m = np.zeros(len(all_df), bool)
        m[idx] = True
        return m

    lh_cache = {}

    def frame(cutoff, with_beh=True):
        key = (cutoff, with_beh)
        if key not in lh_cache:
            while len(lh_cache) >= 2:              # batasi memori (Colab/laptop 8 GB)
                lh_cache.pop(next(iter(lh_cache)))
            lh = pd.concat([label_history(all_df, "id_merchant", cutoff, D, 20, "m"),
                            label_history(all_df, "id_pengguna", cutoff, D, 5, "u")], axis=1)
            A = pd.concat([all_df, lh], axis=1)
            if not with_beh:
                A = A.drop(columns=beh_cols + list(lh.columns))
            lh_cache[key] = A
        return lh_cache[key]

    def data(train_mask, cutoff, others, drop_leak=True, with_beh=True, drop_drift=True):
        A = frame(cutoff, with_beh)
        Xtr, lv = to_model_matrix(A[train_mask], drop_leak=drop_leak, drop_drift=drop_drift)
        return Xtr, [to_model_matrix(A[m], lv, drop_leak=drop_leak, drop_drift=drop_drift)[0] for m in others]

    val_windows = [("2025-04-01", "2025-06-01"), ("2025-05-01", "2025-07-01"),
                   (str((pull - pd.Timedelta(days=D) - pd.DateOffset(months=2)).date()),
                    str((pull - pd.Timedelta(days=D)).date()))]
    folds = time_folds(t_all.iloc[:ntr], val_windows, label_delay_days=D)
    save_table(describe_folds(folds, t_all.iloc[:ntr]), "task2", "fold_waktu")

    # ------------------------------------------------------------ tuning (F1, F2)
    with timer("task2: tuning LightGBM di F1-F2"):
        rows = []
        for f in folds[:2]:
            trm, vam = mask(f.train_idx), mask(f.val_idx)
            Xtr, (Xva,) = data(trm, f.train_end, [vam])
            for ci, cfg in enumerate(GRID):
                m = FraudLGB(**cfg).fit(Xtr, y_all[trm], X_val=Xva, y_val=y_all[vam], early_stopping=200)
                rows.append(dict(config=ci, **cfg, fold=f.name, pr_auc=round(pr_auc(y_all[vam], m.predict_proba(Xva)), 4),
                                 best_iter=m.best_iter_))
        tune = pd.DataFrame(rows)
        save_table(tune, "task2", "tuning_lgb")
        agg = tune.groupby("config").agg(ap=("pr_auc", "mean"), it=("best_iter", "mean"))
        ci = int(agg["ap"].idxmax())
        best = dict(GRID[ci], n_estimators=int(agg.loc[ci, "it"] * 1.1) + 1)
        record_metric("task2", "param_terbaik", best)
        print("  parameter terbaik:", best)

    # ------------------------------------------------------------ perbandingan model F1-F3
    with timer("task2: perbandingan model di F1-F3"):
        rows, keep = [], {}
        for f in folds:
            trm, vam = mask(f.train_idx), mask(f.val_idx)
            Xtr, (Xva,) = data(trm, f.train_end, [vam])
            yv = y_all[vam]
            lr = FraudLogReg().fit(Xtr, y_all[trm])
            rows.append(dict(fold=f.name, model="logistic regression", pr_auc=round(pr_auc(yv, lr.predict_proba(Xva)), 4)))
            m = FraudLGB(**best).fit(Xtr, y_all[trm])
            p = m.predict_proba(Xva)
            rows.append(dict(fold=f.name, model="LightGBM", pr_auc=round(pr_auc(yv, p), 4)))
            if f.name == "F3":
                Xtr_nb, (Xva_nb,) = data(trm, f.train_end, [vam], with_beh=False)
                p_nb = FraudLGB(**best).fit(Xtr_nb, y_all[trm]).predict_proba(Xva_nb)
                rows.append(dict(fold=f.name, model="LightGBM tanpa fitur perilaku", pr_auc=round(pr_auc(yv, p_nb), 4)))
                keep = dict(f=f, trm=trm, vam=vam, Xtr=Xtr, Xva=Xva, p=p, m=m)
        comp = pd.DataFrame(rows)
        prev = {f.name: round(float(y_all[f.val_idx].mean()), 5) for f in folds}
        comp["prevalensi"] = comp["fold"].map(prev)
        save_table(comp, "task2", "perbandingan_model")
        record_metric("task2", "pr_auc_per_fold", comp.pivot(index="model", columns="fold", values="pr_auc").to_dict(orient="index"))

    f, trm, vam = keep["f"], keep["trm"], keep["vam"]
    yv, nomv = y_all[vam], all_df.loc[vam, "nominal"].to_numpy()
    cal_start = f.train_end - pd.Timedelta(days=CAL_DAYS)
    cfit = trm & (t_all < cal_start).to_numpy()
    cwin = trm & (t_all >= cal_start).to_numpy()
    Xc_tr, (Xc_win,) = data(cfit, cal_start, [cwin])
    yc, nomc = y_all[cwin], all_df.loc[cwin, "nominal"].to_numpy()

    # ------------------------------------------------------------ 2a ablasi flag_review_manual
    with timer("task2: ablasi flag_review_manual (F3)"):
        Xtr_l, (Xva_l,) = data(trm, f.train_end, [vam], drop_leak=False)
        p_l = FraudLGB(**best).fit(Xtr_l, y_all[trm]).predict_proba(Xva_l)
        Xtr_d, (Xva_d,) = data(trm, f.train_end, [vam], drop_drift=False)
        p_d = FraudLGB(**best).fit(Xtr_d, y_all[trm]).predict_proba(Xva_d)
        abl = pd.DataFrame([dict(fitur="fitur final (tanpa flag_review_manual, v15, hitungan kumulatif)",
                                 pr_auc=round(pr_auc(yv, keep["p"]), 4)),
                            dict(fitur="+ flag_review_manual (leakage)", pr_auc=round(pr_auc(yv, p_l), 4)),
                            dict(fitur="+ v15 dan hitungan kumulatif (drift)", pr_auc=round(pr_auc(yv, p_d), 4))])
        save_table(abl, "task2", "ablasi_leakage")
        record_metric("task2", "ablasi_leakage", abl.set_index("fitur")["pr_auc"].to_dict())

    # ------------------------------------------------------------ 2d class imbalance
    with timer("task2: perbandingan penanganan imbalance (F3)"):
        ytr_c, ytr = y_all[cfit], y_all[trm]
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        approaches = {
            "tanpa penanganan": dict(kw={}, under=None),
            "class weight (scale_pos_weight)": dict(kw={"scale_pos_weight": spw}, under=None),
            "undersampling negatif 1:20": dict(kw={}, under=20),
        }
        rows, store = [], {}
        for name, a in approaches.items():
            def fit(X, yy):
                if a["under"]:
                    idx = undersample_idx(yy, a["under"])
                    return FraudLGB(**best, **a["kw"]).fit(X.iloc[idx], yy[idx])
                return FraudLGB(**best, **a["kw"]).fit(X, yy)
            mc = fit(Xc_tr, ytr_c)
            pc = mc.predict_proba(Xc_win)
            cal = PlattCalibrator().fit(pc, yc)
            mm = keep["m"] if name == "tanpa penanganan" else fit(keep["Xtr"], ytr)
            pv_raw = mm.predict_proba(keep["Xva"])
            pv = cal.transform(pv_raw)
            thr = _best_threshold(cal.transform(pc), yc, nomc)
            rows.append(dict(pendekatan=name, pr_auc=round(pr_auc(yv, pv_raw), 4),
                             brier_mentah=round(brier(yv, pv_raw), 5), brier_kalibrasi=round(brier(yv, pv), 5),
                             cost_aturan_biaya_juta=round(total_cost(yv, cost_rule_flag(pv, nomv), nomv) / 1e6, 1),
                             cost_ambang_global_juta=round(total_cost(yv, pv >= thr, nomv) / 1e6, 1)))
            store[name] = dict(pc=pc, pv_raw=pv_raw)
        imb = pd.DataFrame(rows)
        save_table(imb, "task2", "imbalance")
        record_metric("task2", "imbalance", imb.set_index("pendekatan").to_dict(orient="index"))
        chosen = imb.sort_values(["cost_aturan_biaya_juta", "pr_auc"], ascending=[True, False]).iloc[0]["pendekatan"]
        record_metric("task2", "pendekatan_dipilih", chosen)
        print("  pendekatan imbalance:", chosen)

    # ------------------------------------------------------------ 2f kalibrasi + keputusan
    with timer("task2: kalibrasi + aturan keputusan (F3)"):
        pc, pv_raw = store[chosen]["pc"], store[chosen]["pv_raw"]
        cals = {"tanpa kalibrasi": IdentityCalibrator(), "Platt": PlattCalibrator(), "isotonic": IsotonicCalibrator()}
        rows, pv_by = [], {}
        for name, c in cals.items():
            c.fit(pc, yc)
            pv = np.clip(c.transform(pv_raw), 1e-7, 1 - 1e-7)
            pv_by[name] = pv
            rows.append(dict(kalibrasi=name, brier=round(brier(yv, pv), 5), log_loss=round(log_loss(yv, pv), 5),
                             pr_auc=round(pr_auc(yv, pv), 4), mean_prob=round(float(pv.mean()), 5),
                             fraud_rate_aktual=round(float(yv.mean()), 5),
                             cost_aturan_biaya_juta=round(total_cost(yv, cost_rule_flag(pv, nomv), nomv) / 1e6, 1)))
        calt = pd.DataFrame(rows)
        save_table(calt, "task2", "kalibrasi")
        record_metric("task2", "kalibrasi", calt.set_index("kalibrasi").to_dict(orient="index"))
        analysis.reliability_plot(yv, pv_by)
        cal_name = calt[calt.kalibrasi != "tanpa kalibrasi"].sort_values("brier").iloc[0]["kalibrasi"]
        record_metric("task2", "kalibrator_dipilih", cal_name)
        pv = pv_by[cal_name]
        cal_obj = cals[cal_name]
        thr = _best_threshold(cal_obj.transform(pc), yc, nomc)
        flag_rule = cost_rule_flag(pv, nomv)
        mults = [0.5, 0.75, 1.0, 1.5, 2.0]
        dec = [dict(strategi="tidak ada yang di-flag", n_flag=0, cost=total_cost(yv, np.zeros_like(yv), nomv)),
               dict(strategi="semua di-flag", n_flag=len(yv), cost=total_cost(yv, np.ones_like(yv), nomv)),
               dict(strategi="ambang probabilitas global (dipilih di jendela kalibrasi)", n_flag=int((pv >= thr).sum()),
                    cost=total_cost(yv, pv >= thr, nomv)),
               dict(strategi="aturan biaya: p x nominal > 25.000", n_flag=int(flag_rule.sum()),
                    cost=total_cost(yv, flag_rule, nomv))]
        for mu in mults:
            fl = cost_rule_flag(pv, nomv, multiplier=mu)
            dec.append(dict(strategi=f"sensitivitas: p x nominal > {mu} x 25.000", n_flag=int(fl.sum()),
                            cost=total_cost(yv, fl, nomv)))
        dect = pd.DataFrame(dec)
        dect["cost_juta"] = (dect["cost"] / 1e6).round(1)
        dect = dect.drop(columns="cost")
        save_table(dect, "task2", "strategi_keputusan")
        record_metric("task2", "strategi_keputusan", dect.set_index("strategi")["cost_juta"].to_dict())
        record_metric("task2", "F3_pr_auc_final", round(pr_auc(yv, pv), 4))
        record_metric("task2", "F3_prevalensi", round(float(yv.mean()), 5))

    # ------------------------------------------------------------ 2e pergeseran + adaptasi
    with timer("task2: pergeseran pola + adaptasi recency (F3)"):
        age = (f.train_end - t_all[trm]).dt.days.to_numpy()
        w = 0.5 ** (age / 60.0)
        kw = approaches[chosen]["kw"]
        p_w = FraudLGB(**best, **kw).fit(keep["Xtr"], y_all[trm], sample_weight=w).predict_proba(keep["Xva"])
        age_c = (cal_start - t_all[cfit]).dt.days.to_numpy()
        pc_w = FraudLGB(**best, **kw).fit(Xc_tr, ytr_c, sample_weight=0.5 ** (age_c / 60.0)).predict_proba(Xc_win)
        pv_w = type(cal_obj)().fit(pc_w, yc).transform(p_w)
        adapt = pd.DataFrame([
            dict(skema="tanpa bobot", pr_auc=round(pr_auc(yv, pv), 4),
                 cost_juta=round(total_cost(yv, flag_rule, nomv) / 1e6, 1)),
            dict(skema="bobot recency (half-life 60 hari)", pr_auc=round(pr_auc(yv, pv_w), 4),
                 cost_juta=round(total_cost(yv, cost_rule_flag(pv_w, nomv), nomv) / 1e6, 1))])
        save_table(adapt, "task2", "adaptasi_recency")
        record_metric("task2", "adaptasi_recency", adapt.set_index("skema").to_dict(orient="index"))
        use_recency = adapt.iloc[1]["pr_auc"] > adapt.iloc[0]["pr_auc"] and adapt.iloc[1]["cost_juta"] <= adapt.iloc[0]["cost_juta"]
        record_metric("task2", "pakai_bobot_recency", bool(use_recency))
        # PR-AUC per bulan di validasi F3
        mon = t_all[vam].dt.to_period("M").astype(str).to_numpy()
        pm = pd.DataFrame([dict(bulan=b, n=int((mon == b).sum()), fraud_rate=round(float(yv[mon == b].mean()), 5),
                                pr_auc=round(pr_auc(yv[mon == b], pv[mon == b]), 4) if yv[mon == b].sum() > 0 else np.nan)
                           for b in np.unique(mon)])
        save_table(pm, "task2", "pr_auc_per_bulan_F3")

    # ------------------------------------------------------------ 2g analisis error
    with timer("task2: analisis error (F3)"):
        dv = all_df[vam].reset_index(drop=True)
        analysis.group_errors(dv, yv, pv, flag_rule, "kanal")
        analysis.group_errors(dv, yv, pv, flag_rule, "kategori_merchant")
        analysis.fraud_patterns(dv, yv, flag_rule)
        analysis.top_false_negatives(dv, yv, pv, flag_rule)
        imp = pd.DataFrame({"fitur": keep["Xtr"].columns,
                            "gain": keep["m"].model_.booster_.feature_importance("gain")})
        imp["gain_%"] = (100 * imp.gain / imp.gain.sum()).round(2)
        save_table(imp.sort_values("gain", ascending=False).head(25), "task2", "importance_gain")

    # ------------------------------------------------------------ final + test
    with timer("task2: model final + prediksi test"):
        final_end = pull - pd.Timedelta(days=D)
        cstart = final_end - pd.Timedelta(days=CAL_DAYS)
        trn = ~is_test
        m_tr = trn & (t_all < final_end).to_numpy()
        m_cfit = trn & (t_all < cstart).to_numpy()
        m_cwin = trn & (t_all >= cstart).to_numpy() & (t_all < final_end).to_numpy()
        record_metric("task2", "final_latih_sampai", str((final_end - pd.Timedelta(days=1)).date()))
        record_metric("task2", "final_dibuang_label_belum_matang", int((trn & (t_all >= final_end).to_numpy()).sum()))
        kw = approaches[chosen]["kw"]

        def fitw(X, yy, tt, end):
            sw = 0.5 ** ((end - tt).dt.days.to_numpy() / 60.0) if use_recency else None
            if approaches[chosen]["under"]:
                idx = undersample_idx(yy, approaches[chosen]["under"])
                return FraudLGB(**best, **kw).fit(X.iloc[idx], yy[idx], sample_weight=None if sw is None else sw[idx])
            return FraudLGB(**best, **kw).fit(X, yy, sample_weight=sw)

        Xa, (Xb,) = data(m_cfit, cstart, [m_cwin])
        mc = fitw(Xa, y_all[m_cfit], t_all[m_cfit], cstart)
        cal_final = type(cal_obj)().fit(mc.predict_proba(Xb), y_all[m_cwin])
        Xf, (Xt,) = data(m_tr, final_end, [is_test])
        mf = fitw(Xf, y_all[m_tr], t_all[m_tr], final_end)
        prob = np.clip(cal_final.transform(mf.predict_proba(Xt)), 0, 1)
        nom_t = all_df.loc[is_test, "nominal"].to_numpy()
        flag = cost_rule_flag(prob, nom_t)
        out = pd.DataFrame({"id_transaksi": all_df.loc[is_test, "id_transaksi"].to_numpy(),
                            "prob": np.round(prob, 6), "tandai": flag.astype(int)})
        if not quick:                      # --quick (subsampel) tidak boleh menimpa prediksi final
            out.to_csv(config.PRED_KLASIFIKASI, index=False)
        record_metric("task2", "test_n_flag", int(flag.sum()))
        record_metric("task2", "test_persen_flag", round(100 * float(flag.mean()), 2))
        record_metric("task2", "test_mean_prob", round(float(prob.mean()), 5))

        # adversarial validation: bisakah model membedakan train (Jan-Agu) dari test (Sep-Okt)?
        Xf_all, (Xt_all,) = data(m_tr, final_end, [is_test], drop_drift=False)
        samp = Xf_all.sample(min(150_000, len(Xf_all)), random_state=config.SEED)
        A = pd.concat([samp, Xt_all], ignore_index=True)
        lab = np.r_[np.zeros(len(samp)), np.ones(len(Xt_all))]
        res = {}
        for name, use in [("semua kolom (termasuk v15 dan hitungan kumulatif)", list(Xf_all.columns)),
                          ("fitur model final", list(Xf.columns))]:
            oof = np.zeros(len(A))
            for a_idx, b_idx in StratifiedKFold(3, shuffle=True, random_state=config.SEED).split(A, lab):
                clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, verbose=-1,
                                         random_state=config.SEED, n_jobs=config.N_JOBS, deterministic=True,
                                         force_row_wise=True).fit(A.iloc[a_idx][use], lab[a_idx])
                oof[b_idx] = clf.predict_proba(A.iloc[b_idx][use])[:, 1]
            res[name] = round(float(roc_auc_score(lab, oof)), 4)
            gi = pd.DataFrame({"fitur": use, "gain": clf.booster_.feature_importance("gain")})
            gi["gain_%"] = (100 * gi.gain / gi.gain.sum()).round(2)
            tag = "semua" if name.startswith("semua") else "final"
            save_table(gi.sort_values("gain", ascending=False).head(10), "task2", f"adversarial_top_fitur_{tag}")
        record_metric("task2", "adversarial_auc", res)
