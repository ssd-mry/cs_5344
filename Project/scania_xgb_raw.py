"""
SCANIA RUL: XGBoost on raw last-readout features (StarterKit-style).

All training vehicles (repaired + non-repaired); XGB gives P(y|x); θ* minimizes total cost on
validation set D_val; metrics printed for D_val. Refit XGB on train + val for test CSV
(θ* fixed, not re-tuned on full train).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scania_xgb_common import (
    COST,
    build_model_pipeline,
    get_scania_dir,
    infer_feature_columns,
    last_readout,
    predict_with_prob_threshold,
    print_eval_metrics,
    print_fit_val_shapes,
    print_label_counts,
    print_theta_star_dval,
    repair_flag_01,
    rul_to_ordinal,
    tune_prob_threshold,
)


def main() -> None:
    scania = get_scania_dir()

    # ── Train: last readout + specs + TTE (all vehicles) ──
    ops_tr  = pd.read_csv(scania / "train_operational_readouts.csv")
    spec_tr = pd.read_csv(scania / "train_specifications.csv")
    tte_tr  = pd.read_csv(scania / "train_tte.csv")

    df_tr = (
        last_readout(ops_tr)
        .merge(spec_tr, on="vehicle_id", how="inner")
        .merge(tte_tr,  on="vehicle_id", how="inner")
    )
    if len(df_tr) == 0:
        raise ValueError("No training rows after merge; check train_tte / paths.")

    t_cur = df_tr["time_step"].to_numpy()
    T     = df_tr["length_of_study_time_step"].to_numpy()
    rep   = repair_flag_01(df_tr["in_study_repair"])
    rul   = np.where(rep == 1, np.maximum(T - t_cur, 0.0), np.inf)
    df_tr["y"] = np.where(np.isfinite(rul), rul_to_ordinal(rul), 0)

    drop_cols = {"vehicle_id", "time_step", "in_study_repair", "length_of_study_time_step", "y"}
    cat_cols, num_cols, X_tr = infer_feature_columns(df_tr, drop_cols)
    y_tr = df_tr["y"].astype(int).to_numpy()

    # ── Real validation set: last readout + specs + official labels ──
    ops_va    = pd.read_csv(scania / "validation_operational_readouts.csv")
    spec_va   = pd.read_csv(scania / "validation_specifications.csv")
    labels_va = pd.read_csv(scania / "validation_labels.csv")

    df_va = (
        last_readout(ops_va)
        .merge(spec_va, on="vehicle_id", how="inner")
        .merge(
            labels_va.rename(columns={"class_label": "y"})[["vehicle_id", "y"]],
            on="vehicle_id", how="inner",
        )
    )
    X_va = df_va.drop(columns=[c for c in drop_cols if c in df_va.columns], errors="ignore")
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=np.nan)
    y_va = df_va["y"].astype(int).to_numpy()

    print_fit_val_shapes(
        int(rep.sum()), len(X_tr), X_tr.shape[1],
        df_tr["vehicle_id"].nunique(), df_va["vehicle_id"].nunique(),
    )
    print_label_counts("Labels (train set, all vehicles)", y_tr)
    print_label_counts("Labels (real validation set, one row per vehicle)", y_va)

    # ── Fit on train → tune θ* on real val ──
    pipe = build_model_pipeline(cat_cols, num_cols)
    pipe.fit(X_tr, y_tr)

    proba_va      = pipe.predict_proba(X_va)
    theta_star, _ = tune_prob_threshold(proba_va, y_va, COST)
    print_theta_star_dval(theta_star)

    pred_va = predict_with_prob_threshold(proba_va, theta_star)
    print_eval_metrics(
        "Validation metrics (real validation set, one row per vehicle)",
        y_va, pred_va, proba_va,
    )



if __name__ == "__main__":
    main()
