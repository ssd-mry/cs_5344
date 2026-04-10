"""
SCANIA RUL: XGBoost on time-aggregated features (train_features_w10).

Train on all temporal windows per vehicle. θ* and validation metrics use only the last window
per validation vehicle (same as test: last_row_per_vehicle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scania_xgb_common import (
    COST,
    build_model_pipeline,
    get_scania_dir,
    infer_feature_columns,
    last_row_per_vehicle,
    predict_with_threshold_vector,
    print_eval_metrics,
    print_fit_val_shapes,
    print_label_counts,
    print_theta_star_dval,
    project_dir,
    tune_threshold_vector,
    write_scania_submission_csv,
)

WINDOW_SIZE = 10


def main() -> None:
    scania = get_scania_dir()

    # ── Train features (all time windows, all vehicles) ──
    df_tr = pd.read_csv(scania / f"train_features_w{WINDOW_SIZE}.csv")

    drop_cols = {"vehicle_id", "time_step", "label", "in_study_repair"}
    cat_cols, num_cols, X_tr = infer_feature_columns(df_tr, drop_cols)
    y_tr = df_tr["label"].astype(int).to_numpy()

    # ── Real validation set (one row per vehicle, label from official labels CSV) ──
    df_va = pd.read_csv(scania / f"validation_features_w{WINDOW_SIZE}.csv")
    df_va = df_va.drop(columns=["label"], errors="ignore")  # 统一用官方标注，与其他方法保持一致
    labels_va = pd.read_csv(scania / "validation_labels.csv")
    df_va = df_va.merge(
        labels_va.rename(columns={"class_label": "label"})[["vehicle_id", "label"]],
        on="vehicle_id", how="inner",
    )
    X_va  = df_va.drop(columns=[c for c in drop_cols if c in df_va.columns], errors="ignore")
    X_va  = X_va.reindex(columns=X_tr.columns, fill_value=np.nan)
    y_va  = df_va["label"].astype(int).to_numpy()

    print_fit_val_shapes(
        len(df_tr), len(X_tr), X_tr.shape[1],
        df_tr["vehicle_id"].nunique(), df_va["vehicle_id"].nunique(),
    )
    print_label_counts("Labels (real validation set, one row per vehicle)", y_va)

    # ── Fit on train → tune θ* on real val ──
    pipe = build_model_pipeline(cat_cols, num_cols)
    pipe.fit(X_tr, y_tr)

    proba_va      = pipe.predict_proba(X_va)
    theta_star, _ = tune_threshold_vector(proba_va, y_va, COST)
    print_theta_star_dval(theta_star)

    pred_va = predict_with_threshold_vector(proba_va, theta_star)
    print_eval_metrics(
        "Validation metrics (real validation set, one row per vehicle)",
        y_va, pred_va, proba_va,
    )

    # ── Refit on train + val combined, then predict test ──
    X_full = pd.concat([X_tr, X_va], ignore_index=True)
    y_full = np.concatenate([y_tr, y_va])
    pipe.fit(X_full, y_full)

    df_te   = pd.read_csv(scania / f"test_features_w{WINDOW_SIZE}.csv")
    df_te   = last_row_per_vehicle(df_te)
    X_te    = df_te.drop(columns=[c for c in drop_cols if c in df_te.columns], errors="ignore")
    X_te    = X_te.reindex(columns=X_tr.columns, fill_value=np.nan)
    pred_te = predict_with_threshold_vector(pipe.predict_proba(X_te), theta_star)
    write_scania_submission_csv(
        project_dir() / "scania_test_predictions_xgb_features.csv",
        df_te["vehicle_id"].to_numpy(),
        pred_te,
    )


if __name__ == "__main__":
    main()
