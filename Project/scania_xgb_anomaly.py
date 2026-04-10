"""
SCANIA RUL: XGBoost on agg features + anomaly_score_* columns (train_features_w*_agg.csv).

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
    predict_with_prob_threshold,
    print_eval_metrics,
    print_fit_val_shapes,
    print_label_counts,
    print_theta_star_dval,
    tune_prob_threshold,
)

WINDOW_SIZE = 10


def main() -> None:
    scania = get_scania_dir()

    # ── Train features + anomaly score (all time windows, all vehicles) ──
    train_path = scania / f"train_features_w{WINDOW_SIZE}_agg.csv"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Missing {train_path}. Run rfod (scania_clean) + feature pipeline first."
        )
    df_tr = pd.read_csv(train_path)
    anomaly_cols = [c for c in df_tr.columns if c.startswith("anomaly_score")]
    if not anomaly_cols:
        raise ValueError("Expected anomaly_score_* columns in agg train CSV.")

    drop_cols = {"vehicle_id", "time_step", "label", "in_study_repair"}
    cat_cols, num_cols, X_tr = infer_feature_columns(df_tr, drop_cols)
    y_tr = df_tr["label"].astype(int).to_numpy()

    # ── Real validation set (one row per vehicle, label from official labels CSV) ──
    val_path = scania / f"validation_features_w{WINDOW_SIZE}_agg.csv"
    if not val_path.is_file():
        raise FileNotFoundError(
            f"Missing {val_path}. Run rfod (scania_clean) + feature pipeline first."
        )
    df_va = pd.read_csv(val_path)
    if not any(c.startswith("anomaly_score") for c in df_va.columns):
        raise ValueError("Expected anomaly_score_* columns in agg val CSV.")
    df_va = df_va.drop(columns=["label"], errors="ignore")  # 防止 merge 产生 label_x/label_y
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
    theta_star, _ = tune_prob_threshold(proba_va, y_va, COST)
    print_theta_star_dval(theta_star)

    pred_va = predict_with_prob_threshold(proba_va, theta_star)
    print_eval_metrics(
        "Validation metrics (real validation set, one row per vehicle)",
        y_va, pred_va, proba_va,
    )



if __name__ == "__main__":
    main()
