"""Shared helpers for SCANIA RUL XGBoost (paths, cost matrix, splits, metrics)."""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def get_scania_dir() -> Path:
    cwd = Path(__file__).resolve().parent
    root = cwd / "Datasets" if (cwd / "Datasets").exists() else cwd.parent / "Datasets"
    return root / "SCANIA"


def last_readout(df_ops: pd.DataFrame) -> pd.DataFrame:
    df_ops = df_ops.sort_values(["vehicle_id", "time_step"])
    return df_ops.drop_duplicates("vehicle_id", keep="last")


def rul_to_ordinal(rul: np.ndarray) -> np.ndarray:
    rul = np.asarray(rul, dtype=float)
    y = np.zeros_like(rul, dtype=int)
    y[rul <= 48] = 1
    y[rul <= 24] = 2
    y[rul <= 12] = 3
    y[rul <= 6] = 4
    return y


# Same as StarterKit demo_scania
COST = np.array(
    [
        [0, 20, 30, 40, 50],
        [200, 0, 20, 30, 40],
        [300, 200, 0, 20, 30],
        [400, 300, 200, 0, 20],
        [500, 400, 300, 200, 0],
    ],
    dtype=np.float64,
)


def total_cost(y_true, y_pred, cost: np.ndarray = COST) -> int:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return 0
    return int(cost[y_true, y_pred].sum())


def repair_flag_01(series: pd.Series) -> np.ndarray:
    """Coerce in_study_repair to 0/1 bool mask.

    If the column is read as strings (e.g. \"1\"), ``series == 1`` is all-False, RUL
    becomes inf, labels become all 0, and in-sample total cost can spuriously be 0.
    """
    return (pd.to_numeric(series, errors="coerce").fillna(0).astype(int) == 1).to_numpy()


def repaired_vehicle_ids(tte: pd.DataFrame) -> set:
    mask = repair_flag_01(tte["in_study_repair"])
    return set(tte.loc[mask, "vehicle_id"].tolist())


def keep_in_study_repair_rows(df: pd.DataFrame, tte: pd.DataFrame) -> pd.DataFrame:
    """Keep rows for vehicles with ``in_study_repair == 1`` in ``train_tte``.

    If ``in_study_repair`` is missing (e.g. feature CSV), merge from ``tte`` then filter.
    Equivalent intent to ``vehicle_id.isin(repaired_vehicle_ids(tte))`` but enforces the flag explicitly.
    """
    out = df.copy()
    if "in_study_repair" not in out.columns:
        tte_s = tte[["vehicle_id", "in_study_repair"]].drop_duplicates(
            subset=["vehicle_id"], keep="last"
        )
        out = out.merge(tte_s, on="vehicle_id", how="inner")
    mask = repair_flag_01(out["in_study_repair"])
    return out.loc[mask].reset_index(drop=True)


def split_vehicle_ids(
    df: pd.DataFrame,
    *,
    vehicle_col: str = "vehicle_id",
    label_col: str = "label",
    time_col: str = "time_step",
    val_fraction: float = 0.15,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split on each vehicle's **last time_step** label (one label per vehicle)."""
    if time_col in df.columns:
        last = df.sort_values(time_col).groupby(vehicle_col, as_index=False).last()
    else:
        last = df.groupby(vehicle_col, as_index=False).last()
    ids = last[vehicle_col].to_numpy()
    y = last[label_col].astype(int).to_numpy()
    if len(ids) < 2:
        raise ValueError("Need at least two vehicles for a train/validation split.")
    try:
        train_ids, val_ids = train_test_split(
            ids,
            test_size=val_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=y,
        )
    except ValueError as e:
        warnings.warn(
            f"Stratified vehicle split failed ({e!s}); falling back to unstratified split.",
            UserWarning,
            stacklevel=2,
        )
        train_ids, val_ids = train_test_split(
            ids,
            test_size=val_fraction,
            random_state=random_state,
            shuffle=True,
        )
    return np.asarray(train_ids, dtype=np.int64), np.asarray(val_ids, dtype=np.int64)


def compute_sample_weights(y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Inverse-frequency sample weights so each class contributes equally to training loss."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero for absent classes
    class_weight = y.size / (n_classes * counts)
    return class_weight[y]


def build_model_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", clf)])


def infer_feature_columns(
    df: pd.DataFrame, drop_cols: set[str]
) -> tuple[list[str], list[str], pd.DataFrame]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols, X


def ordinal_risk_score(proba: np.ndarray) -> np.ndarray:
    """Risk score s(x) = E[Y|x] = sum_k k * P(k|x) in [0, 4]."""
    proba = np.asarray(proba, dtype=np.float64)
    return (proba * np.arange(proba.shape[1], dtype=np.float64)).sum(axis=1)


def predict_with_threshold_vector(proba: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Map probabilities to class 0..4 using monotone cut points on s(x).

    theta = (tau_1,...,tau_4) sorted internally: [0,tau_1)->0, [tau_1,tau_2)->1, ...
    """
    proba = np.asarray(proba, dtype=np.float64)
    s = ordinal_risk_score(proba)
    t1, t2, t3, t4 = np.sort(np.asarray(theta, dtype=np.float64).ravel())[:4]
    pred = np.zeros(len(s), dtype=int)
    pred[s < t1] = 0
    pred[(s >= t1) & (s < t2)] = 1
    pred[(s >= t2) & (s < t3)] = 2
    pred[(s >= t3) & (s < t4)] = 3
    pred[s >= t4] = 4
    return pred


def tune_threshold_vector(
    proba: np.ndarray,
    y_true: np.ndarray,
    cost: np.ndarray = COST,
    *,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find theta* minimizing sum_i C[y_i, yhat_i] over a grid of monotone (tau_1..tau_4)."""
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    if grid is None:
        grid = np.linspace(0.0, 4.0, 21)

    best_theta = np.zeros(4)
    best_tc = np.inf
    best_pred: np.ndarray | None = None

    for theta_tuple in combinations_with_replacement(grid, 4):
        theta = np.array(theta_tuple, dtype=np.float64)
        pred = predict_with_threshold_vector(proba, theta)
        tc = total_cost(y_true, pred, cost)
        if tc < best_tc:
            best_tc = tc
            best_theta = np.sort(theta)
            best_pred = pred.copy()

    assert best_pred is not None
    return best_theta, best_pred


def safe_roc_auc_ovr(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    classes = np.array([0, 1, 2, 3, 4])
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except ValueError:
        present = np.unique(y_true)
        if len(present) < 2:
            return float("nan")
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=present,
            )
        )


def print_report(title: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None):
    """Full metric block (optional; not used by default scania_xgb_* scripts)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    print(f"\n--- {title} ---")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.6f}")
    print(f"  Macro-F1:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.6f}")
    print(
        f"  Weighted-F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.6f}"
    )
    print(
        f"  Macro-Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.6f}"
    )
    print(
        f"  Weighted-Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.6f}"
    )
    print(f"  Total cost: {total_cost(y_true, y_pred)}")
    if proba is not None:
        auc = safe_roc_auc_ovr(y_true, proba)
        print(f"  AUC-ROC (macro-OVR): {auc:.6f}" if not np.isnan(auc) else "  AUC-ROC: n/a (degenerate)")


def print_label_counts(title: str, y: np.ndarray, *, n_classes: int = 5) -> None:
    """One compact line: class 0..n_classes-1 with counts and percentages."""
    y = np.asarray(y, dtype=int)
    print(f"\n{title}:")
    if y.size == 0:
        print("  (empty)")
        return
    parts = []
    for k in range(n_classes):
        cnt = int((y == k).sum())
        pct = 100.0 * cnt / y.size
        parts.append(f"class {k}: {cnt} ({pct:.2f}%)")
    print("  " + "  ".join(parts))


def print_fit_val_shapes(
    n_total_rows: int,
    n_fit_rows: int,
    n_feats: int,
    n_train_vehicles: int,
    n_val_vehicles: int,
) -> None:
    """Total repaired rows; fit rows/vehicles; val size as vehicle count (last-window / one-row-per-veh)."""
    print("\nData shape")
    print(f"  All repaired rows: {n_total_rows:,}  |  feature columns: {n_feats}")
    print(f"  Fit: {n_fit_rows:,} rows ({n_train_vehicles:,} vehicles)")
    print(f"  Val (D_val, last time_step per vehicle): {n_val_vehicles:,} vehicles")


def print_theta_star_dval(theta_star: np.ndarray) -> None:
    print("\nTheta (cost-optimal on D_val, vehicle-level):")
    print(f"  theta* = [{', '.join(f'{x:.4f}' for x in theta_star)}]")


def print_eval_metrics(
    section_title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    *,
    n_features: int | None = None,
) -> None:
    """Validation-style block: accuracy, F1 (macro/weighted), total cost, AUC. Ignores ``n_features``."""
    _ = n_features  # optional; kept for backward-compatible call sites
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    print(f"\n{section_title}")
    n = len(y_true)
    if n == 0:
        print("  (no rows; skipping metrics)")
        return
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Weighted-F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Total cost: {total_cost(y_true, y_pred)}")
    if proba is not None:
        auc = safe_roc_auc_ovr(y_true, proba)
        print(
            f"  AUC-ROC (macro-OVR): {auc:.4f}"
            if not np.isnan(auc)
            else "  AUC-ROC: n/a (degenerate labels)"
        )


def project_dir() -> Path:
    """Directory that contains ``scania_xgb_common.py`` (the Project folder)."""
    return Path(__file__).resolve().parent


def last_row_per_vehicle(
    df: pd.DataFrame,
    vehicle_col: str = "vehicle_id",
    time_col: str = "time_step",
) -> pd.DataFrame:
    """One row per vehicle (last time_step), matching validation/test construction."""
    if time_col in df.columns:
        return df.sort_values([vehicle_col, time_col]).groupby(vehicle_col, as_index=False).last()
    return df.drop_duplicates(vehicle_col, keep="last")


def mask_last_time_step_per_vehicle(
    vehicle_ids: np.ndarray,
    time_steps: np.ndarray,
) -> np.ndarray:
    """Aligned with ``X_val`` / ``y_val`` row order: True on each vehicle's last ``time_step``.

    Same rule as ``last_row_per_vehicle`` (max time; last row in sort order if tied).
    """
    n = len(vehicle_ids)
    if n == 0:
        return np.zeros(0, dtype=bool)
    g = pd.DataFrame(
        {
            "v": np.asarray(vehicle_ids),
            "t": np.asarray(time_steps),
            "i": np.arange(n, dtype=np.int64),
        }
    )
    last_i = g.sort_values(["v", "t"]).groupby("v", sort=False)["i"].last().to_numpy()
    mask = np.zeros(n, dtype=bool)
    mask[last_i] = True
    return mask


def write_scania_submission_csv(
    out_path: Path | str,
    vehicle_ids: np.ndarray,
    labels: np.ndarray,
) -> Path:
    """Write competition file: columns ``id``, ``label`` (same as StarterKit demo_scania)."""
    out_path = Path(out_path)
    df = pd.DataFrame(
        {"id": vehicle_ids.astype(np.int64), "label": labels.astype(np.int64)}
    )
    df.to_csv(out_path, index=False)
    print("\nSubmission CSV (id, label):")
    print(f"  Written: {out_path.name}")
    print(f"  Rows: {len(df):,}")
    return out_path
