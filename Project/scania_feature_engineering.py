import hashlib

import numpy as np
import pandas as pd
from pathlib import Path
import time

# ── Configuration ────
WINDOW_SIZE = 10
MIN_PERIODS = 1

_cwd = Path(__file__).resolve().parent
DATASETS_ROOT = _cwd / "Datasets" if (_cwd / "Datasets").exists() else _cwd.parent / "Datasets"
SCANIA_DIR = DATASETS_ROOT / "SCANIA" 

# ── Helper Functions ────

def compute_slope(values: np.ndarray) -> float:
    """Linear regression slope β over equally-spaced time indices.

    β = Σ(t - t̄)(x - x̄) / Σ(t - t̄)²
    NaN values are dropped before fitting.
    """
    mask = ~np.isnan(values)
    y = values[mask]
    if len(y) < 2:
        return np.nan
    t = np.arange(len(y), dtype=np.float64)
    t_bar = t.mean()
    y_bar = y.mean()
    denom = np.sum((t - t_bar) ** 2)
    if denom == 0:
        return 0.0
    return np.sum((t - t_bar) * (y - y_bar)) / denom


def rul_to_ordinal(rul: np.ndarray) -> np.ndarray:
    """Map RUL values to ordinal risk classes 0-4.

    0: Safe       (RUL > 48)
    1: Low risk   (24 < RUL ≤ 48)
    2: Medium     (12 < RUL ≤ 24)
    3: High risk  ( 6 < RUL ≤ 12)
    4: Critical   (RUL ≤ 6)
    """
    rul = np.asarray(rul, dtype=float)
    y = np.zeros_like(rul, dtype=int)
    y[rul <= 48] = 1
    y[rul <= 24] = 2
    y[rul <= 12] = 3
    y[rul <=  6] = 4
    return y


def patch_missing_inplace(dfs: list[pd.DataFrame], sensor_cols: list[str],
                          fill_values: dict[str, float]):
    """Fill NaN in sensor columns using pre-computed fill values (in-place)."""
    for df in dfs:
        for col in sensor_cols:
            if col in fill_values and pd.notna(fill_values[col]):
                df[col] = df[col].fillna(fill_values[col])


def build_features(ops_df: pd.DataFrame, sensor_cols: list[str],
                   w: int = 10, min_p: int = 3) -> pd.DataFrame:

    ops_df = ops_df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

    grouped = ops_df.groupby("vehicle_id", sort=False)[sensor_cols]

    # ── Vectorised rolling mean ──
    rolling_stats = grouped.rolling(window=w, min_periods=min_p).mean()
    rolling_stats.columns = [f"{col}_mean" for col in rolling_stats.columns]
    rolling_stats = rolling_stats.reset_index(level=0, drop=True)

    # ── Combine with identity columns ──
    features = pd.concat([
        ops_df[["vehicle_id", "time_step"]].reset_index(drop=True),
        rolling_stats.reset_index(drop=True),
    ], axis=1)

    return features


def encode_specifications(specs_df: pd.DataFrame,
                          fit_categories: dict | None = None):
    """One-hot encode Spec_0 ~ Spec_7.

    When fit_categories is None (training), categories are learned.
    When provided (val/test), columns are aligned to the training schema.
    """
    spec_cols = sorted(c for c in specs_df.columns if c.startswith("Spec_"))

    if fit_categories is None:
        fit_categories = {
            col: sorted(specs_df[col].dropna().unique().tolist())
            for col in spec_cols
        }

    parts = [specs_df[["vehicle_id"]].copy()]
    for col in spec_cols:
        dummies = pd.get_dummies(specs_df[col], prefix=col)
        for cat_val in fit_categories[col]:
            dummy_col = f"{col}_{cat_val}"
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0
        keep = [f"{col}_{cv}" for cv in fit_categories[col]]
        parts.append(dummies[keep])

    return pd.concat(parts, axis=1), fit_categories


def find_cols_to_keep(df, exclude_cols, missing_threshold=0.8):
    """Decide which columns to keep based on training data."""
    exclude = set(exclude_cols)
    cands = [c for c in df.columns if c not in exclude]
    steps = []

    # 1. High-missing (subsumes All-NaN since ratio 1.0 > threshold)
    miss = df[cands].isna().mean()
    drop = miss[miss > missing_threshold].index.tolist()
    if drop:
        steps.append(("High-missing", drop))
        cands = [c for c in cands if c not in set(drop)]

    # 2. Constant
    nuniq = df[cands].nunique()
    drop = nuniq[nuniq <= 1].index.tolist()
    if drop:
        steps.append(("Constant", drop))
        cands = [c for c in cands if c not in set(drop)]

    # 3. Duplicate (hash-based, avoids costly transpose)
    seen, drop = {}, []
    for c in cands:
        h = hashlib.sha256(pd.util.hash_pandas_object(df[c]).values.tobytes()).hexdigest()
        if h in seen:
            drop.append(c)
        else:
            seen[h] = c
    if drop:
        steps.append(("Duplicate", drop))
        cands = [c for c in cands if c not in set(drop)]

    keep = [c for c in df.columns if c in exclude or c in set(cands)]

    for name, dropped in steps:
        preview = str(dropped[:5]) + ("..." if len(dropped) > 5 else "")
        print(f"  {name}: dropped {len(dropped)}  {preview}")
    print(f"  Result: {df.shape[1]} cols \u2192 {len(keep)} cols\n")
    return keep


def clean_split(train, val, test, exclude_cols, missing_threshold=0.8):
    """Fit on train, align val/test columns."""
    keep = find_cols_to_keep(train, exclude_cols, missing_threshold)
    return train[keep], val.reindex(columns=keep), test.reindex(columns=keep)


def clean_dataset():
    ops_tr = pd.read_csv(SCANIA_DIR / "train_operational_readouts.csv")
    ops_va = pd.read_csv(SCANIA_DIR / "validation_operational_readouts.csv")
    ops_te = pd.read_csv(SCANIA_DIR / "test_operational_readouts.csv")

    print("Scania Cleaning:")
    ops_tr, ops_va, ops_te = clean_split(
        ops_tr, ops_va, ops_te,
        exclude_cols=["vehicle_id", "time_step"],
    )

    ops_tr.to_csv(SCANIA_DIR / "train_ops_cleaned.csv", index=False)
    ops_va.to_csv(SCANIA_DIR / "validation_ops_cleaned.csv", index=False)
    ops_te.to_csv(SCANIA_DIR / "test_ops_cleaned.csv", index=False)
    print(f"Saved. Train: {ops_tr.shape} | Val: {ops_va.shape} | Test: {ops_te.shape}")

# ── Main Pipeline ────


def main():
    t_start = time.time()
    print("=" * 64)
    print("  SCANIA Feature Engineering Pipeline")
    print(f"  Window size (w) = {WINDOW_SIZE}  |  Min periods = {MIN_PERIODS}")
    print("=" * 64)

    clean_dataset()

    # ── Step 1: Load cleaned operational readouts ──
    print("\n[Step 1] Loading cleaned operational readouts ...")
    train_ops = pd.read_csv(SCANIA_DIR / "train_ops_cleaned.csv")
    val_ops   = pd.read_csv(SCANIA_DIR / "validation_ops_cleaned.csv")
    test_ops  = pd.read_csv(SCANIA_DIR / "test_ops_cleaned.csv")

    sensor_cols = [c for c in train_ops.columns if c not in ("vehicle_id", "time_step")]
    print(f"  Train : {train_ops.shape[0]:>10,} rows  |  {train_ops['vehicle_id'].nunique():,} vehicles")
    print(f"  Val   : {val_ops.shape[0]:>10,} rows  |  {val_ops['vehicle_id'].nunique():,} vehicles")
    print(f"  Test  : {test_ops.shape[0]:>10,} rows  |  {test_ops['vehicle_id'].nunique():,} vehicles")
    print(f"  Sensor columns: {len(sensor_cols)}")

    # ── Step 2: Patch missing values (fit on train, apply to all) ──
    print("\n[Step 2] Patching missing values (numeric=median, categorical=max, fit on train) ...")
    fill_values = {col: train_ops[col].median() for col in sensor_cols}

    for name, df in [("Train", train_ops), ("Val", val_ops), ("Test", test_ops)]:
        before = df[sensor_cols].isna().sum().sum()
        patch_missing_inplace([df], sensor_cols, fill_values)
        after = df[sensor_cols].isna().sum().sum()
        print(f"  {name:5s}: {before:>12,} NaN  →  {after:>12,} NaN")

    # ── Step 3: Temporal aggregation (sliding window) ──
    print(f"\n[Step 3] Temporal aggregation (window={WINDOW_SIZE}, min_periods={MIN_PERIODS}) ...")

    t0 = time.time()
    train_feat = build_features(train_ops, sensor_cols, WINDOW_SIZE, MIN_PERIODS)
    print(f"  Train : {train_feat.shape}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    val_feat = build_features(val_ops, sensor_cols, WINDOW_SIZE, MIN_PERIODS)
    print(f"  Val   : {val_feat.shape}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    test_feat = build_features(test_ops, sensor_cols, WINDOW_SIZE, MIN_PERIODS)
    print(f"  Test  : {test_feat.shape}  ({time.time() - t0:.1f}s)")

    # ── Step 4: Compute ordinal labels (0-4) from RUL ──
    print("\n[Step 4] Computing ordinal labels from RUL ...")
    train_tte  = pd.read_csv(SCANIA_DIR / "train_tte.csv")
    val_labels = pd.read_csv(SCANIA_DIR / "validation_labels.csv")

    # --- Training labels: per-row RUL ---
    train_feat = train_feat.merge(
        train_tte[["vehicle_id", "length_of_study_time_step", "in_study_repair"]],
        on="vehicle_id", how="left"
    )
    rep  = train_feat["in_study_repair"].to_numpy()
    T    = train_feat["length_of_study_time_step"].to_numpy()
    t_cur = train_feat["time_step"].to_numpy()

    rul = np.where(rep == 1, np.maximum(T - t_cur, 0.0), np.inf)
    train_feat["label"] = np.where(np.isfinite(rul), rul_to_ordinal(rul), 0)
    train_feat.drop(columns=["length_of_study_time_step", "in_study_repair"], inplace=True)

    # --- Validation / Test: keep last row per vehicle (labels are per-vehicle) ---
    val_feat  = val_feat.groupby("vehicle_id").tail(1).reset_index(drop=True)
    test_feat = test_feat.groupby("vehicle_id").tail(1).reset_index(drop=True)

    val_labels = val_labels.rename(columns={"class_label": "label"})
    val_feat = val_feat.merge(val_labels[["vehicle_id", "label"]], on="vehicle_id", how="left")

    for split, df in [("Train", train_feat), ("Val", val_feat)]:
        print(f"  {split} label distribution:")
        for lbl, cnt in df["label"].value_counts().sort_index().items():
            pct = 100 * cnt / len(df)
            print(f"    class {lbl}: {cnt:>6,}  ({pct:.1f}%)")

    # ── Step 5: Merge vehicle specifications (raw categorical) ──
    print("\n[Step 5] Merging vehicle specifications (raw categorical) ...")
    train_specs = pd.read_csv(SCANIA_DIR / "train_specifications.csv")
    val_specs   = pd.read_csv(SCANIA_DIR / "validation_specifications.csv")
    test_specs  = pd.read_csv(SCANIA_DIR / "test_specifications.csv")

    spec_cols = sorted(c for c in train_specs.columns if c.startswith("Spec_"))
    cat_fill = {col: train_specs[col].mode()[0] for col in spec_cols}
    for specs_df in [train_specs, val_specs, test_specs]:
        for col in spec_cols:
            specs_df[col] = specs_df[col].fillna(cat_fill[col])

    train_feat = train_feat.merge(train_specs[["vehicle_id"] + spec_cols], on="vehicle_id", how="left")
    val_feat   = val_feat.merge(val_specs[["vehicle_id"] + spec_cols],   on="vehicle_id", how="left")
    test_feat  = test_feat.merge(test_specs[["vehicle_id"] + spec_cols], on="vehicle_id", how="left")

    print(f"  Added {len(spec_cols)} raw categorical specification columns: {spec_cols}")

    # ── Step 6: Save ──
    print("\n[Step 6] Saving feature datasets ...")
    out_train = SCANIA_DIR / f"train_features_w{WINDOW_SIZE}.csv"
    out_val   = SCANIA_DIR / f"validation_features_w{WINDOW_SIZE}.csv"
    out_test  = SCANIA_DIR / f"test_features_w{WINDOW_SIZE}.csv"

    train_feat.to_csv(out_train, index=False)
    val_feat.to_csv(out_val, index=False)
    test_feat.to_csv(out_test, index=False)

    print(f"  {out_train.name:40s}  {train_feat.shape}")
    print(f"  {out_val.name:40s}  {val_feat.shape}")
    print(f"  {out_test.name:40s}  {test_feat.shape}")

    # summary
    n_sensor_feats = len(sensor_cols)
    print(f"\n{'─' * 64}")
    print(f"  Features per row:")
    print(f"    {len(sensor_cols)} sensors × 1 aggregation (mean) = {n_sensor_feats}")
    print(f"    + {len(spec_cols)} raw categorical specification columns")
    print(f"    + 2 identity columns (vehicle_id, time_step)")
    print(f"    + 1 label column (train/val only)")
    print(f"    = {n_sensor_feats + len(spec_cols) + 2 + 1} total columns (train/val)")
    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
