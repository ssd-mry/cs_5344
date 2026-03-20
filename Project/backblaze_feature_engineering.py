from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_COLUMNS = ("date", "serial_number", "failure")
ROLLING_AGGREGATIONS = ("mean", "median", "min", "max", "std", "first", "last")
SUPPORTED_ROLLING_AGGREGATIONS = ("mean", "median", "min", "max", "std")

DATA_PATH = "Datasets/Backblaze/backblaze_data/train_set.csv"
CLEAN_OUTPUT_DIR = Path("Datasets/Backblaze/backblaze_data/clean")

@dataclass(frozen=True)
class CleaningReport:
    kept_columns: list[str]
    dropped_high_missing: list[str]
    dropped_constant: list[str]
    dropped_low_variance: list[str]
    dropped_duplicate: list[str]
    dropped_redundant_smart: list[str]


FillStrategy = str | Callable[[pd.Series], object]


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _hash_series(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=True)
    return hashlib.sha256(hashed.values.tobytes()).hexdigest()


def _smart_metric_base(column_name: str) -> tuple[str, str] | None:
    if column_name.endswith("_raw"):
        return column_name[: -len("_raw")], "raw"
    if column_name.endswith("_normalized"):
        return column_name[: -len("_normalized")], "normalized"
    return None


def find_cols_to_keep(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = REQUIRED_COLUMNS,
    missing_threshold: float = 0.8,
    low_variance_threshold: float = 1e-8,
    prefer_raw_smart: bool = True,
) -> CleaningReport:
    """
    Decide which columns to keep using train-only statistics.

    The required columns in ``exclude_cols`` are always preserved.
    """
    exclude = set(exclude_cols)
    missing_required = exclude.difference(df.columns)
    if missing_required:
        raise ValueError(f"Excluded columns not present in dataframe: {sorted(missing_required)}")

    candidates = [column for column in df.columns if column not in exclude]

    dropped_high_missing: list[str] = []
    dropped_constant: list[str] = []
    dropped_low_variance: list[str] = []
    dropped_duplicate: list[str] = []
    dropped_redundant_smart: list[str] = []

    missing_ratio = df[candidates].isna().mean()
    dropped_high_missing = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if dropped_high_missing:
        dropped = set(dropped_high_missing)
        candidates = [column for column in candidates if column not in dropped]

    nunique = df[candidates].nunique(dropna=True)
    dropped_constant = nunique[nunique <= 1].index.tolist()
    if dropped_constant:
        dropped = set(dropped_constant)
        candidates = [column for column in candidates if column not in dropped]

    # numeric_candidates = [column for column in candidates if is_numeric_dtype(df[column])]
    # if numeric_candidates:
    #     variance = df[numeric_candidates].var(skipna=True)
    #     dropped_low_variance = variance[variance.fillna(0.0) <= low_variance_threshold].index.tolist()
    #     print(variance)
    #     if dropped_low_variance:
    #         dropped = set(dropped_low_variance)
    #         print("drop low variance:", dropped)
    #         candidates = [column for column in candidates if column not in dropped]

    seen_hashes: dict[str, str] = {}
    for column in list(candidates):
        column_hash = _hash_series(df[column])
        if column_hash in seen_hashes:
            dropped_duplicate.append(column)
        else:
            seen_hashes[column_hash] = column
    if dropped_duplicate:
        dropped = set(dropped_duplicate)
        candidates = [column for column in candidates if column not in dropped]

    if prefer_raw_smart:
        candidate_set = set(candidates)
        grouped_metrics: dict[str, set[str]] = {}
        for column in candidates:
            metric_info = _smart_metric_base(column)
            if metric_info is None:
                continue
            metric_base, metric_kind = metric_info
            grouped_metrics.setdefault(metric_base, set()).add(metric_kind)

        for metric_base, metric_kinds in grouped_metrics.items():
            raw_column = f"{metric_base}_raw"
            normalized_column = f"{metric_base}_normalized"
            if {"raw", "normalized"}.issubset(metric_kinds) and raw_column in candidate_set:
                dropped_redundant_smart.append(normalized_column)

        if dropped_redundant_smart:
            dropped = set(dropped_redundant_smart)
            candidates = [column for column in candidates if column not in dropped]

    kept_columns = [column for column in df.columns if column in exclude or column in set(candidates)]

    return CleaningReport(
        kept_columns=kept_columns,
        dropped_high_missing=dropped_high_missing,
        dropped_constant=dropped_constant,
        dropped_low_variance=dropped_low_variance,
        dropped_duplicate=dropped_duplicate,
        dropped_redundant_smart=dropped_redundant_smart,
    )


def clean_dataset(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = REQUIRED_COLUMNS,
    missing_threshold: float = 0.8,
    low_variance_threshold: float = 1e-8,
    prefer_raw_smart: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply column filtering while preserving required identifier columns."""
    report = find_cols_to_keep(
        df=df,
        exclude_cols=exclude_cols,
        missing_threshold=missing_threshold,
        low_variance_threshold=low_variance_threshold,
        prefer_raw_smart=prefer_raw_smart,
    )
    return df.loc[:, report.kept_columns].copy(), report


def _resolve_fill_value(series: pd.Series, strategy: FillStrategy, column_name: str) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA

    if callable(strategy):
        fill_value = strategy(non_null)
    elif strategy == "mode":
        modes = non_null.mode(dropna=True)
        fill_value = modes.iloc[0] if not modes.empty else pd.NA
    else:
        try:
            fill_value = non_null.agg(strategy)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"Unable to apply fill strategy '{strategy}' to column '{column_name}'"
            ) from exc

    if isinstance(fill_value, pd.Series):
        fill_value = fill_value.iloc[0] if not fill_value.empty else pd.NA

    return fill_value


def patch_missing_values(
    df: pd.DataFrame,
    numeric_fill_strategy: FillStrategy = "median",
    categorical_fill_strategy: FillStrategy = "max",
    exclude_cols: Iterable[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """
    Patch missing values after column filtering and before RUL computation.

    Required identifier/target columns are excluded from imputation to avoid
    changing the core event semantics of the dataset.
    """
    processed = df.copy()
    excluded = set(exclude_cols)

    numeric_patched_columns = 0
    categorical_patched_columns = 0

    for column in processed.columns:
        if column in excluded or not processed[column].isna().any():
            continue

        if is_numeric_dtype(processed[column]):
            fill_value = _resolve_fill_value(processed[column], numeric_fill_strategy, column)
            if pd.isna(fill_value):
                continue
            processed[column] = processed[column].fillna(fill_value)
            numeric_patched_columns += 1
        else:
            fill_value = _resolve_fill_value(processed[column], categorical_fill_strategy, column)
            if pd.isna(fill_value):
                continue
            processed[column] = processed[column].fillna(fill_value)
            categorical_patched_columns += 1

    print(
        "Patched missing values in "
        f"{numeric_patched_columns} numeric columns using '{numeric_fill_strategy}' "
        f"and {categorical_patched_columns} categorical columns using "
        f"'{categorical_fill_strategy}'"
    )
    return processed


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date to datetime and enforce per-disk time ordering."""
    _validate_required_columns(df)

    processed = df.copy()
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce")

    if processed["date"].isna().any():
        invalid_rows = int(processed["date"].isna().sum())
        raise ValueError(f"Found {invalid_rows} rows with invalid date values")

    processed = processed.sort_values(["serial_number", "date"], kind="mergesort").reset_index(drop=True)
    return processed


def compute_rul(
    df: pd.DataFrame,
    censored_rul_value: int | None = 9999,
    drop_censored: bool = False,
) -> pd.DataFrame:
    """
    Compute failure date and row-level RUL in days.

    If a disk has no failure event:
    - ``drop_censored=True`` removes its rows.
    - otherwise ``rul_days`` is filled with ``censored_rul_value``.
    """
    processed = preprocess_dataframe(df)

    failure_rows = processed["failure"].fillna(0).eq(1)
    failure_dates = (
        processed.loc[failure_rows, ["serial_number", "date"]]
        .groupby("serial_number", sort=False)["date"]
        .min()
        .rename("failure_date")
    )

    processed = processed.join(failure_dates, on="serial_number")
    processed["rul_days"] = (processed["failure_date"] - processed["date"]).dt.days

    if drop_censored:
        processed = processed.loc[processed["failure_date"].notna()].copy()
    else:
        if censored_rul_value is None:
            raise ValueError("censored_rul_value cannot be None when drop_censored is False")
        processed["rul_days"] = processed["rul_days"].fillna(censored_rul_value)

    # Some datasets retain observations after the first recorded failure date.
    # RUL is time remaining, so post-failure rows are capped at zero.
    processed["rul_days"] = processed["rul_days"].clip(lower=0)

    processed["rul_days"] = processed["rul_days"].astype("int64")
    return processed


def generate_labels(
    df: pd.DataFrame,
    rul_column: str = "rul_days",
    label_column: str = "label",
) -> pd.DataFrame:
    """Create 3-class labels from RUL thresholds."""
    if rul_column not in df.columns:
        raise ValueError(f"Column '{rul_column}' not found in dataframe")

    labeled = df.copy()
    labeled[label_column] = 0
    labeled.loc[labeled[rul_column] < 20, label_column] = 1
    labeled.loc[labeled[rul_column] < 10, label_column] = 2
    labeled[label_column] = labeled[label_column].astype("int8")
    return labeled


def temporal_aggregation(
    df: pd.DataFrame,
    window_size: int = 7,
    min_periods: int = 1,
    include_identity_columns: Sequence[str] = REQUIRED_COLUMNS,
    rul_column: str = "rul_days",
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Build rolling window features per disk using only current and past rows.

    Each output row represents the window ending at the current observation date.
    """
    _validate_required_columns(df)

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    aggregated_source = preprocess_dataframe(df)

    reserved_columns = set(include_identity_columns) | {"failure_date", rul_column, label_column}
    numeric_feature_columns = [
        column
        for column in aggregated_source.columns
        if column not in reserved_columns and is_numeric_dtype(aggregated_source[column])
    ]

    if not numeric_feature_columns:
        raise ValueError("No numeric feature columns available for temporal aggregation")

    grouped_numeric = aggregated_source.groupby("serial_number", sort=False)[numeric_feature_columns]

    rolling_features = grouped_numeric.rolling(window=window_size, min_periods=min_periods).agg(
        SUPPORTED_ROLLING_AGGREGATIONS
    )
    rolling_features.columns = [
        f"{feature_name}_{aggregation_name}"
        for feature_name, aggregation_name in rolling_features.columns.to_flat_index()
    ]
    rolling_features = rolling_features.reset_index(level=0, drop=True)

    first_features = grouped_numeric.rolling(window=window_size, min_periods=min_periods).apply(
        lambda values: values[0],
        raw=True,
    )
    first_features = first_features.reset_index(level=0, drop=True)
    first_features = first_features.rename(columns=lambda column: f"{column}_first")

    last_features = aggregated_source.loc[:, numeric_feature_columns].copy()
    last_features = last_features.rename(columns=lambda column: f"{column}_last")

    rolling_features = pd.concat(
        [rolling_features.reset_index(drop=True), first_features.reset_index(drop=True), last_features.reset_index(drop=True)],
        axis=1,
    )

    identity_columns = [column for column in include_identity_columns if column in aggregated_source.columns]
    passthrough_columns = identity_columns + [
        column
        for column in ("failure_date", rul_column, label_column)
        if column in aggregated_source.columns
    ]

    result = pd.concat(
        [aggregated_source.loc[:, passthrough_columns].reset_index(drop=True), rolling_features.reset_index(drop=True)],
        axis=1,
    )

    print(
        "Generated "
        f"{len(rolling_features.columns)} rolling features from "
        f"{len(numeric_feature_columns)} base numeric columns using window_size={window_size}"
    )
    return result


def build_feature_dataset(
    df: pd.DataFrame,
    window_size: int = 7,
    censored_rul_value: int | None = 9999,
    drop_censored: bool = False,
    numeric_fill_strategy: FillStrategy = "median",
    categorical_fill_strategy: FillStrategy = "max",
) -> pd.DataFrame:
    """End-to-end helper for RUL computation, label generation, and rolling features."""
    patched_df = patch_missing_values(
        df,
        numeric_fill_strategy=numeric_fill_strategy,
        categorical_fill_strategy=categorical_fill_strategy,
    )
    with_rul = compute_rul(
        patched_df,
        censored_rul_value=censored_rul_value,
        drop_censored=drop_censored,
    )
    with_labels = generate_labels(with_rul)
    return temporal_aggregation(with_labels, window_size=window_size)


def load_data(path: str = DATA_PATH, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def run_case(
    name: str,
    df: pd.DataFrame,
    *,
    window_size: int,
    censored_rul_value: int | None,
    drop_censored: bool,
    numeric_fill_strategy: str = "mean",
    categorical_fill_strategy: str = "max",
) -> pd.DataFrame:
    print(f"\n=== {name} ===")
    print(
        f"window_size={window_size}, "
        f"censored_rul_value={censored_rul_value}, "
        f"drop_censored={drop_censored}, "
        f"numeric_fill_strategy={numeric_fill_strategy}, "
        f"categorical_fill_strategy={categorical_fill_strategy}"
    )

    feature_df = build_feature_dataset(
        df,
        window_size=window_size,
        censored_rul_value=censored_rul_value,
        drop_censored=drop_censored,
        numeric_fill_strategy=numeric_fill_strategy,
        categorical_fill_strategy=categorical_fill_strategy,
    )

    print("shape:", feature_df.shape)
    print("label counts:", feature_df["label"].value_counts(dropna=False).sort_index().to_dict())
    print("rul range:", (feature_df["rul_days"].min(), feature_df["rul_days"].max()))
    print(
        feature_df[
            ["serial_number", "date", "failure", "rul_days", "label"]
        ].head(10).to_string(index=False)
    )

    sample_feature_cols = [
        c for c in feature_df.columns
        if c.endswith("_mean") or c.endswith("_first") or c.endswith("_last")
    ][:10]
    print("sample feature columns:", sample_feature_cols)

    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CLEAN_OUTPUT_DIR / f"train_set_window_{window_size}_{timestamp}.csv"
    feature_df.to_csv(output_path, index=False)
    print("saved feature dataset to:", output_path)

    return feature_df


def main() -> None:
    raw_df = load_data(nrows=999999)
    print("raw shape: ", raw_df.shape)

    clean_df, report = clean_dataset(raw_df)
    print("cleaned shape:", clean_df.shape)
    print(
        "cleaning report:",
        {
            "high_missing": len(report.dropped_high_missing),
            "constant": len(report.dropped_constant),
            "low_variance": len(report.dropped_low_variance),
            "duplicate": len(report.dropped_duplicate),
            "redundant_smart": len(report.dropped_redundant_smart),
        },
    )

    run_case(
        "default config",
        clean_df,
        window_size=7,
        censored_rul_value=9999,
        drop_censored=False,
        numeric_fill_strategy="median",
        categorical_fill_strategy="max",
    )

    # run_case(
    #     "shorter window",
    #     clean_df,
    #     window_size=3,
    #     censored_rul_value=9999,
    #     drop_censored=False,
    # )
    #
    # run_case(
    #     "longer window",
    #     clean_df,
    #     window_size=14,
    #     censored_rul_value=9999,
    #     drop_censored=False,
    # )
    #
    # run_case(
    #     "drop censored disks",
    #     clean_df,
    #     window_size=7,
    #     censored_rul_value=9999,
    #     drop_censored=True,
    # )
    #
    # run_case(
    #     "smaller censored fallback",
    #     clean_df,
    #     window_size=7,
    #     censored_rul_value=365,
    #     drop_censored=False,
    # )


if __name__ == '__main__':
    main()