from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from backblaze_feature_engineering import clean_train_val_datasets


RANDOM_STATE = 42
COST_MATRIX = np.array(
    [
        [0, 1, 3],
        [4, 0, 2],
        [15, 5, 0],
    ],
    dtype=int,
)


@dataclass(frozen=True)
class DatasetBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    train_groups: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series


def total_cost(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> int:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    return int(COST_MATRIX[y_true_array, y_pred_array].sum())


def last_readout(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["serial_number", "date"], kind="mergesort")
    return df.drop_duplicates("serial_number", keep="last")


def prepare_raw_training_frame(df: pd.DataFrame, horizon_days: int = 60) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Match the reference notebook's raw-data training logic.

    This is equivalent in spirit to the notebook:
    - use failed disks only
    - compute days to failure
    - keep rows inside the requested horizon
    - bucket labels with the same 10/20 day thresholds
    """
    df = df.sort_values(["serial_number", "date"], kind="mergesort")

    failure_dates = (
        df.loc[df["failure"] == 1, ["serial_number", "date"]]
        .drop_duplicates("serial_number", keep="last")
        .set_index("serial_number")["date"]
    )

    df = df[df["serial_number"].isin(failure_dates.index)].copy()
    df["failure_date"] = df["serial_number"].map(failure_dates)
    df["days_to_failure"] = (df["failure_date"] - df["date"]).dt.days

    mask = (df["days_to_failure"] >= 0) & (df["days_to_failure"] <= horizon_days)
    df = df.loc[mask].copy()

    df["label"] = np.select(
        [df["days_to_failure"] <= 10, df["days_to_failure"] <= 20],
        [2, 1],
        default=0,
    ).astype(int)

    feature_columns_to_drop = [
        "date",
        "serial_number",
        "failure",
        "failure_date",
        "days_to_failure",
        "label",
    ]
    X = df.drop(columns=feature_columns_to_drop)
    y = df["label"].astype(int)
    groups = df["serial_number"]
    return X, y, groups


def prepare_raw_training_set(path: Path, horizon_days: int = 60) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path, parse_dates=["date"])
    return prepare_raw_training_frame(df, horizon_days=horizon_days)


def prepare_raw_eval_frame(df: pd.DataFrame, id_path: Path, label_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df_last = last_readout(df)
    ids = pd.read_csv(id_path)
    labels = pd.read_csv(label_path)
    merged = df_last.merge(ids, on="serial_number", how="inner").merge(labels, on="id", how="inner")
    X = merged.drop(columns=["date", "serial_number", "id", "label", "failure"])
    y = merged["label"].astype(int)
    return X, y


def prepare_raw_eval_split(data_path: Path, id_path: Path, label_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path, parse_dates=["date"])
    return prepare_raw_eval_frame(df, id_path=id_path, label_path=label_path)


def load_raw_train_val(
    train_path: Path,
    val_path: Path,
    val_ids_path: Path,
    val_labels_path: Path,
    horizon_days: int = 60,
    apply_feature_engineering_preprocess: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(train_path, parse_dates=["date"], low_memory=False)
    val_df = pd.read_csv(val_path, parse_dates=["date"], low_memory=False)

    if apply_feature_engineering_preprocess:
        train_df, val_df, report = clean_train_val_datasets(train_df, val_df)
        print(
            "Applied feature_engineering preprocessing to raw baseline:",
            {
                "high_missing": len(report.dropped_high_missing),
                "constant": len(report.dropped_constant),
                "low_variance": len(report.dropped_low_variance),
                "duplicate": len(report.dropped_duplicate),
                "redundant_smart": len(report.dropped_redundant_smart),
            },
        )

    X_train, y_train, groups = prepare_raw_training_frame(train_df, horizon_days=horizon_days)
    X_val, y_val = prepare_raw_eval_frame(val_df, id_path=val_ids_path, label_path=val_labels_path)
    return X_train, y_train, groups, X_val, y_val


def load_aggregated_train_val(
    train_path: Path,
    val_path: Path,
    train_failed_disks_only: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(train_path, parse_dates=["date"], low_memory=False)
    val_df = pd.read_csv(val_path, parse_dates=["date"], low_memory=False)
    val_df = last_readout(val_df)

    if train_failed_disks_only:
        if "failure_date" in train_df.columns:
            train_df = train_df.loc[train_df["failure_date"].notna()].copy()
        else:
            failed_serials = (
                train_df.groupby("serial_number", sort=False)["failure"]
                .max()
                .loc[lambda series: series.eq(1)]
                .index
            )
            train_df = train_df.loc[train_df["serial_number"].isin(failed_serials)].copy()
        print(
            "Filtered aggregated train set to failed disks only:",
            {
                "rows": int(len(train_df)),
                "unique_disks": int(train_df["serial_number"].nunique()),
                "label_counts": train_df["label"].value_counts().sort_index().to_dict(),
            },
        )

    train_feature_columns_to_drop = [
        "date",
        "serial_number",
        "failure",
        "failure_date",
        "rul_days",
        "label",
    ]
    val_feature_columns_to_drop = [
        "date",
        "serial_number",
        "failure",
        "label",
    ]

    X_train = train_df.drop(columns=[column for column in train_feature_columns_to_drop if column in train_df.columns])
    y_train = train_df["label"].astype(int)
    groups = train_df["serial_number"]

    X_val = val_df.drop(columns=[column for column in val_feature_columns_to_drop if column in val_df.columns])
    y_val = val_df["label"].astype(int)
    return X_train, y_train, groups, X_val, y_val


def align_feature_frames(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_only_all_nan = [column for column in X_train.columns if X_train[column].isna().all()]
    if train_only_all_nan:
        X_train = X_train.drop(columns=train_only_all_nan)
        X_val = X_val.drop(columns=train_only_all_nan, errors="ignore")

    X_val = X_val.reindex(columns=X_train.columns)
    return X_train, X_val


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = [
        column for column in X_train.columns if X_train[column].dtype == "object"
    ]
    numeric_columns = [column for column in X_train.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_cv(y_train: pd.Series, groups: pd.Series, n_splits: int) -> Any:
    try:
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        list(cv.split(np.zeros(len(y_train)), y_train, groups))
        return cv
    except Exception:
        return GroupKFold(n_splits=n_splits)


def tune_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_splits: int,
    n_iter: int,
) -> RandomizedSearchCV:
    cv = build_cv(y_train, groups, n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__n_estimators": randint(150, 450),
            "model__max_depth": randint(3, 9),
            "model__learning_rate": uniform(0.03, 0.17),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__min_child_weight": randint(1, 8),
            "model__gamma": uniform(0.0, 2.0),
            "model__reg_lambda": uniform(0.5, 2.5),
            "model__reg_alpha": uniform(0.0, 1.0),
        },
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, Any]:
    labels = [0, 1, 2]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    multiclass_ovr_auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")
    multiclass_ovo_auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo", average="macro")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "aucroc_ovr_macro": multiclass_ovr_auc,
        "aucroc_ovo_macro": multiclass_ovo_auc,
        "total_cost": total_cost(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def run_experiment(
    name: str,
    dataset_bundle: DatasetBundle,
    n_splits: int,
    n_iter: int,
) -> dict[str, Any]:
    X_train, X_val = align_feature_frames(dataset_bundle.X_train.copy(), dataset_bundle.X_val.copy())
    pipeline = build_model_pipeline(X_train)
    search = tune_pipeline(
        pipeline=pipeline,
        X_train=X_train,
        y_train=dataset_bundle.y_train,
        groups=dataset_bundle.train_groups,
        n_splits=n_splits,
        n_iter=n_iter,
    )

    best_estimator = search.best_estimator_
    y_val_pred = best_estimator.predict(X_val)
    y_val_pred_proba = best_estimator.predict_proba(X_val)
    metrics = evaluate_predictions(dataset_bundle.y_val, y_val_pred, y_val_pred_proba)

    return {
        "experiment": name,
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "val_shape": [int(X_val.shape[0]), int(X_val.shape[1])],
        "train_label_counts": dataset_bundle.y_train.value_counts().sort_index().to_dict(),
        "val_label_counts": dataset_bundle.y_val.value_counts().sort_index().to_dict(),
        "best_params": search.best_params_,
        "best_cv_score_macro_f1": float(search.best_score_),
        "validation_metrics": metrics,
    }


def save_results(results: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"xgboost_comparison_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2, default=float))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare Backblaze XGBoost models.")
    parser.add_argument("--train-raw", default="data/backblaze/train_set.csv")
    parser.add_argument("--val-raw", default="data/backblaze/val_set.csv")
    parser.add_argument("--val-ids", default="data/backblaze/val_serial_number_id.csv")
    parser.add_argument("--val-labels", default="data/backblaze/val_label.csv")
    parser.add_argument("--train-agg", default="data/backblaze/clean/backblaze_clean_train_feature_agg.csv")
    parser.add_argument("--val-agg", default="data/backblaze/clean/backblaze_clean_val_feature_agg.csv")
    parser.add_argument("--horizon-days", type=int, default=60)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--search-iter", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--apply-raw-feature-engineering-preprocess",
        action="store_true",
        help="Apply the project's train-fitted column filtering and missing-value patching to the raw baseline.",
    )
    parser.add_argument(
        "--agg-train-failed-disks-only",
        action="store_true",
        help="Train the aggregated-feature model using only disks that eventually fail in the train set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_X_train, raw_y_train, raw_groups, raw_X_val, raw_y_val = load_raw_train_val(
        train_path=Path(args.train_raw),
        val_path=Path(args.val_raw),
        val_ids_path=Path(args.val_ids),
        val_labels_path=Path(args.val_labels),
        horizon_days=args.horizon_days,
        apply_feature_engineering_preprocess=args.apply_raw_feature_engineering_preprocess,
    )
    raw_bundle = DatasetBundle(
        X_train=raw_X_train,
        y_train=raw_y_train,
        train_groups=raw_groups,
        X_val=raw_X_val,
        y_val=raw_y_val,
    )

    agg_X_train, agg_y_train, agg_groups, agg_X_val, agg_y_val = load_aggregated_train_val(
        Path(args.train_agg),
        Path(args.val_agg),
        train_failed_disks_only=args.agg_train_failed_disks_only,
    )
    agg_bundle = DatasetBundle(
        X_train=agg_X_train,
        y_train=agg_y_train,
        train_groups=agg_groups,
        X_val=agg_X_val,
        y_val=agg_y_val,
    )

    results = {
        "raw_baseline": run_experiment(
            name="raw_baseline",
            dataset_bundle=raw_bundle,
            n_splits=args.cv_splits,
            n_iter=args.search_iter,
        ),
        "temporal_aggregation_rfod": run_experiment(
            name="temporal_aggregation_rfod",
            dataset_bundle=agg_bundle,
            n_splits=args.cv_splits,
            n_iter=args.search_iter,
        ),
    }

    output_path = save_results(results, Path(args.output_dir))
    print(json.dumps(results, indent=2, default=float))
    print(f"Saved comparison report to: {output_path}")


if __name__ == "__main__":
    main()
