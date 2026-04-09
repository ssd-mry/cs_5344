import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import time
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, make_scorer, ndcg_score, r2_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, log_loss
import argparse
import sys
import builtins
import os
from badgers.generators.tabular_data.outliers import DecompositionAndOutlierGenerator
from sklearn.utils import shuffle
from numpy.random import default_rng

base_dir = "/home/ruiyao/cs_5344/Project"
parser = argparse.ArgumentParser(description="ndcg + std experiments")

parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--msg", type=str, required=True, help="comment for the run")
parser.add_argument("--frac", type=float, required=True, help="Train_frac")
args = parser.parse_args()

dataset_name = args.dataset
sample_frac = args.frac

logs_all_folder = os.path.join("results/logs_all")
logs_result_folder = os.path.join("results/logs_result")
scores_folder = os.path.join("results/Scores")
os.makedirs(logs_all_folder, exist_ok=True)
os.makedirs(logs_result_folder, exist_ok=True)
logs_all_folder = os.path.join(logs_all_folder, "rfod_2025_10")
logs_result_folder = os.path.join(logs_result_folder, "rfod_2025_10")
os.makedirs(scores_folder, exist_ok=True)
scores_folder = os.path.join(scores_folder, f"{dataset_name}")
os.makedirs(scores_folder, exist_ok=True)

os.makedirs(logs_all_folder, exist_ok=True)
os.makedirs(logs_result_folder, exist_ok=True)
stdout_file = os.path.join(logs_all_folder, f"{dataset_name}.txt")
resout_file = os.path.join(logs_result_folder, f"{dataset_name}.txt")
open(resout_file, "w").close()  # 每次运行清空结果文件


def print_to_log(*args, **kwargs):
    print(*args, **kwargs)
    with open(resout_file, "a") as f:
        print(*args, **kwargs, file=f)


def smape(y_true, y_pred):
    # print('smape', y_true.shape, y_pred.shape, y_true[0], y_pred[0])
    return np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 0.00001)
    )


alpha = [
    0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.25,
]  # ignore outlier data


def alphaquantile(
    matrix1,
    matrix2,
    categorical_columns,
    continuous_columns,
    column_names,
    model_scores,
    residual,
    weighted=True,
):
    """
    Calculate improved Gower distance between two matrices.

    matrix1, matrix2: Two 2D matrices to compare, with the same shape. Matrix1: prediction, Matrix2: true dirty data.
    categorical_columns: List of indices representing categorical columns.
    continuous_columns: List of indices representing continuous columns.

    Returns: A distance matrix.
    """
    n_rows, n_cols = matrix1.shape
    result = {}
    for a in alpha:
        diff_matrix = np.zeros((n_rows, n_cols))

        # Calculate the difference for each continuous column
        for col in continuous_columns:
            col_name = column_names[col]
            iqr = np.percentile(residual[col_name], 100 * (1 - a)) * 2
            if iqr == 0:
                scaled_diff = np.zeros_like(matrix1[:, col])
            else:
                scaled_diff = np.abs(matrix1[:, col] - matrix2[:, col]) / iqr
                scaled_diff = np.clip(scaled_diff, 0, 1)
            diff_matrix[:, col] = scaled_diff

        # Calculate the difference for each categorical column
        for col_idx in categorical_columns:
            col_name = column_names[col_idx]
            diff_matrix[:, col_idx] = (
                1 - matrix1[:, col_idx]
            )  # false proba for categorical

        if weighted:
            model_scores = np.array(model_scores)

            weighted_diff = diff_matrix * model_scores

            mean_diff = np.sum(weighted_diff, axis=1)
        else:
            mean_diff = np.mean(diff_matrix, axis=1)
            
        result[a] = (diff_matrix, mean_diff)

    return result


def compute_ndcg(y_true, y_scores, k=10):
    # Reshape as 2D arrays for scikit-learn's ndcg_score
    y_true = np.array(y_true).reshape(1, -1)
    y_scores = np.array(y_scores).reshape(1, -1)
    return ndcg_score(y_true, y_scores, k=k)


def top_k_precision(y, y_scores, k=5):
    y = np.asarray(y)
    y_scores = np.asarray(y_scores)
    # Get the top k indices of the predicted scores
    top_k_indices = np.argsort(y_scores)[-k:]
    precision_at_k = np.mean(y[top_k_indices])
    return precision_at_k


def tree_scoring(tree, X, y):
    if col_name in num_cols:
        y_pred = tree.predict(X)
        error = smape(y, y_pred)
    else:
        y_pred = tree.predict_proba(X)
        n_classes = y_pred.shape[1]
        if n_classes != len(np.unique(y)):
            score = 0
        elif n_classes == 1:
            score = 0
        elif n_classes == 2:
            score = roc_auc_score(y, y_pred[:, 1])
        else:
            score = roc_auc_score(y, y_pred, multi_class="ovr")
        error = 1 - score
    return error, y_pred


def trees_scoring(y_preds, y):
    y_preds = np.mean(y_preds, axis=0)
    if col_name in num_cols:
        error = smape(y, y_preds)
    else:
        n_classes = y_preds.shape[1]
        if n_classes != len(np.unique(y)):
            score = 0
        elif n_classes == 1:
            score = 0
        elif n_classes == 2:
            score = roc_auc_score(y, y_preds[:, 1])
        else:
            score = roc_auc_score(y, y_preds, multi_class="ovr")
        error = 1 - score
    return error


def calc_resi(col_name, Xy, treeid, model):
    X_valid_resi = Xy.drop(columns=[col_name])
    y_valid_resi = Xy[col_name]
    top_tree_preds = np.zeros_like(y_valid_resi)
    trees = model.estimators_
    for idx in treeid:
        top_tree_preds += trees[idx].predict(X_valid_resi.values)
    top_tree_preds = top_tree_preds / len(treeid)
    return np.abs(top_tree_preds - y_valid_resi.values)


sys.stdout = open(stdout_file, "w")

fitonly = []
if dataset_name == "liver":
    data = pd.read_csv("./datasets/medical/liver_time.csv")
    Paras = {
        "门脉高压": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "n_estimators": 315,
        },
        "食管胃底静脉曲张": {
            "max_depth": 14,
            "min_samples_leaf": 4,
            "min_samples_split": 5,
            "n_estimators": 362,
        },
        "ALT": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "AST": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "ALP": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
            "n_estimators": 195,
        },
        "GGT": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "n_estimators": 315,
        },
        "Tbil": {
            "max_depth": 16,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
            "n_estimators": 157,
        },
        "Alb": {
            "max_depth": 14,
            "min_samples_leaf": 4,
            "min_samples_split": 5,
            "n_estimators": 362,
        },
        "PT": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "腹水": {
            "max_depth": 12,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
            "n_estimators": 230,
        },
        "肝性脑病": {
            "max_depth": 12,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
            "n_estimators": 230,
        },
        "child": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "INR": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "Plt": {
            "max_depth": 16,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
            "n_estimators": 157,
        },
        "HBV": {
            "max_depth": 13,
            "min_samples_leaf": 5,
            "min_samples_split": 6,
            "n_estimators": 351,
        },
        "AFP": {
            "max_depth": 13,
            "min_samples_leaf": 5,
            "min_samples_split": 6,
            "n_estimators": 351,
        },
        "自免肝": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
            "n_estimators": 195,
        },
        "性别": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "n_estimators": 315,
        },
        "age": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "n_estimators": 315,
        },
    }
    categ_cols = ["门脉高压", "食管胃底静脉曲张", "HBV", "自免肝", "性别"]
    target = "bleeding"
    X = data.drop(columns=[target, "time", "mrn"])
    num_cols = [col for col in X.columns if col not in categ_cols]
elif dataset_name == "scania_label":
    data = pd.read_csv(
        "/home/yushuo/Longitudinal/data/scania/train_merged_filtered.csv"
    )
    target = "label"
    X = data.drop(columns=[target, "vehicle_id"])

    categ_cols = [
        "Spec_0",
        "Spec_1",
        "Spec_2",
        "Spec_3",
        "Spec_4",
        "Spec_5",
        "Spec_6",
        "Spec_7",
    ]
    num_cols = [col for col in X.columns if col not in categ_cols]
elif dataset_name == "beth":
    data = pd.read_csv("/home/yushuo/Longitudinal/data/beth/beth.csv")
    target = "evil"
    X = data.drop(columns=[target, "sus", "timestamp"])

    fitonly = ["hostName"]
    num_cols = ["argsNum", "returnValue", "processId", "parentProcessId", "threadId"]
    categ_cols = [col for col in X.columns if col not in num_cols]
elif dataset_name == "beth_longi":
    data = pd.read_csv("/home/yushuo/Longitudinal/data/beth/beth_longi.csv")
    target = "evil"
    X = data.drop(columns=[target, "sus"])

    num_cols = [
        "argsNum",
        "returnValue",
        "processId",
        "parentProcessId",
        "threadId",
        "lastThreadId",
        "lastParentProcessId",
        "lastProcessId",
        "Interval",
    ]
    categ_cols = [col for col in X.columns if col not in num_cols]
elif dataset_name == "loan":
    data = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan.csv")
    data["CurrentLoanDelinquencyStatus"] = data["CurrentLoanDelinquencyStatus"].astype(
        "string"
    )
    data["target"] = data["CurrentLoanDelinquencyStatus"] != "0"
    target = "target"
    data.drop(
        columns=[
            "CurrentLoanDelinquencyStatus",
            "LoanSequenceNumber",
            "MonthlyReportingPeriod",
        ],
        inplace=True,
    )
    X = data.drop(columns=[target])

    categ_cols = []
    num_cols = [col for col in X.columns if col not in categ_cols]
elif dataset_name == "loan_deri":
    data = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_derivative.csv")
    data["CurrentLoanDelinquencyStatus"] = data["CurrentLoanDelinquencyStatus"].astype(
        "string"
    )
    data["target"] = data["CurrentLoanDelinquencyStatus"] != "0"
    target = "target"
    data.drop(
        columns=["CurrentLoanDelinquencyStatus", "LoanSequenceNumber"], inplace=True
    )
    X = data.drop(columns=[target])

    categ_cols = []
    num_cols = [col for col in X.columns if col not in categ_cols]
elif dataset_name == "loan_merge":
    data = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_merged.csv")
    target = "target"
    data.drop(columns=["LoanSequenceNumber"], inplace=True)
    X = data.drop(columns=[target])

    categ_cols = []
    num_cols = [col for col in X.columns if col not in categ_cols]

elif dataset_name == "loan_sub":
    data = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_subject.csv")
    target = "target"
    data.drop(
        columns=[
            "LoanSequenceNumber",
            "FirstPaymentDate",
            "MaturityDate",
            "SellerName",
            "ServicerName",
            "PostalCode",
            "SuperConformingFlag",
            "PreHARP_Flag",
            "ReliefRefinanceIndicator",
        ],
        inplace=True,
    )
    X = data.drop(columns=[target])

    categ_cols = [
        "OccupancyStatus",
        "Channel",
        "FirstTimeHomebuyerFlag",
        "PPM_Flag",
        "ProductType",
        "PropertyState",
        "PropertyType",
        "LoanPurpose",
        "MSA",
        "ProgramIndicator",
        "PropertyValMethod",
        "InterestOnlyFlag",
        "BalloonIndicator",
    ]
    num_cols = [col for col in X.columns if col not in categ_cols]

elif dataset_name == "scania2":
    train_vehicle = pd.read_csv(
        "/home/yushuo/Longitudinal/data/scania/vehicle_train.csv"
    )
    train_record = pd.read_csv("/home/yushuo/Longitudinal/data/scania/record_train.csv")
    valipri_vehicle = pd.read_csv(
        "/home/yushuo/Longitudinal/data/scania/vehicle_valipri.csv"
    )
    valipri_record = pd.read_csv(
        "/home/yushuo/Longitudinal/data/scania/record_valipri.csv"
    )
    valipub_vehicle = pd.read_csv(
        "/home/yushuo/Longitudinal/data/scania/vehicle_valipub.csv"
    )
    valipub_record = pd.read_csv(
        "/home/yushuo/Longitudinal/data/scania/record_valipub.csv"
    )
    test_vehicle = pd.read_csv("/home/yushuo/Longitudinal/data/scania/vehicle_test.csv")
    test_record = pd.read_csv("/home/yushuo/Longitudinal/data/scania/record_test.csv")

    train = train_record.join(train_vehicle.set_index("index"), on="index")
    valipri = valipri_record.join(valipri_vehicle.set_index("index"), on="index")
    valipub = valipub_record.join(valipub_vehicle.set_index("index"), on="index")
    test = test_record.join(test_vehicle.set_index("index"), on="index")

    data = pd.concat([train, valipri, valipub, test], ignore_index=True)

    categ_cols = [
        "Spec_0",
        "Spec_1",
        "Spec_2",
        "Spec_3",
        "Spec_4",
        "Spec_5",
        "Spec_6",
        "Spec_7",
    ]
    num_cols = [
        col
        for col in data.columns
        if col not in categ_cols and col != "target" and col != "index"
    ]
    target = "target"

elif dataset_name == "loan_split":
    train = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_train.csv")
    valipri = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_valipri.csv")
    valipub = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_valipub.csv")
    test = pd.read_csv("/home/yushuo/Longitudinal/data/loanstd/loan_test.csv")
    data = pd.concat([train, valipri, valipub, test], ignore_index=True)

    data.drop(
        columns=[
            "LoanSequenceNumber",
            "FirstPaymentDate",
            "MaturityDate",
            "SellerName",
            "ServicerName",
            "PostalCode",
            "SuperConformingFlag",
            "PreHARP_Flag",
            "ReliefRefinanceIndicator",
        ],
        inplace=True,
    )
    target = "target"

    categ_cols = [
        "OccupancyStatus",
        "Channel",
        "FirstTimeHomebuyerFlag",
        "PPM_Flag",
        "ProductType",
        "PropertyState",
        "PropertyType",
        "LoanPurpose",
        "MSA",
        "ProgramIndicator",
        "PropertyValMethod",
        "InterestOnlyFlag",
        "BalloonIndicator",
    ]
    num_cols = [
        col for col in data.columns if col not in categ_cols and col != "target"
    ]
    X = data[num_cols]

elif dataset_name == "beth_split":
    train = pd.read_csv("/home/yushuo/Longitudinal/data/beth/beth_train.csv")
    valipri = pd.read_csv("/home/yushuo/Longitudinal/data/beth/beth_valipri.csv")
    valipub = pd.read_csv("/home/yushuo/Longitudinal/data/beth/beth_valipub.csv")
    test = pd.read_csv("/home/yushuo/Longitudinal/data/beth/beth_test.csv")
    data = pd.concat([train, valipri, valipub, test], ignore_index=True)
    target = "target"
    X = data.drop(
        columns=[target, "index", "args", "stackAddresses", "hostName", "timestamp"]
    )
    num_cols = ["argsNum", "returnValue", "processId", "parentProcessId", "threadId"]
    categ_cols = [col for col in X.columns if col not in num_cols]
if dataset_name == "thyroid":
    target = "target"
    data = pd.read_csv(os.path.join("datasets/thyroid/thyroid.csv"), sep=";")
    X = data.drop(columns=["target"])
    y = data["target"]

    # Transform labels: 'o' -> 1, 'n' -> 0
    y = np.where(y == "o", 1, 0)

    categ_cols = [
        "Sex",
        "on_thyroxine",
        "query_on_thyroxine",
        "on_antithyroid_medication",
        "sick",
        "pregnant",
        "thyroid_surgery",
        "I131_treatment",
        "query_hypothyroid",
        "query_hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
    ]
    num_cols = [
        "Age",
        "TSH",
        "T3_measured",
        "TT4_measured",
        "T4U_measured",
        "FTI_measured",
    ]
    if sample_frac == 0.05:
        Paras = {
            "Age": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Sex": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "on_thyroxine": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "query_on_thyroxine": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "on_antithyroid_medication": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "sick": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "pregnant": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "thyroid_surgery": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "I131_treatment": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hypothyroid": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hyperthyroid": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "lithium": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "goitre": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "tumor": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hypopituitary": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "psych": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "TSH": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "T3_measured": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "TT4_measured": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "T4U_measured": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FTI_measured": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "Age": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "Sex": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "on_thyroxine": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "query_on_thyroxine": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "on_antithyroid_medication": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "sick": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "pregnant": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "thyroid_surgery": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "I131_treatment": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hypothyroid": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hyperthyroid": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "lithium": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "goitre": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "tumor": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hypopituitary": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "psych": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "TSH": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "T3_measured": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "TT4_measured": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "T4U_measured": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "FTI_measured": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "Age": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "Sex": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "on_thyroxine": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "query_on_thyroxine": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "on_antithyroid_medication": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "sick": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "pregnant": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "thyroid_surgery": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "I131_treatment": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hypothyroid": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hyperthyroid": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "lithium": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "goitre": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "tumor": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hypopituitary": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "psych": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "TSH": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "T3_measured": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "TT4_measured": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "T4U_measured": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FTI_measured": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "Age": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Sex": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "on_thyroxine": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "query_on_thyroxine": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "on_antithyroid_medication": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "sick": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "pregnant": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "thyroid_surgery": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "I131_treatment": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hypothyroid": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hyperthyroid": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "lithium": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "goitre": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "tumor": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hypopituitary": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "psych": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "TSH": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "T3_measured": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "TT4_measured": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "T4U_measured": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "FTI_measured": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
        }
    else:
        Paras = {
            "Age": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "Sex": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "on_thyroxine": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "query_on_thyroxine": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "on_antithyroid_medication": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "sick": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "pregnant": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "thyroid_surgery": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "I131_treatment": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "query_hypothyroid": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "query_hyperthyroid": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "lithium": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "goitre": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "tumor": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hypopituitary": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "psych": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "TSH": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "T3_measured": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "TT4_measured": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "T4U_measured": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "FTI_measured": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
        }
elif dataset_name == "arrhythmia":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    # Transform labels: 1 -> 0, all other values -> 1 (anomaly)
    y = np.where(y == 1, 0, 1)

    categ_cols_indices = [
        1,
        21,
        22,
        23,
        24,
        25,
        26,
        33,
        34,
        35,
        36,
        37,
        38,
        45,
        46,
        47,
        48,
        49,
        50,
        57,
        58,
        59,
        60,
        61,
        62,
        69,
        70,
        71,
        72,
        73,
        74,
        81,
        82,
        83,
        84,
        85,
        86,
        93,
        94,
        95,
        96,
        97,
        98,
        105,
        106,
        107,
        108,
        109,
        110,
        117,
        118,
        119,
        120,
        121,
        122,
        129,
        130,
        131,
        132,
        133,
        134,
        141,
        142,
        143,
        144,
        145,
        146,
        153,
        154,
        155,
        156,
        157,
        158,
    ]
    categ_cols = [X.columns[i] for i in categ_cols_indices]
    num_cols = [col for col in X.columns if col not in categ_cols]
    if sample_frac == 0.05:
        Paras = {
            "age": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "sex": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "weight": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "qrs_duration": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "p-r_interval": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "q-t_interval": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_interval": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "qrs": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "T": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "P": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "QRST": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "J": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "heart_rate": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "q_wave": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "r_wave": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "s_wave": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "R'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "S'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AA": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "AB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AI": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "AJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AM": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "AN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AU": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "AV": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "AY": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "AZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AB'": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BB": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "BC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BI": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "BJ": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BN": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "BO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BV": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "BY": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "BZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CC": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "CD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Cf": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CK": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "CL": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "CM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CZ": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "DA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DL": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "DM": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "DN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "DR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EA": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "EB": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "EC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ED": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EF": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EN": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EO": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "EP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ER": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ES": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "ET": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FC": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "FD": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "FE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FO": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "FP": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "FR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FT": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "FU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GC": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GD": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "GE": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "GF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GI": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GJ": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GL": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GO": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GP": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GT": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "GU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GV": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "GY": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GZ": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HA": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "HB": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HC": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "HD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HF": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HG": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HH": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "HI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HK": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "HL": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HP": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "HR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HS": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HU": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "HV": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "HY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IC": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "ID": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "IE": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "IF": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "IG": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "IH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "II": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "IJ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IM": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "IN": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "IO": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "IP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IR": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "IS": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "IT": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "IU": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "IV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JA": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "JB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JC": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JD": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JF": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JK": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "JL": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "JM": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "JN": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "JO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JP": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "JR": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "JS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JV": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "JY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KA": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "KB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KD": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "KE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KG": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "KH": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "KI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KJ": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KM": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "KN": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "KO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KR": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KS": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KT": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "KU": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KZ": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "LA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LD": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "LE": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "LF": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "LG": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "age": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "sex": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "height": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "weight": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "qrs_duration": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "p-r_interval": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "q-t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "t_interval": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "p_interval": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "qrs": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "T": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "P": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "QRST": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "J": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "heart_rate": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "q_wave": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "r_wave": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "s_wave": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "R'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "S'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AU": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "AV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AB'": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BV": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "BY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Cf": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EB": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "EC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ED": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ER": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ES": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ET": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FG": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "FH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GJ": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GL": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GU": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "GV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HA": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "HB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HF": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "HG": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HJ": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "HK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HM": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "HN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IC": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "ID": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "IE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IG": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "IH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "II": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IM": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JM": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JR": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JY": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JZ": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "KA": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KR": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KS": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "KT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "age": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "sex": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "weight": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "qrs_duration": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p-r_interval": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "q-t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "p_interval": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "qrs": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "T": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "P": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "QRST": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "J": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "heart_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "q_wave": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "r_wave": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "s_wave": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "R'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "S'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AB'": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BJ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BY": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "BZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Cf": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CJ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ED": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EF": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "EG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "EO": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ER": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ES": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ET": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FC": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "FD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GC": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GE": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GF": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "GG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GI": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GJ": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "GK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GM": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GO": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GP": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GZ": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "HA": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "HB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HJ": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HN": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "HO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HU": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HV": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ID": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "IE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "II": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IJ": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "IK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IN": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IR": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "IS": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IZ": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "JA": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JC": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "JD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JJ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JN": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "JO": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JU": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "JV": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "JY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JZ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KA": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KN": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "KO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KV": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "LB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "LE": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "LF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LG": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "age": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "sex": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "height": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "weight": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "qrs_duration": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "p-r_interval": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "q-t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "t_interval": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "p_interval": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "qrs": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "T": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "P": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "QRST": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "J": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "heart_rate": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "q_wave": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "r_wave": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "s_wave": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "R'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "S'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AI": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "AJ": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "AK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AM": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "AN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AV": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "AY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AB'": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BB": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BJ": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BL": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "BM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BN": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "BO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "BY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Cf": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CJ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CM": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "CN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CZ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "DA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DM": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "DN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "EB": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ED": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EF": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "EN": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ER": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ES": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ET": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FO": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "FP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GD": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GE": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GI": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "GJ": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GY": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "GZ": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HJ": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "HK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HS": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HU": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "HV": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IC": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "ID": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IG": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "II": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IO": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "IP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IR": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "IS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IU": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IZ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "JA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JB": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JC": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "JD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JE": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "JF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JJ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JK": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "JL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JM": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "JN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JP": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "JR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JZ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KC": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KD": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "KE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KH": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "KI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KJ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KK": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KM": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "KN": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KR": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KS": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KT": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KZ": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "LA": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "LB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "LE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    else:
        Paras = {
            "age": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "sex": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "weight": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "qrs_duration": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "p-r_interval": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "q-t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "t_interval": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_interval": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "qrs": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "T": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "P": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "QRST": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "J": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "heart_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "q_wave": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "r_wave": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "s_wave": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "R'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "S'_wave": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "AB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AI": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "AJ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "AK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "AV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AY": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AB'": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BJ": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "BK": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "BL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "BO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BV": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "BY": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "BZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CD": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Cf": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CJ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CO": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "CY": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "CZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DA": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "DB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DN": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "EA": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "EB": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "EC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ED": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "EG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EM": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "EN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "EO": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "EP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ER": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ES": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ET": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "EZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FC": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "FD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FE": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FG": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "FH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FJ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FN": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "FO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FP": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "FR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FU": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "FZ": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GC": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "GD": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GF": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "GG": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GJ": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GL": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "GM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GN": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "GO": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GR": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "GT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "GU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "GY": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "GZ": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "HA": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HB": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HE": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HH": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HJ": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "HK": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HL": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "HM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HN": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "HP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HR": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "HS": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "HU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "HV": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "HY": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "HZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IA": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IC": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "ID": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IE": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IF": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IG": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "IH": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "II": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IJ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IK": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IL": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IN": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "IO": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IP": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IR": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "IT": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "IU": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "IV": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IY": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "IZ": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "JA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JD": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JE": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JF": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "JG": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JH": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JI": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JJ": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "JK": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "JL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JM": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JN": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JO": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "JP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JR": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "JS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JT": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "JU": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "JV": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "JY": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "JZ": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KA": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "KB": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KD": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KE": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KF": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KG": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "KH": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "KI": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KJ": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "KK": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KM": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KN": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "KO": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "KR": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KT": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "KU": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "KY": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "KZ": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "LA": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "LB": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "LD": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LE": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "LF": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "LG": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
elif dataset_name == "spambase":
    data = pd.read_csv(
        os.path.join(base_dir, "datasets/spambase/spambase.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = []
    num_cols = X.columns.tolist()
    if sample_frac == 0.05:
        Paras = {
            "word_freq_make": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_address": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_all": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_3d": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_our": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "word_freq_over": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_remove": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_internet": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_order": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_mail": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_receive": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "word_freq_will": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_people": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_report": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_addresses": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_free": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_business": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "word_freq_email": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_you": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_credit": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_your": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_font": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_000": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_money": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_hp": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_hpl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_george": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_650": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_lab": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_labs": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_telnet": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_857": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_data": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_415": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_85": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_technology": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_1999": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_parts": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_pm": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_direct": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_cs": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_meeting": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_original": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_project": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_re": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_edu": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_table": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_conference": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "char_freq_;": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "char_freq_(": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "char_freq_[": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "char_freq_!": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "char_freq_$": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "char_freq_#": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_average": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "capital_run_length_longest": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "capital_run_length_total": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "word_freq_make": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_address": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "word_freq_all": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_3d": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_our": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_over": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_remove": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_internet": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_order": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_mail": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_receive": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_will": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_people": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_report": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_addresses": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_free": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_business": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_email": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_you": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_credit": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_your": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_font": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_000": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_money": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_hp": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_hpl": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_george": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_650": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_lab": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_labs": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_telnet": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_857": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_data": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_415": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_85": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_technology": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_1999": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_parts": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_pm": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_direct": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_cs": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_meeting": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_original": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_project": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_re": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_edu": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_table": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_conference": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "char_freq_;": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "char_freq_(": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "char_freq_[": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "char_freq_!": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "char_freq_$": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "char_freq_#": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "capital_run_length_average": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "capital_run_length_longest": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_total": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "word_freq_make": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_address": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_all": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_3d": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_our": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_over": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_remove": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_internet": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_order": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_mail": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_receive": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_will": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_people": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_report": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_addresses": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_free": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_business": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_email": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_you": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_credit": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_your": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_font": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_000": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_money": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_hp": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_hpl": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_george": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_650": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_lab": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_labs": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_telnet": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_857": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_data": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_415": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_85": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_technology": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_1999": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_parts": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_pm": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_direct": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_cs": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_meeting": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_original": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_project": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_re": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_edu": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_table": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_conference": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "char_freq_;": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "char_freq_(": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "char_freq_[": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "char_freq_!": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "char_freq_$": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "char_freq_#": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "capital_run_length_average": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_longest": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_total": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "word_freq_make": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_address": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_all": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_3d": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_our": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_over": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_remove": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_internet": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_order": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "word_freq_mail": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_receive": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_will": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_people": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_report": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_addresses": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_free": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_business": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_email": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_you": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_credit": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_your": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_font": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_000": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_money": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_hp": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_hpl": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_george": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_650": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_lab": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_labs": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_telnet": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_857": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_data": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_415": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_85": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_technology": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_1999": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_parts": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_pm": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_direct": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_cs": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_meeting": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_original": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_project": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_re": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_edu": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_table": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_conference": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "char_freq_;": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "char_freq_(": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "char_freq_[": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "char_freq_!": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "char_freq_$": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "char_freq_#": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "capital_run_length_average": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "capital_run_length_longest": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_total": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    else:
        Paras = {
            "word_freq_make": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_address": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_all": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_3d": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_our": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_over": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_remove": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_internet": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_order": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "word_freq_mail": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_receive": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_will": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_people": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_report": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_addresses": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_free": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_business": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "word_freq_email": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_you": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "word_freq_credit": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_your": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_font": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_000": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_money": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_hp": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_hpl": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "word_freq_george": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_650": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_lab": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_labs": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_telnet": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_857": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_data": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_415": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_85": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_technology": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "word_freq_1999": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_parts": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_pm": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_direct": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_cs": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "word_freq_meeting": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "word_freq_original": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_project": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "word_freq_re": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_edu": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "word_freq_table": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "word_freq_conference": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "char_freq_;": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "char_freq_(": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "char_freq_[": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "char_freq_!": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "char_freq_$": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "char_freq_#": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_average": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "capital_run_length_longest": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "capital_run_length_total": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
elif dataset_name == "pima":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = []
    num_cols = X.columns.tolist()
    if sample_frac == 0.05:
        Paras = {
            "Pregnancies": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Glucose": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BloodPressure": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "SkinThickness": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Insulin": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "BMI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DiabetesPedigreeFunction": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Age": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "Pregnancies": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Glucose": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "BloodPressure": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "SkinThickness": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Insulin": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BMI": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "DiabetesPedigreeFunction": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Age": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "Pregnancies": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Glucose": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "BloodPressure": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "SkinThickness": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Insulin": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "BMI": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "DiabetesPedigreeFunction": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Age": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "Pregnancies": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Glucose": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "BloodPressure": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "SkinThickness": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Insulin": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BMI": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "DiabetesPedigreeFunction": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "Age": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
        }
    else:
        Paras = {
            "Pregnancies": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Glucose": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BloodPressure": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "SkinThickness": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Insulin": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "BMI": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DiabetesPedigreeFunction": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "Age": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
        }
elif dataset_name == "breastw":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    data = data.iloc[:, 1:]  # Drop first column
    X = data.drop(columns=["target"])
    y = data["target"]

    # Transform labels: 'M' -> 1, 'B' -> 0
    y = np.where(y == "M", 1, 0)

    categ_cols = []
    num_cols = X.columns.tolist()
    if sample_frac == 0.05:
        Paras = {
            "radius_mean": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "texture_mean": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "perimeter_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_mean": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "smoothness_mean": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "compactness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_mean": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "concave points_mean": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "symmetry_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "fractal_dimension_mean": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "radius_se": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "texture_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "perimeter_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_se": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "smoothness_se": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "compactness_se": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "concavity_se": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "concave points_se": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "symmetry_se": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "fractal_dimension_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "radius_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "texture_worst": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "perimeter_worst": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "area_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_worst": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "compactness_worst": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "concavity_worst": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "concave points_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "symmetry_worst": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "fractal_dimension_worst": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "radius_mean": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "texture_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "perimeter_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "compactness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "symmetry_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "fractal_dimension_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "radius_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "texture_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "perimeter_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "compactness_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "concavity_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "fractal_dimension_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "radius_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "texture_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "perimeter_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "compactness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "symmetry_worst": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "fractal_dimension_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "radius_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "texture_mean": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "perimeter_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "smoothness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "compactness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "fractal_dimension_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "radius_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "texture_se": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "perimeter_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_se": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "smoothness_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "compactness_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "concavity_se": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "concave points_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_se": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "fractal_dimension_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "radius_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "texture_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "perimeter_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "compactness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_worst": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "fractal_dimension_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "radius_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "texture_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "perimeter_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_mean": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "compactness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "concavity_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "concave points_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "fractal_dimension_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "radius_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "texture_se": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "perimeter_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_se": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "compactness_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "concave points_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "fractal_dimension_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "radius_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "texture_worst": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "perimeter_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smoothness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "compactness_worst": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "concavity_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "fractal_dimension_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    else:
        Paras = {
            "radius_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "texture_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "perimeter_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "smoothness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "compactness_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_mean": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "symmetry_mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "fractal_dimension_mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "radius_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "texture_se": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "perimeter_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "area_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "smoothness_se": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "compactness_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_se": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concave points_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "symmetry_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "fractal_dimension_se": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "radius_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "texture_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "perimeter_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "smoothness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "compactness_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "concavity_worst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "concave points_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "symmetry_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "fractal_dimension_worst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
elif dataset_name == "adult":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ]
    num_cols = [
        "age",
        "fnlwgt",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    Paras = {
        "age": {
            "max_depth": 13,
            "min_samples_leaf": 5,
            "min_samples_split": 6,
            "n_estimators": 351,
        },
        "workclass": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "fnlwgt": {
            "max_depth": 5,
            "min_samples_leaf": 4,
            "min_samples_split": 8,
            "n_estimators": 176,
        },
        "education": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "educational-num": {
            "max_depth": 18,
            "min_samples_leaf": 5,
            "min_samples_split": 4,
            "n_estimators": 385,
        },
        "marital-status": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "occupation": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "relationship": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "race": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
        "gender": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "n_estimators": 315,
        },
        "capital-gain": {
            "max_depth": 5,
            "min_samples_leaf": 4,
            "min_samples_split": 8,
            "n_estimators": 176,
        },
        "capital-loss": {
            "max_depth": 5,
            "min_samples_leaf": 4,
            "min_samples_split": 8,
            "n_estimators": 176,
        },
        "hours-per-week": {
            "max_depth": 12,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
            "n_estimators": 230,
        },
        "native-country": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
            "n_estimators": 341,
        },
    }

elif dataset_name == "mushrooms":
    X = data.drop(columns=["target"])
    y = data["target"]

    # Transform labels: 'p' -> 1, 'e' -> 0
    y = np.where(y == "p", 1, 0)

    categ_cols = X.columns.tolist()
    num_cols = []
elif dataset_name == "default":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    data = data.dropna(subset=[target])
    data = data.iloc[:, 1:]  # Drop first column
    # print(data.columns)
    X = data.drop(columns=[target])
    y = data[target]

    categ_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
elif dataset_name == "wine":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = []
    num_cols = X.columns.tolist()

    y = np.where(y.isin([5, 6]), 0, 1)
    n_estimators = 62
    max_depth = 64
    min_samples_split = 4
    min_samples_leaf = 1
    if sample_frac == 0.05:
        Paras = {
            "fixed acidity": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "volatile acidity": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "citric acid": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "residual sugar": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "chlorides": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "free sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "total sulfur dioxide": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "density": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "pH": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "sulphates": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "alcohol": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "fixed acidity": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "volatile acidity": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "citric acid": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "residual sugar": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "chlorides": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "free sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "total sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "density": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "pH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "sulphates": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "alcohol": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "fixed acidity": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "volatile acidity": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "citric acid": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "residual sugar": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "chlorides": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "free sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "total sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "density": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "pH": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "sulphates": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "alcohol": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "fixed acidity": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "volatile acidity": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "citric acid": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "residual sugar": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "chlorides": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "free sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "total sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "density": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "pH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "sulphates": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "alcohol": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    else:
        Paras = {
            "fixed acidity": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "volatile acidity": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "citric acid": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "residual sugar": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "chlorides": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "free sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "total sulfur dioxide": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "density": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "pH": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "sulphates": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "alcohol": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
elif dataset_name == "dos" or dataset_name == "backdoor":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    data = data.rename(columns={"attack_cat": "target"})
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]
    categ_cols = ["proto", "service", "state", "is_sm_ips_ports", "is_ftp_login"]
    num_cols = [col for col in X.columns if col not in categ_cols]

    y = np.where(y == "Normal", 0, 1)
    if dataset_name == "dos":
        Paras = {
            "dur": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "proto": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "state": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "spkts": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dpkts": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sbytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dbytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sttl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dttl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sload": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dload": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sloss": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dloss": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sinpkt": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dinpkt": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sjit": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "djit": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "swin": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "stcpb": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "dtcpb": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "dwin": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "tcprtt": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "synack": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ackdat": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dmean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "trans_depth": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "response_body_len": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_srv_src": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ct_state_ttl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_dst_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ct_src_dport_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_dst_sport_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_dst_src_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "is_ftp_login": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_ftp_cmd": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ct_flw_http_mthd": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_src_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ct_srv_dst": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "is_sm_ips_ports": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
        }
    else:
        Paras = {
            "dur": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "proto": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "state": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "spkts": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dpkts": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sbytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dbytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rate": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "sttl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dttl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sload": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dload": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sloss": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dloss": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sinpkt": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dinpkt": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "sjit": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "djit": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "swin": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "stcpb": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "dtcpb": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "dwin": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "tcprtt": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "synack": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ackdat": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "smean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dmean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "trans_depth": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "response_body_len": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_srv_src": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_state_ttl": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_dst_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ct_src_dport_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_dst_sport_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_dst_src_ltm": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "is_ftp_login": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ct_ftp_cmd": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ct_flw_http_mthd": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "ct_src_ltm": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "ct_srv_dst": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "is_sm_ips_ports": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
        }
elif dataset_name == "ionosphere":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = []
    num_cols = X.columns.tolist()
    if sample_frac == 0.05:
        Paras = {
            "0": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "1": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "2": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "3": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "4": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "5": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "6": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "7": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "8": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "9": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "10": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "11": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "12": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "13": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "14": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "15": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "16": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "17": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "18": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "19": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "20": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "21": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "22": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "23": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "24": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "25": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "26": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "27": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "28": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "29": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "30": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "31": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "32": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "33": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "0": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "1": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "2": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "3": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "4": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "5": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "6": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "7": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "8": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "9": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "10": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "11": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "12": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "13": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "14": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "15": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "16": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "17": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "18": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "19": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "20": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "21": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "22": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "23": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "24": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "25": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "26": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "27": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "28": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "29": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "30": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "31": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "32": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "33": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "0": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "1": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "2": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "3": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "4": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "5": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "6": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "7": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "8": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "9": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "10": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "11": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "12": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "13": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "14": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "15": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "16": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "17": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "18": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "19": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "20": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "21": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "22": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "23": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "24": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "25": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "26": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "27": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "28": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "29": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "30": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "31": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "32": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "33": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "0": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "1": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "2": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "3": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "4": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "5": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "6": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "7": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "8": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "9": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "10": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "11": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "12": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "13": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "14": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "15": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "16": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "17": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "18": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "19": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "20": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "21": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "22": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "23": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "24": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "25": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "26": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "27": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "28": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "29": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "30": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "31": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "32": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "33": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    else:
        Paras = {
            "0": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "1": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "2": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "3": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "4": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "5": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "6": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "7": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "8": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "9": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "10": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "11": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "12": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "13": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "14": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "15": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "16": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "17": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "18": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "19": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "20": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "21": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "22": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "23": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "24": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "25": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "26": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "27": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "28": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "29": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "30": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "31": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "32": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "33": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }

elif dataset_name == "cardio":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    y = np.where(y == 1, 0, 1)

    categ_cols = ["CLASS"]
    num_cols = [col for col in X.columns if col not in categ_cols]
    if sample_frac == 0.05:
        Paras = {
            "LB": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "AC": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "FM": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "UC": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "ASTV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "MSTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ALTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "MLTV": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Width": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Min": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "Max": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Nmax": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "Nzeros": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "Mode": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Median": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Variance": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "Tendency": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CLASS": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "LB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FM": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "UC": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ASTV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "MSTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ALTV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "MLTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Width": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Min": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Max": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Nmax": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Nzeros": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Mode": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Median": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Variance": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Tendency": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "CLASS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "LB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FM": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "UC": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "ASTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "MSTV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ALTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "MLTV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Width": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Min": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Max": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Nmax": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
            "Nzeros": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "Mode": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "Mean": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Median": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Variance": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Tendency": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CLASS": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "LB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FM": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "UC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "ASTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "MSTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ALTV": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "MLTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Width": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Min": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Max": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Nmax": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Nzeros": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Mode": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Median": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Variance": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Tendency": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "CLASS": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
        }
    else:
        Paras = {
            "LB": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "AC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "FM": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "UC": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "DL": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "DS": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "DP": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "ASTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "MSTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "ALTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "MLTV": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Width": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Min": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Max": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Nmax": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Nzeros": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Mode": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "Mean": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Median": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "Variance": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "Tendency": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "CLASS": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
        }
elif dataset_name == "fraud":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = []
    num_cols = X.columns.tolist()
elif dataset_name == "pageblocks":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]
    y = np.where(y == 1, 0, 1)

    categ_cols = []
    num_cols = X.columns.tolist()
    if sample_frac == 0.05:
        Paras = {
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "length": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "eccen": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_black": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "p_and": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "mean_tr": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "blackpix": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "blackand": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "wb_trans": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "height": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "length": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "eccen": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_black": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "p_and": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "mean_tr": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "blackpix": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "blackand": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "wb_trans": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "length": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "eccen": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_black": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "p_and": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "mean_tr": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "blackpix": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "blackand": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "wb_trans": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "length": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "eccen": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_black": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_and": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "mean_tr": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "blackpix": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "blackand": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "wb_trans": {
                "max_depth": 18,
                "min_samples_leaf": 5,
                "min_samples_split": 4,
                "n_estimators": 385,
            },
        }
    else:
        Paras = {
            "height": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "length": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "area": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "eccen": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_black": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "p_and": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "mean_tr": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "blackpix": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "blackand": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "wb_trans": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
elif dataset_name == "bank":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv"), sep=";"
    )
    target = "target"
    X = data.drop(columns=["target"])
    y = data["target"]

    y = np.where(y == "no", 0, 1)

    categ_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    num_cols = [col for col in X.columns if col not in categ_cols]
    Paras = {
        "age": {"max_depth": 13, "min_samples_leaf": 5, "min_samples_split": 6},
        "job": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
        "marital": {"max_depth": 17, "min_samples_leaf": 3, "min_samples_split": 6},
        "education": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
        "default": {"max_depth": 11, "min_samples_leaf": 4, "min_samples_split": 9},
        "balance": {"max_depth": 5, "min_samples_leaf": 4, "min_samples_split": 8},
        "housing": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
        "loan": {"max_depth": 17, "min_samples_leaf": 3, "min_samples_split": 6},
        "contact": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
        "day": {"max_depth": 14, "min_samples_leaf": 4, "min_samples_split": 5},
        "month": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
        "duration": {"max_depth": 13, "min_samples_leaf": 5, "min_samples_split": 6},
        "campaign": {"max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 3},
        "pdays": {"max_depth": 13, "min_samples_leaf": 5, "min_samples_split": 6},
        "previous": {"max_depth": 13, "min_samples_leaf": 5, "min_samples_split": 6},
        "poutcome": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
    }
elif dataset_name == "kdd":
    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv")
    )
    target = "target"
    service_to_category = {
        "ecr_i": "ecr_i",
        "private": "private",
        "http": "http",
        # 1. 专有/私有协议
        "eco_i": "proprietary",
        "urp_i": "proprietary",
        "red_i": "proprietary",
        "tim_i": "proprietary",
        "other": "proprietary",
        # 2. Web 服务
        "http_443": "web",
        "http_8001": "web",
        "http_2784": "web",
        # 3. 邮件服务
        "smtp": "mail",
        "pop_3": "mail",
        "pop_2": "mail",
        "imap4": "mail",
        "courier": "mail",
        # 4. FTP 服务
        "ftp": "ftp",
        "ftp_data": "ftp",
        "tftp_u": "ftp",
        # 5. 远程登录
        "ssh": "remote_login",
        "telnet": "remote_login",
        "login": "remote_login",
        "rlogin": "remote_login",
        "kshell": "remote_login",
        "klogin": "remote_login",
        # 6. DNS
        "domain": "dns",
        "domain_u": "dns",
        "hostnames": "dns",
        "csnet_ns": "dns",
        # 7. 网络管理
        "ntp_u": "net_mgmt",
        "netstat": "net_mgmt",
        "daytime": "net_mgmt",
        "time": "net_mgmt",
        "systat": "net_mgmt",
        "discard": "net_mgmt",
        "echo": "net_mgmt",
        # 8. 认证授权
        "auth": "auth",
        "exec": "auth",
        "printer": "auth",
        # 9. 信息查询
        "finger": "info_query",
        "whois": "info_query",
        "netbios_ns": "info_query",
        "netbios_dgm": "info_query",
        "netbios_ssn": "info_query",
        # 10. RPC / 中间件
        "sunrpc": "rpc",
        "sql_net": "rpc",
        "vmnet": "rpc",
        "ldap": "rpc",
        "iso_tsap": "rpc",
        # 11. 新闻类
        "nntp": "news",
        "nnsp": "news",
        # 12. 老旧服务
        "efs": "legacy",
        "uucp": "legacy",
        "uucp_path": "legacy",
        "rje": "legacy",
        "remote_job": "legacy",
        "shell": "legacy",
        # 13. 图形界面
        "X11": "gui",
        # 14. P2P / 聊天
        "IRC": "p2p_chat",
        "aol": "p2p_chat",
        # 15. 低频其他
        "harvest": "misc_low_freq",
        "Z39_50": "misc_low_freq",
        "gopher": "misc_low_freq",
        "supdup": "misc_low_freq",
        "link": "misc_low_freq",
        "ctf": "misc_low_freq",
        "mtp": "misc_low_freq",
        "bgp": "misc_low_freq",
    }
    X = data.drop(columns=["target"])
    y = data["target"]
    categ_cols = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]
    num_cols = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]
    y = np.where(y == "normal.", 1, 0)
    X["service"] = X["service"].map(service_to_category).fillna("misc")
    if sample_frac == 0.01:
        Paras = {
            "duration": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "protocol_type": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "flag": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "src_bytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_bytes": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "land": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "wrong_fragment": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "urgent": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hot": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_failed_logins": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "logged_in": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_compromised": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "root_shell": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "su_attempted": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_root": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_file_creations": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_shells": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_access_files": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_outbound_cmds": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_host_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_guest_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "count": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "srv_serror_rate": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_rerror_rate": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_same_src_port_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_serror_rate": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "dst_host_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.05:
        Paras = {
            "duration": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "protocol_type": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "flag": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "src_bytes": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "dst_bytes": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "land": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "wrong_fragment": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "urgent": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hot": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_failed_logins": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "logged_in": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_compromised": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "root_shell": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "su_attempted": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_root": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_file_creations": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_shells": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_access_files": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_outbound_cmds": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_host_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_guest_login": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_count": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "dst_host_same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_same_src_port_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_serror_rate": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "dst_host_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    elif sample_frac == 0.1:
        Paras = {
            "duration": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "protocol_type": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "flag": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "src_bytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_bytes": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "land": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "wrong_fragment": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "urgent": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hot": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "num_failed_logins": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "logged_in": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "num_compromised": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "root_shell": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "su_attempted": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_root": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_file_creations": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_shells": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_access_files": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_outbound_cmds": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_host_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_guest_login": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_count": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "dst_host_same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_same_src_port_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.2:
        Paras = {
            "duration": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "protocol_type": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "flag": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "src_bytes": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "dst_bytes": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 230,
            },
            "land": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "wrong_fragment": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "urgent": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "hot": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_failed_logins": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "logged_in": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_compromised": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "root_shell": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "su_attempted": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_root": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_file_creations": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_shells": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_access_files": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_outbound_cmds": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_host_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_guest_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_count": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "dst_host_same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_same_src_port_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
        }
    elif sample_frac == 0.5:
        Paras = {
            "duration": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "protocol_type": {
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "n_estimators": 362,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "flag": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "src_bytes": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_bytes": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "land": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "wrong_fragment": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "urgent": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "hot": {
                "max_depth": 5,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 176,
            },
            "num_failed_logins": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "logged_in": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_compromised": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "root_shell": {
                "max_depth": 16,
                "min_samples_leaf": 5,
                "min_samples_split": 3,
                "n_estimators": 157,
            },
            "su_attempted": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_root": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "num_file_creations": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_shells": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "num_access_files": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_outbound_cmds": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_host_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_guest_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_count": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "dst_host_same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_same_src_port_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
    else:
        Paras = {
            "duration": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "protocol_type": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "service": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "flag": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "src_bytes": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_bytes": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "land": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "wrong_fragment": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "urgent": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "hot": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_failed_logins": {
                "max_depth": 13,
                "min_samples_leaf": 5,
                "min_samples_split": 6,
                "n_estimators": 351,
            },
            "logged_in": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_compromised": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "root_shell": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "su_attempted": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "num_root": {
                "max_depth": 19,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 225,
            },
            "num_file_creations": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_shells": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "num_access_files": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "num_outbound_cmds": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_host_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "is_guest_login": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_count": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_count": {
                "max_depth": 11,
                "min_samples_leaf": 4,
                "min_samples_split": 9,
                "n_estimators": 195,
            },
            "dst_host_same_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_diff_srv_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_same_src_port_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_diff_host_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "n_estimators": 315,
            },
            "dst_host_srv_serror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
            "dst_host_srv_rerror_rate": {
                "max_depth": 17,
                "min_samples_leaf": 2,
                "min_samples_split": 7,
                "n_estimators": 341,
            },
        }
elif dataset_name == "credit":
    data.drop(columns=["Time"], inplace=True)
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = []
    num_cols = X.columns.tolist()

elif dataset_name == "etherum":

    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv")
    )
    target = "target"
    data.drop(columns=["s", "Index", "Address"], inplace=True)
    X = data.drop(columns=["target"])
    y = data["target"]

    categ_cols = [" ERC20 most sent token type", " ERC20_most_rec_token_type"]
    num_cols = [col for col in X.columns if col not in categ_cols]
    Paras = {
        "Avg min between sent tnx": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "Avg min between received tnx": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "Time Diff between first and last (Mins)": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "Sent tnx": {"max_depth": 17, "min_samples_leaf": 3, "min_samples_split": 6},
        "Received Tnx": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "Number of Created Contracts": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "Unique Received From Addresses": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "Unique Sent To Addresses": {
            "max_depth": 14,
            "min_samples_leaf": 4,
            "min_samples_split": 5,
        },
        "min value received": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "max value received ": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "avg val received": {
            "max_depth": 12,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
        },
        "min val sent": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "max val sent": {
            "max_depth": 19,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        "avg val sent": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "min value sent to contract": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "max val sent to contract": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "avg value sent to contract": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "total transactions (including tnx to create contract": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "total Ether sent": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "total ether received": {
            "max_depth": 14,
            "min_samples_leaf": 4,
            "min_samples_split": 5,
        },
        "total ether sent contracts": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "total ether balance": {
            "max_depth": 18,
            "min_samples_leaf": 5,
            "min_samples_split": 4,
        },
        " Total ERC20 tnxs": {
            "max_depth": 5,
            "min_samples_leaf": 4,
            "min_samples_split": 8,
        },
        " ERC20 total Ether received": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 total ether sent": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 total Ether sent contract": {
            "max_depth": 13,
            "min_samples_leaf": 5,
            "min_samples_split": 6,
        },
        " ERC20 uniq sent addr": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        " ERC20 uniq rec addr": {
            "max_depth": 19,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 uniq sent addr.1": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        " ERC20 uniq rec contract addr": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 avg time between sent tnx": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 avg time between rec tnx": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 avg time between rec 2 tnx": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 avg time between contract tnx": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 min val rec": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 max val rec": {
            "max_depth": 18,
            "min_samples_leaf": 5,
            "min_samples_split": 4,
        },
        " ERC20 avg val rec": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 min val sent": {
            "max_depth": 18,
            "min_samples_leaf": 5,
            "min_samples_split": 4,
        },
        " ERC20 max val sent": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 avg val sent": {
            "max_depth": 16,
            "min_samples_leaf": 5,
            "min_samples_split": 3,
        },
        " ERC20 min val sent contract": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 max val sent contract": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 avg val sent contract": {
            "max_depth": 11,
            "min_samples_leaf": 4,
            "min_samples_split": 9,
        },
        " ERC20 uniq sent token name": {
            "max_depth": 18,
            "min_samples_leaf": 5,
            "min_samples_split": 4,
        },
        " ERC20 uniq rec token name": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20 most sent token type": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        " ERC20_most_rec_token_type": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
    }

elif dataset_name == "online_fraud":

    data = pd.read_csv(
        os.path.join(base_dir, f"datasets/{dataset_name}/{dataset_name}.csv")
    )
    target = "target"
    # sample 80000 rows
    data = data.sample(n=80000, random_state=42)
    data.drop(columns=["step", "nameOrig", "nameDest", "isFlaggedFraud"], inplace=True)
    X = data.drop(columns=["target"])
    y = data["target"]
    categ_cols = ["type"]
    num_cols = [col for col in X.columns if col not in categ_cols]
    Paras = {
        "type": {"max_depth": 14, "min_samples_leaf": 4, "min_samples_split": 5},
        "amount": {"max_depth": 17, "min_samples_leaf": 2, "min_samples_split": 7},
        "oldbalanceOrg": {
            "max_depth": 17,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
        },
        "newbalanceOrig": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "oldbalanceDest": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
        "newbalanceDest": {
            "max_depth": 17,
            "min_samples_leaf": 2,
            "min_samples_split": 7,
        },
    }

elif dataset_name == "backblaze":
    bb_train_path = os.path.join(base_dir, "Datasets/Backblaze/backblaze_data/train_set.csv")
    bb_val_path = os.path.join(base_dir, "Datasets/Backblaze/backblaze_data/val_set.csv")
    bb_val_ids_path = os.path.join(base_dir, "Datasets/Backblaze/backblaze_data/val_serial_number_id.csv")
    bb_val_labels_path = os.path.join(base_dir, "Datasets/Backblaze/backblaze_data/val_label.csv")

    def _bb_prepare_training_set(path, horizon_days=60):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values(["serial_number", "date"])
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
            default=0
        ).astype(int)
        feat = df.drop(columns=["date", "serial_number", "failure", "failure_date", "days_to_failure", "label"])
        lbl = df["label"]
        return feat, lbl

    def _bb_last_readout(df):
        df = df.sort_values(["serial_number", "date"])
        return df.drop_duplicates("serial_number", keep="last")

    def _bb_prepare_eval_split(data_path, id_path, label_path):
        df = pd.read_csv(data_path, parse_dates=["date"])
        df_last = _bb_last_readout(df)
        ids = pd.read_csv(id_path)
        labels = pd.read_csv(label_path)
        merged = (df_last
                  .merge(ids, on="serial_number", how="inner")
                  .merge(labels, on="id", how="inner"))
        feat = merged.drop(columns=["date", "serial_number", "id", "label", "failure"])
        lbl = merged["label"].astype(int)
        return feat, lbl

    X_bb_train, y_bb_train = _bb_prepare_training_set(bb_train_path, horizon_days=60)
    X_bb_val, y_bb_val = _bb_prepare_eval_split(bb_val_path, bb_val_ids_path, bb_val_labels_path)

    X = pd.concat([X_bb_train, X_bb_val], ignore_index=True)
    y_raw = pd.concat([y_bb_train, y_bb_val], ignore_index=True)

    # Drop columns that are always NaN across all data
    always_nan_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=always_nan_cols)

    # Binary label: 0 = normal, 1 = at-risk (failing within 20 days)
    y = (y_raw > 0).astype(int)

    categ_cols = ["model"]
    num_cols = [col for col in X.columns if col not in categ_cols]
    target = "label"

elif dataset_name == "backblaze_clean":
    _bb_clean_dir    = os.path.join(base_dir, "Datasets/Backblaze/clean")
    _bb_val_ids_path = os.path.join(base_dir, "Datasets/Backblaze/backblaze_data/val_serial_number_id.csv")
    _bb_val_lbl_path = os.path.join(base_dir, "Datasets/Backblaze/backblaze_data/val_label.csv")

    _bb_files      = sorted(os.listdir(_bb_clean_dir))
    _bb_train_file = next(f for f in _bb_files if f.startswith("train_set"))
    _bb_val_file   = next(f for f in _bb_files if f.startswith("val_set"))

    _bb_train = pd.read_csv(os.path.join(_bb_clean_dir, _bb_train_file), low_memory=False)
    _bb_val   = pd.read_csv(os.path.join(_bb_clean_dir, _bb_val_file),   low_memory=False)

    # 只保留故障前 60 天的数据（与原始 backblaze 处理一致）
    _bb_train = _bb_train[_bb_train["rul_days"] <= 60].reset_index(drop=True)

    # 训练集：去掉 meta 列，提取特征和标签
    _meta_cols  = {"date", "serial_number", "failure", "failure_date", "rul_days", "label"}
    _feat_cols  = [c for c in _bb_train.columns if c not in _meta_cols]
    X_bb_train  = _bb_train[_feat_cols].reset_index(drop=True)
    y_bb_train  = _bb_train["label"].reset_index(drop=True)

    # 验证集：取每个磁盘最后一行的滚动特征，join val_label.csv 获取标签
    _bb_val["date"] = pd.to_datetime(_bb_val["date"], errors="coerce")
    _bb_val_last = (
        _bb_val.sort_values(["serial_number", "date"])
               .drop_duplicates("serial_number", keep="last")
    )
    _bb_ids    = pd.read_csv(_bb_val_ids_path)
    _bb_labels = pd.read_csv(_bb_val_lbl_path)
    _bb_val_merged = (
        _bb_val_last
        .merge(_bb_ids,    on="serial_number", how="inner")
        .merge(_bb_labels, on="id",            how="inner")
    )
    _val_feat_cols = [c for c in _feat_cols if c in _bb_val_merged.columns]
    X_bb_val = _bb_val_merged[_val_feat_cols].reset_index(drop=True)
    y_bb_val = _bb_val_merged["label"].astype(int).reset_index(drop=True)

    X     = pd.concat([X_bb_train[_val_feat_cols], X_bb_val], ignore_index=True)
    y_raw = pd.concat([y_bb_train, y_bb_val], ignore_index=True)

    always_nan_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=always_nan_cols)

    y = (y_raw > 0).astype(int)

    categ_cols = [c for c in ["model"] if c in X.columns]
    num_cols   = [c for c in X.columns if c not in categ_cols]
    target     = "label"

    # For feature agg
    _agg_feat_cols       = list(X.columns)
    _orig_id_cols        = ["serial_number", "date", "rul_days"]
    _orig_train_csv_path = os.path.join(_bb_clean_dir, _bb_train_file)

    _bb_val_full = pd.read_csv(os.path.join(_bb_clean_dir, _bb_val_file), low_memory=False)
    _bb_val_full["date"] = pd.to_datetime(_bb_val_full["date"], errors="coerce")
    _bb_val_label_map = (
        _bb_val_full.sort_values(["serial_number", "date"])
        .drop_duplicates("serial_number", keep="last")
        .merge(_bb_ids,    on="serial_number", how="inner")
        .merge(_bb_labels, on="id",            how="inner")
        [["serial_number", "label"]].rename(columns={"label": "_lbl"})
    )
    _bb_val_full = _bb_val_full.merge(_bb_val_label_map, on="serial_number", how="left")
    _bb_val_full["label"] = _bb_val_full["_lbl"].fillna(-1).astype(int)
    _orig_val_df = _bb_val_full.drop(columns=["_lbl"], errors="ignore").copy().reset_index(drop=True)

    _orig_train_agg_path = os.path.join(_bb_clean_dir, "backblaze_clean_train_feature_agg.csv")
    _orig_val_agg_path   = os.path.join(_bb_clean_dir, "backblaze_clean_val_feature_agg.csv")

elif dataset_name == "scania":
    sc_train_ops_path   = os.path.join(base_dir, "Datasets/SCANIA/SCANIA/train_operational_readouts.csv")
    sc_train_spec_path  = os.path.join(base_dir, "Datasets/SCANIA/SCANIA/train_specifications.csv")
    sc_train_tte_path   = os.path.join(base_dir, "Datasets/SCANIA/SCANIA/train_tte.csv")
    sc_val_ops_path     = os.path.join(base_dir, "Datasets/SCANIA/SCANIA/validation_operational_readouts.csv")
    sc_val_spec_path    = os.path.join(base_dir, "Datasets/SCANIA/SCANIA/validation_specifications.csv")
    sc_val_labels_path  = os.path.join(base_dir, "Datasets/SCANIA/SCANIA/validation_labels.csv")

    def _sc_last_readout(df_ops):
        df_ops = df_ops.sort_values(["vehicle_id", "time_step"])
        return df_ops.drop_duplicates("vehicle_id", keep="last")

    def _sc_rul_to_ordinal(rul):
        rul = np.asarray(rul, dtype=float)
        y = np.zeros_like(rul, dtype=int)
        y[rul <= 48] = 1
        y[rul <= 24] = 2
        y[rul <= 12] = 3
        y[rul <=  6] = 4
        return y

    # --- Training set ---
    ops_tr  = pd.read_csv(sc_train_ops_path)
    spec_tr = pd.read_csv(sc_train_spec_path)
    tte_tr  = pd.read_csv(sc_train_tte_path)

    df_tr = (
        _sc_last_readout(ops_tr)
        .merge(spec_tr, on="vehicle_id", how="inner")
        .merge(tte_tr,  on="vehicle_id", how="inner")
    )

    t_cur = df_tr["time_step"].to_numpy()
    T     = df_tr["length_of_study_time_step"].to_numpy()
    rep   = df_tr["in_study_repair"].to_numpy()
    rul   = np.where(rep == 1, np.maximum(T - t_cur, 0.0), np.inf)
    df_tr["y"] = np.where(np.isfinite(rul), _sc_rul_to_ordinal(rul), 0)

    drop_cols_sc = ["vehicle_id", "time_step", "in_study_repair", "length_of_study_time_step"]
    X_sc_train = df_tr.drop(columns=drop_cols_sc + ["y"], errors="ignore")
    y_sc_train = df_tr["y"].astype(int)

    # --- Validation set ---
    ops_va  = pd.read_csv(sc_val_ops_path)
    spec_va = pd.read_csv(sc_val_spec_path)
    lab_va  = pd.read_csv(sc_val_labels_path)

    y_col = "class_label" if "class_label" in lab_va.columns else lab_va.columns[-1]
    df_va = (
        _sc_last_readout(ops_va)
        .merge(spec_va, on="vehicle_id", how="inner")
        .merge(lab_va[["vehicle_id", y_col]], on="vehicle_id", how="inner")
        .rename(columns={y_col: "y"})
    )

    X_sc_val = df_va.drop(columns=drop_cols_sc + ["y"], errors="ignore")
    y_sc_val = df_va["y"].astype(int)

    X = pd.concat([X_sc_train, X_sc_val], ignore_index=True)
    y_raw = pd.concat([y_sc_train, y_sc_val], ignore_index=True)

    # Binary label: 0 = healthy, 1 = at-risk (y > 0)
    y = (y_raw > 0).astype(int)

    categ_cols = ["Spec_0", "Spec_1", "Spec_2", "Spec_3", "Spec_4", "Spec_5", "Spec_6", "Spec_7"]
    num_cols = [col for col in X.columns if col not in categ_cols]
    target = "y"

elif dataset_name == "scania_clean":
    _sc_clean_dir  = os.path.join(base_dir, "Datasets/SCANIA/SCANIA")
    _sc_window     = 10
    _sc_train = pd.read_csv(os.path.join(_sc_clean_dir, f"train_features_w{_sc_window}.csv"), low_memory=False)
    _sc_val   = pd.read_csv(os.path.join(_sc_clean_dir, f"validation_features_w{_sc_window}.csv"), low_memory=False)

    # 只保留维修过的车辆（in_study_repair == 1）
    _sc_tte = pd.read_csv(os.path.join(_sc_clean_dir, "train_tte.csv"))
    _repaired_ids = set(_sc_tte.loc[_sc_tte["in_study_repair"] == 1, "vehicle_id"])
    _sc_train = _sc_train[_sc_train["vehicle_id"].isin(_repaired_ids)].reset_index(drop=True)

    _meta_cols = {"vehicle_id", "time_step", "label"}
    _feat_cols = [c for c in _sc_train.columns if c not in _meta_cols]

    X_sc_train = _sc_train[_feat_cols].reset_index(drop=True)
    y_sc_train = _sc_train["label"].reset_index(drop=True)

    _val_feat_cols = [c for c in _feat_cols if c in _sc_val.columns]
    X_sc_val = _sc_val[_val_feat_cols].reset_index(drop=True)
    y_sc_val = _sc_val["label"].astype(int).reset_index(drop=True)

    X     = pd.concat([X_sc_train[_val_feat_cols], X_sc_val], ignore_index=True)
    y_raw = pd.concat([y_sc_train, y_sc_val], ignore_index=True)

    always_nan_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=always_nan_cols)

    y = (y_raw > 0).astype(int)

    categ_cols = [c for c in X.columns if c.startswith("Spec_")]
    num_cols   = [c for c in X.columns if c not in categ_cols]
    target     = "label"

    # For feature agg
    _agg_feat_cols       = list(X.columns)
    _orig_id_cols        = ["vehicle_id", "time_step"]
    _orig_train_csv_path = os.path.join(_sc_clean_dir, f"train_features_w{_sc_window}.csv")
    _orig_val_df         = _sc_val.copy().reset_index(drop=True)
    _orig_train_agg_path = os.path.join(_sc_clean_dir, f"train_features_w{_sc_window}_agg.csv")
    _orig_val_agg_path   = os.path.join(_sc_clean_dir, f"validation_features_w{_sc_window}_agg.csv")

X = X.replace("?", np.nan)
# y = data[target]
print(X.columns)
print("X shape: ", X.shape)
_label_encoders = {}
_num_medians = {}
for col in X.columns:
    if col in categ_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        _label_encoders[col] = le
    else:
        _num_medians[col] = X[col].median()
        X[col] = X[col].fillna(_num_medians[col])

print("Number of categorical columns: ", len(categ_cols))
print("Number of numerical columns: ", len(num_cols))
print("Number of rows: ", X.shape[0])
print("Number of columns: ", X.shape[1])
print("Contamination percentage: ", y.sum() / y.shape[0])

y = pd.Series(y)

train_randomly = False
if train_randomly == True:
    X_train = X.sample(frac=0.6, random_state=42)
    X_test = X.drop(X_train.index)
    X_valid = X_train.sample(frac=0.1 / 0.6, random_state=42)

else:
    X_train = X[y == 0].sample(frac=0.6, random_state=42)
    X_test_0 = X[y == 0].drop(X_train.index)
    X_valid = X_train.sample(frac=0.1 / 0.6, random_state=42)

    X_test_1 = X[y == 1]
    X_test = pd.concat([X_test_0, X_test_1])

y_train = y.loc[X_train.index]
y_test = y.loc[X_test.index]
y_valid = y.loc[X_valid.index]

X_train = X_train.sample(frac=sample_frac, random_state=42)
X_valid = X_valid.sample(frac=sample_frac, random_state=42)

test_ids = X_test.index.to_numpy()

X_train.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(y_test)

print_to_log(f"X_train shape: {X_train.shape}")
print_to_log(f"X_test shape: {X_test.shape}")
print_to_log(f"y_train shape: {y_train.shape}")
print_to_log(f"y_test shape: {y_test.shape}")
print_to_log(f"y_test_positive_rate: {y_test.sum() / y_test.shape[0]}")

#### preprocess (/one hot encoding)
X_processed_train = pd.DataFrame()
X_processed_valid = pd.DataFrame()
X_processed_test = pd.DataFrame()

process_list = {}
_onehot_encoders = {}
for col_name in categ_cols:
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    onehot_col_train = encoder.fit_transform(X_train[[col_name]])
    _onehot_encoders[col_name] = encoder
    onehot_col_valid = encoder.transform(X_valid[[col_name]])
    onehot_col_test = encoder.transform(X_test[[col_name]])
    process_list[col_name] = [f"{col_name}_{cat}" for cat in encoder.categories_[0][1:]]
    onehot_col_train = pd.DataFrame(onehot_col_train, columns=process_list[col_name])
    onehot_col_valid = pd.DataFrame(onehot_col_valid, columns=process_list[col_name])
    onehot_col_test = pd.DataFrame(onehot_col_test, columns=process_list[col_name])
    X_processed_train = pd.concat([X_processed_train, onehot_col_train], axis=1)
    X_processed_valid = pd.concat([X_processed_valid, onehot_col_valid], axis=1)
    X_processed_test = pd.concat([X_processed_test, onehot_col_test], axis=1)

for col_name in num_cols:
    process_list[col_name] = [col_name]

scaler = StandardScaler()
X_numerical_scaled_train = pd.DataFrame(
    scaler.fit_transform(X_train[num_cols]), columns=num_cols
)
X_numerical_scaled_valid = pd.DataFrame(
    scaler.transform(X_valid[num_cols]), columns=num_cols
)
X_numerical_scaled_test = pd.DataFrame(
    scaler.transform(X_test[num_cols]), columns=num_cols
)
X_processed_train = pd.concat([X_processed_train, X_numerical_scaled_train], axis=1)
X_processed_valid = pd.concat([X_processed_valid, X_numerical_scaled_valid], axis=1)
X_processed_test = pd.concat([X_processed_test, X_numerical_scaled_test], axis=1)
####


def train_rf(col_name, num_cols, categ_cols):
    print(f"Target: column {col_name}")
    rf_X_train = X_processed_train.drop(columns=process_list[col_name])
    rf_X_valid = X_processed_valid.drop(columns=process_list[col_name])
    rf_X_test = X_processed_test.drop(columns=process_list[col_name])
    rf_y_train = X_train[col_name]
    rf_y_valid = X_valid[col_name]
    rf_y_test = X_test[col_name]

    if col_name in num_cols:
        # print("Training regression model for:", col_name)
        try:
            Paras
        except NameError:
            rf_model = RandomForestRegressor(n_estimators=400, n_jobs=-1)
        else:
            rf_model = RandomForestRegressor(
                n_estimators=400,
                max_depth=Paras[col_name]["max_depth"],
                min_samples_split=Paras[col_name]["min_samples_split"],
                min_samples_leaf=Paras[col_name]["min_samples_leaf"],
                n_jobs=-1,
            )
        TranStr = time.time()
        rf_model.fit(rf_X_train.values, rf_y_train)
        TranEnd = time.time()

    elif col_name in categ_cols:
        # print("Training classification model for:", col_name)
        try:
            Paras
        except NameError:
            rf_model = RandomForestClassifier(n_jobs=-1, n_estimators=400)
        else:
            rf_model = RandomForestClassifier(
                n_estimators=400,
                max_depth=Paras[col_name]["max_depth"],
                min_samples_split=Paras[col_name]["min_samples_split"],
                min_samples_leaf=Paras[col_name]["min_samples_leaf"],
                n_jobs=-1,
            )
        TranStr = time.time()
        rf_model.fit(rf_X_train.values, rf_y_train)
        TranEnd = time.time()

    # The list of trees in the forest
    # print("Train", col_name, TranEnd - TranStr, sep=";")
    trees = rf_model.estimators_

    if np.min(rf_y_train) < 0:
        rf_y_train = rf_y_train - np.min(rf_y_train)

    PrStr = time.time()

    output_trees = Parallel(n_jobs=-1)(
        delayed(tree_scoring)(tree, rf_X_valid.values, rf_y_valid.values)
        for tree in trees
    )
    errors, preds = zip(*output_trees)
    tree_oob_errors = np.asarray(errors, float)

    y_preds = np.stack(preds, axis=0)

    tree_indices = np.argsort(tree_oob_errors)
    sorted_errors = np.array(tree_oob_errors)[tree_indices]

    trees_oob_errors = {}
    lef = 0
    rig = 399
    while lef + 3 <= rig:
        midl = rig - (rig - lef) * 0.61803398875
        midr = lef + (rig - lef) * 0.61803398875
        midl = int(midl)
        midr = int(midr)
        # print(lef, midl, midr, rig)
        if trees_oob_errors.get(midl) is None:
            trees_oob_errors[midl] = trees_scoring(
                y_preds[tree_indices[: midl + 1]], rf_y_valid.values
            ) + sorted_errors[0] * 0.3 / np.sqrt((midl + 1) / 10)
        if trees_oob_errors.get(midr) is None:
            trees_oob_errors[midr] = trees_scoring(
                y_preds[tree_indices[: midr + 1]], rf_y_valid.values
            ) + sorted_errors[0] * 0.3 / np.sqrt((midr + 1) / 10)
        if trees_oob_errors[midl] < trees_oob_errors[midr]:
            rig = midr
        else:
            lef = midl
    Best_trees = 5
    if trees_oob_errors.get(Best_trees) is None:
        trees_oob_errors[Best_trees] = trees_scoring(
            y_preds[tree_indices[: Best_trees + 1]], rf_y_valid.values
        ) + sorted_errors[0] * 0.3 / np.sqrt((Best_trees + 1) / 10)
    for i in range(lef, rig + 1):
        if trees_oob_errors.get(i) is None:
            trees_oob_errors[i] = trees_scoring(
                y_preds[tree_indices[: i + 1]], rf_y_valid.values
            ) + sorted_errors[0] * 0.3 / np.sqrt((i + 1) / 10)
        if trees_oob_errors[i] < trees_oob_errors[Best_trees]:
            Best_trees = i

    tree_used = max(Best_trees, 5)
    feature_perf = np.mean(sorted_errors[: max(1, Best_trees - 1)])
    PrEnd = time.time()

    return {
        "tree_indices": tree_indices,
        "rf_model": rf_model,
        "tr_time": TranEnd - TranStr,
        "pr_time": PrEnd - PrStr,
        "tree_used": tree_used,
        "feature_performance": feature_perf,
    }


def predict(col_name, tree_used, true_value, tree_indices, rf_model, num_cols):
    top_tree_indices = tree_indices[:tree_used]
    trees = rf_model.estimators_
    predict_X = X_processed_test.drop(columns=process_list[col_name]).to_numpy()
    if col_name in num_cols:
        # print("Predicting for num:", col_name)
        # Regression: average the predictions
        top_tree_preds = []
        PredSt = time.time()
        for idx in top_tree_indices:
            top_tree_preds.append(trees[idx].predict(predict_X))
        PredEnd = time.time()
        top_tree_preds = np.array(top_tree_preds)
        # print(top_tree_preds.shape) # 400, n_samples
        final_pred = top_tree_preds.mean(axis=0)
        # calculate confidence scores
        confidence_scores = np.std(top_tree_preds, axis=0, ddof=1)

    else:
        # Classification: average the probability predictions and then choose the class with highest probability

        class_idx_map = {label: i for i, label in enumerate(rf_model.classes_)}
        map_func = np.vectorize(lambda x: class_idx_map.get(x, -1))
        true_class_indices = map_func(true_value)
        valid = true_class_indices != -1
        # print(valid.sum())
        # print(valid.shape)
        # print(true_class_indices)
        PredSt = time.time()
        true_proba = np.zeros((predict_X.shape[0], len(top_tree_indices)))
        for i, idx in enumerate(top_tree_indices):
            tree_proba = trees[idx].predict_proba(predict_X)
            tree_true = np.zeros((predict_X.shape[0],))
            tree_true[valid] = tree_proba[
                np.arange(predict_X.shape[0])[valid], true_class_indices[valid]
            ]
            true_proba[:, i] = tree_true
        final_pred = true_proba.mean(axis=1)
        # print(final_pred)
        PredEnd = time.time()

        confidence_scores = np.std(true_proba, axis=1, ddof=1)
    return {
        "confidence_scores": confidence_scores,
        "final_pred": final_pred,
        "time": PredEnd - PredSt,
    }


start_time = time.time()

Tid = {}
label_encoder = {}
rFM = {}
Feature_tree_used = {}
Feature_Perf = {}
errors_output = {}
tot_prun = 0
tot_train = 0
for col_name in X_train.columns:
    print(col_name)
    if col_name in fitonly:
        continue
    result_dict = train_rf(col_name, num_cols, categ_cols)
    Tid[col_name] = result_dict["tree_indices"]
    rFM[col_name] = result_dict["rf_model"]
    print_to_log("Train", col_name, result_dict["tr_time"], sep=";")
    print_to_log("Pruning", col_name, result_dict["pr_time"], sep=";")
    tot_train += result_dict["tr_time"]
    tot_prun += result_dict["pr_time"]
    Feature_tree_used[col_name] = result_dict["tree_used"]
    Feature_Perf[col_name] = result_dict["feature_performance"]
    # errors_output[f'{col_name}_single'] = result_dict["sorted_errors"]
    # errors_output[f'{col_name}_combined'] = result_dict["trees_errors"]

print_to_log("Total", "Train", tot_train, sep=";")
print_to_log("Total", "Pruning", tot_prun, sep=";")

end_time = time.time()
print(f"Total time for training RF: {end_time - start_time} seconds")

print("################################")
# print(f"Total time: {end_time - start_time} seconds")

all_y_pred_rf = pd.DataFrame()
conf_combined = pd.DataFrame()
model_scores = []
start_time = time.time()
Conf_Pre = {}
residual = {}

for col_name in num_cols:
    residual[col_name] = calc_resi(
        col_name,
        X_processed_valid,
        Tid[col_name][: Feature_tree_used[col_name]],
        rFM[col_name],
    )
    residual[col_name] = np.sort(residual[col_name])

for col_name in X_train.columns:
    if col_name in fitonly:
        continue
    rf_model = rFM[col_name]
    Conf_Pre[col_name] = predict(
        col_name,
        Feature_tree_used[col_name],
        X_test[col_name].values,
        Tid[col_name],
        rf_model,
        num_cols,
    )

for col_name in X_train.columns:
    if col_name in fitonly:
        continue
    conf_combined[col_name] = Conf_Pre[col_name]["confidence_scores"]
    # print(col_name, np.isnan(conf_combined[col_name]).sum())
    all_y_pred_rf[col_name] = Conf_Pre[col_name]["final_pred"]
    print_to_log(
        "Predict",
        col_name,
        Conf_Pre[col_name]["time"],
        Feature_tree_used[col_name],
        sep=";",
    )

predcol = X_train.columns.difference(fitonly)
conf_combined = conf_combined[predcol]

# end_time = time.time()
# print(f"Total time for prediction: {end_time - start_time} seconds")


print("################################")
print("Conf Combined:", conf_combined.shape)
print(conf_combined)

# feature_weights = np.zeros(X_train.shape[1])
# for i, col_name in enumerate(X_train.columns):
#   feature_weights[i] = Feature_Perf[col_name]
# feature_weights = feature_weights * np.array(conf_combined)
# feature_weights = feature_weights / np.sum(feature_weights)
feature_weights = conf_combined.to_numpy()
for i in range(feature_weights.shape[0]):
    feature_weights[i, :] = (
        feature_weights[i, :] / np.sum(feature_weights[i, :])
        if np.sum(feature_weights[i, :]) != 0
        else feature_weights[i, :]
    )
feature_weights = 1 - feature_weights
for i in range(feature_weights.shape[0]):
    feature_weights[i, :] = (
        feature_weights[i, :] / np.sum(feature_weights[i, :])
        if np.sum(feature_weights[i, :]) != 0
        else feature_weights[i, :]
    )
feature_weights = np.square(feature_weights)
feature_weights = feature_weights / np.sum(feature_weights, axis=1, keepdims=True)
print("Feature weights:", feature_weights)
print("Min:", np.min(feature_weights), "Max:", np.max(feature_weights))

weighted = True if "weighted" in args.msg else False
X_test = X_test.drop(columns=fitonly)
categ_cols_idx = [i for i, x in enumerate(X_test.columns.isin(categ_cols)) if x]
numerical_cols_idx = [i for i, x in enumerate(X_test.columns.isin(num_cols)) if x]
result = alphaquantile(
    all_y_pred_rf.values,  # predict values of numerical; true proba of categorical
    X_test.values,  # true dirty data
    categ_cols_idx,  # categorical columns index
    numerical_cols_idx,  # numerical columns index
    X_test.columns,  # column names
    feature_weights,
    residual,
    weighted=weighted,
)
end_time = time.time()
print(f"Test time: {end_time - start_time} seconds")

# Initialize variables to store the best metrics
best_f1 = -1
best_accuracy = -1
best_log_loss = float("inf")  # Log Loss 越低越好
best_aucroc = -1
best_avpr = -1
best_recall = -1
best_precision = -1
best_ndcg = -1
best_top_5 = -1

# Initialize variables to store the corresponding alpha and threshold for each best metric
best_f1_alpha = None
best_f1_threshold = None

best_accuracy_alpha = None
best_accuracy_threshold = None

best_recall_threshold = None
best_precision_threshold = None

best_log_loss_alpha = None
best_aucroc_alpha = None
best_avpr_alpha = None
best_ndcg_alpha = None
best_top_5_alpha = None

for a in result:
    improved_gower_df, improved_gower_df_row = result[a]

    CellLevel = pd.DataFrame(improved_gower_df, columns=X_test.columns)
    CellLevel["row_score"] = improved_gower_df_row
    CellLevel["target"] = y_test.values
    CellLevel.to_csv(
        os.path.join(scores_folder, f"{dataset_name}_{a}_{train_randomly}_random.csv"),
        index=False,
    )

    thresholds = np.arange(0.001, 1.001, 0.001)
    f1_alpha = -1
    acc_alpha = -1
    f1_thre = 0
    acc_thre = 0
    #  = np.argsort(improved_gower_df_row)
    # improved_gower_df_row = improved_gower_df_row * improved_gower_df_row
    for threshold in thresholds:
        outlier_score_row_binary = (improved_gower_df_row > threshold).astype(int)

        # Calculate F1-score, Accuracy
        f1_rf = f1_score(y_test, outlier_score_row_binary)
        accuracy_rf = accuracy_score(y_test, outlier_score_row_binary)

        tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(
            y_test, outlier_score_row_binary
        ).ravel()
        recall_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0
        precision_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0
        # Update the best F1-score
        if f1_rf > f1_alpha:
            f1_alpha = f1_rf
            f1_thre = threshold

        # Update the best Accuracy
        if accuracy_rf > acc_alpha:
            acc_alpha = accuracy_rf
            acc_thre = threshold

        if recall_rf > best_recall:
            best_recall = recall_rf
            best_recall_threshold = threshold
            best_recall_alpha = a

        if precision_rf > best_precision:
            best_precision = precision_rf
            best_precision_threshold = threshold
            best_precision_alpha = a
    log_loss_rf = log_loss(y_test, improved_gower_df_row)
    aucroc_rf = roc_auc_score(y_test, improved_gower_df_row)
    avpr_rf = average_precision_score(y_test, improved_gower_df_row)
    top_5_rf = top_k_precision(y_test, improved_gower_df_row, k=5)

    if best_f1 < f1_alpha:
        best_f1 = f1_alpha
        best_f1_alpha = a
        best_f1_threshold = f1_thre
    if best_accuracy < acc_alpha:
        best_accuracy = acc_alpha
        best_accuracy_alpha = a
        best_accuracy_threshold = acc_thre

    if log_loss_rf < best_log_loss:
        best_log_loss = log_loss_rf
        best_log_loss_alpha = a

    if aucroc_rf > best_aucroc:
        best_aucroc = aucroc_rf
        best_aucroc_alpha = a

    if avpr_rf > best_avpr:
        best_avpr = avpr_rf
        best_avpr_alpha = a

    if top_5_rf > best_top_5:
        best_top_5 = top_5_rf
        best_top_5_alpha = a

    scores = {
        "best_aucroc": best_aucroc,
        "best_avpr": best_avpr,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
        "best_accuracy": best_accuracy,
        "best_accuracy_threshold": best_accuracy_threshold,
        "best_log_loss": best_log_loss,
    }
    print_to_log(
        f"Alpha;{a};AUCROC;{aucroc_rf};AVPR;{avpr_rf};Log Loss;{log_loss_rf};Top 5 Precision;{top_5_rf};F1;{f1_alpha};Accuracy;{acc_alpha}"
    )

print_to_log("Train random: ", train_randomly)
print_to_log(f'Best AUCROC: {scores["best_aucroc"]} (Alpha: {best_aucroc_alpha})')
print_to_log(f'Best AVPR: {scores["best_avpr"]} (Alpha: {best_avpr_alpha})')
print_to_log(f'Best log loss: {scores["best_log_loss"]} (Alpha: {best_log_loss_alpha})')
print_to_log(f"Best top 5 precision: {best_top_5} (Alpha: {best_top_5_alpha})")

print_to_log(
    f'Best F1 score: {scores["best_f1"]} (Threshold: {scores["best_f1_threshold"]} Alpha: {best_f1_alpha})'
)
print_to_log(
    f'Best accuracy: {scores["best_accuracy"]} (Threshold: {scores["best_accuracy_threshold"]} Alpha: {best_accuracy_alpha})'
)
print_to_log(
    f"Best recall: {best_recall} (Threshold: {best_recall_threshold} Alpha: {best_recall_alpha})"
)
print_to_log(
    f"Best precision: {best_precision} (Threshold: {best_precision_threshold} Alpha: {best_precision_alpha})"
)


# 3. 输出异常 id
try:
    scores = improved_gower_df_row
except NameError:
    raise RuntimeError("Anomaly scores not found. Make sure RFOD score computation is executed.")

thresholds = {
    "q95": np.quantile(scores, 0.95),
    "q97": np.quantile(scores, 0.97),
    "q99": np.quantile(scores, 0.99),
}

anomaly_results = {}

for name, t in thresholds.items():
    mask = scores > t
    anomaly_results[name] = {
        "threshold": float(t),
        "num_anomalies": int(mask.sum()),
        "anomaly_ids": test_ids[mask].tolist(),
    }

print(anomaly_results)

# Save results
os.makedirs(scores_folder, exist_ok=True)
out_path = os.path.join(scores_folder, "rfod.json")
with open(out_path, "w") as f:
    json.dump(anomaly_results, f, indent=2)

print(f"[RFOD-Unsupervised] Anomaly IDs saved to: {out_path}")
sys.stdout.flush()
sys.stderr = sys.stdout  # redirect stderr to log so exceptions are visible


# ──────────────────────────────────────────────────────────────────────────────
# Save feature-augmented train/val CSVs with multi-threshold anomaly scores
# anomaly_score_k: labels 0..k treated as normal, labels > k treated as anomaly
# ──────────────────────────────────────────────────────────────────────────────
if dataset_name in ("backblaze_clean", "scania_clean"):

    def _safe_le_transform(le, values):
        """Vectorized LabelEncode; map unseen categories to -1."""
        arr = np.asarray(values)
        known_mask = np.array([v in set(le.classes_) for v in arr])
        result = np.full(len(arr), -1, dtype=np.int64)
        if known_mask.any():
            result[known_mask] = le.transform(arr[known_mask])
        return result

    def _preprocess_for_scoring(df_orig):
        """Apply same preprocessing pipeline (LabelEncode + StandardScale) as training."""
        df = df_orig[_agg_feat_cols].copy().replace("?", np.nan)
        for col in _agg_feat_cols:
            if col in categ_cols:
                fill = df[col].mode()
                df[col] = df[col].fillna(fill[0] if len(fill) > 0 else 0)
                df[col] = _safe_le_transform(_label_encoders[col], df[col].values)
            else:
                df[col] = df[col].fillna(_num_medians.get(col, 0))
        df.reset_index(drop=True, inplace=True)

        X_proc = pd.DataFrame()
        for col_name in categ_cols:
            ohe_vals = _onehot_encoders[col_name].transform(df[[col_name]])
            X_proc = pd.concat([X_proc, pd.DataFrame(ohe_vals, columns=process_list[col_name])], axis=1)
        _nc = [c for c in num_cols if c in _agg_feat_cols]
        X_num_sc = pd.DataFrame(scaler.transform(df[_nc]), columns=_nc)
        X_proc = pd.concat([X_proc, X_num_sc], axis=1)
        X_proc.reset_index(drop=True, inplace=True)
        return df, X_proc

    def _compute_anomaly_scores(X_raw, X_proc, alpha_val):
        """Score rows in X_raw using the original (k=0) trained RFOD models."""
        global X_processed_test
        _saved_proc = X_processed_test
        X_processed_test = X_proc

        _Conf = {}
        for _col in X_train.columns:
            if _col in fitonly or _col not in X_raw.columns:
                continue
            _Conf[_col] = predict(
                _col, Feature_tree_used[_col],
                X_raw[_col].values, Tid[_col], rFM[_col], num_cols,
            )

        X_processed_test = _saved_proc

        _pcols = [c for c in X_train.columns if c not in fitonly and c in X_raw.columns]
        _conf  = pd.DataFrame({c: _Conf[c]["confidence_scores"] for c in _pcols})[_pcols]
        _pred  = pd.DataFrame({c: _Conf[c]["final_pred"]        for c in _pcols})[_pcols]

        _fw = _conf.to_numpy().copy()
        for i in range(_fw.shape[0]):
            s = _fw[i].sum()
            if s != 0:
                _fw[i] /= s
        _fw = 1 - _fw
        for i in range(_fw.shape[0]):
            s = _fw[i].sum()
            if s != 0:
                _fw[i] /= s
        _fw = np.square(_fw)
        _fw = _fw / np.sum(_fw, axis=1, keepdims=True)

        _cat_idx = [i for i, c in enumerate(X_raw.columns) if c in categ_cols]
        _num_idx = [i for i, c in enumerate(X_raw.columns) if c in num_cols]
        _res = alphaquantile(
            _pred.values, X_raw.values,
            _cat_idx, _num_idx, X_raw.columns,
            _fw, residual, weighted=weighted,
        )
        _, _scores = _res[alpha_val]
        return _scores

    def _retrain_for_threshold(k):
        """Retrain RFOD treating y_raw <= k as normal, y_raw > k as anomaly.
        Temporarily overrides training globals, then restores them.
        Returns (Ftree_k, rFM_k, Tid_k, resid_k, scaler_k, ohe_k, plist_k).
        """
        global X_processed_train, X_processed_valid, X_processed_test
        global X_train, X_valid, X_test, process_list

        _y_k = pd.Series((y_raw.values > k).astype(int), index=X.index)

        # Split: normal-only for train/valid, mix for test
        _Xtr = X[_y_k == 0].sample(frac=0.6, random_state=42)
        _Xvl = _Xtr.sample(frac=0.1 / 0.6, random_state=42)
        _Xtr_rest = X[_y_k == 0].drop(_Xtr.index)
        _Xte = pd.concat([_Xtr_rest, X[_y_k == 1]])

        # Apply sample_frac consistent with original training
        _Xtr = _Xtr.sample(frac=sample_frac, random_state=42)
        _Xvl = _Xvl.sample(frac=sample_frac, random_state=42)

        _Xtr = _Xtr.reset_index(drop=True)
        _Xvl = _Xvl.reset_index(drop=True)
        _Xte = _Xte.reset_index(drop=True)

        # OHE for categorical cols (refit on new X_train_k)
        _plist_k = {}
        _ohe_k = {}
        _Xp_tr = pd.DataFrame()
        _Xp_vl = pd.DataFrame()
        _Xp_te = pd.DataFrame()
        for _c in categ_cols:
            _enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            _tr_ohe = _enc.fit_transform(_Xtr[[_c]])
            _ohe_k[_c] = _enc
            _plist_k[_c] = [f"{_c}_{cat}" for cat in _enc.categories_[0][1:]]
            _Xp_tr = pd.concat([_Xp_tr, pd.DataFrame(_tr_ohe,                       columns=_plist_k[_c])], axis=1)
            _Xp_vl = pd.concat([_Xp_vl, pd.DataFrame(_enc.transform(_Xvl[[_c]]),   columns=_plist_k[_c])], axis=1)
            _Xp_te = pd.concat([_Xp_te, pd.DataFrame(_enc.transform(_Xte[[_c]]),   columns=_plist_k[_c])], axis=1)

        # process_list entries for numerical cols (mirrors line 18043-18044)
        for _c in num_cols:
            _plist_k[_c] = [_c]

        # Scale numerical cols (refit StandardScaler on new X_train_k)
        _sc_k = StandardScaler()
        _nc_k = [c for c in num_cols if c in _Xtr.columns]
        _Xp_tr = pd.concat([_Xp_tr, pd.DataFrame(_sc_k.fit_transform(_Xtr[_nc_k]),  columns=_nc_k)], axis=1)
        _Xp_vl = pd.concat([_Xp_vl, pd.DataFrame(_sc_k.transform(_Xvl[_nc_k]),      columns=_nc_k)], axis=1)
        _Xp_te = pd.concat([_Xp_te, pd.DataFrame(_sc_k.transform(_Xte[_nc_k]),      columns=_nc_k)], axis=1)
        _Xp_tr.reset_index(drop=True, inplace=True)
        _Xp_vl.reset_index(drop=True, inplace=True)
        _Xp_te.reset_index(drop=True, inplace=True)

        # Save current globals, override with threshold-k data
        _sv = (X_processed_train, X_processed_valid, X_processed_test,
               X_train, X_valid, X_test, process_list)
        X_processed_train = _Xp_tr
        X_processed_valid = _Xp_vl
        X_processed_test  = _Xp_te
        X_train = _Xtr
        X_valid = _Xvl
        X_test  = _Xte
        process_list = _plist_k

        # Retrain RFOD per feature
        # NOTE: use 'col_name' as loop variable (not '_col') because tree_scoring /
        # trees_scoring reference the global 'col_name' to decide regression vs classification.
        global col_name
        _Ftree_k, _rFM_k, _Tid_k = {}, {}, {}
        for col_name in X_train.columns:
            if col_name in fitonly:
                continue
            _r = train_rf(col_name, num_cols, categ_cols)
            _Tid_k[col_name]   = _r["tree_indices"]
            _rFM_k[col_name]   = _r["rf_model"]
            _Ftree_k[col_name] = _r["tree_used"]

        # Compute residuals on new validation set
        _resid_k = {}
        for _col in num_cols:
            if _col in fitonly:
                continue
            _resid_k[_col] = np.sort(calc_resi(
                _col, X_processed_valid,
                _Tid_k[_col][: _Ftree_k[_col]], _rFM_k[_col],
            ))

        # Restore globals
        (X_processed_train, X_processed_valid, X_processed_test,
         X_train, X_valid, X_test, process_list) = _sv

        return _Ftree_k, _rFM_k, _Tid_k, _resid_k, _sc_k, _ohe_k, _plist_k

    def _preprocess_for_scoring_k(df_orig, sc_k, ohe_k, plist_k):
        """Preprocess using threshold-k scaler/encoders (LabelEncode same as base)."""
        df = df_orig[_agg_feat_cols].copy().replace("?", np.nan)
        for col in _agg_feat_cols:
            if col in categ_cols:
                fill = df[col].mode()
                df[col] = df[col].fillna(fill[0] if len(fill) > 0 else 0)
                df[col] = _safe_le_transform(_label_encoders[col], df[col].values)
            else:
                df[col] = df[col].fillna(_num_medians.get(col, 0))
        df.reset_index(drop=True, inplace=True)

        X_proc = pd.DataFrame()
        for col_name in categ_cols:
            ohe_vals = ohe_k[col_name].transform(df[[col_name]])
            X_proc = pd.concat([X_proc, pd.DataFrame(ohe_vals, columns=plist_k[col_name])], axis=1)
        _nc = [c for c in num_cols if c in _agg_feat_cols]
        X_num_sc = pd.DataFrame(sc_k.transform(df[_nc]), columns=_nc)
        X_proc = pd.concat([X_proc, X_num_sc], axis=1)
        X_proc.reset_index(drop=True, inplace=True)
        return df, X_proc

    def _compute_anomaly_scores_k(X_raw, X_proc, alpha_val, Ftree_k, rFM_k, Tid_k, resid_k, plist_k=None):
        """Score rows using threshold-k RFOD models (no global mutation of models)."""
        global X_processed_test, process_list
        _saved_proc = X_processed_test
        _saved_plist = process_list
        X_processed_test = X_proc
        if plist_k is not None:
            process_list = plist_k

        _Conf = {}
        for _col in X_raw.columns:
            if _col in fitonly or _col not in Ftree_k:
                continue
            _Conf[_col] = predict(
                _col, Ftree_k[_col],
                X_raw[_col].values, Tid_k[_col], rFM_k[_col], num_cols,
            )

        X_processed_test = _saved_proc
        process_list = _saved_plist

        _pcols = [c for c in X_raw.columns if c not in fitonly and c in Ftree_k]
        _conf  = pd.DataFrame({c: _Conf[c]["confidence_scores"] for c in _pcols})[_pcols]
        _pred  = pd.DataFrame({c: _Conf[c]["final_pred"]        for c in _pcols})[_pcols]

        _fw = _conf.to_numpy().copy()
        for i in range(_fw.shape[0]):
            s = _fw[i].sum()
            if s != 0:
                _fw[i] /= s
        _fw = 1 - _fw
        for i in range(_fw.shape[0]):
            s = _fw[i].sum()
            if s != 0:
                _fw[i] /= s
        _fw = np.square(_fw)
        _fw = _fw / np.sum(_fw, axis=1, keepdims=True)

        _cat_idx = [i for i, c in enumerate(X_raw.columns) if c in categ_cols]
        _num_idx = [i for i, c in enumerate(X_raw.columns) if c in num_cols]
        _res = alphaquantile(
            _pred.values, X_raw.values,
            _cat_idx, _num_idx, X_raw.columns,
            _fw, resid_k, weighted=weighted,
        )
        _, _scores = _res[alpha_val]
        return _scores

    def _score_csv_chunked(csv_path, out_path, id_cols, chunk_size=5000):
        """Read csv_path in chunks, score with all threshold models, stream-write to out_path."""
        first = True
        total = 0
        for _chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            # k=0: use original trained models
            _xr0, _xp0 = _preprocess_for_scoring(_chunk)
            _chunk["anomaly_score_0"] = _compute_anomaly_scores(_xr0, _xp0, best_aucroc_alpha)
            # k>=1: use retrained threshold-k models
            for _k, _arts in _extra_models.items():
                _Ftree_k, _rFM_k, _Tid_k, _resid_k, _sc_k, _ohe_k, _plist_k = _arts
                _xrk, _xpk = _preprocess_for_scoring_k(_chunk, _sc_k, _ohe_k, _plist_k)
                _chunk[f"anomaly_score_{_k}"] = _compute_anomaly_scores_k(
                    _xrk, _xpk, best_aucroc_alpha, _Ftree_k, _rFM_k, _Tid_k, _resid_k, _plist_k
                )
            _chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
            first = False
            total += len(_chunk)
            print(f"  [Feature Agg] train processed {total} rows")
            sys.stdout.flush()
        return total

    try:
        print(f"[Feature Agg] Best alpha = {best_aucroc_alpha}")
        sys.stdout.flush()

        # Determine all label thresholds from y_raw
        # e.g. labels [0,1,2,3,4] → thresholds [0,1,2,3] (exclude max: no anomalies left)
        _all_label_vals = sorted(int(v) for v in y_raw.unique() if pd.notna(v))
        _score_thresholds = _all_label_vals[:-1]
        print(f"[Feature Agg] Label values: {_all_label_vals}  →  score thresholds: {_score_thresholds}")
        sys.stdout.flush()

        # Retrain for thresholds k >= 1 (k=0 uses the already-trained globals)
        _extra_models = {}
        for _k in _score_thresholds:
            if _k == 0:
                continue
            print(f"[Feature Agg] Retraining RFOD for threshold k={_k} (normal = labels 0..{_k})...")
            sys.stdout.flush()
            _extra_models[_k] = _retrain_for_threshold(_k)
            print(f"[Feature Agg] Retraining done for k={_k}")
            sys.stdout.flush()

        # Train CSV — chunked (may be very large)
        print(f"[Feature Agg] Scoring train CSV (chunked): {_orig_train_csv_path}")
        sys.stdout.flush()
        _n_train = _score_csv_chunked(_orig_train_csv_path, _orig_train_agg_path, _orig_id_cols)
        print(f"[Feature Agg] Train saved: {_n_train} rows → {_orig_train_agg_path}")
        sys.stdout.flush()

        # Val CSV — at once (small)
        print("[Feature Agg] Scoring val CSV...")
        sys.stdout.flush()
        _val_raw0, _val_proc0 = _preprocess_for_scoring(_orig_val_df)
        _out_val = _orig_val_df.drop(columns=["label"], errors="ignore").copy()
        _out_val["anomaly_score_0"] = _compute_anomaly_scores(_val_raw0, _val_proc0, best_aucroc_alpha)
        for _k, _arts in _extra_models.items():
            _Ftree_k, _rFM_k, _Tid_k, _resid_k, _sc_k, _ohe_k, _plist_k = _arts
            _val_rawK, _val_procK = _preprocess_for_scoring_k(_orig_val_df, _sc_k, _ohe_k, _plist_k)
            _out_val[f"anomaly_score_{_k}"] = _compute_anomaly_scores_k(
                _val_rawK, _val_procK, best_aucroc_alpha, _Ftree_k, _rFM_k, _Tid_k, _resid_k, _plist_k
            )
        _out_val.to_csv(_orig_val_agg_path, index=False)
        print(f"[Feature Agg] Val   saved: {_out_val.shape[0]} rows → {_orig_val_agg_path}")
        sys.stdout.flush()

    except Exception as _e:
        import traceback
        print(f"[Feature Agg] ERROR: {_e}")
        traceback.print_exc()
        sys.stdout.flush()


sys.stdout.close()
sys.stdout = sys.__stdout__

# python3 exp_ndcg_Pure.py --dataset wine --msg weighted && python3 exp_ndcg_Pure.py --dataset pima --msg weighted && python3 exp_ndcg_Pure.py --dataset ionosphere --msg weighted && python3 exp_ndcg_Pure.py --dataset spambase --msg weighted && python3 exp_ndcg_Pure.py --dataset cardio --msg weighted && python3 exp_ndcg_Pure.py --dataset dos --msg weighted
# python3 exp_ndcg_Pure.py --dataset breastw --msg weighted && python3 exp_ndcg_Pure.py --dataset pageblocks --msg weighted && python3 exp_ndcg_Pure.py --dataset arrhythmia --msg weighted && python3 exp_ndcg_Pure.py --dataset thyroid --msg weighted && python3 exp_ndcg_Pure.py --dataset backdoor --msg weighted
