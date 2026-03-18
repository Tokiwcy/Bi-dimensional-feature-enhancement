#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost / LightGBM / XGBoost baseline 超参自动调优。
使用与主脚本相同的数据工程与划分协议，Optuna 搜索最优参数，输出可复用到主脚本的配置。

USAGE:
  python tune_baseline.py --dataset ames
  python tune_baseline.py --dataset boston
  python tune_baseline.py --dataset kc_house
  python tune_baseline.py --dataset california
  python tune_baseline.py --dataset brazilian_houses
  python tune_baseline.py --dataset ames_rag --data-path ames_train_rag.csv   # 先用 Ames.py --mode rag 生成 CSV
  python tune_baseline.py --dataset ames_rag_price --data-path ames_train_rag_price.csv
  python tune_baseline.py --dataset ames --model catboost --trials 50
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

SEED = 42
OUTPUT_DIR = Path("runs/tune_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_and_prepare_data(dataset: str, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """加载数据、清洗、特征工程、划分，返回 (X_train, X_val, X_test, y_train, y_val, y_test, y_is_log)。
    y_is_log: True 表示 y 已在 log 空间（如 Ames 的 log1p(PricePerSqrt)），优化时直接用 sqrt(mse(y_val, pred))。"""
    set_seed(SEED)

    if dataset == "ames":
        from Ames import clean_data, feature_engineering
        path = data_path or "train.csv"
        df = pd.read_csv(path).drop(columns=["Id"], errors="ignore")
        df = clean_data(df)
        df = feature_engineering(df)
        # Match Ames.py baseline: target = log1p(PricePerSqrt), one-hot encoding
        df["PricePerSqrt"] = df["SalePrice"] / df["OverallQual_TotalSF"]
        y = np.log1p(df["PricePerSqrt"].values.astype(np.float64))
        X = df.drop(columns=["SalePrice", "PricePerSqrt"])
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            tr_cat = pd.DataFrame(ohe.fit_transform(X[cat_cols].astype(str).fillna("None")), columns=ohe.get_feature_names_out(cat_cols), index=X.index)
            X = pd.concat([X.drop(columns=cat_cols), tr_cat], axis=1)
        idx = np.arange(len(df))
        train_idx, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        val_idx, test_idx = train_test_split(rest, test_size=1.0 / 3.0, random_state=SEED)  # 70/20/10
        use_scaler = False  # Ames baseline: one-hot, no scaling

    elif dataset == "kc_house":
        from kc_house import clean_data, feature_engineering
        path = data_path or "kc_house_data.csv"
        df = pd.read_csv(path).drop(columns=["id", "date"], errors="ignore")
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = clean_data(df)
        df = feature_engineering(df)
        target = "price"
        y = df[target].clip(lower=0).values.astype(np.float64)
        X = df.drop(columns=[target])
        idx = np.arange(len(df))
        rest, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
        train_idx, val_idx = train_test_split(rest, test_size=0.125, random_state=SEED)
        use_scaler = True  # kc_house baseline: StandardScaler

    elif dataset == "california":
        try:
            from california_housing_opt import clean_data, feature_engineering
        except ImportError:
            try:
                from cal import clean_data, feature_engineering
            except ImportError:
                raise ImportError("Need california_housing_opt or cal module")
        path = data_path or "california_housing.csv"
        df = pd.read_csv(path)
        df = clean_data(df)
        df = feature_engineering(df)
        target = "median_house_value"
        y = df[target].clip(lower=0).values.astype(np.float64)
        X = df.drop(columns=[target])
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("None").values)
        idx = np.arange(len(df))
        train_idx, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        val_idx, test_idx = train_test_split(rest, test_size=2/3, random_state=SEED)
        use_scaler = False  # California baseline: label encoding only, no scaling

    elif dataset == "boston":
        path = data_path or "boston_housing_dataset.csv"
        df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
        target = "MEDV"
        if target not in df.columns:
            raise ValueError(f"Boston dataset must have column {target}")
        y = df[target].values.astype(np.float64)
        X = df.drop(columns=[target])
        # Impute: numeric -> median, categorical -> mode
        for col in X.columns:
            if X[col].dtype in (np.float64, np.int64, "float64", "int64"):
                med = X[col].median()
                X[col] = X[col].fillna(med)
            else:
                mode_val = X[col].mode()
                fill = mode_val.iloc[0] if len(mode_val) else "None"
                X[col] = X[col].fillna(fill)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            tr_cat = pd.DataFrame(
                ohe.fit_transform(X[cat_cols].astype(str).fillna("None")),
                columns=ohe.get_feature_names_out(cat_cols),
                index=X.index,
            )
            X = pd.concat([X.drop(columns=cat_cols), tr_cat], axis=1)
        idx = np.arange(len(df))
        train_idx, rest = train_test_split(idx, test_size=0.5, random_state=SEED)
        val_idx, test_idx = train_test_split(rest, test_size=0.5, random_state=SEED)  # 50/25/25
        use_scaler = True  # Boston: StandardScaler

    elif dataset == "ames_rag":
        # Precomputed RAG features from Ames.py (ames_train_rag*.csv): features + target column
        path = data_path or "ames_train_rag.csv"
        df = pd.read_csv(path)
        if "target" not in df.columns:
            raise ValueError(f"ames_rag CSV must have 'target' column (log1p(PricePerSqrt)). Got: {list(df.columns)[:5]}...")
        y = df["target"].values.astype(np.float64)
        X = df.drop(columns=["target"])
        idx = np.arange(len(df))
        train_idx, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        val_idx, test_idx = train_test_split(rest, test_size=1.0 / 3.0, random_state=SEED)  # 70/20/10
        use_scaler = False
        y_is_log = True

    elif dataset == "ames_rag_price":
        # Precomputed RAG_PRICE features from Ames.py (ames_train_rag_price*.csv): features + target column
        path = data_path or "ames_train_rag_price.csv"
        df = pd.read_csv(path)
        if "target" not in df.columns:
            raise ValueError(f"ames_rag_price CSV must have 'target' column (log1p(PricePerSqrt)). Got: {list(df.columns)[:5]}...")
        y = df["target"].values.astype(np.float64)
        X = df.drop(columns=["target"])
        idx = np.arange(len(df))
        train_idx, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        val_idx, test_idx = train_test_split(rest, test_size=1.0 / 3.0, random_state=SEED)  # 70/20/10
        use_scaler = False
        y_is_log = True

    elif dataset == "brazilian_houses":
        try:
            from scipy.io import arff
        except ImportError:
            raise ImportError("brazilian_houses 需要 scipy: pip install scipy")
        path = data_path or "brazilian_houses.arff"
        try:
            data, _ = arff.loadarff(path)
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode("utf-8")
                    except Exception:
                        pass
        except Exception:
            df = pd.read_csv(path.replace(".arff", ".csv"))
        for col in ["city", "animal", "furniture"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        if "floor" in df.columns:
            floor_raw = df["floor"]
            floor_str = floor_raw.astype(str).str.strip()
            unknown = floor_raw.isna() | floor_str.isin(["-", "", "nan", "None", "null", "NULL"])
            df["floor_is_unknown"] = unknown.astype(int)
            df["floor"] = floor_str.mask(unknown, "0")
        df = df.drop_duplicates().reset_index(drop=True)
        target_col = "total"
        leakage = ["hoa", "rent_amount", "property_tax", "fire_insurance"]
        drop_cols = [c for c in leakage if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if target_col not in df.columns:
            raise ValueError(f"brazilian_houses 需包含列 {target_col}")
        y = np.log1p(pd.to_numeric(df[target_col], errors="coerce").clip(lower=0).values.astype(np.float64))
        X = df.drop(columns=[target_col])
        cat_cols = [c for c in ["city", "floor", "animal", "furniture"] if c in X.columns]
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("__missing__").values)
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        X = X.astype(np.float64)
        idx = np.arange(len(df))
        train_idx, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        val_idx, test_idx = train_test_split(rest, test_size=1.0 / 3.0, random_state=SEED)
        use_scaler = True
        y_is_log = True

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use ames|boston|kc_house|california|brazilian_houses|ames_rag|ames_rag_price")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # 全部转为数值矩阵（label encoding 后均为 numeric）
    X_train_num = X_train.fillna(0).values.astype(np.float64)
    X_val_num = X_val.fillna(0).values.astype(np.float64)
    X_test_num = X_test.fillna(0).values.astype(np.float64)

    if use_scaler:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_val_s = scaler.transform(X_val_num)
        X_test_s = scaler.transform(X_test_num)
    else:
        X_train_s, X_val_s, X_test_s = X_train_num, X_val_num, X_test_num

    y_is_log = dataset in ("ames", "brazilian_houses", "ames_rag", "ames_rag_price")
    print(f"  {dataset}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={X_train_s.shape[1]} (scaled={use_scaler}, y_log={y_is_log})")
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, y_is_log


def suggest_catboost(trial: "optuna.Trial") -> dict:
    return {
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "iterations": trial.suggest_int("iterations", 100, 3000),
        "random_seed": SEED,
        "verbose": False,
    }


def suggest_xgb(trial: "optuna.Trial") -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
    }


def suggest_lgbm(trial: "optuna.Trial") -> dict:
    return {
        "objective": "regression",
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 5000),
        "max_bin": trial.suggest_int("max_bin", 63, 255),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "random_state": SEED,
        "verbose": -1,
    }


def run_tune(
    dataset: str,
    model_name: str,
    n_trials: int,
    data_path: Optional[str],
) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test, y_is_log = load_and_prepare_data(dataset, data_path)

    def objective(trial):
        if model_name == "catboost":
            params = suggest_catboost(trial)
            m = CatBoostRegressor(**params)
            m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
        elif model_name == "xgboost":
            params = suggest_xgb(trial)
            m = XGBRegressor(**params)
            # XGBoost 2.x: early_stopping_rounds 已从 fit() 移除，仅用 eval_set 监控
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "lightgbm":
            params = suggest_lgbm(trial)
            m = LGBMRegressor(**params)
            from lightgbm import early_stopping
            m.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(30, verbose=False)],
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        pred = m.predict(X_val)
        eps = 1e-8
        if y_is_log:
            return np.sqrt(mean_squared_error(y_val, pred))  # RMSLE on expm1 scale
        return np.sqrt(mean_squared_error(
            np.log(np.maximum(y_val, eps)),
            np.log(np.maximum(pred, eps)),
        ))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED, n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_val_rmse = study.best_value

    m_final = None
    if model_name == "catboost":
        p = suggest_catboost(optuna.trial.FixedTrial(best_params))
        m_final = CatBoostRegressor(**p)
    elif model_name == "xgboost":
        p = suggest_xgb(optuna.trial.FixedTrial(best_params))
        m_final = XGBRegressor(**p)
    elif model_name == "lightgbm":
        p = suggest_lgbm(optuna.trial.FixedTrial(best_params))
        m_final = LGBMRegressor(**p)

    m_final.fit(X_train, y_train)
    pred_test = m_final.predict(X_test)
    eps = 1e-8
    if y_is_log:
        test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    else:
        test_rmse = np.sqrt(mean_squared_error(
            np.log(np.maximum(y_test, eps)),
            np.log(np.maximum(pred_test, eps)),
        ))

    result = {
        "dataset": dataset,
        "model": model_name,
        "best_params": best_params,
        "val_rmse": float(best_val_rmse),
        "test_rmse": float(test_rmse),
    }
    return result


def params_to_code(model_name: str, best_params: dict) -> str:
    """生成可粘贴到主脚本的代码片段。"""
    if model_name == "catboost":
        return (
            f'CatBoostRegressor(\n'
            f'    depth={best_params.get("depth", "?")}, learning_rate={best_params.get("learning_rate", "?")}, '
            f'l2_leaf_reg={best_params.get("l2_leaf_reg", "?")}, iterations={best_params.get("iterations", "?")},\n'
            f'    random_seed=seed, verbose=False\n)'
        )
    if model_name == "xgboost":
        return (
            f'XGBRegressor(\n'
            f'    n_estimators={best_params.get("n_estimators", "?")}, max_depth={best_params.get("max_depth", "?")}, '
            f'learning_rate={best_params.get("learning_rate", "?")},\n'
            f'    subsample={best_params.get("subsample", "?")}, colsample_bytree={best_params.get("colsample_bytree", "?")},\n'
            f'    reg_alpha={best_params.get("reg_alpha", "?")}, reg_lambda={best_params.get("reg_lambda", "?")},\n'
            f'    random_state=seed, n_jobs=-1\n)'
        )
    if model_name == "lightgbm":
        return (
            f'LGBMRegressor(\n'
            f'    objective="regression", num_leaves={best_params.get("num_leaves", "?")}, '
            f'learning_rate={best_params.get("learning_rate", "?")}, n_estimators={best_params.get("n_estimators", "?")},\n'
            f'    max_bin={best_params.get("max_bin", "?")}, bagging_fraction={best_params.get("bagging_fraction", "?")}, '
            f'bagging_freq={best_params.get("bagging_freq", "?")}, feature_fraction={best_params.get("feature_fraction", "?")},\n'
            f'    random_state=seed, verbose=-1\n)'
        )
    return ""


def main():
    parser = argparse.ArgumentParser(description="CatBoost/LightGBM/XGBoost baseline 超参调优")
    parser.add_argument("--dataset", type=str, default="ames", choices=["ames", "boston", "kc_house", "california", "brazilian_houses", "ames_rag", "ames_rag_price"])
    parser.add_argument("--data-path", type=str, default=None, help="数据文件路径，默认按 dataset 自动选择")
    parser.add_argument("--model", type=str, default="all", choices=["all", "catboost", "xgboost", "lightgbm"])
    parser.add_argument("--trials", type=int, default=80, help="每个模型的 Optuna 试验次数")
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("请安装 Optuna: pip install optuna")
        sys.exit(1)

    set_seed(SEED)
    models = ["catboost", "xgboost", "lightgbm"] if args.model == "all" else [args.model]
    all_results = []

    print(f"\n>>> Tuning baseline for dataset={args.dataset}, models={models}, trials={args.trials}\n")

    for model_name in models:
        print(f"\n{'='*60}\n>>> {model_name.upper()}\n{'='*60}")
        res = run_tune(args.dataset, model_name, args.trials, args.data_path)
        all_results.append(res)
        print(f"  Best val Log RMSE: {res['val_rmse']:.4f}  |  test Log RMSE: {res['test_rmse']:.4f}")
        print(f"  Params: {res['best_params']}")

    out_path = OUTPUT_DIR / f"best_params_{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Saved to {out_path}")

    print("\n" + "=" * 60)
    print("可粘贴到主脚本的代码片段:")
    print("=" * 60)
    for res in all_results:
        print(f"\n# {res['model'].upper()} (val_log_rmse={res['val_rmse']:.4f})")
        print(params_to_code(res["model"], res["best_params"]))


if __name__ == "__main__":
    main()
