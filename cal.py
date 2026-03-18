#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
California Housing: Aligned with Ames.py architecture.
Single file; split 70% train / 10% val / 20% test (same as Ames.py), over seeds.
Target: median_house_value. No Id column - uses DataFrame index for test_ids.

Feature engineering: rooms_per_household, bedrooms_per_room, population_per_household,
bedrooms_per_household, rooms_per_person, bedrooms_per_person, geo_cluster (KMeans),
distance_to_sf/la/san_diego/sacramento, lat_lon_sum/diff, coastal_flag, age_bucket,
income_bucket, income_per_room, income_x_rooms_per_household.

Embedding template: --openai-template structured_short_plus (default), hybrid, bucket_only_semantic, structured.

Modes: baseline, emb, rag, rag_price, fni, fni+rag, fni_with_rag, fni_dual, emb+rag, emb_with_rag, emb_with_rag+rag.
--mode all runs baseline, emb, rag, rag_price, emb+rag, emb_with_rag, emb_with_rag+rag (no fni series).
RAG modes (--rag-mode): geo (default, longitude/latitude/distance_to_coast), hybrid (all numeric).
Models: CatBoost, XGBoost, LightGBM, TabPFN only.
"""
import os
import math
import time
import random
import hashlib
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 地理聚类与锚点城市（用于 distance 特征）
GEO_CLUSTER_K = 12  # KMeans 聚类数，8~20 均可
ANCHOR_CITIES = {
    "sf": (37.7749, -122.4194),
    "la": (34.0522, -118.2437),
    "san_diego": (32.7157, -117.1611),
    "sacramento": (38.5816, -121.4944),
}

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except Exception:
    TABPFN_AVAILABLE = False


SEED = 42
DATA_PATH = "data/california_housing.csv"

OPENAI_EMB_MODEL = "text-embedding-3-small"
OPENAI_TEMPLATE = "structured_short_plus"
PCA_DIM = 16

USE_CATBOOST = True
USE_XGB = True
USE_LGBM = True
USE_TABPFN = True

EMB_CACHE_DIR = Path("results/embedding_cache")
PCA_CACHE_DIR = Path("results/pca_cache")

RAG_IMPORTANT_NUMERIC_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "log_median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "rooms_per_population",
    "other_rooms",
    "distance_to_coast",
    "avg_household_size",
    "geo_cluster",
]
RAG_IMPORTANT_CAT_COLS = ["ocean_proximity", "coastal_flag", "age_bucket", "income_bucket"]

# RAG 近邻检索：geo=仅地理，all=全部数值列
RAG_GEO_COLS = ["longitude", "latitude", "distance_to_coast"]

# RAG 邻居模式: hybrid=全部数值, geo=仅地理
RAG_MODE_CONFIG = {
    "hybrid": {"search_numeric": RAG_IMPORTANT_NUMERIC_COLS, "search_cat": []},
    "geo": {"search_numeric": RAG_GEO_COLS, "search_cat": []},
}

# Best config per model (rag_k, pca_dim, rag_pca_dim). Used when --best_config.
BEST_CONFIG_PER_MODEL = {
    "catboost": {"rag_k": 16, "pca_dim": 20, "rag_pca_dim": 20},
    "lgbm": {"rag_k": 20, "pca_dim": 20, "rag_pca_dim": 20},
    "tabpfn": {"rag_k": 12, "pca_dim": 20, "rag_pca_dim": 16},
    "xgb": {"rag_k": 12, "pca_dim": 16, "rag_pca_dim": 16},
}
# rag_price uses same rag_k as rag for filtering in best_config


def _should_record_model(best_config: bool, mode: str, name: str, pca_dim, rag_k, rag_pca_dim) -> bool:
    """When --best-config, only record if (pca_dim, rag_k, rag_pca_dim) matches model's config."""
    if not best_config:
        return True
    cfg = BEST_CONFIG_PER_MODEL.get(name)
    if not cfg:
        return True
    if mode == "emb":
        return pca_dim == cfg["pca_dim"]
    if mode == "rag":
        return rag_k == cfg["rag_k"]
    if mode == "emb+rag":
        return pca_dim == cfg["pca_dim"] and rag_k == cfg["rag_k"]
    if mode == "emb_with_rag":
        return rag_pca_dim == cfg["rag_pca_dim"]
    if mode == "emb_with_rag+rag":
        return rag_pca_dim == cfg["rag_pca_dim"] and rag_k == cfg["rag_k"]
    if mode == "rag_price":
        return rag_k == cfg["rag_k"]
    return True


def _parse_int_list(s: str, default: List[int]) -> List[int]:
    """Parse '8,12,16' -> [8,12,16]. Single int also OK."""
    if isinstance(s, (list, tuple)):
        return list(s)
    s = str(s).strip()
    if not s:
        return default
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def evaluate_model(y_true, y_pred, name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    y_t = np.maximum(np.asarray(y_true, dtype=float).ravel(), 0)
    y_p = np.maximum(np.asarray(y_pred, dtype=float).ravel(), -1)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_t), np.log1p(y_p)))
    print(f"  {name}: RMSLE = {rmsle:.4f}, RMSE = {rmse:.4f}")
    return {"mse": mse, "rmse": rmse, "rmsle": rmsle}


def safe_val(v) -> str:
    if pd.isna(v):
        return "unknown"
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.2f}"
    return str(v)


# California-specific: region inference
def _infer_region(latitude: float, longitude: float) -> str:
    if latitude >= 38.0:
        return "Northern California"
    if latitude >= 36.0 and longitude <= -121.0:
        return "Bay Area / Central Coast"
    if latitude >= 36.0:
        return "Central Valley"
    if latitude >= 34.5:
        return "Southern California (LA region)"
    return "Southern California (San Diego / Desert)"


def _ocean_prose(ocean: str) -> str:
    m = {
        "NEAR BAY": "near the bay",
        "INLAND": "inland",
        "NEAR OCEAN": "by the ocean",
        "<1H OCEAN": "within an hour of the ocean",
        "ISLAND": "on an island",
    }
    return m.get(str(ocean).upper(), "inland")


def _income_prose(income: float) -> str:
    usd = income * 10000
    if usd < 30000:
        return f"low income area (median ${usd/1000:.0f}k)"
    if usd < 60000:
        return f"moderate income (median ${usd/1000:.0f}k)"
    if usd < 100000:
        return f"upper-middle income (median ${usd/1000:.0f}k)"
    return f"high income area (median ${usd/1000:.0f}k)"


def _age_prose(age: int) -> str:
    if age < 15:
        return "newer homes"
    if age < 35:
        return "moderately aged housing"
    return "older housing stock"


def _rooms_prose(rooms_per_hh: float) -> str:
    if rooms_per_hh < 4:
        return "compact homes"
    if rooms_per_hh < 6:
        return "average-sized homes"
    return "spacious homes"


def _bpr_semantic(bpr: float) -> str:
    """Bedrooms per room -> balanced / high / low bedroom ratio."""
    if pd.isna(bpr):
        return "unknown bedroom ratio"
    b = float(bpr)
    if b < 0.14:
        return "low bedroom ratio"
    if b <= 0.22:
        return "balanced bedroom ratio"
    return "high bedroom ratio"


def _density_semantic(pph: float) -> str:
    """Population per household -> low / moderate / high household density."""
    if pd.isna(pph):
        return "unknown density"
    p = float(pph)
    if p < 2.5:
        return "low household density"
    if p <= 3.5:
        return "moderate household density"
    return "high household density"


def _coast_phrase(coastal_flag_or_ocean) -> str:
    """Close to coast / inland / near bay for template prose."""
    if pd.isna(coastal_flag_or_ocean):
        return "inland"
    s = str(coastal_flag_or_ocean).lower()
    if "coastal" in s or "ocean" in s or "1h" in s:
        return "close to the coast"
    if "bay" in s:
        return "near the bay"
    if "island" in s:
        return "on the island"
    return "inland"


def _income_bucket_prose(bucket) -> str:
    """Map income_bucket to prose: upper-middle income, low income, etc."""
    if pd.isna(bucket):
        return "unknown income"
    s = str(bucket).lower().replace(" ", "_")
    if s == "low":
        return "low income"
    if s == "moderate":
        return "moderate income"
    if "upper" in s or s == "upper_middle":
        return "upper-middle income"
    if s == "high":
        return "high income"
    return str(bucket)


def _age_bucket_prose(bucket) -> str:
    """Map age_bucket to template prose: newer, moderately aged, mid-aged, older."""
    if pd.isna(bucket):
        return "unknown age"
    s = str(bucket).lower()
    if s in ("new", "relatively new"):
        return "newer" if s == "new" else "relatively new"
    if s == "middle-aged":
        return "moderately aged"
    if s in ("old", "very old"):
        return "older"
    return s.replace("_", " ")


def _market_region_phrase(lat, lon, ocean) -> str:
    """Bay Area type / Southern California type housing market."""
    if pd.isna(lat) or pd.isna(lon):
        return _ocean_prose(str(ocean)) if not pd.isna(ocean) else "California"
    region = _infer_region(float(lat), float(lon))
    if "Bay" in region or "Central Coast" in region:
        return "Bay Area type"
    if "Northern" in region:
        return "Northern California type"
    if "Southern" in region:
        return "Southern California type"
    return "Central Valley type"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy: impute on full df (use only for non-split pipelines). Prefer clean_data_fit_transform."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna("None")
    return df


def clean_data_fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Impute missing values: fit median (numeric) on train only, then fill train/val/test. Cat → 'None'. No leakage."""
    num_cols = [c for c in train_df.select_dtypes(include=["number"]).columns if c in train_df.columns]
    cat_cols = [c for c in train_df.select_dtypes(include=["object", "category"]).columns if c in train_df.columns]
    medians = train_df[num_cols].median()
    train_clean = train_df.copy()
    val_clean = val_df.copy()
    test_clean = test_df.copy()
    for c in num_cols:
        train_clean[c] = train_clean[c].fillna(medians[c])
        val_clean[c] = val_clean[c].fillna(medians[c])
        test_clean[c] = test_clean[c].fillna(medians[c])
    for c in cat_cols:
        train_clean[c] = train_clean[c].fillna("None").astype(str)
        val_clean[c] = val_clean[c].fillna("None").astype(str)
        test_clean[c] = test_clean[c].fillna("None").astype(str)
    return train_clean, val_clean, test_clean


def _age_bucket(age: float) -> str:
    """房龄分桶：new / relatively new / middle-aged / old / very old"""
    if pd.isna(age):
        return "unknown"
    a = float(age)
    if a < 10:
        return "new"
    if a < 20:
        return "relatively new"
    if a < 40:
        return "middle-aged"
    if a < 55:
        return "old"
    return "very old"


def _income_bucket(median_income: float) -> str:
    """收入分桶（median_income 单位为万刀）"""
    if pd.isna(median_income):
        return "unknown"
    usd = float(median_income) * 10000
    if usd < 30000:
        return "low"
    if usd < 60000:
        return "moderate"
    if usd < 100000:
        return "upper_middle"
    return "high"


def _coastal_flag(ocean_proximity) -> str:
    """ocean_proximity -> coastal / inland / near bay / island"""
    if pd.isna(ocean_proximity):
        return "inland"
    s = str(ocean_proximity).upper()
    if "ISLAND" in s:
        return "island"
    if "BAY" in s:
        return "near bay"
    if "OCEAN" in s or "1H" in s:
        return "coastal"
    return "inland"


def feature_engineering(df: pd.DataFrame, geo_cluster_k: int = GEO_CLUSTER_K, random_state: int = 42) -> pd.DataFrame:
    """Legacy: FE on full df (KMeans fit on all). Prefer feature_engineering_fit_transform for split data."""
    df = df.copy()
    hh = df["households"].replace(0, np.nan)
    pop = df["population"].replace(0, np.nan)
    rooms = df["total_rooms"].replace(0, np.nan)
    df["rooms_per_household"] = df["total_rooms"] / hh
    df["bedrooms_per_room"] = df["total_bedrooms"] / rooms
    df["bedrooms_ratio"] = df["bedrooms_per_room"]
    df["population_per_household"] = df["population"] / hh
    df["bedrooms_per_household"] = df["total_bedrooms"] / hh
    df["rooms_per_person"] = df["total_rooms"] / pop
    df["bedrooms_per_person"] = df["total_bedrooms"] / pop
    for col in ["rooms_per_household", "bedrooms_per_room", "bedrooms_ratio", "population_per_household",
                "bedrooms_per_household", "rooms_per_person", "bedrooms_per_person"]:
        df[col] = df[col].fillna(df[col].median())
    geo = df[["longitude", "latitude"]].values
    kmeans = KMeans(n_clusters=min(geo_cluster_k, len(df)), random_state=random_state, n_init=10)
    df["geo_cluster"] = kmeans.fit_predict(geo).astype(int)
    for name, (lat_c, lon_c) in ANCHOR_CITIES.items():
        d = np.sqrt((df["latitude"] - lat_c) ** 2 + (df["longitude"] - lon_c) ** 2) * 111.0
        df[f"distance_to_{name}"] = d
    df["lat_lon_sum"] = df["longitude"] + df["latitude"]
    df["lat_lon_diff"] = df["latitude"] - df["longitude"]
    df["coastal_flag"] = df["ocean_proximity"].map(lambda x: _coastal_flag(x))
    df["income_per_room"] = df["median_income"] / df["rooms_per_household"].replace(0, np.nan)
    df["income_per_room"] = df["income_per_room"].fillna(df["income_per_room"].median())
    df["income_x_rooms_per_household"] = df["median_income"] * df["rooms_per_household"]
    df["age_bucket"] = df["housing_median_age"].map(lambda x: _age_bucket(x))
    df["income_bucket"] = df["median_income"].map(lambda x: _income_bucket(x))
    df["log_median_income"] = np.log1p(df["median_income"])
    df["rooms_per_population"] = df["total_rooms"] / pop
    df["rooms_per_population"] = df["rooms_per_population"].fillna(df["rooms_per_population"].median())
    df["other_rooms"] = df["total_rooms"] - df["total_bedrooms"]
    df["avg_household_size"] = df["population"] / hh
    df["avg_household_size"] = df["avg_household_size"].fillna(df["avg_household_size"].median())
    coast_lon = -124 + 0.7 * (42 - np.clip(df["latitude"].values, 32, 42))
    df["distance_to_coast"] = np.abs(df["longitude"].values - coast_lon)
    return df


def feature_engineering_fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    geo_cluster_k: int = GEO_CLUSTER_K,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """FE with fit on train only: KMeans on train geo; derived column medians from train. No leakage."""
    def _ratio_cols(tdf):
        hh = tdf["households"].replace(0, np.nan)
        pop = tdf["population"].replace(0, np.nan)
        rooms = tdf["total_rooms"].replace(0, np.nan)
        out = tdf.copy()
        out["rooms_per_household"] = out["total_rooms"] / hh
        out["bedrooms_per_room"] = out["total_bedrooms"] / rooms
        out["bedrooms_ratio"] = out["bedrooms_per_room"]
        out["population_per_household"] = out["population"] / hh
        out["bedrooms_per_household"] = out["total_bedrooms"] / hh
        out["rooms_per_person"] = out["total_rooms"] / pop
        out["bedrooms_per_person"] = out["total_bedrooms"] / pop
        return out, hh, pop, rooms

    train_fe, hh_t, pop_t, rooms_t = _ratio_cols(train_df)
    val_fe, _, _, _ = _ratio_cols(val_df)
    test_fe, _, _, _ = _ratio_cols(test_df)

    ratio_cols = ["rooms_per_household", "bedrooms_per_room", "bedrooms_ratio", "population_per_household",
                  "bedrooms_per_household", "rooms_per_person", "bedrooms_per_person"]
    medians_ratio = train_fe[ratio_cols].median()
    for c in ratio_cols:
        train_fe[c] = train_fe[c].fillna(medians_ratio[c])
        val_fe[c] = val_fe[c].fillna(medians_ratio[c])
        test_fe[c] = test_fe[c].fillna(medians_ratio[c])

    kmeans = KMeans(n_clusters=min(geo_cluster_k, len(train_df)), random_state=random_state, n_init=10)
    kmeans.fit(train_df[["longitude", "latitude"]].values)
    train_fe["geo_cluster"] = kmeans.predict(train_df[["longitude", "latitude"]].values).astype(int)
    val_fe["geo_cluster"] = kmeans.predict(val_df[["longitude", "latitude"]].values).astype(int)
    test_fe["geo_cluster"] = kmeans.predict(test_df[["longitude", "latitude"]].values).astype(int)

    for name, (lat_c, lon_c) in ANCHOR_CITIES.items():
        for tdf, src in [(train_fe, train_df), (val_fe, val_df), (test_fe, test_df)]:
            d = np.sqrt((src["latitude"] - lat_c) ** 2 + (src["longitude"] - lon_c) ** 2) * 111.0
            tdf[f"distance_to_{name}"] = d
    for tdf, src in [(train_fe, train_df), (val_fe, val_df), (test_fe, test_df)]:
        tdf["lat_lon_sum"] = src["longitude"] + src["latitude"]
        tdf["lat_lon_diff"] = src["latitude"] - src["longitude"]
        tdf["coastal_flag"] = src["ocean_proximity"].map(lambda x: _coastal_flag(x))

    train_fe["income_per_room"] = train_fe["median_income"] / train_fe["rooms_per_household"].replace(0, np.nan)
    val_fe["income_per_room"] = val_fe["median_income"] / val_fe["rooms_per_household"].replace(0, np.nan)
    test_fe["income_per_room"] = test_fe["median_income"] / test_fe["rooms_per_household"].replace(0, np.nan)
    med_income_room = train_fe["income_per_room"].median()
    train_fe["income_per_room"] = train_fe["income_per_room"].fillna(med_income_room)
    val_fe["income_per_room"] = val_fe["income_per_room"].fillna(med_income_room)
    test_fe["income_per_room"] = test_fe["income_per_room"].fillna(med_income_room)
    train_fe["income_x_rooms_per_household"] = train_fe["median_income"] * train_fe["rooms_per_household"]
    val_fe["income_x_rooms_per_household"] = val_fe["median_income"] * val_fe["rooms_per_household"]
    test_fe["income_x_rooms_per_household"] = test_fe["median_income"] * test_fe["rooms_per_household"]

    for tdf, src in [(train_fe, train_df), (val_fe, val_df), (test_fe, test_df)]:
        tdf["age_bucket"] = src["housing_median_age"].map(lambda x: _age_bucket(x))
        tdf["income_bucket"] = src["median_income"].map(lambda x: _income_bucket(x))
        tdf["log_median_income"] = np.log1p(src["median_income"])
        tdf["rooms_per_population"] = src["total_rooms"] / src["population"].replace(0, np.nan)
        tdf["other_rooms"] = src["total_rooms"] - src["total_bedrooms"]
        tdf["avg_household_size"] = src["population"] / src["households"].replace(0, np.nan)
    med_rpp = train_fe["rooms_per_population"].median()
    med_ahs = train_fe["avg_household_size"].median()
    for tdf in [train_fe, val_fe, test_fe]:
        tdf["rooms_per_population"] = tdf["rooms_per_population"].fillna(med_rpp)
        tdf["avg_household_size"] = tdf["avg_household_size"].fillna(med_ahs)
    # coast_lon 按行依赖 latitude，各集用自己的 latitude 计算
    for tdf, src in [(train_fe, train_df), (val_fe, val_df), (test_fe, test_df)]:
        coast_lon = -124 + 0.7 * (42 - np.clip(src["latitude"].values, 32, 42))
        tdf["distance_to_coast"] = np.abs(src["longitude"].values - coast_lon)
    return train_fe, val_fe, test_fe


def verbalize_row_structured(row: pd.Series) -> str:
    vals = row.to_dict()
    sections = {"Location": [], "Building": [], "Amenities": [], "Quality": []}

    # Location
    lat = vals.get("latitude")
    lon = vals.get("longitude")
    if not pd.isna(lat) and not pd.isna(lon):
        region = _infer_region(float(lat), float(lon))
        sections["Location"].append(f"in {region} at ({float(lat):.2f}, {float(lon):.2f})")

    ocean = vals.get("ocean_proximity")
    if not pd.isna(ocean):
        sections["Location"].append(f"ocean proximity {safe_val(ocean)} ({_ocean_prose(str(ocean))})")

    # Building
    age = vals.get("housing_median_age")
    if not pd.isna(age):
        sections["Building"].append(f"housing median age {int(age)} years ({_age_prose(int(age))})")

    rooms = vals.get("total_rooms")
    beds = vals.get("total_bedrooms")
    if not pd.isna(rooms) and not pd.isna(beds):
        sections["Building"].append(f"{int(rooms)} total rooms, {int(beds)} bedrooms")

    rooms_per_hh = vals.get("rooms_per_household")
    if not pd.isna(rooms_per_hh):
        sections["Building"].append(f"{rooms_per_hh:.1f} rooms per household ({_rooms_prose(rooms_per_hh)})")

    # Amenities / Demographics
    pop = vals.get("population")
    hh = vals.get("households")
    if not pd.isna(pop) and not pd.isna(hh):
        sections["Amenities"].append(f"population {int(pop)}, {int(hh)} households")

    # Quality / Income
    income = vals.get("median_income")
    if not pd.isna(income):
        sections["Quality"].append(_income_prose(income))

    parts = []
    for items in sections.values():
        if items:
            parts.append(", ".join(items))

    return ". ".join(parts) + "."


def verbalize_row_structured_short_plus(row: pd.Series) -> str:
    """Template 1: Structured Short Plus. Location, market/income, housing age, density/layout, coast/region."""
    vals = row.to_dict()
    ocean = vals.get("ocean_proximity", "unknown")
    ocean_prose = _ocean_prose(str(ocean)).replace("the ", "").strip()  # e.g. "near the bay" -> "near bay"
    geo_cluster = vals.get("geo_cluster", -1)
    lon = vals.get("longitude")
    lat = vals.get("latitude")
    age_bucket = vals.get("age_bucket", "unknown")
    income_bucket = vals.get("income_bucket", "unknown")
    rph = vals.get("rooms_per_household")
    bpr = vals.get("bedrooms_per_room")
    pph = vals.get("population_per_household")
    coastal_flag = vals.get("coastal_flag", ocean)

    loc = f"A census block in a {ocean_prose} California area, geo cluster {geo_cluster}"
    if not pd.isna(lon) and not pd.isna(lat):
        loc += f", located at longitude {float(lon):.2f} and latitude {float(lat):.2f}"
    loc += "."

    income_prose = _income_bucket_prose(income_bucket)
    age_prose = _age_bucket_prose(age_bucket)
    article = "an" if str(income_prose).startswith("upper") else "a"
    market = f"This is {article} {income_prose} neighborhood with {age_prose} housing stock."

    rooms_adj = (_rooms_prose(rph).replace(" homes", "") if not pd.isna(rph) else "typical")
    bpr_sem = _bpr_semantic(bpr)
    density_sem = _density_semantic(pph)
    layout = f"Homes are {rooms_adj} with {rph:.1f} rooms per household, a {bpr_sem}, and {density_sem}." if not pd.isna(rph) else f"Homes with a {bpr_sem} and {density_sem}."

    coast_ph = _coast_phrase(coastal_flag)
    region_ph = _market_region_phrase(lat, lon, ocean)
    coast_region = f"The block is {coast_ph} and belongs to a {region_ph} housing market."

    return " ".join([loc, market, layout, coast_region])


def verbalize_row_bucket_only_semantic(row: pd.Series) -> str:
    """Template 2: Bucket-Only Semantic. Minimal raw numbers, semantic labels only."""
    vals = row.to_dict()
    ocean = vals.get("ocean_proximity", "unknown")
    coastal_flag = vals.get("coastal_flag", ocean)
    geo_cluster = vals.get("geo_cluster", -1)
    income_bucket = vals.get("income_bucket", "unknown")
    age_bucket = vals.get("age_bucket", "unknown")
    rph = vals.get("rooms_per_household")
    bpr = vals.get("bedrooms_per_room")
    pph = vals.get("population_per_household")

    ocean_prose = _ocean_prose(str(ocean)).replace("the ", "").strip()
    if "inland" in ocean_prose:
        area = "Inland"
    elif "bay" in ocean_prose:
        area = "Near-bay"
    elif "ocean" in ocean_prose or "hour" in ocean_prose:
        area = "Coastal"
    else:
        area = "California"
    parts = [f"{area} California neighborhood", f"{ocean_prose} market", _income_bucket_prose(income_bucket)]

    age_prose = _age_bucket_prose(age_bucket)
    if age_prose in ("moderately aged", "relatively new", "newer", "older"):
        parts.append("mid-aged housing" if age_prose == "moderately aged" else f"{age_prose} housing")
    else:
        parts.append(f"{age_prose} housing")

    parts.append(_rooms_prose(rph) if not pd.isna(rph) else "homes")
    parts.append(_bpr_semantic(bpr))
    parts.append(_density_semantic(pph))
    parts.append(_coast_phrase(coastal_flag))
    parts.append(f"geo cluster {geo_cluster}")
    return ", ".join(parts) + "."


def verbalize_row_hybrid(row: pd.Series) -> str:
    """Template 3: Hybrid. Some raw numbers + bucket explanations. Balanced for embedding."""
    vals = row.to_dict()
    ocean = vals.get("ocean_proximity", "unknown")
    coastal_flag = vals.get("coastal_flag", ocean)
    geo_cluster = vals.get("geo_cluster", -1)
    lat = vals.get("latitude")
    lon = vals.get("longitude")
    income = vals.get("median_income")
    income_bucket = vals.get("income_bucket", "unknown")
    age_bucket = vals.get("age_bucket", "unknown")
    rph = vals.get("rooms_per_household")
    bpr = vals.get("bedrooms_per_room")
    pph = vals.get("population_per_household")

    region_ph = _market_region_phrase(lat, lon, ocean)
    coast_ph = _coast_phrase(coastal_flag)
    lead = f"A {region_ph} / {coast_ph} housing block"
    if not pd.isna(income):
        lead += f" with median income {float(income):.1f} ({_income_bucket_prose(income_bucket)})"
    lead += "."

    age_prose = _age_bucket_prose(age_bucket)
    layout_parts = [f"Homes are {age_prose}"]
    if not pd.isna(rph) and not pd.isna(bpr) and not pd.isna(pph):
        layout_parts.append(f"with {rph:.1f} rooms per household, {bpr:.2f} bedrooms per room, and {pph:.1f} people per household, indicating {_rooms_prose(rph)} with {_density_semantic(pph)}")
    elif not pd.isna(rph):
        layout_parts.append(f"with {rph:.1f} rooms per household ({_rooms_prose(rph)})")
    layout = ". ".join(layout_parts) + "."

    tail = f"The block is {coast_ph} and belongs to geo cluster {geo_cluster}."
    return " ".join([lead, layout, tail])


def build_texts(df: pd.DataFrame, template: str = "structured") -> List[str]:
    if template == "structured":
        return [verbalize_row_structured(row) for _, row in df.iterrows()]
    elif template == "structured_short_plus":
        return [verbalize_row_structured_short_plus(row) for _, row in df.iterrows()]
    elif template == "bucket_only_semantic":
        return [verbalize_row_bucket_only_semantic(row) for _, row in df.iterrows()]
    elif template == "hybrid":
        return [verbalize_row_hybrid(row) for _, row in df.iterrows()]
    else:
        raise ValueError("template must be 'structured', 'structured_short_plus', 'bucket_only_semantic', or 'hybrid'")


def _base_fn_for_template(template: str):
    """Return the verbalize function for the given template (for RAG and non-RAG)."""
    if template == "structured":
        return verbalize_row_structured
    if template == "structured_short_plus":
        return verbalize_row_structured_short_plus
    if template == "bucket_only_semantic":
        return verbalize_row_bucket_only_semantic
    if template == "hybrid":
        return verbalize_row_hybrid
    raise ValueError("template must be 'structured', 'structured_short_plus', 'bucket_only_semantic', or 'hybrid'")


def _rag_row_to_prose(rag_row: pd.Series, label_encoders: Optional[dict] = None) -> str:
    """Format RAG neighbor stats as one short prose sentence (structured_short_plus style)."""
    parts = []
    # Income (semantic)
    key = "rag_mean_median_income"
    if key in rag_row and not pd.isna(rag_row[key]):
        inc = float(rag_row[key])
        bucket = _income_bucket(inc)
        parts.append(_income_bucket_prose(bucket))
    if not parts and "rag_mode_income_bucket" in rag_row and not pd.isna(rag_row.get("rag_mode_income_bucket")):
        v = rag_row["rag_mode_income_bucket"]
        if label_encoders and "income_bucket" in label_encoders:
            try:
                v = label_encoders["income_bucket"].inverse_transform([int(v)])[0]
            except (ValueError, IndexError):
                pass
        parts.append(_income_bucket_prose(v))
    # Age (semantic)
    key = "rag_mean_housing_median_age"
    if key in rag_row and not pd.isna(rag_row[key]):
        age = float(rag_row[key])
        parts.append(_age_bucket_prose(_age_bucket(age)) + " housing")
    # Rooms per household + density
    rph_key = "rag_mean_rooms_per_household"
    pph_key = "rag_mean_population_per_household"
    if rph_key in rag_row and not pd.isna(rag_row[rph_key]):
        rph = float(rag_row[rph_key])
        parts.append(f"{rph:.1f} rooms per household ({_rooms_prose(rph).replace(' homes', '')})")
    if pph_key in rag_row and not pd.isna(rag_row[pph_key]):
        pph = float(rag_row[pph_key])
        parts.append(_density_semantic(pph))
    # Geo
    gc_key = "rag_mean_geo_cluster"
    if gc_key in rag_row and not pd.isna(rag_row[gc_key]):
        parts.append(f"geo cluster {int(rag_row[gc_key])}")
    if not parts:
        # Fallback: short list of key rag means (structured_short_plus style)
        fallback = []
        for col in ["median_income", "rooms_per_household", "housing_median_age", "geo_cluster"]:
            k = f"rag_mean_{col}"
            if k in rag_row and not pd.isna(rag_row[k]):
                v = rag_row[k]
                fallback.append(f"{col} {float(v):.1f}" if col != "geo_cluster" else f"geo cluster {int(v)}")
        if fallback:
            return "Similar nearby blocks: " + ", ".join(fallback) + "."
        return ""
    return "Similar nearby blocks average " + ", ".join(parts) + "."


def build_texts_with_rag(
    df: pd.DataFrame,
    rag_df: pd.DataFrame,
    template: str = "structured",
    label_encoders: Optional[dict] = None
) -> List[str]:
    base_fn = _base_fn_for_template(template)
    texts = []
    for idx, row in df.iterrows():
        base = base_fn(row)
        if rag_df is not None and idx in rag_df.index:
            rag_row = rag_df.loc[idx]
            prose = _rag_row_to_prose(rag_row, label_encoders)
            if prose:
                base = base + " " + prose
        texts.append(base)
    return texts


def _texts_fingerprint(texts: List[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def embed_texts_openai_full_cache(
    all_texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    normalize: bool = True
) -> np.ndarray:
    """Embed all texts once and cache as single full array. Uses per-text cache as fallback."""
    cache_root = Path(os.getenv("OPENAI_EMBEDDING_CACHE_DIR", "results/embedding_cache"))
    full_cache_dir = cache_root / model / "full"
    full_cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _texts_fingerprint(all_texts)
    full_path = full_cache_dir / f"{fp}.npy"

    if full_path.exists():
        emb = np.load(full_path)
        print(f"    Loaded full embedding cache: {full_path} ({emb.shape[0]} rows, {emb.shape[1]} dims)")
        return emb

    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    per_text_dir = cache_root / model
    per_text_dir.mkdir(parents=True, exist_ok=True)
    embeddings = [None] * len(all_texts)
    missing_idx = []

    for i, t in enumerate(all_texts):
        key = hashlib.sha256(f"{model}|{t}".encode("utf-8")).hexdigest()
        p = per_text_dir / f"{key}.npy"
        if p.exists():
            embeddings[i] = np.load(p)
        else:
            missing_idx.append(i)

    if missing_idx:
        for s in range(0, len(missing_idx), batch_size):
            batch_ids = missing_idx[s:s + batch_size]
            batch_texts = [all_texts[i] for i in batch_ids]
            resp = client.embeddings.create(model=model, input=batch_texts)
            for j, item in enumerate(resp.data):
                emb = np.array(item.embedding, dtype=np.float32)
                idx = batch_ids[j]
                embeddings[idx] = emb
                np.save(per_text_dir / f"{hashlib.sha256(f'{model}|{all_texts[idx]}'.encode()).hexdigest()}.npy", emb)
            time.sleep(0.05)

    emb_dim = 1536
    for e in embeddings:
        if e is not None:
            emb_dim = len(e)
            break
    embeddings = [e if e is not None else np.zeros(emb_dim, dtype=np.float32) for e in embeddings]
    emb = np.array(embeddings, dtype=np.float32)
    if normalize:
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    np.save(full_path, emb)
    print(f"    Saved full embedding cache: {full_path}")
    return emb


def _embedding_store_dir(
    mode: str,
    template: str,
    model: str,
    pca_dim: int,
    seed: int,
    fold_idx: Optional[int] = None,
) -> Path:
    base = Path("cache") / "embedding_pca" / mode / f"model_{model}" / f"template_{template}" / f"pca_{pca_dim}" / f"seed_{seed}"
    if fold_idx is not None:
        base = base / f"fold_{fold_idx}"
    return base


def _try_load_pca_cache(
    cache_dir: Path,
    full_fingerprint: str,
    pca_dim: int,
    seed: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    meta_path = cache_dir / "meta.json"
    train_path = cache_dir / "train.npy"
    val_path = cache_dir / "val.npy"
    test_path = cache_dir / "test.npy"
    if not (meta_path.exists() and train_path.exists() and val_path.exists() and test_path.exists()):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("pca_dim") != pca_dim or meta.get("seed") != seed or meta.get("full_fingerprint") != full_fingerprint:
            return None
        train_emb = np.load(train_path)
        val_emb = np.load(val_path)
        test_emb = np.load(test_path)
        return train_emb, val_emb, test_emb
    except Exception:
        return None


def _save_pca_cache(
    cache_dir: Path,
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    test_emb: np.ndarray,
    pca_model: PCA,
    full_fingerprint: str,
    pca_dim: int,
    seed: int
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train.npy", train_emb)
    np.save(cache_dir / "val.npy", val_emb)
    np.save(cache_dir / "test.npy", test_emb)
    with open(cache_dir / "pca.pkl", "wb") as f:
        pickle.dump(pca_model, f)
    meta = {"pca_dim": pca_dim, "seed": seed, "full_fingerprint": full_fingerprint}
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def fit_transform_pca_three(
    train_emb: np.ndarray, val_emb: np.ndarray, test_emb: np.ndarray, pca_dim: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=pca_dim, random_state=seed)
    train_pca = pca.fit_transform(train_emb)
    val_pca = pca.transform(val_emb)
    test_pca = pca.transform(test_emb)
    return train_pca, val_pca, test_pca, pca


def _safe_le_transform(le: LabelEncoder, vals: np.ndarray) -> np.ndarray:
    """Transform with LabelEncoder; unseen labels map to -1."""
    return np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in vals])


def _label_encode_three(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    label_encoders: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply label encoding to categorical columns. Uses pre-fitted label_encoders. Val/test use safe transform for unseen labels."""
    cat_cols = [c for c in label_encoders.keys() if c in train_df.columns]
    if len(cat_cols) == 0:
        return train_df.copy(), val_df.copy(), test_df.copy()
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    for col in cat_cols:
        le = label_encoders[col]
        train_vals = train_df[col].astype(str).fillna("None").values
        val_vals = val_df[col].astype(str).fillna("None").values
        test_vals = test_df[col].astype(str).fillna("None").values
        train_out[col] = le.transform(train_vals)
        val_out[col] = _safe_le_transform(le, val_vals)
        test_out[col] = _safe_le_transform(le, test_vals)
    return train_out, val_out, test_out


def _compute_rag_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    numeric_cols: List[str],
    cat_cols: List[str],
    label_encoders: dict,
    val_df: Optional[pd.DataFrame] = None,
    rag_mode: str = "geo",
    train_target: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    empty_val = pd.DataFrame(index=val_df.index) if val_df is not None else None
    if k <= 0:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index), empty_val

    numeric_cols = [c for c in numeric_cols if c in train_df.columns]
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    cfg = RAG_MODE_CONFIG.get(rag_mode, RAG_MODE_CONFIG["geo"])
    search_numeric = [c for c in cfg["search_numeric"] if c in train_df.columns]
    search_cat = [c for c in cfg["search_cat"] if c in train_df.columns]
    agg_numeric_cols = numeric_cols  # 输出特征仍用全部数值列聚合

    # Build search matrix: numeric (scaled) + cat (label-encoded then scaled)
    parts_train, parts_test, parts_val = [], [], []
    if search_numeric:
        scaler_n = StandardScaler()
        tr_n = scaler_n.fit_transform(train_df[search_numeric].fillna(0).values)
        te_n = scaler_n.transform(test_df[search_numeric].fillna(0).values)
        parts_train.append(tr_n)
        parts_test.append(te_n)
        if val_df is not None:
            parts_val.append(scaler_n.transform(val_df[search_numeric].fillna(0).values))
    if search_cat:
        tr_c = np.column_stack([
            label_encoders[c].transform(train_df[c].astype(str).fillna("None").values)
            for c in search_cat if c in label_encoders
        ])
        te_c = np.column_stack([
            _safe_le_transform(label_encoders[c], test_df[c].astype(str).fillna("None").values)
            for c in search_cat if c in label_encoders
        ])
        scaler_c = StandardScaler()
        parts_train.append(scaler_c.fit_transform(tr_c))
        parts_test.append(scaler_c.transform(te_c))
        if val_df is not None:
            va_c = np.column_stack([
                _safe_le_transform(label_encoders[c], val_df[c].astype(str).fillna("None").values)
                for c in search_cat if c in label_encoders
            ])
            parts_val.append(scaler_c.transform(va_c))
    if not parts_train:
        raise ValueError("No search columns available for RAG similarity.")

    train_scaled = np.hstack(parts_train)
    test_scaled = np.hstack(parts_test)
    val_scaled = np.hstack(parts_val) if val_df is not None and parts_val else None

    from sklearn.neighbors import NearestNeighbors

    k_train = min(k + 1, len(train_df))
    nn = NearestNeighbors(n_neighbors=k_train, metric="euclidean", n_jobs=-1)
    nn.fit(train_scaled)

    train_nei = nn.kneighbors(train_scaled, return_distance=False)
    test_k = min(k, len(train_df))
    test_nei = nn.kneighbors(test_scaled, n_neighbors=test_k, return_distance=False)
    val_nei = nn.kneighbors(val_scaled, n_neighbors=test_k, return_distance=False) if val_scaled is not None else None

    def _neighbors_for_train_row(i: int) -> np.ndarray:
        idxs = train_nei[i].tolist()
        if i in idxs:
            idxs.remove(i)
        if len(idxs) > k:
            idxs = idxs[:k]
        return np.array(idxs, dtype=int)

    def _rag_row_from_neighbors(nei_idx: np.ndarray) -> dict:
        row = {}
        for col in agg_numeric_cols:
            vals = train_df.iloc[nei_idx][col].dropna().values
            if len(vals) > 0:
                row[f"rag_mean_{col}"] = float(np.mean(vals))
        for col in cat_cols:
            vals = train_df.iloc[nei_idx][col].dropna().astype(str).values
            if len(vals) > 0:
                mode_val = pd.Series(vals).mode(dropna=True)
                if not mode_val.empty:
                    mode_str = mode_val.iloc[0]
                    if col in label_encoders:
                        le = label_encoders[col]
                        label_map = {label: idx for idx, label in enumerate(le.classes_)}
                        default_label = "None" if "None" in label_map else le.classes_[0]
                        row[f"rag_mode_{col}"] = label_map.get(mode_str, label_map[default_label])
                    else:
                        row[f"rag_mode_{col}"] = mode_str
        if train_target is not None and len(nei_idx) > 0:
            row["neighbour_price"] = float(np.mean(train_target[nei_idx]))
        return row

    train_rows = []
    for i in range(len(train_df)):
        nei_idx = _neighbors_for_train_row(i)
        if len(nei_idx) == 0:
            train_rows.append({})
        else:
            train_rows.append(_rag_row_from_neighbors(nei_idx))

    test_rows = []
    for i in range(len(test_df)):
        nei_idx = test_nei[i]
        if len(nei_idx) == 0:
            test_rows.append({})
        else:
            test_rows.append(_rag_row_from_neighbors(nei_idx))

    val_rows = []
    if val_nei is not None and val_df is not None:
        for i in range(len(val_df)):
            nei_idx = val_nei[i]
            if len(nei_idx) == 0:
                val_rows.append({})
            else:
                val_rows.append(_rag_row_from_neighbors(nei_idx))
        val_rag = pd.DataFrame(val_rows, index=val_df.index)
    else:
        val_rag = None

    return pd.DataFrame(train_rows, index=train_df.index), pd.DataFrame(test_rows, index=test_df.index), val_rag


# FNI: Feature-Neighborhood Interaction
FNI_K = 5
FNI_BINARY_EXCLUDE: List[str] = []  # California Housing 无 binary 列
FNI_CORE_COLS = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "log_median_income", "rooms_per_population", "other_rooms",
    "distance_to_coast", "avg_household_size", "geo_cluster",
]
FNI_MIN_CORR = 0.15
FNI_WEIGHTED = True
FNI_SCALE_OUTPUT = True
FNI_SQRT_STABILIZE = True
# Target-aware 邻接、仅正相关、输入裁剪
FNI_TARGET_AWARE = True   # 传入 y_train 时按 target 相关度调整邻接权重
FNI_TARGET_PRIORITY = 0.3  # 邻接选取时混入 corr(j,y)，对 target 更相关的特征优先
FNI_POSITIVE_CORR_ONLY = True  # 乘积仅用 corr>0 邻接，负相关易引入噪声
FNI_CLIP_INPUT = 3.0     # 乘积前裁剪到 ±clip，减少极端值影响；0=不裁剪
FNI_ROBUST_SCALE = True  # 输出用 RobustScaler（对异常值更鲁棒）


def compute_fni_features(
    train_mat: np.ndarray,
    val_mat: np.ndarray,
    test_mat: np.ndarray,
    k: int = FNI_K,
    exclude_col_idx: Optional[List[int]] = None,
    min_corr: float = FNI_MIN_CORR,
    weighted: bool = FNI_WEIGHTED,
    scale_output: bool = FNI_SCALE_OUTPUT,
    sqrt_stabilize: bool = FNI_SQRT_STABILIZE,
    y_train: Optional[np.ndarray] = None,
    target_aware: bool = FNI_TARGET_AWARE,
    target_priority: float = FNI_TARGET_PRIORITY,
    positive_corr_only: bool = FNI_POSITIVE_CORR_ONLY,
    clip_input: float = FNI_CLIP_INPUT,
    robust_scale: bool = FNI_ROBUST_SCALE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature-Neighborhood Interaction: target-aware、邻接补齐、RobustScaler。"""
    train_mat = np.asarray(train_mat, dtype=np.float64)
    val_mat = np.asarray(val_mat, dtype=np.float64)
    test_mat = np.asarray(test_mat, dtype=np.float64)
    if exclude_col_idx:
        keep = [i for i in range(train_mat.shape[1]) if i not in exclude_col_idx]
        train_mat = train_mat[:, keep]
        val_mat = val_mat[:, keep]
        test_mat = test_mat[:, keep]
    train_mat = np.nan_to_num(train_mat, nan=0.0, posinf=0.0, neginf=0.0)
    val_mat = np.nan_to_num(val_mat, nan=0.0, posinf=0.0, neginf=0.0)
    test_mat = np.nan_to_num(test_mat, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_input > 0:
        train_mat = np.clip(train_mat, -clip_input, clip_input)
        val_mat = np.clip(val_mat, -clip_input, clip_input)
        test_mat = np.clip(test_mat, -clip_input, clip_input)

    n_feat = train_mat.shape[1]
    if n_feat < 2:
        return (
            np.empty((train_mat.shape[0], 0), dtype=np.float64),
            np.empty((val_mat.shape[0], 0), dtype=np.float64),
            np.empty((test_mat.shape[0], 0), dtype=np.float64),
        )

    corr = np.corrcoef(train_mat.T)
    np.nan_to_num(corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, -2)

    # target-aware: corr(j, y) 用于调整邻接排序/权重
    corr_y = None
    if target_aware and y_train is not None and len(y_train) == train_mat.shape[0]:
        y = np.asarray(y_train, dtype=np.float64).ravel()
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        corr_y = np.corrcoef(np.column_stack([train_mat, y.reshape(-1, 1)]).T)[-1, :-1]

    k_actual = min(k, n_feat - 1)
    if k_actual <= 0:
        return (
            np.empty((train_mat.shape[0], 0), dtype=np.float64),
            np.empty((val_mat.shape[0], 0), dtype=np.float64),
            np.empty((test_mat.shape[0], 0), dtype=np.float64),
        )

    neighbor_idx: List[List[int]] = []
    neighbor_weights: List[np.ndarray] = []
    for i in range(n_feat):
        sim = np.abs(corr[i])
        raw_corr = corr[i].copy()
        if positive_corr_only:
            candidate = [j for j in range(n_feat) if j != i and raw_corr[j] > min_corr]
        else:
            candidate = [j for j in range(n_feat) if j != i and sim[j] >= min_corr]
        if not candidate:
            other = [j for j in np.argsort(sim)[::-1] if j != i]
            candidate = [other[0]] if other else [0]
        # target_priority: 邻接选取时混入 corr_y，使对 target 更相关的特征优先
        if corr_y is not None and target_priority > 0:
            corr_y_abs = np.abs(corr_y)
            score = lambda j: sim[j] + target_priority * corr_y_abs[j]
            topk = sorted(candidate, key=score, reverse=True)[:k_actual]
        else:
            topk = sorted(candidate, key=lambda j: sim[j], reverse=True)[:k_actual]
        # 当 candidate 不足 k_actual 时，用剩余特征按 |corr| 补齐，避免 k=6,7,8 结果相同
        if len(topk) < k_actual:
            used = set(topk)
            rest = [j for j in np.argsort(sim)[::-1] if j != i and j not in used]
            topk = topk + rest[: k_actual - len(topk)]
        nei = topk

        w = np.array([max(0.1, sim[j]) for j in nei], dtype=np.float64)
        if corr_y is not None:
            for idx, j in enumerate(nei):
                target_bonus = 0.5 + 0.5 * min(1.0, abs(corr_y[j]))
                w[idx] *= target_bonus
        neighbor_idx.append(nei)
        neighbor_weights.append(w)

    def _stabilize(x: np.ndarray) -> np.ndarray:
        if not sqrt_stabilize:
            return x
        return np.sign(x) * np.sqrt(np.abs(x) + 1.0)

    def _fni_for_matrix(mat: np.ndarray) -> np.ndarray:
        n_samples = mat.shape[0]
        out_cols = []
        for i in range(n_feat):
            x_i = mat[:, i]
            nei = neighbor_idx[i]
            w = neighbor_weights[i]
            if not nei:
                continue
            mult_vals = np.column_stack([x_i * mat[:, j] for j in nei])
            if weighted:
                w_sum = w.sum()
                w_norm = w / (w_sum + 1e-10) if w_sum > 0 else np.ones_like(w) / len(w)
                mult_agg = (mult_vals * w_norm).sum(axis=1)
            else:
                mult_agg = np.nanmean(mult_vals, axis=1)
            mult_agg = _stabilize(mult_agg)
            out_cols.append(mult_agg)
        if not out_cols:
            return np.empty((n_samples, 0))
        return np.column_stack(out_cols)

    train_fni = _fni_for_matrix(train_mat)
    val_fni = _fni_for_matrix(val_mat)
    test_fni = _fni_for_matrix(test_mat)

    if scale_output and train_fni.shape[1] > 0:
        scaler = RobustScaler() if robust_scale else StandardScaler()
        train_fni = scaler.fit_transform(train_fni)
        val_fni = scaler.transform(val_fni)
        test_fni = scaler.transform(test_fni)

    return train_fni, val_fni, test_fni


def _train_eval_and_predict(
    label: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    test_ids: pd.Series,
    seed: int,
    models_to_run: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, dict, dict, dict]:
    """Trains on log1p(y), validates/predicts in original scale. MSE on log minimizes RMSLE.
    When models_to_run is set (e.g. --best-config), only run those models."""
    y_train_log = np.log1p(np.maximum(y_train, 0))
    y_val_log = np.log1p(np.maximum(y_val, 0))
    y_test_log = np.log1p(np.maximum(y_test, 0))

    def _to_original(pred_log):
        return np.expm1(np.clip(np.asarray(pred_log, dtype=float), -20, 20))

    val_metrics = {}
    test_metrics = {}
    per_model_preds = {}
    first_pred_test = None

    if USE_CATBOOST and (models_to_run is None or "catboost" in models_to_run):
        cat = CatBoostRegressor(
            depth=10, learning_rate=0.024229331857231032, l2_leaf_reg=0.0010719864040714892, iterations=2870,
            random_seed=seed, verbose=False
        )
        cat.fit(X_train, y_train_log)
        pred_val = _to_original(cat.predict(X_val))
        pred_test = _to_original(cat.predict(X_test))
        val_metrics["catboost"] = evaluate_model(y_val, pred_val, f"CatBoost (val, {label})")
        test_metrics["catboost"] = evaluate_model(y_test, pred_test, f"CatBoost (test, {label})")
        per_model_preds["catboost"] = pd.Series(pred_test, index=test_ids)
        if first_pred_test is None:
            first_pred_test = pred_test

    if USE_XGB and (models_to_run is None or "xgb" in models_to_run):
        xgb = XGBRegressor(
            n_estimators=1957, max_depth=8, learning_rate=0.012949573419995286,
            subsample=0.8776284325308382, colsample_bytree=0.735930364666962,
            reg_alpha=0.008583519739639377, reg_lambda=0.011775061448687748,
            random_state=seed, n_jobs=-1
        )
        xgb.fit(X_train, y_train_log)
        pred_val = _to_original(xgb.predict(X_val))
        pred_test = _to_original(xgb.predict(X_test))
        val_metrics["xgb"] = evaluate_model(y_val, pred_val, f"XGBoost (val, {label})")
        test_metrics["xgb"] = evaluate_model(y_test, pred_test, f"XGBoost (test, {label})")
        per_model_preds["xgb"] = pd.Series(pred_test, index=test_ids)
        if first_pred_test is None:
            first_pred_test = pred_test

    if USE_LGBM and (models_to_run is None or "lgbm" in models_to_run):
        lgbm = LGBMRegressor(
            objective="regression", num_leaves=54, learning_rate=0.028558725396760045, n_estimators=1129,
            max_bin=169, bagging_fraction=0.9278314148139136, bagging_freq=2, feature_fraction=0.8009940160028424,
            random_state=seed, verbose=-1
        )
        lgbm.fit(X_train, y_train_log)
        pred_val = _to_original(lgbm.predict(X_val))
        pred_test = _to_original(lgbm.predict(X_test))
        val_metrics["lgbm"] = evaluate_model(y_val, pred_val, f"LightGBM (val, {label})")
        test_metrics["lgbm"] = evaluate_model(y_test, pred_test, f"LightGBM (test, {label})")
        per_model_preds["lgbm"] = pd.Series(pred_test, index=test_ids)
        if first_pred_test is None:
            first_pred_test = pred_test

    if USE_TABPFN and (models_to_run is None or "tabpfn" in models_to_run):
        if not TABPFN_AVAILABLE:
            print("\n[WARN] TabPFN not available; skipping TabPFN.")
        else:
            os.environ["TABPFN_DEVICE"] = "cuda"
            os.environ["TABPFN_ALLOW_LARGE_DATASETS"] = "1"
            tabpfn = TabPFNRegressor(device="cuda", n_estimators=16, random_state=seed)
            tabpfn.fit(X_train, y_train_log)
            pred_val = _to_original(tabpfn.predict(X_val))
            pred_test = _to_original(tabpfn.predict(X_test))
            val_metrics["tabpfn"] = evaluate_model(y_val, pred_val, f"TabPFN (val, {label})")
            test_metrics["tabpfn"] = evaluate_model(y_test, pred_test, f"TabPFN (test, {label})")
            per_model_preds["tabpfn"] = pd.Series(pred_test, index=test_ids)
            if first_pred_test is None:
                first_pred_test = pred_test

    if len(val_metrics) == 0:
        raise ValueError("No learner enabled. Set one of USE_CATBOOST/USE_XGB/USE_LGBM = True")

    pred_for_sub = first_pred_test if first_pred_test is not None else list(per_model_preds.values())[0].values
    sub_df = pd.DataFrame({"index": test_ids, "median_house_value": pred_for_sub})
    return sub_df, per_model_preds, val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(description="California Housing: 5-fold CV, median_house_value target, full embedding cache")
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--data-ratio", "--data_ratio", dest="data_ratio", type=float, default=1.0,
                        help="数据采样比例 (0,1]，用于快速测试；1.0=全量")
    parser.add_argument("--seed", type=str, default=str(SEED), help="多个 seed，逗号分隔，如 42,43,44")
    parser.add_argument("--openai-emb-model", type=str, default=OPENAI_EMB_MODEL)
    parser.add_argument("--openai-template", type=str, default=OPENAI_TEMPLATE,
                        choices=["structured", "structured_short_plus", "bucket_only_semantic", "hybrid"],
                        help="structured_short_plus=短结构+区位/市场/密度(默认); hybrid=数值+语义; bucket_only_semantic=纯语义; structured=四段式")
    parser.add_argument("--pca-dim", type=str, default=str(PCA_DIM), help="PCA dim(s), comma-separated: 8,12,16")
    parser.add_argument("--rag-pca-dim", type=str, default=str(PCA_DIM), help="RAG PCA dim(s): 8,12,16")
    parser.add_argument("--rag-k", "--rag_k", dest="rag_k", type=str, default="8", help="多个 RAG K，逗号分隔，如 4,8,12,16")
    parser.add_argument("--rag-mode", "--rag_mode", dest="rag_mode", type=str, default="geo",
                        choices=list(RAG_MODE_CONFIG.keys()),
                        help="RAG 邻居模式: geo=仅地理(默认), hybrid=全部数值")
    parser.add_argument("--fni-k", "--fni_k", dest="fni_k", type=str, default=str(FNI_K), help="多个 FNI K，逗号分隔，如 3,5,8")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline,emb,rag",
        help="Comma-separated: baseline,emb,rag,rag_price,fni,fni+rag,fni_with_rag,fni_dual,emb+rag,emb_with_rag,emb_with_rag+rag (or 'all' = no fni)"
    )
    parser.add_argument(
        "--best-config",
        action="store_true",
        dest="best_config",
        help="Use fixed best config per model; only record each model at its best config"
    )
    parser.add_argument(
        "--no-tabpfn",
        "--no_tabpfn",
        dest="no_tabpfn",
        action="store_true",
        help="禁用 TabPFN 模型（无 GPU 或节省显存时使用）",
    )
    parser.add_argument("--output-csv", "-o", dest="output_csv", type=str, default="california_sweep_results.csv",
                        help="结果保存的 CSV 路径（默认 california_sweep_results.csv）")
    args = parser.parse_args()

    if getattr(args, "no_tabpfn", False):
        globals()["USE_TABPFN"] = False
        print("[--no-tabpfn] TabPFN 已禁用")

    pca_dims = _parse_int_list(args.pca_dim, [PCA_DIM])
    rag_pca_dims = _parse_int_list(args.rag_pca_dim, [PCA_DIM])
    seeds = _parse_int_list(args.seed, [SEED])
    best_config = getattr(args, "best_config", False)
    if best_config:
        combos = sorted(set((c["pca_dim"], c["rag_k"], c["rag_pca_dim"]) for c in BEST_CONFIG_PER_MODEL.values()))
        print(f"\n[--best-config] Using fixed config per model: {list(BEST_CONFIG_PER_MODEL.items())}")
    else:
        rag_ks = _parse_int_list(args.rag_k, [8])
        combos = list(product(pca_dims, rag_ks, rag_pca_dims))

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Missing: {args.data_path}")
    df_full = pd.read_csv(args.data_path)
    df = df_full.copy()
    if 0 < getattr(args, "data_ratio", 1.0) < 1.0:
        ratio = args.data_ratio
        n = int(len(df) * ratio)
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
        print(f"  采样: {n} 条 ({ratio*100:.0f}%)")
    n = len(df)

    modes = {m.strip().lower() for m in args.mode.split(",") if m.strip()}
    if "all" in modes:
        modes = {"baseline", "emb", "rag", "rag_price", "emb+rag", "emb_with_rag", "emb_with_rag+rag"}
        print("\n[mode=all] Running all modes (no fni): baseline, emb, rag, rag_price, emb+rag, emb_with_rag, emb_with_rag+rag")

    fni_ks = _parse_int_list(getattr(args, "fni_k", str(FNI_K)), [FNI_K])

    if best_config:
        print(f"\n[参数] seeds={seeds}  rag_k=best_config  fni_k={fni_ks}  combos={len(combos)} 组")
    else:
        print(f"\n[参数] seeds={seeds}  rag_k={rag_ks}  fni_k={fni_ks}  combos={len(combos)} 组")

    results = []
    if not OPENAI_AVAILABLE and any(m in modes for m in ["emb", "emb+rag", "emb_with_rag", "emb_with_rag+rag"]):
        print("\n[WARN] OpenAI not available; skipping embedding modes.")

    for seed in seeds:
        set_seed(seed)
        all_indices = np.arange(n)
        # 与 Ames 一致: 70% train, 10% val, 20% test；先划分，再在 train 上 fit，避免泄露
        train_idx, rest = train_test_split(all_indices, test_size=0.3, random_state=seed)
        val_idx, test_idx = train_test_split(rest, test_size=2.0 / 3.0, random_state=seed)
        fold_idx = 0

        train_raw = df.iloc[train_idx].copy()
        val_raw = df.iloc[val_idx].copy()
        test_raw = df.iloc[test_idx].copy()
        train_clean, val_clean, test_clean = clean_data_fit_transform(train_raw, val_raw, test_raw)
        train_fe, val_fe, test_fe = feature_engineering_fit_transform(
            train_clean, val_clean, test_clean,
            geo_cluster_k=GEO_CLUSTER_K,
            random_state=seed,
        )

        # LabelEncoder 仅 fit 在 train 上
        cat_cols = [c for c in train_fe.select_dtypes(include=["object", "category"]).columns if c in train_fe.columns]
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(train_fe[col].astype(str).fillna("None").values)
            label_encoders[col] = le

        train_df = train_fe.copy()
        val_df = val_fe.copy()
        test_df = test_fe.copy()
        y_train = train_fe["median_house_value"].values
        y_val = val_fe["median_house_value"].values
        y_test = test_fe["median_house_value"].values
        y_train_log = np.log1p(np.maximum(y_train, 0))
        y_val_log = np.log1p(np.maximum(y_val, 0))
        y_test_log = np.log1p(np.maximum(y_test, 0))

        numeric_cols_list = [c for c in train_fe.select_dtypes(include=[np.number]).columns if c != "median_house_value"]
        train_numeric = train_fe[numeric_cols_list].reset_index(drop=True)
        val_numeric = val_fe[numeric_cols_list].reset_index(drop=True)
        test_numeric = test_fe[numeric_cols_list].reset_index(drop=True)
        def _safe_transform(le, vals):
            return np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in vals])

        train_cat = pd.DataFrame({
            col: label_encoders[col].transform(train_fe[col].astype(str).fillna("None").values)
            for col in cat_cols if col in label_encoders
        }, columns=[c for c in cat_cols if c in label_encoders])
        val_cat = pd.DataFrame({
            col: _safe_transform(label_encoders[col], val_fe[col].astype(str).fillna("None").values)
            for col in cat_cols if col in label_encoders
        }, columns=[c for c in cat_cols if c in label_encoders])
        test_cat = pd.DataFrame({
            col: _safe_transform(label_encoders[col], test_fe[col].astype(str).fillna("None").values)
            for col in cat_cols if col in label_encoders
        }, columns=[c for c in cat_cols if c in label_encoders])

        # 保留原始类别值，供 _label_encode_three 统一编码（baseline 等）
        X_train_base = train_fe.drop(columns=["median_house_value"]).copy()
        X_val_base = val_fe.drop(columns=["median_house_value"]).copy()
        X_test_base = test_fe.drop(columns=["median_house_value"]).copy()

        # Embedding：在当次划分后的 train/val/test 上构建（顺序 train, val, test）
        full_emb = None
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
        if OPENAI_AVAILABLE and any(m in modes for m in ["emb", "emb+rag", "emb_with_rag", "emb_with_rag+rag"]):
            all_texts = list(build_texts(train_df, template=args.openai_template)) + list(build_texts(val_df, template=args.openai_template)) + list(build_texts(test_df, template=args.openai_template))
            full_emb = embed_texts_openai_full_cache(all_texts, model=args.openai_emb_model)

        print(f"\n{'='*60}\nseed={seed}  70% train / 10% val / 20% test: train={n_train}, val={n_val}, test={n_test} (fit on train only)")

        test_ids = df_full.index[test_idx].copy()
        # 供 RAG/embedding 使用的全表（顺序 train, val, test）
        df_for_text = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # FNI 预计算（fni / fni+rag 需要 fni_by_k；fni_with_rag 需要 train_num_s）
        fni_by_k: dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        train_num_s = val_num_s = test_num_s = None
        if "fni" in modes or "fni+rag" in modes or "fni_with_rag" in modes or "fni_dual" in modes:
            num_cols = train_numeric.columns.tolist()
            scaler_num = StandardScaler()
            train_num_s = scaler_num.fit_transform(train_numeric.fillna(0).values)
            val_num_s = scaler_num.transform(val_numeric.fillna(0).values)
            test_num_s = scaler_num.transform(test_numeric.fillna(0).values)
        if "fni" in modes or "fni+rag" in modes:
            if FNI_CORE_COLS:
                fni_include_idx = [i for i, c in enumerate(num_cols) if c in FNI_CORE_COLS]
                _train = train_num_s[:, fni_include_idx]
                _val = val_num_s[:, fni_include_idx]
                _test = test_num_s[:, fni_include_idx]
            else:
                _train, _val, _test = train_num_s, val_num_s, test_num_s
            for fk in fni_ks:
                fni_by_k[fk] = compute_fni_features(_train, _val, _test, k=fk, exclude_col_idx=None, y_train=y_train_log)
            if FNI_CORE_COLS:
                _core = [c for c in FNI_CORE_COLS if c in num_cols]
                print(f"    FNI 核心特征 ({len(_core)} 列): {_core}")

        # Baseline
        if "baseline" in modes:
            X_tr_oh, X_va_oh, X_te_oh = _label_encode_three(X_train_base, X_val_base, X_test_base, label_encoders)
            print("\n" + "=" * 60)
            print("Baseline (NO OpenAI Embedding)")
            print("=" * 60)
            sub_b, _, val_m, test_m = _train_eval_and_predict(
                "baseline", X_tr_oh, X_va_oh, X_te_oh, y_train, y_val, y_test, test_ids, seed
            )
            for name, m in val_m.items():
                t = test_m.get(name, {})
                results.append((seed, fold_idx, None, None, None, None, "baseline", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))
            sub_b.to_csv("california_submission_baseline.csv", index=False)
            print("Saved: california_submission_baseline.csv")

        # FNI mode（独立于 combo，每个 fni_k 运行一次）
        if "fni" in modes and fni_by_k and train_num_s is not None:
            for fk in fni_ks:
                if fk not in fni_by_k:
                    continue
                ftr, fva, fte = fni_by_k[fk]
                train_num_df = pd.DataFrame(train_num_s, columns=train_numeric.columns)
                val_num_df = pd.DataFrame(val_num_s, columns=val_numeric.columns)
                test_num_df = pd.DataFrame(test_num_s, columns=test_numeric.columns)
                fni_cols = [f"fni_{i}" for i in range(ftr.shape[1])]
                X_tr_f = pd.concat([
                    train_num_df.reset_index(drop=True),
                    train_cat.reset_index(drop=True),
                    pd.DataFrame(ftr, columns=fni_cols)
                ], axis=1)
                X_va_f = pd.concat([
                    val_num_df.reset_index(drop=True),
                    val_cat.reset_index(drop=True),
                    pd.DataFrame(fva, columns=fni_cols)
                ], axis=1)
                X_te_f = pd.concat([
                    test_num_df.reset_index(drop=True),
                    test_cat.reset_index(drop=True),
                    pd.DataFrame(fte, columns=fni_cols)
                ], axis=1)
                print(f"\n" + "=" * 60)
                print(f"FNI k={fk}")
                print("=" * 60)
                _, _, val_m, test_m = _train_eval_and_predict("fni", X_tr_f, X_va_f, X_te_f, y_train, y_val, y_test, test_ids, seed)
                for name, m in val_m.items():
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, None, None, None, fk, "fni", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

        # Parameter sweep
        for pca_dim, rag_k, rag_pca_dim in combos:
            tag = f"p{pca_dim}_k{rag_k}_rp{rag_pca_dim}"
            if best_config:
                combo_modes = [m for m in ["rag", "rag_price", "emb", "emb+rag", "emb_with_rag", "emb_with_rag+rag", "fni", "fni+rag", "fni_with_rag", "fni_dual"] if m in modes]
                models_to_run = [n for n in ["catboost", "xgb", "lgbm", "tabpfn"] if any(_should_record_model(True, m, n, pca_dim, rag_k, rag_pca_dim) for m in combo_modes)]
            else:
                models_to_run = None

            rag_mode = getattr(args, "rag_mode", "geo")
            rag_tr, rag_te, rag_va = _compute_rag_features(
                train_df, test_df, rag_k, RAG_IMPORTANT_NUMERIC_COLS, RAG_IMPORTANT_CAT_COLS, label_encoders, val_df, rag_mode=rag_mode, train_target=y_train
            )

            train_emb_df = val_emb_df = test_emb_df = None
            if full_emb is not None and ("emb" in modes or "emb+rag" in modes):
                train_emb_raw = full_emb[:n_train]
                val_emb_raw = full_emb[n_train : n_train + n_val]
                test_emb_raw = full_emb[n_train + n_val :]
                cache_dir = _embedding_store_dir("emb", args.openai_template, args.openai_emb_model, pca_dim, seed, fold_idx)
                fp = _texts_fingerprint(all_texts)
                cached = _try_load_pca_cache(cache_dir, fp, pca_dim, seed)
                if cached:
                    train_emb_pca, val_emb_pca, test_emb_pca = cached
                    if len(val_emb_pca) != len(val_idx):
                        cached = None
                if not cached:
                    train_emb_pca, val_emb_pca, test_emb_pca, pca_m = fit_transform_pca_three(train_emb_raw, val_emb_raw, test_emb_raw, pca_dim, seed)
                    _save_pca_cache(cache_dir, train_emb_pca, val_emb_pca, test_emb_pca, pca_m, fp, pca_dim, seed)
                emb_cols = [f"emb_{i}" for i in range(pca_dim)]
                train_emb_df = pd.DataFrame(train_emb_pca, columns=emb_cols)
                val_emb_df = pd.DataFrame(val_emb_pca, columns=emb_cols)
                test_emb_df = pd.DataFrame(test_emb_pca, columns=emb_cols)

            train_emb_rag_df = val_emb_rag_df = test_emb_rag_df = None
            if OPENAI_AVAILABLE and any(m in modes for m in ["emb_with_rag", "emb_with_rag+rag"]):
                parts = [rag_tr, rag_va, rag_te] if rag_va is not None else [rag_tr, rag_te]
                rag_combined = pd.concat(parts, ignore_index=True)
                texts_rag = build_texts_with_rag(df_for_text, rag_combined, args.openai_template, label_encoders)
                emb_rag_full = embed_texts_openai_full_cache(texts_rag, model=args.openai_emb_model)
                train_er = emb_rag_full[:n_train]
                val_er = emb_rag_full[n_train : n_train + n_val]
                test_er = emb_rag_full[n_train + n_val :]
                cache_dir_r = _embedding_store_dir("emb_with_rag", args.openai_template, args.openai_emb_model, rag_pca_dim, seed, fold_idx)
                fp_r = _texts_fingerprint(texts_rag)
                cached_r = _try_load_pca_cache(cache_dir_r, fp_r, rag_pca_dim, seed)
                if cached_r:
                    train_emb_rag_pca, val_emb_rag_pca, test_emb_rag_pca = cached_r
                    if len(val_emb_rag_pca) != len(val_idx):
                        cached_r = None
                if not cached_r:
                    train_emb_rag_pca, val_emb_rag_pca, test_emb_rag_pca, pca_m = fit_transform_pca_three(train_er, val_er, test_er, rag_pca_dim, seed)
                    _save_pca_cache(cache_dir_r, train_emb_rag_pca, val_emb_rag_pca, test_emb_rag_pca, pca_m, fp_r, rag_pca_dim, seed)
                emb_rag_cols = [f"emb_rag_{i}" for i in range(rag_pca_dim)]
                train_emb_rag_df = pd.DataFrame(train_emb_rag_pca, columns=emb_rag_cols)
                val_emb_rag_df = pd.DataFrame(val_emb_rag_pca, columns=emb_rag_cols)
                test_emb_rag_df = pd.DataFrame(test_emb_rag_pca, columns=emb_rag_cols)

            if "rag" in modes and rag_k > 0:
                print(f"\n[{tag}] RAG k={rag_k} mode={rag_mode}")
                X_tr_r = pd.concat([train_numeric, train_cat, rag_tr.reset_index(drop=True)], axis=1)
                X_va_r = pd.concat([val_numeric, val_cat, rag_va.reset_index(drop=True)], axis=1) if rag_va is not None else pd.concat([val_numeric, val_cat], axis=1)
                X_te_r = pd.concat([test_numeric, test_cat, rag_te.reset_index(drop=True)], axis=1)
                _, _, val_m, test_m = _train_eval_and_predict("rag", X_tr_r, X_va_r, X_te_r, y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                for name, m in val_m.items():
                    if not _should_record_model(getattr(args, "best_config", False), "rag", name, pca_dim, rag_k, rag_pca_dim):
                        continue
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, None, "rag", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            if "rag_price" in modes and rag_k > 0 and "neighbour_price" in rag_tr.columns:
                print(f"\n[{tag}] RAG price k={rag_k} mode={rag_mode}")
                rp_scaler = StandardScaler()
                tr_rp = rp_scaler.fit_transform(rag_tr[["neighbour_price"]].values)
                va_rp = rp_scaler.transform(rag_va[["neighbour_price"]].values) if rag_va is not None else np.zeros((len(val_numeric), 1))
                te_rp = rp_scaler.transform(rag_te[["neighbour_price"]].values)
                X_tr_rp = pd.concat([train_numeric, train_cat, pd.DataFrame(tr_rp, columns=["neighbour_price"], index=train_numeric.index)], axis=1)
                X_va_rp = pd.concat([val_numeric, val_cat, pd.DataFrame(va_rp, columns=["neighbour_price"], index=val_numeric.index)], axis=1)
                X_te_rp = pd.concat([test_numeric, test_cat, pd.DataFrame(te_rp, columns=["neighbour_price"], index=test_numeric.index)], axis=1)
                _, _, val_m, test_m = _train_eval_and_predict("rag_price", X_tr_rp, X_va_rp, X_te_rp, y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                for name, m in val_m.items():
                    if not _should_record_model(getattr(args, "best_config", False), "rag_price", name, pca_dim, rag_k, rag_pca_dim):
                        continue
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, None, "rag_price", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            if "emb" in modes and train_emb_df is not None:
                _, _, val_m, test_m = _train_eval_and_predict("emb",
                    pd.concat([train_numeric, train_cat, train_emb_df], axis=1),
                    pd.concat([val_numeric, val_cat, val_emb_df], axis=1),
                    pd.concat([test_numeric, test_cat, test_emb_df], axis=1),
                    y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                for name, m in val_m.items():
                    if not _should_record_model(getattr(args, "best_config", False), "emb", name, pca_dim, rag_k, rag_pca_dim):
                        continue
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, None, "emb", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            if "emb+rag" in modes and train_emb_df is not None and rag_k > 0:
                _, _, val_m, test_m = _train_eval_and_predict("emb+rag",
                    pd.concat([train_numeric, train_cat, rag_tr.reset_index(drop=True), train_emb_df], axis=1),
                    pd.concat([val_numeric, val_cat, rag_va.reset_index(drop=True), val_emb_df], axis=1) if rag_va is not None else pd.concat([val_numeric, val_cat, val_emb_df], axis=1),
                    pd.concat([test_numeric, test_cat, rag_te.reset_index(drop=True), test_emb_df], axis=1),
                    y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                for name, m in val_m.items():
                    if not _should_record_model(getattr(args, "best_config", False), "emb+rag", name, pca_dim, rag_k, rag_pca_dim):
                        continue
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, None, "emb+rag", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            if "emb_with_rag" in modes and train_emb_rag_df is not None:
                _, _, val_m, test_m = _train_eval_and_predict("emb_with_rag",
                    pd.concat([train_numeric, train_cat, train_emb_rag_df], axis=1),
                    pd.concat([val_numeric, val_cat, val_emb_rag_df], axis=1),
                    pd.concat([test_numeric, test_cat, test_emb_rag_df], axis=1),
                    y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                for name, m in val_m.items():
                    if not _should_record_model(getattr(args, "best_config", False), "emb_with_rag", name, pca_dim, rag_k, rag_pca_dim):
                        continue
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, None, "emb_with_rag", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            if "emb_with_rag+rag" in modes and train_emb_rag_df is not None and rag_k > 0:
                _, _, val_m, test_m = _train_eval_and_predict("emb_with_rag+rag",
                    pd.concat([train_numeric, train_cat, rag_tr.reset_index(drop=True), train_emb_rag_df], axis=1),
                    pd.concat([val_numeric, val_cat, rag_va.reset_index(drop=True), val_emb_rag_df], axis=1) if rag_va is not None else pd.concat([val_numeric, val_cat, val_emb_rag_df], axis=1),
                    pd.concat([test_numeric, test_cat, rag_te.reset_index(drop=True), test_emb_rag_df], axis=1),
                    y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                for name, m in val_m.items():
                    if not _should_record_model(getattr(args, "best_config", False), "emb_with_rag+rag", name, pca_dim, rag_k, rag_pca_dim):
                        continue
                    t = test_m.get(name, {})
                    results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, None, "emb_with_rag+rag", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            # fni+rag: scaled_numeric + cat + fni + rag
            if "fni+rag" in modes and fni_by_k and train_num_s is not None and rag_k > 0:
                for fk in fni_ks:
                    if fk not in fni_by_k:
                        continue
                    ftr, fva, fte = fni_by_k[fk]
                    train_num_df = pd.DataFrame(train_num_s, columns=train_numeric.columns)
                    val_num_df = pd.DataFrame(val_num_s, columns=val_numeric.columns)
                    test_num_df = pd.DataFrame(test_num_s, columns=test_numeric.columns)
                    fni_cols = [f"fni_{i}" for i in range(ftr.shape[1])]
                    X_tr_fr = pd.concat([
                        train_num_df.reset_index(drop=True), train_cat.reset_index(drop=True),
                        pd.DataFrame(ftr, columns=fni_cols), rag_tr.reset_index(drop=True)
                    ], axis=1)
                    X_va_fr = pd.concat([
                        val_num_df.reset_index(drop=True), val_cat.reset_index(drop=True),
                        pd.DataFrame(fva, columns=fni_cols), rag_va.reset_index(drop=True)
                    ], axis=1) if rag_va is not None else pd.concat([
                        val_num_df.reset_index(drop=True), val_cat.reset_index(drop=True),
                        pd.DataFrame(fva, columns=fni_cols)
                    ], axis=1)
                    X_te_fr = pd.concat([
                        test_num_df.reset_index(drop=True), test_cat.reset_index(drop=True),
                        pd.DataFrame(fte, columns=fni_cols), rag_te.reset_index(drop=True)
                    ], axis=1)
                    print(f"\n[{tag}] fni+rag fni_k={fk} rag_k={rag_k}")
                    _, _, val_m, test_m = _train_eval_and_predict("fni+rag", X_tr_fr, X_va_fr, X_te_fr, y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                    for name, m in val_m.items():
                        if not _should_record_model(getattr(args, "best_config", False), "fni+rag", name, pca_dim, rag_k, rag_pca_dim):
                            continue
                        t = test_m.get(name, {})
                        results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, fk, "fni+rag", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            # fni_with_rag: RAG 后对 (numeric + rag_numeric) 做 FNI
            if "fni_with_rag" in modes and rag_k > 0:
                rag_num_cols = [c for c in rag_tr.columns if c.startswith("rag_mean_")]
                mat_train = np.hstack([train_numeric.fillna(0).values, rag_tr[rag_num_cols].fillna(0).values])
                mat_val = np.hstack([val_numeric.fillna(0).values, rag_va[rag_num_cols].fillna(0).values]) if rag_va is not None else train_numeric.fillna(0).values
                mat_test = np.hstack([test_numeric.fillna(0).values, rag_te[rag_num_cols].fillna(0).values])
                fni_wr_scaler = StandardScaler()
                mat_train_s = fni_wr_scaler.fit_transform(mat_train)
                mat_val_s = fni_wr_scaler.transform(mat_val) if rag_va is not None else mat_train_s[:len(val_numeric)]
                mat_test_s = fni_wr_scaler.transform(mat_test)
                for fk in fni_ks:
                    train_fni_wr, val_fni_wr, test_fni_wr = compute_fni_features(
                        mat_train_s, mat_val_s, mat_test_s, k=fk, exclude_col_idx=None, y_train=y_train_log
                    )
                    train_num_df = pd.DataFrame(train_num_s, columns=train_numeric.columns)
                    val_num_df = pd.DataFrame(val_num_s, columns=val_numeric.columns)
                    test_num_df = pd.DataFrame(test_num_s, columns=test_numeric.columns)
                    fni_wr_cols = [f"fni_wr_{i}" for i in range(train_fni_wr.shape[1])]
                    X_tr_fwr = pd.concat([
                        train_num_df.reset_index(drop=True), train_cat.reset_index(drop=True),
                        rag_tr.reset_index(drop=True),
                        pd.DataFrame(train_fni_wr, columns=fni_wr_cols)
                    ], axis=1)
                    X_va_fwr = pd.concat([
                        val_num_df.reset_index(drop=True), val_cat.reset_index(drop=True),
                        rag_va.reset_index(drop=True),
                        pd.DataFrame(val_fni_wr, columns=fni_wr_cols)
                    ], axis=1) if rag_va is not None else pd.concat([
                        val_num_df.reset_index(drop=True), val_cat.reset_index(drop=True),
                        pd.DataFrame(val_fni_wr, columns=fni_wr_cols)
                    ], axis=1)
                    X_te_fwr = pd.concat([
                        test_num_df.reset_index(drop=True), test_cat.reset_index(drop=True),
                        rag_te.reset_index(drop=True),
                        pd.DataFrame(test_fni_wr, columns=fni_wr_cols)
                    ], axis=1)
                    print(f"\n[{tag}] fni_with_rag fni_k={fk} rag_k={rag_k}")
                    _, _, val_m, test_m = _train_eval_and_predict("fni_with_rag", X_tr_fwr, X_va_fwr, X_te_fwr, y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                    for name, m in val_m.items():
                        if not _should_record_model(getattr(args, "best_config", False), "fni_with_rag", name, pca_dim, rag_k, rag_pca_dim):
                            continue
                        t = test_m.get(name, {})
                        results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, fk, "fni_with_rag", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

            # fni_dual: base + rag + FNI(base) + FNI(rag)
            if "fni_dual" in modes and fni_by_k and rag_k > 0:
                rag_num_cols = [c for c in rag_tr.columns if c.startswith("rag_mean_")]
                if len(rag_num_cols) >= 2:
                    rag_mat_tr = rag_tr[rag_num_cols].fillna(0).values
                    rag_mat_va = rag_va[rag_num_cols].fillna(0).values if rag_va is not None else rag_mat_tr[:len(val_numeric)]
                    rag_mat_te = rag_te[rag_num_cols].fillna(0).values
                    rag_scaler = StandardScaler()
                    rag_mat_tr_s = rag_scaler.fit_transform(rag_mat_tr)
                    rag_mat_va_s = rag_scaler.transform(rag_mat_va) if rag_va is not None else rag_mat_tr_s[:len(val_numeric)]
                    rag_mat_te_s = rag_scaler.transform(rag_mat_te)
                    for fk in fni_ks:
                        if fk not in fni_by_k:
                            continue
                        ftr, fva, fte = fni_by_k[fk]
                        train_fni_rag, val_fni_rag, test_fni_rag = compute_fni_features(
                            rag_mat_tr_s, rag_mat_va_s, rag_mat_te_s, k=fk, exclude_col_idx=None, y_train=y_train_log
                        )
                        train_num_df = pd.DataFrame(train_num_s, columns=train_numeric.columns)
                        val_num_df = pd.DataFrame(val_num_s, columns=val_numeric.columns)
                        test_num_df = pd.DataFrame(test_num_s, columns=test_numeric.columns)
                        fni_cols = [f"fni_{i}" for i in range(ftr.shape[1])]
                        fni_rag_cols = [f"fni_rag_{i}" for i in range(train_fni_rag.shape[1])]
                        X_tr_fd = pd.concat([
                            train_num_df.reset_index(drop=True), train_cat.reset_index(drop=True),
                            rag_tr.reset_index(drop=True),
                            pd.DataFrame(ftr, columns=fni_cols), pd.DataFrame(train_fni_rag, columns=fni_rag_cols)
                        ], axis=1)
                        X_va_fd = pd.concat([
                            val_num_df.reset_index(drop=True), val_cat.reset_index(drop=True),
                            rag_va.reset_index(drop=True),
                            pd.DataFrame(fva, columns=fni_cols), pd.DataFrame(val_fni_rag, columns=fni_rag_cols)
                        ], axis=1) if rag_va is not None else pd.concat([
                            val_num_df.reset_index(drop=True), val_cat.reset_index(drop=True),
                            pd.DataFrame(fva, columns=fni_cols), pd.DataFrame(val_fni_rag, columns=fni_rag_cols)
                        ], axis=1)
                        X_te_fd = pd.concat([
                            test_num_df.reset_index(drop=True), test_cat.reset_index(drop=True),
                            rag_te.reset_index(drop=True),
                            pd.DataFrame(fte, columns=fni_cols), pd.DataFrame(test_fni_rag, columns=fni_rag_cols)
                        ], axis=1)
                        print(f"\n[{tag}] fni_dual fni_k={fk} rag_k={rag_k}")
                        _, _, val_m, test_m = _train_eval_and_predict("fni_dual", X_tr_fd, X_va_fd, X_te_fd, y_train, y_val, y_test, test_ids, seed, models_to_run=models_to_run)
                        for name, m in val_m.items():
                            if not _should_record_model(getattr(args, "best_config", False), "fni_dual", name, pca_dim, rag_k, rag_pca_dim):
                                continue
                            t = test_m.get(name, {})
                            results.append((seed, fold_idx, pca_dim, rag_k, rag_pca_dim, fk, "fni_dual", name, m.get("rmsle"), t.get("rmsle"), m.get("rmse"), t.get("rmse")))

    if results:
        output_csv = getattr(args, "output_csv", "california_sweep_results.csv")
        df_res = pd.DataFrame(results, columns=["seed", "fold", "pca_dim", "rag_k", "rag_pca_dim", "fni_k", "mode", "model", "val_rmsle", "test_rmsle", "val_rmse", "test_rmse"])
        df_res.to_csv(output_csv, index=False)
        print(f"\nFull results saved: {output_csv}  ({len(df_res)} rows)")

        def _cfg(r):
            p, k, rp = r["pca_dim"], r["rag_k"], r["rag_pca_dim"]
            fk = r.get("fni_k")
            base = f"p{int(p)}_k{int(k)}_rp{int(rp)}" if pd.notna(p) and pd.notna(k) and pd.notna(rp) else "-"
            if pd.notna(fk):
                return f"{base}_fk{int(fk)}" if base != "-" else f"fk{int(fk)}"
            return base
        df_res["config"] = df_res.apply(_cfg, axis=1)

        out_base = Path(output_csv).stem
        for metric, metric_name in [("rmsle", "RMSLE"), ("rmse", "RMSE")]:
            # aggregate over seed -> mean ± std
            agg = df_res.groupby(["mode", "model", "config"], as_index=True).agg(
                mean=(f"val_{metric}", "mean"),
                std=(f"val_{metric}", "std")
            ).reset_index()
            agg["val_mean_std"] = agg["mean"].round(4).astype(str) + " ± " + agg["std"].fillna(0).round(4).astype(str)
            val_summary = agg[["mode", "model", "config", "val_mean_std"]].sort_values(["mode", "model", "config"])
            val_path = Path(output_csv).parent / f"{out_base}_val_{metric}.csv"
            val_summary.to_csv(val_path, index=False)
            print("\n" + "=" * 80)
            print(f"Val {metric_name} (mean ± std over seeds)")
            print("=" * 80)
            print(val_summary.to_string(index=False))
            print(f"\nSaved: {val_path}")

            agg_te = df_res.groupby(["mode", "model", "config"], as_index=True).agg(
                mean=(f"test_{metric}", "mean"),
                std=(f"test_{metric}", "std")
            ).reset_index()
            agg_te["test_mean_std"] = agg_te["mean"].round(4).astype(str) + " ± " + agg_te["std"].fillna(0).round(4).astype(str)
            test_summary = agg_te[["mode", "model", "config", "test_mean_std"]].sort_values(["mode", "model", "config"])
            test_path = Path(output_csv).parent / f"{out_base}_test_{metric}.csv"
            test_summary.to_csv(test_path, index=False)
            print("\n" + "=" * 80)
            print(f"Test {metric_name} (mean ± std over seeds)")
            print("=" * 80)
            print(test_summary.to_string(index=False))
            print(f"\nSaved: {test_path}")


if __name__ == "__main__":
    main()
