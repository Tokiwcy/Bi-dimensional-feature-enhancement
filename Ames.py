import os
import math
import time
import random
import hashlib
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
TRAIN_PATH = "data/train.csv"
OUTPUT_PATH = "submission.csv"

OPENAI_EMB_MODEL = "text-embedding-3-large"
OPENAI_TEMPLATE = "structured"
PCA_DIM = 16

USE_CATBOOST = True
USE_XGB = True
USE_LGBM = True
USE_TABPFN = True

# Model-specific best config (from tuning notes)
BEST_CONFIG_PER_MODEL = {
    "catboost": {"rag_k": 15, "pca_dim": 12},
    "xgb": {"rag_k": 6, "pca_dim": 16},
    "lgbm": {"rag_k": 6, "pca_dim": 20},
    "tabpfn": {"rag_k": 8, "pca_dim": 16},
}

EMB_CACHE_DIR = Path("results/embedding_cache")
PCA_CACHE_DIR = Path("results/pca_cache")

RAG_IMPORTANT_NUMERIC_COLS = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "TotRmsAbvGrd",
    "LotArea",
    "1stFlrSF",
    "2ndFlrSF",
]
RAG_IMPORTANT_CAT_COLS = [
    "Neighborhood",
    "KitchenQual",
]

# Short labels for emb_with_rag compare/delta (Chinese, kept for reference)
RAG_COL_LABELS = {
    "OverallQual": "品质",
    "GrLivArea": "面积",
    "TotalBsmtSF": "地下室面积",
    "GarageCars": "车库车位数",
    "GarageArea": "车库面积",
    "YearBuilt": "建造年份",
    "YearRemodAdd": "改造年份",
    "FullBath": "全卫",
    "TotRmsAbvGrd": "房间数",
    "LotArea": "地块面积",
    "1stFlrSF": "一层面积",
    "2ndFlrSF": "二层面积",
    "Neighborhood": "街区",
    "KitchenQual": "厨房品质",
}

# English labels for compare/delta output (column name -> display label)
RAG_COL_LABELS_EN = {
    "OverallQual": "quality",
    "GrLivArea": "area",
    "TotalBsmtSF": "basement area",
    "GarageCars": "garage cars",
    "GarageArea": "garage area",
    "YearBuilt": "year built",
    "YearRemodAdd": "year remodeled",
    "FullBath": "full baths",
    "TotRmsAbvGrd": "rooms",
    "LotArea": "lot area",
    "1stFlrSF": "1st flr area",
    "2ndFlrSF": "2nd flr area",
    "Neighborhood": "neighborhood",
    "KitchenQual": "kitchen quality",
}

# RAG mode configs: hybrid (current), quality_year (B), location (C), stratified_neighborhood (D1), stratified_bldg (D2)
RAG_MODE_CONFIG = {
    "hybrid": {
        "numeric": RAG_IMPORTANT_NUMERIC_COLS,
        "cat": RAG_IMPORTANT_CAT_COLS,
        "stratify": None,
        "extra_derived": None,
    },
    "quality_year": {
        "numeric": ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"],
        "cat": ["ExterQual", "KitchenQual", "BsmtQual", "HeatingQC", "GarageQual", "GarageCond"],
        "stratify": None,
        "extra_derived": ["years_vs_neighbors"],
    },
    "location": {
        "numeric": [],
        "cat": ["Neighborhood", "MSZoning", "Condition1", "Condition2", "Street", "LotConfig", "LandContour", "LandSlope"],
        "stratify": None,
        "extra_derived": None,
    },
    "stratified_neighborhood": {
        "numeric": ["GrLivArea", "TotalBsmtSF", "GarageArea", "OverallQual", "YearBuilt"],
        "cat": [],
        "stratify": ["Neighborhood"],
        "extra_derived": None,
    },
    "stratified_bldg": {
        "numeric": RAG_IMPORTANT_NUMERIC_COLS,
        "cat": [],
        "stratify": ["BldgType", "HouseStyle"],
        "extra_derived": None,
    },
}

def _parse_comma_int(value) -> List[int]:
    """Parse int or comma-separated ints into list, e.g. 42 -> [42], '42,123' -> [42, 123]."""
    if isinstance(value, int):
        return [value]
    s = str(value).strip()
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def evaluate_model(y_true, y_pred, name="Model", quiet=False):
    y_true_log = np.log1p(np.maximum(y_true, 0))
    y_pred_log = np.log1p(np.maximum(y_pred, 0))
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    if not quiet:
        print(f"  {name}: RMSLE = {rmsle:.6f}")
    return {"rmsle": rmsle}


def safe_val(v) -> str:
    if pd.isna(v):
        return "unknown"
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.2f}"
    return str(v)


def qual_to_text(qual: str) -> str:
    qual_map = {"Ex": "Excellent", "Gd": "Good", "TA": "Average", "Fa": "Fair", "Po": "Poor"}
    return qual_map.get(str(qual), "Average")


def bucket_area(area: float) -> str:
    if pd.isna(area):
        return "unknown size"
    area = float(area)
    if area < 900:
        return "compact"
    if area < 1500:
        return "cozy"
    if area < 2200:
        return "spacious"
    if area < 3200:
        return "very spacious"
    return "estate-sized"


def bucket_year(year: float) -> str:
    if pd.isna(year):
        return "unknown year"
    year = int(year)
    if year < 1940:
        return "pre-war"
    if year < 1970:
        return "mid-century"
    if year < 2000:
        return "modern"
    return "recent"


def get_clean_fit(train_df: pd.DataFrame) -> dict:
    """Compute fill values from training subset only (for no leakage)."""
    num_cols = train_df.select_dtypes(include=["number"]).columns
    cat_fill = "None"
    medians = train_df[num_cols].median()
    return {"medians": medians, "cat_fill": cat_fill, "num_cols": list(num_cols)}


def apply_clean(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """Apply clean using fit from get_clean_fit (no fit on df)."""
    df = df.copy()
    num_cols = [c for c in fit["num_cols"] if c in df.columns]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(fit["medians"][c])
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(fit["cat_fill"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy: fit and transform on same df. Prefer get_clean_fit + apply_clean after split."""
    fit = get_clean_fit(df)
    return apply_clean(df, fit)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    df["BuildingAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsNew"] = (df["BuildingAge"] <= 2).astype(int)

    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"] + df["GrLivArea"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    df["TotalPorch"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]

    return df


def verbalize_row_structured(row: pd.Series) -> str:
    vals = row.to_dict()
    sections = {"Location": [], "Building": [], "Amenities": [], "Quality": []}
    
    # Location
    nbhd = vals.get("Neighborhood")
    if not pd.isna(nbhd):
        sections["Location"].append(f"neighborhood {safe_val(nbhd)}")
    
    # Building
    beds = vals.get("BedroomAbvGr")
    baths = vals.get("FullBath")
    if not pd.isna(beds) and not pd.isna(baths):
        sections["Building"].append(f"{int(beds)} bedrooms, {int(baths)} full baths")
    
    area = vals.get("GrLivArea")
    if not pd.isna(area):
        sections["Building"].append(f"{int(area)} sqft living area ({bucket_area(area)})")
    
    yr_built = vals.get("YearBuilt")
    if not pd.isna(yr_built):
        sections["Building"].append(f"built {int(yr_built)} ({bucket_year(yr_built)})")
    
    yr_remod = vals.get("YearRemodAdd")
    if not pd.isna(yr_remod) and yr_remod != yr_built:
        sections["Building"].append(f"remodeled {int(yr_remod)}")
    
    # Amenities
    garage_cars = vals.get("GarageCars")
    if not pd.isna(garage_cars) and garage_cars > 0:
        sections["Amenities"].append(f"{int(garage_cars)}-car garage")
    
    fireplace = vals.get("Fireplaces")
    if not pd.isna(fireplace) and fireplace > 0:
        sections["Amenities"].append(f"{int(fireplace)} fireplace(s)")
    
    kitchen = vals.get("KitchenQual")
    if not pd.isna(kitchen):
        sections["Amenities"].append(f"kitchen quality {qual_to_text(kitchen)}")
    
    # Quality
    qual = vals.get("OverallQual")
    cond = vals.get("OverallCond")
    if not pd.isna(qual):
        sections["Quality"].append(f"overall quality {safe_val(qual)}/10")
    if not pd.isna(cond):
        sections["Quality"].append(f"condition {safe_val(cond)}/10")
    
    parts = []
    for items in sections.values():
        if items:
            parts.append(", ".join(items))
    
    return ". ".join(parts) + "."


def verbalize_row_descriptive(row: pd.Series) -> str:
    """Descriptive template: concise narrative emphasizing price drivers."""
    vals = row.to_dict()
    parts = []

    # 1. 位置和建筑（合并）
    nbhd = vals.get("Neighborhood")
    style = vals.get("HouseStyle")
    yr_built = vals.get("YearBuilt")
    yr_remod = vals.get("YearRemodAdd")
    location_building = []
    if not pd.isna(nbhd):
        location_building.append(f"located in {safe_val(nbhd)}")
    if not pd.isna(style):
        location_building.append(f"a {safe_val(style)}")
    if not pd.isna(yr_built):
        location_building.append(f"built in {int(yr_built)}")
    if not pd.isna(yr_remod) and not pd.isna(yr_built) and yr_remod != yr_built:
        location_building.append(f"renovated in {int(yr_remod)}")
    if location_building:
        value_note = "increasing its value" if (not pd.isna(yr_remod) and not pd.isna(yr_built) and yr_remod != yr_built) else "with location being a major price factor"
        parts.append("This property is " + ", ".join(location_building) + f", {value_note}.")

    # 2. 面积和布局（精简）
    beds = vals.get("BedroomAbvGr")
    full_baths = vals.get("FullBath")
    area = vals.get("GrLivArea")
    layout_parts = []
    if not pd.isna(beds) and not pd.isna(full_baths):
        layout_parts.append(f"{int(beds)} bedrooms and {int(full_baths)} bathrooms")
    if not pd.isna(area):
        layout_parts.append(f"{int(area)} sqft living space")
    if layout_parts:
        parts.append("It features " + ", ".join(layout_parts) + ", with size being a key price factor.")

    # 3. 整体质量（强调 OverallQual 对价格的影响）
    qual = vals.get("OverallQual")
    if not pd.isna(qual):
        qual_desc = "excellent" if qual >= 8 else "good" if qual >= 6 else "average" if qual >= 4 else "fair"
        value_impact = "significantly drives the property value" if qual >= 8 else "strongly impacts the home's worth" if qual >= 6 else "affects the market price" if qual >= 4 else "influences the property value"
        parts.append(f"Overall quality is {qual_desc} (rated {safe_val(qual)}/10), which {value_impact}.")

    # 4. 车库和地下室（合并简化）
    garage_cars = vals.get("GarageCars")
    bsmt_sf = vals.get("TotalBsmtSF")
    features = []
    if not pd.isna(garage_cars) and garage_cars > 0:
        features.append(f"{int(garage_cars)}-car garage")
    if not pd.isna(bsmt_sf) and bsmt_sf > 0:
        features.append(f"{int(bsmt_sf)} sqft basement")
    if features:
        parts.append("It includes " + ", ".join(features) + ", adding value.")

    # 5. 设施（精简）
    fireplace = vals.get("Fireplaces")
    has_pool = vals.get("HasPool")
    amenities = []
    if not pd.isna(fireplace) and fireplace > 0:
        amenities.append(f"{int(fireplace)} fireplace(s)")
    if not pd.isna(has_pool) and has_pool > 0:
        amenities.append("pool")
    if amenities:
        parts.append("Premium features include " + ", ".join(amenities) + ", increasing the property's value.")

    # 收尾
    if parts:
        return ". ".join(parts)
    return "This property's value is determined by location, size, and quality."


def verbalize_row_compact_core(row: pd.Series) -> str:
    """Compact template using a small core feature subset."""
    vals = row.to_dict()
    nbhd = safe_val(vals.get("Neighborhood", "unknown"))
    qual = vals.get("OverallQual")
    area = vals.get("GrLivArea")
    year = vals.get("YearBuilt")
    garage = vals.get("GarageCars")
    kitchen = vals.get("KitchenQual")
    bits = [f"Neighborhood {nbhd}"]
    if not pd.isna(qual):
        bits.append(f"OverallQual {int(qual)}/10")
    if not pd.isna(area):
        bits.append(f"GrLivArea {int(area)} sqft")
    if not pd.isna(year):
        bits.append(f"YearBuilt {int(year)}")
    if not pd.isna(garage):
        bits.append(f"GarageCars {int(garage)}")
    if not pd.isna(kitchen):
        bits.append(f"KitchenQual {safe_val(kitchen)}")
    return ". ".join(bits) + "."


def verbalize_row_location_quality(row: pd.Series) -> str:
    """Location + quality oriented template with limited fields."""
    vals = row.to_dict()
    nbhd = safe_val(vals.get("Neighborhood", "unknown"))
    zoning = safe_val(vals.get("MSZoning", "unknown"))
    qual = vals.get("OverallQual")
    cond = vals.get("OverallCond")
    lot = vals.get("LotArea")
    yr = vals.get("YearRemodAdd")
    parts = [f"Home in {nbhd}, zoning {zoning}"]
    if not pd.isna(qual):
        parts.append(f"quality {int(qual)}/10")
    if not pd.isna(cond):
        parts.append(f"condition {int(cond)}/10")
    if not pd.isna(lot):
        parts.append(f"lot {int(lot)} sqft")
    if not pd.isna(yr):
        parts.append(f"last remodel {int(yr)}")
    return ". ".join(parts) + "."


def verbalize_row_size_age(row: pd.Series) -> str:
    """Size + age oriented template with minimal fields."""
    vals = row.to_dict()
    area = vals.get("GrLivArea")
    bsmt = vals.get("TotalBsmtSF")
    first = vals.get("1stFlrSF")
    second = vals.get("2ndFlrSF")
    built = vals.get("YearBuilt")
    remod = vals.get("YearRemodAdd")
    baths = vals.get("FullBath")
    parts = []
    if not pd.isna(area):
        parts.append(f"Living area {int(area)} sqft")
    if not pd.isna(bsmt):
        parts.append(f"Basement {int(bsmt)} sqft")
    if not pd.isna(first) and not pd.isna(second):
        parts.append(f"Floor split {int(first)} + {int(second)} sqft")
    if not pd.isna(built):
        parts.append(f"Built {int(built)}")
    if not pd.isna(remod):
        parts.append(f"Remodeled {int(remod)}")
    if not pd.isna(baths):
        parts.append(f"FullBath {int(baths)}")
    return ". ".join(parts) + "."


def build_texts(df: pd.DataFrame, template: str = "structured") -> List[str]:
    if template == "structured":
        return [verbalize_row_structured(row) for _, row in df.iterrows()]
    elif template in ("descriptive", "narrative"):  # narrative kept as backward-compatible alias
        return [verbalize_row_descriptive(row) for _, row in df.iterrows()]
    elif template == "compact_core":
        return [verbalize_row_compact_core(row) for _, row in df.iterrows()]
    elif template == "location_quality":
        return [verbalize_row_location_quality(row) for _, row in df.iterrows()]
    elif template == "size_age":
        return [verbalize_row_size_age(row) for _, row in df.iterrows()]
    else:
        raise ValueError(
            "template must be one of: 'structured', 'descriptive', 'compact_core', 'location_quality', 'size_age'"
        )


def _rag_compare_parts(row: pd.Series, rag_row: pd.Series, rel_tol: float = 0.05) -> List[str]:
    """Compare self to neighbor mean: (self - nei)/nei; >~5% => larger/smaller, else comparable. Categorical => consistent with majority. Returns English phrases."""
    parts = []
    for col in RAG_IMPORTANT_NUMERIC_COLS:
        if col not in row.index or pd.isna(row[col]):
            continue
        key = f"similar house {col}"
        if key not in rag_row.index or pd.isna(rag_row[key]):
            continue
        self_val = float(row[col])
        nei_val = float(rag_row[key])
        label = RAG_COL_LABELS_EN.get(col, col)
        if nei_val <= 0 or (abs(nei_val) < 1e-12):
            parts.append(f"{label} comparable to neighbors")
            continue
        diff_ratio = (self_val - nei_val) / nei_val
        if diff_ratio > rel_tol:
            parts.append(f"{label} larger than neighbors")
        elif diff_ratio < -rel_tol:
            parts.append(f"{label} smaller than neighbors")
        else:
            parts.append(f"{label} comparable to neighbors")
    for col in RAG_IMPORTANT_CAT_COLS:
        key = f"rag_mode_{col}"
        if key not in rag_row.index or pd.isna(rag_row[key]):
            continue
        label = RAG_COL_LABELS_EN.get(col, col)
        parts.append(f"{label} consistent with majority of neighbors")
    return parts


def _rag_delta_parts(row: pd.Series, rag_row: pd.Series) -> List[str]:
    """Concrete difference vs neighbors: sqft => X sqft larger/smaller than neighbor mean; year => X years earlier/later; other => X higher/lower. Returns English phrases."""
    parts = []
    sqft_cols = ["GrLivArea", "TotalBsmtSF", "LotArea", "GarageArea", "1stFlrSF", "2ndFlrSF"]
    year_cols = ["YearBuilt", "YearRemodAdd"]
    for col in RAG_IMPORTANT_NUMERIC_COLS:
        if col not in row.index or pd.isna(row[col]):
            continue
        key = f"similar house {col}"
        if key not in rag_row.index or pd.isna(rag_row[key]):
            continue
        self_val = float(row[col])
        nei_val = float(rag_row[key])
        delta = self_val - nei_val
        label = RAG_COL_LABELS_EN.get(col, col)
        if abs(delta) < 1e-6:
            parts.append(f"{label} on par with neighbor mean")
            continue
        if col == "YearBuilt":
            d = int(round(abs(delta)))
            parts.append(f"built {d} years {'later' if delta > 0 else 'earlier'} than neighbors")
        elif col == "YearRemodAdd":
            d = int(round(abs(delta)))
            parts.append(f"remodeled {d} years {'later' if delta > 0 else 'earlier'} than neighbors")
        elif col in sqft_cols:
            d = int(round(abs(delta)))
            parts.append(f"{label} {d} sqft {'larger' if delta > 0 else 'smaller'} than neighbor mean")
        else:
            d = abs(delta)
            fmt = f"{d:.1f}" if d != int(d) else str(int(d))
            parts.append(f"{label} {fmt} {'higher' if delta > 0 else 'lower'} than neighbor mean")
    for col in RAG_IMPORTANT_CAT_COLS:
        key = f"rag_mode_{col}"
        if key not in rag_row.index or pd.isna(rag_row[key]):
            continue
        label = RAG_COL_LABELS_EN.get(col, col)
        parts.append(f"{label} consistent with neighbors")
    return parts


def build_texts_with_rag(
    df: pd.DataFrame,
    rag_df: pd.DataFrame,
    base_template: str = "structured",
    rag_template: str = "default",
    rag_k: Optional[int] = None,
) -> List[str]:
    """Build texts with RAG appendix. If rag_k is set, append ' (based on k=N nearest neighbors)' so embedding differs by k (avoids identical text when e.g. location mode gives same mode() for different k)."""
    if base_template != "structured":
        raise ValueError("RAG text base_template only supports 'structured'")
    k_suffix = f" (based on k={rag_k} nearest neighbors)." if rag_k is not None else ""
    texts = []
    for idx, row in df.iterrows():
        base = verbalize_row_structured(row)
        if rag_df is None or idx not in rag_df.index:
            texts.append(base)
            continue
        rag_row = rag_df.loc[idx]
        if rag_template == "default":
            rag_parts = []
            for col in RAG_IMPORTANT_NUMERIC_COLS:
                key = f"similar house {col}"
                if key in rag_row.index and not pd.isna(rag_row[key]):
                    val = rag_row[key]
                    if col in ["GrLivArea", "TotalBsmtSF", "LotArea", "GarageArea", "1stFlrSF", "2ndFlrSF"]:
                        rag_parts.append(f"{col} {val:.0f} sqft")
                    elif col in ["YearBuilt", "YearRemodAdd"]:
                        rag_parts.append(f"{col} {val:.0f}")
                    else:
                        rag_parts.append(f"{col} {val:.2f}")
            for col in RAG_IMPORTANT_CAT_COLS:
                key = f"rag_mode_{col}"
                if key in rag_row.index and not pd.isna(rag_row[key]):
                    rag_parts.append(f"{col} {rag_row[key]}")
            if rag_parts:
                base = base + " Similar homes (avg): " + ", ".join(rag_parts) + "." + k_suffix
        elif rag_template == "compare":
            parts = _rag_compare_parts(row, rag_row)
            if parts:
                base = base + " Compared to similar homes: " + ", ".join(parts) + "." + k_suffix
        elif rag_template == "delta":
            parts = _rag_delta_parts(row, rag_row)
            if parts:
                base = base + " Relative to neighbors: " + ", ".join(parts) + "." + k_suffix
        else:
            raise ValueError(f"rag_template must be 'default', 'compare', or 'delta', got {rag_template!r}")
        texts.append(base)
    return texts


def _sanitize_text_for_api(text: str) -> str:
    """Ensure text is valid UTF-8 and strip control chars that can break JSON (e.g. null byte)."""
    if not isinstance(text, str):
        text = str(text)
    # Normalize to valid UTF-8, replace invalid/control code points
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # Remove control characters (except \n, \r, \t) that can break JSON
    return "".join(c for c in text if c in "\n\r\t" or ord(c) >= 32)


def embed_texts_openai(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 100,
    normalize: bool = True
) -> np.ndarray:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    cache_root = Path(os.getenv("OPENAI_EMBEDDING_CACHE_DIR", "results/embedding_cache"))
    cache_dir = cache_root / model
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize so cache key and API payload are valid (avoids 400 invalid JSON)
    texts = [_sanitize_text_for_api(t) for t in texts]

    def cache_path(text: str) -> Path:
        key = hashlib.sha256(f"{model}|{text}".encode("utf-8")).hexdigest()
        return cache_dir / f"{key}.npy"

    embeddings = [None] * len(texts)
    cache_hits = 0
    missing_idx = []

    for i, t in enumerate(texts):
        p = cache_path(t)
        if p.exists():
            embeddings[i] = np.load(p)
            cache_hits += 1
        else:
            missing_idx.append(i)

    print(f"    Using embedding cache: {cache_hits}/{len(texts)} hits (dir={cache_dir})")

    if missing_idx:
        for s in range(0, len(missing_idx), batch_size):
            batch_ids = missing_idx[s:s + batch_size]
            batch_texts = [texts[i] for i in batch_ids]
            resp = client.embeddings.create(model=model, input=batch_texts)
            for j, item in enumerate(resp.data):
                emb = np.array(item.embedding, dtype=np.float32)
                idx = batch_ids[j]
                embeddings[idx] = emb
                np.save(cache_path(texts[idx]), emb)
            time.sleep(0.05)

    if any(e is None for e in embeddings):
        emb_dim = 1536
        for e in embeddings:
            if e is not None:
                emb_dim = len(e)
                break
        embeddings = [e if e is not None else np.zeros(emb_dim, dtype=np.float32) for e in embeddings]

    emb = np.array(embeddings, dtype=np.float32)
    if normalize:
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


def _texts_fingerprint(texts: List[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _embedding_store_dir(
    mode: str,
    template: str,
    model: str,
    pca_dim: int,
    seed: int,
    rag_k: Optional[int] = None,
) -> Path:
    """Cache path for embedding+PCA. Include rag_k when mode is emb_with_rag so different k do not share cache."""
    base = Path("cache") / "embedding_pca" / mode / f"model_{model}" / f"template_{template}"
    if rag_k is not None:
        base = base / f"rag_k_{rag_k}"
    return base / f"pca_{pca_dim}" / f"seed_{seed}"


def _try_load_pca_cache(
    cache_dir: Path,
    train_texts: List[str],
    test_texts: List[str],
    pca_dim: int,
    seed: int
) -> Optional[tuple]:
    """Load cached embeddings. Returns (train_emb, test_emb, pca_model). If pca_model is not None, arrays are PCA output (full fit); else raw embeddings."""
    meta_path = cache_dir / "meta.json"
    train_path = cache_dir / "train.npy"
    test_path = cache_dir / "test.npy"
    pca_path = cache_dir / "pca.pkl"
    if not (meta_path.exists() and train_path.exists() and test_path.exists()):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("pca_dim") != pca_dim
            or meta.get("seed") != seed
            or meta.get("train_fingerprint") != _texts_fingerprint(train_texts)
            or meta.get("test_fingerprint") != _texts_fingerprint(test_texts)
        ):
            return None
        train_emb = np.load(train_path)
        test_emb = np.load(test_path)
        pca_model = None
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca_model = pickle.load(f)
        return train_emb, test_emb, pca_model
    except Exception:
        return None


def _save_pca_cache(
    cache_dir: Path,
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    train_texts: List[str],
    test_texts: List[str],
    pca_dim: int,
    seed: int,
    pca_model: Optional[PCA] = None,
) -> None:
    """Save embeddings to cache. Full storage: PCA fit on full train, save PCA-transformed train/test and pca_model."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train.npy", train_emb)
    np.save(cache_dir / "test.npy", test_emb)
    if pca_model is not None:
        with open(cache_dir / "pca.pkl", "wb") as f:
            pickle.dump(pca_model, f)
    meta = {
        "pca_dim": pca_dim,
        "seed": seed,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "train_fingerprint": _texts_fingerprint(train_texts),
        "test_fingerprint": _texts_fingerprint(test_texts),
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def fit_transform_pca(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    pca_dim: int,
    seed: int = 42,
    train_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on train portion only if train_idx given, then transform all train and test."""
    pca = PCA(n_components=pca_dim, random_state=seed)
    if train_idx is not None:
        pca.fit(train_emb[train_idx])
        train_pca = pca.transform(train_emb)
    else:
        train_pca = pca.fit_transform(train_emb)
    test_pca = pca.transform(test_emb)
    return train_pca, test_pca, pca


def _one_hot_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categoricals. If train_idx is given, fit only on train portion."""
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) == 0:
        return train_df, test_df
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    fit_df = train_df.iloc[train_idx] if train_idx is not None else train_df
    enc.fit(fit_df[cat_cols])
    tr_cat = pd.DataFrame(enc.transform(train_df[cat_cols]), columns=enc.get_feature_names_out(cat_cols))
    te_cat = pd.DataFrame(enc.transform(test_df[cat_cols]), columns=enc.get_feature_names_out(cat_cols))
    train_out = pd.concat([train_df.drop(columns=cat_cols).reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
    test_out = pd.concat([test_df.drop(columns=cat_cols).reset_index(drop=True), te_cat.reset_index(drop=True)], axis=1)
    return train_out, test_out


def _train_eval_and_predict(
    label: str,
    X_train_all: pd.DataFrame,
    X_test_all: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_ids: pd.Series,
    test_scale: np.ndarray,
    seed: int,
    fit_idx: Optional[np.ndarray] = None,
    models_to_run: Optional[set] = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """fit_idx: indices to use for final refit (train+val only). If None, uses train_idx (no refit on val)."""
    if fit_idx is None:
        fit_idx = np.concatenate([train_idx, val_idx])
    X_tr = X_train_all.iloc[train_idx]
    X_val = X_train_all.iloc[val_idx]
    y_tr = y[train_idx]
    y_val = y[val_idx]
    models = []
    val_metrics = {}
    val_preds = []
    def _run(model_name: str) -> bool:
        return models_to_run is None or model_name in models_to_run

    if USE_CATBOOST and _run("catboost"):
        cat = CatBoostRegressor(
            depth=6,
            learning_rate=0.008999522235187652,
            l2_leaf_reg=0.0052562390910365075,
            iterations=1457,
            random_seed=seed,
            verbose=False
        )
        cat.fit(X_tr, y_tr)
        pred_val = cat.predict(X_val)
        val_metrics["catboost"] = evaluate_model(np.expm1(y_val), np.expm1(pred_val), f"CatBoost (val, {label})")
        val_preds.append(pred_val)
        models.append(("catboost", cat))

    if USE_XGB and _run("xgb"):
        xgb = XGBRegressor(
            n_estimators=1977,
            max_depth=4,
            learning_rate=0.1373050537354267,
            subsample=0.6856210530449924,
            colsample_bytree=0.6770661992767474,
            reg_alpha=2.7410824730098713,
            reg_lambda=5.843059794133128,
            random_state=seed,
            n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        pred_val = xgb.predict(X_val)
        val_metrics["xgb"] = evaluate_model(np.expm1(y_val), np.expm1(pred_val), f"XGBoost (val, {label})")
        val_preds.append(pred_val)
        models.append(("xgb", xgb))

    if USE_LGBM and _run("lgbm"):
        lgbm = LGBMRegressor(
            objective="regression",
            num_leaves=35,
            learning_rate=0.025646636381206286,
            n_estimators=841,
            max_bin=155,
            bagging_fraction=0.4012842862138498,
            bagging_freq=4,
            feature_fraction=0.6690194045906183,
            random_state=seed,
            verbose=-1
        )
        lgbm.fit(X_tr, y_tr)
        pred_val = lgbm.predict(X_val)
        val_metrics["lgbm"] = evaluate_model(np.expm1(y_val), np.expm1(pred_val), f"LightGBM (val, {label})")
        val_preds.append(pred_val)
        models.append(("lgbm", lgbm))

    if USE_TABPFN and _run("tabpfn"):
        if not TABPFN_AVAILABLE:
            print("\n[WARN] TabPFN not available; skipping TabPFN.")
        else:
            os.environ["TABPFN_DEVICE"] = "cuda"
            os.environ["TABPFN_ALLOW_LARGE_DATASETS"] = "1"
            tabpfn = TabPFNRegressor(
                device="cuda",
                n_estimators=8
            )
            tabpfn.fit(X_tr, y_tr)
            pred_val = tabpfn.predict(X_val)
            val_metrics["tabpfn"] = evaluate_model(np.expm1(y_val), np.expm1(pred_val), f"TabPFN (val, {label})")
            val_preds.append(pred_val)
            models.append(("tabpfn", tabpfn))

    if len(models) == 0:
        raise ValueError("No learner enabled. Set one of USE_CATBOOST/USE_XGB/USE_LGBM = True")

    preds = []
    per_model_preds = {}
    # Refit on train+val only (exclude test) to avoid leakage when evaluating test
    X_fit = X_train_all.iloc[fit_idx]
    y_fit = y[fit_idx]
    for name, model in models:
        model.fit(X_fit, y_fit)
        pred_test = model.predict(X_test_all)
        saleprice_pred = np.expm1(pred_test) * test_scale
        preds.append(saleprice_pred)
        per_model_preds[name] = pd.Series(saleprice_pred, index=test_ids)
    pred_final = np.mean(np.vstack(preds), axis=0)
    return pd.DataFrame({"Id": test_ids, "SalePrice": pred_final}), per_model_preds, val_metrics


def _build_rag_distance_matrix(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    label_encoders: dict,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Build (numeric + label-encoded cat) matrix, optionally fit scaler on fit_df. Returns (matrix, scaler)."""
    parts = []
    if numeric_cols:
        part = df[numeric_cols].fillna(fit_df[numeric_cols].median()).values
        parts.append(part)
    if cat_cols:
        enc_list = []
        for col in cat_cols:
            if col not in df.columns or col not in label_encoders:
                continue
            le = label_encoders[col]
            col_clean = df[col].astype(str).fillna("None")
            label_map = {label: idx for idx, label in enumerate(le.classes_)}
            default_label = "None" if "None" in label_map else le.classes_[0]
            default_idx = label_map[default_label]
            enc_list.append(col_clean.map(lambda x: label_map.get(x, default_idx)).values.reshape(-1, 1))
        if enc_list:
            parts.append(np.hstack(enc_list))
    if not parts:
        raise ValueError("No columns available for RAG distance.")
    X = np.hstack(parts).astype(np.float64)
    if scaler is None:
        scaler = StandardScaler()
        fit_parts = []
        if numeric_cols:
            fit_parts.append(fit_df[numeric_cols].fillna(fit_df[numeric_cols].median()).values)
        if cat_cols:
            enc_list = []
            for col in cat_cols:
                if col not in fit_df.columns or col not in label_encoders:
                    continue
                le = label_encoders[col]
                col_clean = fit_df[col].astype(str).fillna("None")
                label_map = {label: idx for idx, label in enumerate(le.classes_)}
                default_label = "None" if "None" in label_map else le.classes_[0]
                default_idx = label_map[default_label]
                enc_list.append(col_clean.map(lambda x: label_map.get(x, default_idx)).values.reshape(-1, 1))
            if enc_list:
                fit_parts.append(np.hstack(enc_list))
        fit_X = np.hstack(fit_parts).astype(np.float64)
        scaler.fit(fit_X)
    X_scaled = scaler.transform(X)
    return X_scaled, scaler


def _get_rag_neighbor_indices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    numeric_cols: List[str],
    cat_cols: List[str],
    stratify_cols: Optional[List[str]],
    label_encoders: dict,
    train_idx: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return (train_nei_list, test_nei_list). Each element is array of indices into train_df."""
    from sklearn.neighbors import NearestNeighbors

    numeric_cols = [c for c in numeric_cols if c in train_df.columns]
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    if not numeric_cols and not cat_cols:
        raise ValueError("No columns available for RAG similarity.")

    fit_idx = train_idx if train_idx is not None else np.arange(len(train_df))
    train_fit = train_df.iloc[fit_idx]

    def _global_nn() -> Tuple[List[np.ndarray], List[np.ndarray]]:
        fit_X, scaler = _build_rag_distance_matrix(train_fit, train_fit, numeric_cols, cat_cols, label_encoders)
        train_X, _ = _build_rag_distance_matrix(train_df, train_fit, numeric_cols, cat_cols, label_encoders, scaler)
        test_X, _ = _build_rag_distance_matrix(test_df, train_fit, numeric_cols, cat_cols, label_encoders, scaler)
        k_fit = min(k + 1, len(train_fit))
        nn = NearestNeighbors(n_neighbors=k_fit, metric="euclidean", n_jobs=-1)
        nn.fit(fit_X)
        train_nei_fit = nn.kneighbors(train_X, return_distance=False)
        test_k = min(k, len(train_fit))
        test_nei_fit = nn.kneighbors(test_X, n_neighbors=test_k, return_distance=False)
        train_nei_list = []
        for i in range(len(train_df)):
            idxs = list(train_nei_fit[i])
            if train_idx is not None:
                pos_in_fit = np.where(fit_idx == i)[0]
                if len(pos_in_fit) > 0 and pos_in_fit[0] in idxs:
                    idxs = [x for x in idxs if x != int(pos_in_fit[0])]
            else:
                if i in idxs:
                    idxs = [x for x in idxs if x != i]
            idxs = idxs[:k]
            train_nei_list.append(fit_idx[np.array(idxs)])
        test_nei_list = [fit_idx[test_nei_fit[j]] for j in range(len(test_df))]
        return (train_nei_list, test_nei_list)

    if not stratify_cols or not all(c in train_df.columns and c in test_df.columns for c in stratify_cols):
        return _global_nn()

    stratify_cols = [c for c in stratify_cols if c in train_df.columns and c in test_df.columns]
    train_keys = list(zip(*[train_df[c].astype(str).fillna("None").values for c in stratify_cols]))
    test_keys = list(zip(*[test_df[c].astype(str).fillna("None").values for c in stratify_cols]))
    fit_keys = list(zip(*[train_fit[c].astype(str).fillna("None").values for c in stratify_cols]))

    train_nei_global, test_nei_global = _global_nn()

    train_nei_list = []
    for i in range(len(train_df)):
        key = train_keys[i]
        subset_mask = np.array([fit_keys[fi] == key for fi in range(len(fit_keys))])
        subset_idx = fit_idx[subset_mask]
        if len(subset_idx) < 2:
            train_nei_list.append(train_nei_global[i])
            continue
        subset_fit = train_df.iloc[subset_idx]
        X_sub, scaler_sub = _build_rag_distance_matrix(subset_fit, subset_fit, numeric_cols, cat_cols, label_encoders)
        row_X, _ = _build_rag_distance_matrix(train_df.iloc[i : i + 1], subset_fit, numeric_cols, cat_cols, label_encoders, scaler_sub)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(subset_idx)), metric="euclidean", n_jobs=-1)
        nn.fit(X_sub)
        nei = nn.kneighbors(row_X, return_distance=False)[0]
        self_pos = np.where(subset_idx == i)[0]
        if len(self_pos) > 0:
            nei = np.array([x for x in nei if x != self_pos[0]])
        nei = nei[:k]
        train_nei_list.append(subset_idx[nei])

    test_nei_list = []
    for j in range(len(test_df)):
        key = test_keys[j]
        subset_mask = np.all(
            np.column_stack([train_fit[c].astype(str).fillna("None").values == key[ci] for ci, c in enumerate(stratify_cols)]),
            axis=1,
        )
        subset_idx = fit_idx[subset_mask]
        if len(subset_idx) < k:
            test_nei_list.append(test_nei_global[j])
            continue
        subset_fit = train_df.iloc[subset_idx]
        X_sub, scaler_sub = _build_rag_distance_matrix(subset_fit, subset_fit, numeric_cols, cat_cols, label_encoders)
        row_X, _ = _build_rag_distance_matrix(test_df.iloc[j : j + 1], subset_fit, numeric_cols, cat_cols, label_encoders, scaler_sub)
        nn = NearestNeighbors(n_neighbors=min(k, len(subset_idx)), metric="euclidean", n_jobs=-1)
        nn.fit(X_sub)
        nei = nn.kneighbors(row_X, return_distance=False)[0]
        test_nei_list.append(subset_idx[nei])

    return (train_nei_list, test_nei_list)


def _compute_rag_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    label_encoders: dict,
    train_idx: Optional[np.ndarray] = None,
    rag_mode: str = "hybrid",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RAG features: use rag_mode config to get neighbors, then aggregate."""
    if k <= 0:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

    cfg = RAG_MODE_CONFIG.get(rag_mode, RAG_MODE_CONFIG["hybrid"])
    numeric_cols = [c for c in cfg["numeric"] if c in train_df.columns]
    cat_cols = [c for c in cfg["cat"] if c in train_df.columns]
    stratify_cols = cfg.get("stratify")
    extra_derived = cfg.get("extra_derived")

    train_nei_list, test_nei_list = _get_rag_neighbor_indices(
        train_df, test_df, k, numeric_cols, cat_cols, stratify_cols, label_encoders, train_idx
    )

    def _rag_row_from_neighbors(nei_idx: np.ndarray, source_df: pd.DataFrame, query_row: Optional[pd.Series] = None) -> dict:
        row = {}
        for col in numeric_cols:
            vals = source_df.iloc[nei_idx][col].dropna().values
            if len(vals) > 0:
                row[f"similar house {col}"] = float(np.mean(vals))
        for col in cat_cols:
            vals = source_df.iloc[nei_idx][col].dropna().astype(str).values
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
        if extra_derived and "years_vs_neighbors" in extra_derived and query_row is not None and "YearBuilt" in query_row.index:
            mean_year = row.get("similar house YearBuilt", np.nan)
            if not np.isnan(mean_year):
                row["years_vs_neighbors"] = float(query_row["YearBuilt"]) - mean_year
        return row

    train_rows = []
    for i in range(len(train_df)):
        nei_idx = train_nei_list[i]
        if len(nei_idx) == 0:
            train_rows.append({})
        else:
            train_rows.append(_rag_row_from_neighbors(nei_idx, train_df, train_df.iloc[i] if extra_derived else None))
    test_rows = []
    for j in range(len(test_df)):
        nei_idx = test_nei_list[j]
        if len(nei_idx) == 0:
            test_rows.append({})
        else:
            test_rows.append(_rag_row_from_neighbors(nei_idx, train_df, test_df.iloc[j] if extra_derived else None))

    return pd.DataFrame(train_rows, index=train_df.index), pd.DataFrame(test_rows, index=test_df.index)


def _compute_rag_price_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    label_encoders: dict,
    price_col: str = "SalePrice",
    train_idx: Optional[np.ndarray] = None,
    rag_mode: str = "hybrid",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Neighbour price: mean(SalePrice) of k nearest neighbours. Uses same rag_mode as _compute_rag_features.
    No leakage: test neighbours are indices into train only; price_arr is from train_df."""
    if k <= 0 or price_col not in train_df.columns:
        return (
            pd.DataFrame({"neighbour_price": np.nan}, index=train_df.index),
            pd.DataFrame({"neighbour_price": np.nan}, index=test_df.index),
        )

    cfg = RAG_MODE_CONFIG.get(rag_mode, RAG_MODE_CONFIG["hybrid"])
    numeric_cols = [c for c in cfg["numeric"] if c in train_df.columns]
    cat_cols = [c for c in cfg["cat"] if c in train_df.columns]
    if not numeric_cols and not cat_cols:
        return (
            pd.DataFrame({"neighbour_price": np.nan}, index=train_df.index),
            pd.DataFrame({"neighbour_price": np.nan}, index=test_df.index),
        )

    # 邻域索引与价格均仅基于 train：test 的邻居来自 train_idx，价格取 train_df[price_col]，无泄露
    train_nei_list, test_nei_list = _get_rag_neighbor_indices(
        train_df, test_df, k, numeric_cols, cat_cols, cfg.get("stratify"), label_encoders, train_idx
    )
    price_arr = train_df[price_col].values.astype(np.float64)

    train_prices = [float(np.nanmean(price_arr[nei_idx])) if len(nei_idx) > 0 else np.nan for nei_idx in train_nei_list]
    test_prices = [float(np.nanmean(price_arr[nei_idx])) if len(nei_idx) > 0 else np.nan for nei_idx in test_nei_list]

    return (
        pd.DataFrame({"neighbour_price": train_prices}, index=train_df.index),
        pd.DataFrame({"neighbour_price": test_prices}, index=test_df.index),
    )


def main():
    parser = argparse.ArgumentParser(description="Baseline model with optional OpenAI embeddings")
    parser.add_argument("--seed", type=str, default=str(SEED), help="Comma-separated seeds, e.g. 42,123,456")
    parser.add_argument("--train-path", type=str, default=TRAIN_PATH, help="Only train.csv is used; train/val/test are split from it")
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--openai-emb-model", type=str, default=OPENAI_EMB_MODEL)
    parser.add_argument(
        "--openai-template",
        type=str,
        default=OPENAI_TEMPLATE,
        choices=["structured", "descriptive", "compact_core", "location_quality", "size_age", "narrative", "all"],
        help="Embedding text template: structured/descriptive/compact_core/location_quality/size_age; 'all' runs all templates",
    )
    parser.add_argument("--pca-dim", type=str, default=str(PCA_DIM), help="Comma-separated PCA dims, e.g. 16,32")
    parser.add_argument("--rag-k", type=str, default="8", help="Comma-separated K for RAG, e.g. 5,8,10")
    parser.add_argument(
        "--rag-mode",
        type=str,
        default="hybrid",
        choices=list(RAG_MODE_CONFIG.keys()) + ["all"],
        help="RAG neighbor mode: hybrid, quality_year, location, stratified_neighborhood, stratified_bldg, or 'all' to run each",
    )
    parser.add_argument(
        "--rag-template",
        type=str,
        default="default",
        choices=["default", "compare", "delta", "all"],
        help="emb_with_rag text: default, compare, delta, or 'all' to run all three",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline,emb,rag",
        help="Comma-separated: baseline,emb,rag,rag_price,rag_with_price,emb+rag,emb+rag_price,emb_with_rag,emb_with_rag+rag,emb_with_rag+rag_price"
    )
    parser.add_argument(
        "--best-config",
        action="store_true",
        help="Run only model-specific best rag_k/pca_dim configs",
    )
    args = parser.parse_args()

    seeds = _parse_comma_int(args.seed)
    mode_all_requested = any(m.strip().lower() == "all" for m in args.mode.split(",") if m.strip())
    rag_k_list = _parse_comma_int(args.rag_k)
    pca_dim_list = _parse_comma_int(args.pca_dim)
    best_config = bool(getattr(args, "best_config", False))
    if best_config:
        combo_to_models: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for model_name, cfg in BEST_CONFIG_PER_MODEL.items():
            combo_to_models[(cfg["rag_k"], cfg["pca_dim"])].append(model_name)
        combo_list = sorted(combo_to_models.keys())
        print(f"[--best-config] Running model-specific combos: {dict(combo_to_models)}")
    else:
        combo_to_models = {}
        combo_list = list(product(rag_k_list, pca_dim_list))
    rag_modes = list(RAG_MODE_CONFIG.keys()) if args.rag_mode == "all" else [args.rag_mode]
    templates = (
        ["structured", "descriptive", "compact_core", "location_quality", "size_age"]
        if args.openai_template == "all"
        else [args.openai_template]
    )
    rag_templates = ["default", "compare", "delta"] if args.rag_template == "all" else [args.rag_template]
    if args.rag_mode == "all":
        print(f"Running all RAG modes: {rag_modes}")
    if args.openai_template == "all":
        print(f"Running all OpenAI templates: {templates}")
    if args.rag_template == "all":
        print(f"Running all RAG templates: {rag_templates}")

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Missing: {args.train_path}")

    train_raw = pd.read_csv(args.train_path)

    # Aggregate: (rag_k, pca_dim) -> (mode, model) -> list of RMSLE over seeds
    all_val: dict = defaultdict(lambda: defaultdict(list))
    all_test: dict = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        set_seed(seed)
        train_full = train_raw.drop(columns=["Id"]).copy()
        all_indices = np.arange(len(train_full))
        # 70% train, 10% val, 20% test (all from train.csv)
        train_idx, rest = train_test_split(all_indices, test_size=0.3, random_state=seed)
        val_idx, test_idx = train_test_split(rest, test_size=2.0 / 3.0, random_state=seed)
        test_ids = train_raw["Id"].iloc[test_idx].copy()

        # Preprocessing: fit on train_idx only, then apply to full train
        clean_fit = get_clean_fit(train_full.iloc[train_idx])
        train = apply_clean(train_full, clean_fit)
        train = feature_engineering(train)

        train["PricePerSqrt"] = train["SalePrice"] / train["OverallQual_TotalSF"]
        y = np.log1p(train["PricePerSqrt"])

        train_for_text = train.copy()
        test = train.iloc[test_idx].reset_index(drop=True)
        test_for_text = test.copy()

        X_train_base = train.drop(columns=["SalePrice", "PricePerSqrt"]).copy()
        X_test_base = test.drop(columns=["SalePrice", "PricePerSqrt"]).copy()

        train_numeric = train_for_text.select_dtypes(include=[np.number]).copy()
        if "PricePerSqrt" in train_numeric.columns:
            train_numeric = train_numeric.drop(["PricePerSqrt"], axis=1)
        if "SalePrice" in train_numeric.columns:
            train_numeric = train_numeric.drop(["SalePrice"], axis=1)
        test_numeric = train_numeric.iloc[test_idx].reset_index(drop=True)

        train_cat = train_for_text.select_dtypes(include=["object", "category"]).copy()
        test_cat = train_cat.iloc[test_idx].reset_index(drop=True)
        label_encoders = {}
        train_cat_fit = train_cat.iloc[train_idx]
        for col in train_cat.columns:
            le = LabelEncoder()
            train_col_clean = train_cat_fit[col].astype(str).fillna("None")
            le.fit(train_col_clean)
            label_encoders[col] = le
            label_map = {label: idx for idx, label in enumerate(le.classes_)}
            default_label = "None" if "None" in label_map else le.classes_[0]
            default_idx = label_map[default_label]
            train_col_clean_all = train_cat[col].astype(str).fillna("None")
            test_col_clean = test_cat[col].astype(str).fillna("None")
            train_cat[col] = train_col_clean_all.map(lambda x: label_map.get(x, default_idx))
            test_cat[col] = test_col_clean.map(lambda x: label_map.get(x, default_idx))

        # Test set ground truth (from train.csv holdout) for test metrics
        gt_internal = pd.DataFrame({
            "Id": test_ids.values,
            "SalePrice": np.expm1(y[test_idx]) * train_for_text["OverallQual_TotalSF"].iloc[test_idx].values,
        })

        modes = {m.strip().lower() for m in args.mode.split(",") if m.strip()}
        ALL_MODES = {
            "baseline", "emb", "rag",
            "emb+rag",
            "emb_with_rag", "emb_with_rag+rag",
        }
        if "all" in modes:
            modes = ALL_MODES.copy()
            print(f"Running all modes: {sorted(modes)}")

        for (rag_k, pca_dim) in combo_list:
            rmsle_val_summary = {}
            rmsle_test_summary = {}
            models_to_run = set(combo_to_models.get((rag_k, pca_dim), [])) if best_config else None

            sub_no_emb = None
            preds_no_emb = None
            if "baseline" in modes:
                X_train_no_emb, X_test_no_emb = _one_hot_encode(X_train_base, X_test_base, train_idx)
                print("\n" + "=" * 60)
                print("Validation (NO OpenAI Embedding)")
                print("=" * 60)
                sub_no_emb, preds_no_emb, val_metrics_no_emb = _train_eval_and_predict(
                    "no-emb",
                    X_train_no_emb,
                    X_test_no_emb,
                    y,
                    train_idx,
                    val_idx,
                    test_ids,
                    test_for_text["OverallQual_TotalSF"].values,
                    seed,
                    models_to_run=models_to_run,
                )
                for name, m in val_metrics_no_emb.items():
                    if name != "ensemble":
                        rmsle_val_summary[("baseline", name)] = m.get("rmsle")
                sub_no_emb.to_csv("submission_no_emb.csv", index=False)
                print("\nSaved submission: submission_no_emb.csv")

            emb_data = {}  # template -> (train_emb_df, test_emb_df, sub_emb, preds_emb)
            sub_emb = None
            preds_emb = None
            train_emb_df = None
            test_emb_df = None
            need_emb = "emb" in modes or "emb+rag" in modes or "emb+rag_price" in modes
            if not need_emb:
                pass
            elif not OPENAI_AVAILABLE:
                print("\n[WARN] OpenAI SDK not available, skipping embedding branch.")
            else:
                for current_template in templates:
                    _t = (lambda b: f"{b}@{current_template}") if len(templates) > 1 else (lambda b: b)
                    _t_suffix = f"_{current_template}" if len(templates) > 1 else ""
                    train_texts = build_texts(train_for_text, template=current_template)
                    test_texts = build_texts(test_for_text, template=current_template)

                    cache_dir = _embedding_store_dir(
                        "emb",
                        current_template,
                        args.openai_emb_model,
                        pca_dim,
                        seed,
                    )
                    cached = _try_load_pca_cache(cache_dir, train_texts, test_texts, pca_dim, seed)
                    if cached is not None:
                        train_emb_cached, test_emb_cached, pca_cached = cached
                        if pca_cached is not None:
                            train_emb_pca, test_emb_pca = train_emb_cached, test_emb_cached
                            print(f"\n📦 [Embedding] template={current_template} Loaded cached PCA embeddings from {cache_dir}")
                        else:
                            train_emb_pca, test_emb_pca, _ = fit_transform_pca(
                                train_emb_cached, test_emb_cached, pca_dim, seed=seed, train_idx=train_idx
                            )
                            print(f"\n📦 [Embedding] template={current_template} Loaded raw cache from {cache_dir}, PCA fit on train split")
                    else:
                        train_emb = embed_texts_openai(train_texts, model=args.openai_emb_model)
                        test_emb = embed_texts_openai(test_texts, model=args.openai_emb_model)
                        train_emb_pca, test_emb_pca, pca_model = fit_transform_pca(
                            train_emb, test_emb, pca_dim, seed=seed, train_idx=train_idx
                        )
                        _save_pca_cache(
                            cache_dir,
                            train_emb_pca,
                            test_emb_pca,
                            train_texts,
                            test_texts,
                            pca_dim,
                            seed,
                            pca_model=pca_model,
                        )
                        print(f"   ✅ [Embedding] template={current_template} Saved PCA cache to: {cache_dir}")

                    emb_cols = [f"emb_{i}" for i in range(pca_dim)]
                    _train_emb_df = pd.DataFrame(train_emb_pca, columns=emb_cols, index=train_numeric.index)
                    _test_emb_df = pd.DataFrame(test_emb_pca, columns=emb_cols, index=test_numeric.index)

                    X_train_all = pd.concat([
                        train_numeric.reset_index(drop=True),
                        train_cat.reset_index(drop=True),
                        _train_emb_df.reset_index(drop=True)
                    ], axis=1)
                    X_test_all = pd.concat([
                        test_numeric.reset_index(drop=True),
                        test_cat.reset_index(drop=True),
                        _test_emb_df.reset_index(drop=True)
                    ], axis=1)

                    print("\n" + "=" * 60)
                    emb_label = f", template={current_template}" if len(templates) > 1 else ""
                    print(f"Validation (WITH OpenAI Embedding{emb_label})")
                    print("=" * 60)
                    _sub_emb, _preds_emb, val_metrics_emb = _train_eval_and_predict(
                        "emb",
                        X_train_all,
                        X_test_all,
                        y,
                        train_idx,
                        val_idx,
                        test_ids,
                        test_for_text["OverallQual_TotalSF"].values,
                        seed,
                        models_to_run=models_to_run,
                    )
                    for name, m in val_metrics_emb.items():
                        if name != "ensemble":
                            rmsle_val_summary[(_t("emb"), name)] = m.get("rmsle")
                    emb_data[current_template] = (_train_emb_df, _test_emb_df, _sub_emb, _preds_emb)
                    train_emb_df = _train_emb_df
                    test_emb_df = _test_emb_df
                    sub_emb = _sub_emb
                    preds_emb = _preds_emb
                    out_path = args.output_path.replace(".csv", f"{_t_suffix}.csv") if _t_suffix else args.output_path
                    if "emb" in modes:
                        _sub_emb.to_csv(out_path, index=False)
                        print(f"\nSaved submission: {out_path}")

            for current_rag_mode in rag_modes:
                _m = (lambda base: f"{base}@{current_rag_mode}") if len(rag_modes) > 1 else (lambda base: base)
                _rag_label = f", rag_mode={current_rag_mode}" if len(rag_modes) > 1 else ""
                _rag_suffix = f"_{current_rag_mode}" if len(rag_modes) > 1 else ""
                rag_train_df = None
                rag_test_df = None
                sub_rag = None
                preds_rag = None
                if "rag" in modes or "rag_with_price" in modes or "emb+rag" in modes or "emb_with_rag" in modes or "emb_with_rag+rag" in modes or "emb_with_rag+rag_price" in modes:
                    print(f"\n[RAG] rag_mode={current_rag_mode}")
                    rag_train_df, rag_test_df = _compute_rag_features(
                        train_for_text,
                        test_for_text,
                        rag_k,
                        label_encoders,
                        train_idx=train_idx,
                        rag_mode=current_rag_mode,
                    )

                rag_price_train_df = None
                rag_price_test_df = None
                if "rag_price" in modes or "rag_with_price" in modes or "emb+rag_price" in modes or "emb_with_rag+rag_price" in modes:
                    rag_price_train_df, rag_price_test_df = _compute_rag_price_features(
                        train_for_text,
                        test_for_text,
                        rag_k,
                        label_encoders,
                        price_col="SalePrice",
                        train_idx=train_idx,
                        rag_mode=current_rag_mode,
                    )

                if "rag" in modes:
                    print("\n" + "=" * 60)
                    print(f"Validation (WITH RAG{_rag_label})")
                    print("=" * 60)
                    X_train_rag = pd.concat(
                        [train_numeric.reset_index(drop=True), train_cat.reset_index(drop=True), rag_train_df.reset_index(drop=True)],
                        axis=1,
                    )
                    X_test_rag = pd.concat(
                        [test_numeric.reset_index(drop=True), test_cat.reset_index(drop=True), rag_test_df.reset_index(drop=True)],
                        axis=1,
                    )
                    sub_rag, preds_rag, val_metrics_rag = _train_eval_and_predict(
                        "rag",
                        X_train_rag,
                        X_test_rag,
                        y,
                        train_idx,
                        val_idx,
                        test_ids,
                        test_for_text["OverallQual_TotalSF"].values,
                        seed,
                        models_to_run=models_to_run,
                    )
                    for name, m in val_metrics_rag.items():
                        if name != "ensemble":
                            rmsle_val_summary[(_m("rag"), name)] = m.get("rmsle")
                    sub_rag.to_csv(f"submission_rag{_rag_suffix}.csv", index=False)
                    print(f"\nSaved submission: submission_rag{_rag_suffix}.csv")

                sub_rag_price = None
                preds_rag_price = None
                if "rag_price" in modes and rag_price_train_df is not None:
                    print("\n" + "=" * 60)
                    print(f"Validation (WITH RAG_PRICE{_rag_label})")
                    print("=" * 60)
                    X_train_rag_price = pd.concat(
                        [train_numeric.reset_index(drop=True), train_cat.reset_index(drop=True), rag_price_train_df.reset_index(drop=True)],
                        axis=1,
                    )
                    X_test_rag_price = pd.concat(
                        [test_numeric.reset_index(drop=True), test_cat.reset_index(drop=True), rag_price_test_df.reset_index(drop=True)],
                        axis=1,
                    )
                    sub_rag_price, preds_rag_price, val_metrics_rag_price = _train_eval_and_predict(
                        "rag_price",
                        X_train_rag_price,
                        X_test_rag_price,
                        y,
                        train_idx,
                        val_idx,
                        test_ids,
                        test_for_text["OverallQual_TotalSF"].values,
                        seed,
                        models_to_run=models_to_run,
                    )
                    for name, m in val_metrics_rag_price.items():
                        if name != "ensemble":
                            rmsle_val_summary[(_m("rag_price"), name)] = m.get("rmsle")
                    sub_rag_price.to_csv(f"submission_rag_price{_rag_suffix}.csv", index=False)
                    print(f"\nSaved submission: submission_rag_price{_rag_suffix}.csv")
                    # RAG_PRICE 导出：全量 neighbour_price + 划分 + 真值 + test 上 ensemble 预测
                    _split = np.empty(len(train_raw), dtype=object)
                    _split[train_idx] = "train"
                    _split[val_idx] = "val"
                    _split[test_idx] = "test"
                    df_rag_price_out = pd.DataFrame({
                        "Id": train_raw["Id"].values,
                        "split": _split,
                        "neighbour_price": rag_price_train_df["neighbour_price"].values.astype(np.float64),
                        "SalePrice_true": train_raw["SalePrice"].values.astype(np.float64),
                    })
                    df_rag_price_out = df_rag_price_out.merge(
                        sub_rag_price.rename(columns={"SalePrice": "SalePrice_pred_rag_price"}),
                        on="Id",
                        how="left",
                    )
                    _out_rag_price = f"rag_price_output{_rag_suffix}.csv"
                    df_rag_price_out.to_csv(_out_rag_price, index=False)
                    print(f"Saved RAG_PRICE export: {_out_rag_price} (neighbour_price + split + true + test pred)")
                    # tune_baseline --dataset ames_rag_price：全特征 + target = log1p(PricePerSqrt)
                    _tune_rag_price = f"ames_train_rag_price{_rag_suffix}.csv"
                    df_rag_price_tune = X_train_rag_price.copy()
                    df_rag_price_tune["target"] = y
                    df_rag_price_tune.to_csv(_tune_rag_price, index=False)
                    print(
                        f"Saved for tune_baseline: {_tune_rag_price} "
                        f"(python tune_baseline.py --dataset ames_rag_price --data-path {_tune_rag_price})"
                    )

                sub_rag_with_price = None
                preds_rag_with_price = None
                if "rag_with_price" in modes and rag_train_df is not None and rag_price_train_df is not None:
                    print("\n" + "=" * 60)
                    print(f"Validation (WITH RAG + neighbour_price{_rag_label})")
                    print("=" * 60)
                    X_train_rag_wp = pd.concat(
                        [
                            train_numeric.reset_index(drop=True),
                            train_cat.reset_index(drop=True),
                            rag_train_df.reset_index(drop=True),
                            rag_price_train_df.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    X_test_rag_wp = pd.concat(
                        [
                            test_numeric.reset_index(drop=True),
                            test_cat.reset_index(drop=True),
                            rag_test_df.reset_index(drop=True),
                            rag_price_test_df.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    sub_rag_with_price, preds_rag_with_price, val_metrics_rag_wp = _train_eval_and_predict(
                        "rag_with_price",
                        X_train_rag_wp,
                        X_test_rag_wp,
                        y,
                        train_idx,
                        val_idx,
                        test_ids,
                        test_for_text["OverallQual_TotalSF"].values,
                        seed,
                        models_to_run=models_to_run,
                    )
                    for name, m in val_metrics_rag_wp.items():
                        if name != "ensemble":
                            rmsle_val_summary[(_m("rag_with_price"), name)] = m.get("rmsle")
                    sub_rag_with_price.to_csv(f"submission_rag_with_price{_rag_suffix}.csv", index=False)
                    print(f"\nSaved submission: submission_rag_with_price{_rag_suffix}.csv")

                sub_emb_rag = None
                preds_emb_rag = None
                sub_emb_rag_price = None
                preds_emb_rag_price = None
                emb_rag_data = {}  # template -> (sub_emb_rag, preds_emb_rag)
                emb_rag_price_data = {}
                if "emb+rag" in modes:
                    for current_template in templates:
                        if current_template not in emb_data:
                            continue
                        _train_emb_df, _test_emb_df, _sub_emb, _ = emb_data[current_template]
                        _emb_key = (lambda b: f"{b}@{current_template}") if len(templates) > 1 else _m
                        _emb_suffix = f"_{current_template}" if len(templates) > 1 else ""
                        print("\n" + "=" * 60)
                        print(f"Validation (WITH EMB + RAG{_rag_label}, template={current_template})")
                        print("=" * 60)
                        X_train_emb_rag = pd.concat(
                            [
                                train_numeric.reset_index(drop=True),
                                train_cat.reset_index(drop=True),
                                rag_train_df.reset_index(drop=True),
                                _train_emb_df.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        X_test_emb_rag = pd.concat(
                            [
                                test_numeric.reset_index(drop=True),
                                test_cat.reset_index(drop=True),
                                rag_test_df.reset_index(drop=True),
                                _test_emb_df.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        _sub_emb_rag, _preds_emb_rag, val_metrics_emb_rag = _train_eval_and_predict(
                            "emb+rag",
                            X_train_emb_rag,
                            X_test_emb_rag,
                            y,
                            train_idx,
                            val_idx,
                            test_ids,
                            test_for_text["OverallQual_TotalSF"].values,
                            seed,
                            models_to_run=models_to_run,
                        )
                        for name, m in val_metrics_emb_rag.items():
                            if name != "ensemble":
                                rmsle_val_summary[(_emb_key("emb+rag"), name)] = m.get("rmsle")
                        sub_emb_rag = _sub_emb_rag
                        preds_emb_rag = _preds_emb_rag
                        emb_rag_data[current_template] = (_sub_emb_rag, _preds_emb_rag)
                        _sub_emb_rag.to_csv(f"submission_emb_rag{_rag_suffix}{_emb_suffix}.csv", index=False)
                        print(f"\nSaved submission: submission_emb_rag{_rag_suffix}{_emb_suffix}.csv")

                if "emb+rag_price" in modes and rag_price_train_df is not None:
                    for current_template in templates:
                        if current_template not in emb_data:
                            continue
                        _train_emb_df, _test_emb_df, _sub_emb, _ = emb_data[current_template]
                        _emb_key = (lambda b: f"{b}@{current_template}") if len(templates) > 1 else _m
                        _emb_suffix = f"_{current_template}" if len(templates) > 1 else ""
                        print("\n" + "=" * 60)
                        print(f"Validation (WITH EMB + RAG_PRICE{_rag_label}, template={current_template})")
                        print("=" * 60)
                        X_train_emb_rag_price = pd.concat(
                            [
                                train_numeric.reset_index(drop=True),
                                train_cat.reset_index(drop=True),
                                rag_price_train_df.reset_index(drop=True),
                                _train_emb_df.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        X_test_emb_rag_price = pd.concat(
                            [
                                test_numeric.reset_index(drop=True),
                                test_cat.reset_index(drop=True),
                                rag_price_test_df.reset_index(drop=True),
                                _test_emb_df.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        _sub_emb_rag_price, _preds_emb_rag_price, val_metrics_emb_rag_price = _train_eval_and_predict(
                            "emb+rag_price",
                            X_train_emb_rag_price,
                            X_test_emb_rag_price,
                            y,
                            train_idx,
                            val_idx,
                            test_ids,
                            test_for_text["OverallQual_TotalSF"].values,
                            seed,
                            models_to_run=models_to_run,
                        )
                        for name, m in val_metrics_emb_rag_price.items():
                            if name != "ensemble":
                                rmsle_val_summary[(_emb_key("emb+rag_price"), name)] = m.get("rmsle")
                        sub_emb_rag_price = _sub_emb_rag_price
                        preds_emb_rag_price = _preds_emb_rag_price
                        emb_rag_price_data[current_template] = (_sub_emb_rag_price, _preds_emb_rag_price)
                        _sub_emb_rag_price.to_csv(f"submission_emb_rag_price{_rag_suffix}{_emb_suffix}.csv", index=False)
                        print(f"\nSaved submission: submission_emb_rag_price{_rag_suffix}{_emb_suffix}.csv")

                train_emb_rag_df = None
                test_emb_rag_df = None
                sub_emb_with_rag = None
                preds_emb_with_rag = None
                sub_emb_with_rag_plus_rag = None
                preds_emb_with_rag_plus_rag = None
                sub_emb_with_rag_plus_rag_price = None
                preds_emb_with_rag_plus_rag_price = None
                emb_with_rag_results = {}  # rag_tmpl -> (sub, preds)
                emb_with_rag_plus_rag_results = {}
                emb_with_rag_plus_rag_price_results = {}

                if "emb_with_rag" in modes or "emb_with_rag+rag" in modes or "emb_with_rag+rag_price" in modes:
                    if not OPENAI_AVAILABLE:
                        print("\n[WARN] emb_with_rag requested but OpenAI embedding is not available; skipping.")
                    else:
                        for current_rag_tmpl in rag_templates:
                            _rt_label = f", rag_template={current_rag_tmpl}" if len(rag_templates) > 1 else ""
                            _rt_suffix = f"_{current_rag_tmpl}" if len(rag_templates) > 1 else ""
                            _rt_mode = f"@{current_rag_tmpl}" if len(rag_templates) > 1 else ""

                            print("\n" + "=" * 60)
                            print(f"Preparing emb_with_rag embeddings (rag_template={current_rag_tmpl})")
                            print("=" * 60)
                            train_texts_rag = build_texts_with_rag(
                                train_for_text, rag_train_df, base_template="structured", rag_template=current_rag_tmpl, rag_k=rag_k
                            )
                            test_texts_rag = build_texts_with_rag(
                                test_for_text, rag_test_df, base_template="structured", rag_template=current_rag_tmpl, rag_k=rag_k
                            )
                            cache_dir = _embedding_store_dir(
                                "emb_with_rag",
                                f"structured_rag_{current_rag_tmpl}",
                                args.openai_emb_model,
                                pca_dim,
                                seed,
                                rag_k=rag_k,
                            )
                            cached = _try_load_pca_cache(cache_dir, train_texts_rag, test_texts_rag, pca_dim, seed)
                            if cached is not None:
                                train_emb_cached_rag, test_emb_cached_rag, pca_cached_rag = cached
                                if pca_cached_rag is not None:
                                    train_emb_pca_rag, test_emb_pca_rag = train_emb_cached_rag, test_emb_cached_rag
                                    print(f"\n📦 [Embedding+RAG] Loaded cached PCA embeddings (full fit) from {cache_dir}")
                                else:
                                    train_emb_pca_rag, test_emb_pca_rag, _ = fit_transform_pca(
                                        train_emb_cached_rag, test_emb_cached_rag, pca_dim, seed=seed, train_idx=train_idx
                                    )
                                    print(f"\n📦 [Embedding+RAG] Loaded raw cache from {cache_dir}, PCA fit on train split")
                            else:
                                train_emb_rag = embed_texts_openai(train_texts_rag, model=args.openai_emb_model)
                                test_emb_rag = embed_texts_openai(test_texts_rag, model=args.openai_emb_model)
                                train_emb_pca_rag, test_emb_pca_rag, pca_model_rag = fit_transform_pca(
                                    train_emb_rag, test_emb_rag, pca_dim, seed=seed, train_idx=train_idx
                                )
                                _save_pca_cache(
                                    cache_dir,
                                    train_emb_pca_rag,
                                    test_emb_pca_rag,
                                    train_texts_rag,
                                    test_texts_rag,
                                    pca_dim,
                                    seed,
                                    pca_model=pca_model_rag,
                                )
                                print(f"   ✅ Saved full PCA embeddings cache to: {cache_dir}")
                            emb_rag_cols = [f"emb_rag_{i}" for i in range(pca_dim)]
                            train_emb_rag_df = pd.DataFrame(train_emb_pca_rag, columns=emb_rag_cols, index=train_numeric.index)
                            test_emb_rag_df = pd.DataFrame(test_emb_pca_rag, columns=emb_rag_cols, index=test_numeric.index)

                            if "emb_with_rag" in modes and train_emb_rag_df is not None:
                                print("\n" + "=" * 60)
                                print(f"Validation (WITH EMBEDDING TEXT + RAG{_rag_label}{_rt_label})")
                                print("=" * 60)
                                X_train_emb_with_rag = pd.concat(
                                    [
                                        train_numeric.reset_index(drop=True),
                                        train_cat.reset_index(drop=True),
                                        train_emb_rag_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                X_test_emb_with_rag = pd.concat(
                                    [
                                        test_numeric.reset_index(drop=True),
                                        test_cat.reset_index(drop=True),
                                        test_emb_rag_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                _sub, _preds, val_metrics_emb_with_rag = _train_eval_and_predict(
                                    "emb_with_rag",
                                    X_train_emb_with_rag,
                                    X_test_emb_with_rag,
                                    y,
                                    train_idx,
                                    val_idx,
                                    test_ids,
                                    test_for_text["OverallQual_TotalSF"].values,
                                    seed,
                                    models_to_run=models_to_run,
                                )
                                for name, m in val_metrics_emb_with_rag.items():
                                    if name != "ensemble":
                                        rmsle_val_summary[(_m("emb_with_rag") + _rt_mode, name)] = m.get("rmsle")
                                _sub.to_csv(f"submission_emb_with_rag{_rag_suffix}{_rt_suffix}.csv", index=False)
                                print(f"\nSaved submission: submission_emb_with_rag{_rag_suffix}{_rt_suffix}.csv")
                                emb_with_rag_results[current_rag_tmpl] = (_sub, _preds)
                                if len(rag_templates) == 1:
                                    sub_emb_with_rag, preds_emb_with_rag = _sub, _preds

                            if "emb_with_rag+rag" in modes and train_emb_rag_df is not None:
                                print("\n" + "=" * 60)
                                print(f"Validation (WITH EMBEDDING TEXT + RAG + RAG FEATURES{_rag_label}{_rt_label})")
                                print("=" * 60)
                                X_train_emb_with_rag_plus = pd.concat(
                                    [
                                        train_numeric.reset_index(drop=True),
                                        train_cat.reset_index(drop=True),
                                        rag_train_df.reset_index(drop=True),
                                        train_emb_rag_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                X_test_emb_with_rag_plus = pd.concat(
                                    [
                                        test_numeric.reset_index(drop=True),
                                        test_cat.reset_index(drop=True),
                                        rag_test_df.reset_index(drop=True),
                                        test_emb_rag_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                _sub, _preds, val_metrics_emb_with_rag_plus = _train_eval_and_predict(
                                    "emb_with_rag+rag",
                                    X_train_emb_with_rag_plus,
                                    X_test_emb_with_rag_plus,
                                    y,
                                    train_idx,
                                    val_idx,
                                    test_ids,
                                    test_for_text["OverallQual_TotalSF"].values,
                                    seed,
                                    models_to_run=models_to_run,
                                )
                                for name, m in val_metrics_emb_with_rag_plus.items():
                                    if name != "ensemble":
                                        rmsle_val_summary[(_m("emb_with_rag+rag") + _rt_mode, name)] = m.get("rmsle")
                                _sub.to_csv(f"submission_emb_with_rag_plus_rag{_rag_suffix}{_rt_suffix}.csv", index=False)
                                print(f"\nSaved submission: submission_emb_with_rag_plus_rag{_rag_suffix}{_rt_suffix}.csv")
                                emb_with_rag_plus_rag_results[current_rag_tmpl] = (_sub, _preds)
                                if len(rag_templates) == 1:
                                    sub_emb_with_rag_plus_rag, preds_emb_with_rag_plus_rag = _sub, _preds

                            if "emb_with_rag+rag_price" in modes and train_emb_rag_df is not None and rag_price_train_df is not None:
                                print("\n" + "=" * 60)
                                print(f"Validation (WITH EMBEDDING TEXT + RAG_PRICE{_rag_label}{_rt_label})")
                                print("=" * 60)
                                X_train_emb_with_rag_plus_price = pd.concat(
                                    [
                                        train_numeric.reset_index(drop=True),
                                        train_cat.reset_index(drop=True),
                                        rag_price_train_df.reset_index(drop=True),
                                        train_emb_rag_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                X_test_emb_with_rag_plus_price = pd.concat(
                                    [
                                        test_numeric.reset_index(drop=True),
                                        test_cat.reset_index(drop=True),
                                        rag_price_test_df.reset_index(drop=True),
                                        test_emb_rag_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                _sub, _preds, val_metrics_emb_with_rag_plus_price = _train_eval_and_predict(
                                    "emb_with_rag+rag_price",
                                    X_train_emb_with_rag_plus_price,
                                    X_test_emb_with_rag_plus_price,
                                    y,
                                    train_idx,
                                    val_idx,
                                    test_ids,
                                    test_for_text["OverallQual_TotalSF"].values,
                                    seed,
                                    models_to_run=models_to_run,
                                )
                                for name, m in val_metrics_emb_with_rag_plus_price.items():
                                    if name != "ensemble":
                                        rmsle_val_summary[(_m("emb_with_rag+rag_price") + _rt_mode, name)] = m.get("rmsle")
                                _sub.to_csv(f"submission_emb_with_rag_plus_rag_price{_rag_suffix}{_rt_suffix}.csv", index=False)
                                print(f"\nSaved submission: submission_emb_with_rag_plus_rag_price{_rag_suffix}{_rt_suffix}.csv")
                                emb_with_rag_plus_rag_price_results[current_rag_tmpl] = (_sub, _preds)
                                if len(rag_templates) == 1:
                                    sub_emb_with_rag_plus_rag_price, preds_emb_with_rag_plus_rag_price = _sub, _preds

                gt = gt_internal
                if sub_rag is not None:
                    merged = sub_rag.merge(gt, on="Id", suffixes=("_pred", "_true"))
                    for name, pred in preds_rag.items():
                        if name == "ensemble":
                            continue
                        aligned_pred = merged["Id"].map(pred).values
                        gt_metrics = evaluate_model(
                            merged["SalePrice_true"].values, aligned_pred, f"Test rag {current_rag_mode} {name}", quiet=True
                        )
                        rmsle_test_summary[(_m("rag"), name)] = gt_metrics.get("rmsle")
                if sub_rag_price is not None:
                    merged = sub_rag_price.merge(gt, on="Id", suffixes=("_pred", "_true"))
                    for name, pred in preds_rag_price.items():
                        if name == "ensemble":
                            continue
                        aligned_pred = merged["Id"].map(pred).values
                        gt_metrics = evaluate_model(
                            merged["SalePrice_true"].values, aligned_pred, f"Test rag_price {current_rag_mode} {name}", quiet=True
                        )
                        rmsle_test_summary[(_m("rag_price"), name)] = gt_metrics.get("rmsle")
                if sub_rag_with_price is not None:
                    merged = sub_rag_with_price.merge(gt, on="Id", suffixes=("_pred", "_true"))
                    for name, pred in preds_rag_with_price.items():
                        if name == "ensemble":
                            continue
                        aligned_pred = merged["Id"].map(pred).values
                        gt_metrics = evaluate_model(
                            merged["SalePrice_true"].values, aligned_pred, f"Test rag_with_price {current_rag_mode} {name}", quiet=True
                        )
                        rmsle_test_summary[(_m("rag_with_price"), name)] = gt_metrics.get("rmsle")
                if len(templates) > 1:
                    for _t in templates:
                        if _t not in emb_rag_data:
                            continue
                        _sub, _preds = emb_rag_data[_t]
                        merged = _sub.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in _preds.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb+rag {_t} {name}", quiet=True
                            )
                            rmsle_test_summary[(f"emb+rag@{_t}", name)] = gt_metrics.get("rmsle")
                    for _t in templates:
                        if _t not in emb_rag_price_data:
                            continue
                        _sub, _preds = emb_rag_price_data[_t]
                        merged = _sub.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in _preds.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb+rag_price {_t} {name}", quiet=True
                            )
                            rmsle_test_summary[(f"emb+rag_price@{_t}", name)] = gt_metrics.get("rmsle")
                else:
                    if sub_emb_rag is not None:
                        merged = sub_emb_rag.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in preds_emb_rag.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb+rag {current_rag_mode} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb+rag"), name)] = gt_metrics.get("rmsle")
                    if sub_emb_rag_price is not None:
                        merged = sub_emb_rag_price.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in preds_emb_rag_price.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb+rag_price {current_rag_mode} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb+rag_price"), name)] = gt_metrics.get("rmsle")
                if len(rag_templates) > 1:
                    for _rt in rag_templates:
                        if _rt not in emb_with_rag_results:
                            continue
                        _sub, _preds = emb_with_rag_results[_rt]
                        merged = _sub.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in _preds.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb_with_rag {current_rag_mode} {_rt} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb_with_rag") + f"@{_rt}", name)] = gt_metrics.get("rmsle")
                    for _rt in rag_templates:
                        if _rt not in emb_with_rag_plus_rag_results:
                            continue
                        _sub, _preds = emb_with_rag_plus_rag_results[_rt]
                        merged = _sub.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in _preds.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb_with_rag+rag {current_rag_mode} {_rt} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb_with_rag+rag") + f"@{_rt}", name)] = gt_metrics.get("rmsle")
                    for _rt in rag_templates:
                        if _rt not in emb_with_rag_plus_rag_price_results:
                            continue
                        _sub, _preds = emb_with_rag_plus_rag_price_results[_rt]
                        merged = _sub.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in _preds.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb_with_rag+rag_price {current_rag_mode} {_rt} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb_with_rag+rag_price") + f"@{_rt}", name)] = gt_metrics.get("rmsle")
                else:
                    if sub_emb_with_rag is not None:
                        merged = sub_emb_with_rag.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in preds_emb_with_rag.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb_with_rag {current_rag_mode} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb_with_rag"), name)] = gt_metrics.get("rmsle")
                    if sub_emb_with_rag_plus_rag is not None:
                        merged = sub_emb_with_rag_plus_rag.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in preds_emb_with_rag_plus_rag.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb_with_rag+rag {current_rag_mode} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb_with_rag+rag"), name)] = gt_metrics.get("rmsle")
                    if sub_emb_with_rag_plus_rag_price is not None:
                        merged = sub_emb_with_rag_plus_rag_price.merge(gt, on="Id", suffixes=("_pred", "_true"))
                        for name, pred in preds_emb_with_rag_plus_rag_price.items():
                            if name == "ensemble":
                                continue
                            aligned_pred = merged["Id"].map(pred).values
                            gt_metrics = evaluate_model(
                                merged["SalePrice_true"].values, aligned_pred, f"Test emb_with_rag+rag_price {current_rag_mode} {name}", quiet=True
                            )
                            rmsle_test_summary[(_m("emb_with_rag+rag_price"), name)] = gt_metrics.get("rmsle")

            gt = gt_internal
            if sub_no_emb is not None:
                merged = sub_no_emb.merge(gt, on="Id", suffixes=("_pred", "_true"))
                for name, pred in preds_no_emb.items():
                    if name == "ensemble":
                        continue
                    aligned_pred = merged["Id"].map(pred).values
                    gt_metrics = evaluate_model(
                        merged["SalePrice_true"].values, aligned_pred, f"Test baseline {name}", quiet=True
                    )
                    rmsle_test_summary[("baseline", name)] = gt_metrics.get("rmsle")
            for current_template in templates:
                if current_template not in emb_data:
                    continue
                _, _, _sub_emb, _preds_emb = emb_data[current_template]
                if _sub_emb is None:
                    continue
                _emb_tkey = f"emb@{current_template}" if len(templates) > 1 else "emb"
                merged = _sub_emb.merge(gt, on="Id", suffixes=("_pred", "_true"))
                for name, pred in _preds_emb.items():
                    if name == "ensemble":
                        continue
                    aligned_pred = merged["Id"].map(pred).values
                    gt_metrics = evaluate_model(
                        merged["SalePrice_true"].values, aligned_pred, f"Test emb {current_template} {name}", quiet=True
                    )
                    rmsle_test_summary[(_emb_tkey, name)] = gt_metrics.get("rmsle")

            for (mode, model), v in rmsle_val_summary.items():
                all_val[(rag_k, pca_dim)][(mode, model)].append(v)
            for (mode, model), v in rmsle_test_summary.items():
                all_test[(rag_k, pca_dim)][(mode, model)].append(v)

    mode_order = [
        "baseline", "emb", "rag", "rag_price", "rag_with_price",
        "emb+rag", "emb+rag_price",
        "emb_with_rag", "emb_with_rag+rag", "emb_with_rag+rag_price",
    ]
    model_order = ["catboost", "xgb", "lgbm", "tabpfn"]
    mode_rank = {m: i for i, m in enumerate(mode_order)}
    model_rank = {m: i for i, m in enumerate(model_order)}
    # For --rag-mode all, mode is e.g. "rag@hybrid"; sort by base mode then full mode
    def _mode_sort_key(k):
        mode, model = k
        base_mode = mode.split("@")[0] if "@" in mode else mode
        return (mode_rank.get(base_mode, 999), mode, model_rank.get(model, 999))
    # 实验设计：所有选择（k / rag_mode / template / 最终配置）必须仅基于 Val_RMSLE；Test 只用于最终报告一次
    for (rag_k, pca_dim) in combo_list:
        val_d = dict(all_val[(rag_k, pca_dim)])
        test_d = dict(all_test[(rag_k, pca_dim)])
        all_keys = sorted(
            set(val_d.keys()) | set(test_d.keys()),
            key=_mode_sort_key,
        )
        if not all_keys:
            continue
        rows = []
        for (mode, model) in all_keys:
            val_list = val_d.get((mode, model), [])
            val_mean = np.mean(val_list) if val_list else None
            rows.append({
                "Mode": mode,
                "Model": model,
                "Val_RMSLE": f"{val_mean:.6f}" if val_mean is not None else "-",
            })
        table_df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print(f"Val_RMSLE only (rag_k={rag_k}, pca_dim={pca_dim}) — mean over {len(seeds)} seed(s). Select config by Val only.")
        print("=" * 80)
        print(table_df.to_string(index=False))
        print("=" * 80)
        if mode_all_requested:
            test_rows = []
            for (mode, model) in all_keys:
                test_list = test_d.get((mode, model), [])
                test_mean = np.mean(test_list) if test_list else None
                test_rows.append({
                    "Mode": mode,
                    "Model": model,
                    "Test_RMSLE": f"{test_mean:.6f}" if test_mean is not None else "-",
                })
            test_table_df = pd.DataFrame(test_rows)
            print("\n" + "=" * 80)
            print(
                f"Test_RMSLE (mode=all only, rag_k={rag_k}, pca_dim={pca_dim}) "
                f"— mean over {len(seeds)} seed(s)"
            )
            print("=" * 80)
            print(test_table_df.to_string(index=False))
            print("=" * 80)

    # 按 Val_RMSLE 选出的最佳配置，仅在此报告一次 Test_RMSLE（避免把 test 当 validation 用）
    best_val_mean = np.inf
    best_key = None
    for (rag_k, pca_dim) in combo_list:
        val_d = dict(all_val[(rag_k, pca_dim)])
        test_d = dict(all_test[(rag_k, pca_dim)])
        for (mode, model) in val_d:
            val_list = val_d[(mode, model)]
            if not val_list:
                continue
            vm = np.mean(val_list)
            if vm < best_val_mean:
                best_val_mean = vm
                test_list = test_d.get((mode, model), [])
                best_test_mean = np.mean(test_list) if test_list else None
                best_key = (rag_k, pca_dim, mode, model, best_test_mean)
    if best_key is not None:
        rag_k_b, pca_dim_b, mode_b, model_b, test_mean_b = best_key
        print("\n" + "=" * 80)
        print("Best config by Val_RMSLE (report Test_RMSLE once only):")
        print(f"  rag_k={rag_k_b}, pca_dim={pca_dim_b}, mode={mode_b}, model={model_b}")
        print(f"  Val_RMSLE = {best_val_mean:.6f}  →  Test_RMSLE = {test_mean_b:.6f}" if test_mean_b is not None else f"  Val_RMSLE = {best_val_mean:.6f}  →  Test_RMSLE = N/A")
        print("=" * 80)


if __name__ == "__main__":
    main()