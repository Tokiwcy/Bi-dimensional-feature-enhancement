import os
import json
import hashlib
import warnings
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.special import boxcox

warnings.filterwarnings("ignore")

# =========================
# Optional models
# =========================
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM未安装，将跳过LightGBM模型")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost未安装，将跳过CatBoost模型")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: OpenAI库未安装，将跳过embedding功能")

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    try:
        import tabpfn  # noqa: F401
        TABPFN_AVAILABLE = True
    except ImportError:
        TABPFN_AVAILABLE = False
        print("警告: TabPFN未安装，将跳过TabPFN模型")
        print("  安装方法: pip install tabpfn")

print("=" * 90)
print("房价预测模型训练 (6-MODE FINAL: baseline/emb/rag/emb+rag/emb_with_rag/emb_with_rag+rag)")
print("=" * 90)

# =========================
# Args
# =========================
parser = argparse.ArgumentParser(description="训练房价预测模型 - 最终整合版")
parser.add_argument("--pca_dim", type=int, default=16, help="embedding降维维度（用于emb/emb+rag）")
parser.add_argument("--rag_pca_dim", type=int, default=0,
                    help="emb_with_rag分支embedding降维维度；0表示与--pca_dim一致")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--data_ratio", type=float, default=1.0, help="使用数据比例(0,1]")
parser.add_argument("--output_file", type=str, default="model_results_final_modes.csv", help="结果输出CSV")
parser.add_argument("--encode", type=str, default="onehot", choices=["onehot", "label"], help="类别编码方式")
parser.add_argument("--embed_template", type=str, default="semantic", choices=["structured", "semantic"], help="embedding文本模板")
parser.add_argument("--dim_reducer", type=str, default="svd", choices=["pca", "svd"], help="embedding降维器")
parser.add_argument("--rag_k", type=int, default=12, help="RAG近邻数")
parser.add_argument("--rag_space", type=str, default="slim", choices=["full", "slim"],
                    help="RAG检索空间: full=多数值+类别one-hot, slim=地理+面积主导(Lat/Long/Distance/Landsize/BuildingArea/Rooms/Bathroom/Type)")
parser.add_argument("--use_rag_text", type=int, default=1, choices=[0, 1], help="是否把RAG证据拼到文本中（对emb_with_rag系列生效）")
parser.add_argument("--mode", type=str, default="all",
                    choices=["all", "baseline", "emb", "rag", "emb+rag", "emb_with_rag", "emb_with_rag+rag"],
                    help="运行模式")
parser.add_argument("--cache_dir", type=str, default="./emb_cache", help="embedding缓存目录")
parser.add_argument("--emb_batch_size", type=int, default=100, help="embedding批大小")
args = parser.parse_args()

PCA_DIM = args.pca_dim
RAG_PCA_DIM = args.rag_pca_dim if args.rag_pca_dim > 0 else args.pca_dim
RANDOM_SEED = args.seed
DATA_RATIO = args.data_ratio
OUTPUT_FILE = args.output_file
ENCODE_METHOD = args.encode
EMBED_TEMPLATE = args.embed_template
DIM_REDUCER = args.dim_reducer
RAG_K = args.rag_k
RAG_SPACE = getattr(args, "rag_space", "slim")
USE_RAG_TEXT = bool(args.use_rag_text)
MODE = args.mode
CACHE_DIR = args.cache_dir
EMB_BATCH_SIZE = args.emb_batch_size

if DATA_RATIO <= 0 or DATA_RATIO > 1.0:
    raise ValueError("--data_ratio 必须在 0.0 到 1.0 之间")

np.random.seed(RANDOM_SEED)

print("\n配置参数:")
print(f"  mode: {MODE}")
print(f"  pca_dim: {PCA_DIM}")
print(f"  rag_pca_dim: {RAG_PCA_DIM} (for emb_with_rag)")
print(f"  seed: {RANDOM_SEED}")
print(f"  data_ratio: {DATA_RATIO}")
print(f"  encode: {ENCODE_METHOD}")
print(f"  embed_template: {EMBED_TEMPLATE}")
print(f"  dim_reducer: {DIM_REDUCER}")
print(f"  rag_k: {RAG_K}")
print(f"  rag_space: {RAG_SPACE}")
print(f"  use_rag_text: {USE_RAG_TEXT}")
print(f"  cache_dir: {CACHE_DIR}")
print(f"  emb_batch_size: {EMB_BATCH_SIZE}")

# =========================
# Utility
# =========================
def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def text_list_hash(texts):
    m = hashlib.md5()
    for t in texts:
        m.update((t + "\n").encode("utf-8"))
    return m.hexdigest()

def load_cached_embeddings(cache_key):
    p = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(p):
        try:
            arr = np.load(p)
            return arr
        except Exception:
            return None
    return None

def save_cached_embeddings(cache_key, arr):
    p = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    np.save(p, arr)

def qbucket(v, bins):
    if pd.isna(v):
        return "unknown"
    x = float(v)
    for thr, lab in bins:
        if x <= thr:
            return lab
    return bins[-1][1]

def reduce_embeddings(train_emb, test_emb, dim, seed, method="svd"):
    n_train, d_raw = train_emb.shape

    if method == "pca":
        max_dim = min(n_train, d_raw)
    else:
        # TruncatedSVD常见稳定上限
        max_dim = min(max(n_train - 1, 1), d_raw)

    used_dim = min(max(1, int(dim)), max_dim)

    # 请求维度>=原始维度时，直接不降维返回
    if used_dim >= d_raw:
        return train_emb, test_emb, np.nan

    if method == "pca":
        reducer = PCA(n_components=used_dim, random_state=seed)
    else:
        reducer = TruncatedSVD(n_components=used_dim, random_state=seed)

    tr = reducer.fit_transform(train_emb)
    te = reducer.transform(test_emb)
    evr = getattr(reducer, "explained_variance_ratio_", None)
    evr_sum = float(np.sum(evr)) if evr is not None else np.nan
    return tr, te, evr_sum

def inverse_boxcox(transformed_data, lambda_param, shift=0):
    if lambda_param is None:
        return transformed_data
    if lambda_param == 0:
        inverse_data = np.exp(transformed_data) - shift
    else:
        inverse_data = np.power(transformed_data * lambda_param + 1, 1 / lambda_param) - shift
    inverse_data = np.maximum(inverse_data, 0)
    return inverse_data

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_rmsle(y_true, y_pred):
    """RMSLE = sqrt(mean((log1p(pred) - log1p(true))^2)). 价格尺度下使用。"""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(y_t) & np.isfinite(y_p) & (y_t >= 0) & (y_p >= -1)
    if not np.any(valid):
        return float("nan")
    return np.sqrt(mean_squared_error(np.log1p(np.maximum(y_t[valid], 0)), np.log1p(np.maximum(y_p[valid], 0))))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# =========================
# 1) Load data
# =========================
print("\n1. 加载数据...")
data_path = "/mnt/data/melb_data.csv" if os.path.exists("/mnt/data/melb_data.csv") else "melb_data.csv"
df = pd.read_csv(data_path)
print(f"数据文件: {data_path}")
print(f"原始数据形状: {df.shape}")

# =========================
# 2) Cleaning duplicates/inconsistency
# =========================
print("\n2. 处理不一致值和重复值...")
if "CouncilArea" in df.columns:
    df["CouncilArea"] = df["CouncilArea"].astype(str).str.strip().str.title()
    council_corrections = {"nan": None, "None": None, "": None}
    for old_val, new_val in council_corrections.items():
        df.loc[df["CouncilArea"] == old_val, "CouncilArea"] = new_val
    print("  ✓ 修正了CouncilArea列的不一致值")

original_len = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicates_removed = original_len - len(df)
print(f"  ✓ 删除重复: {duplicates_removed}")

if DATA_RATIO < 1.0:
    print(f"  ✓ 采样数据比例: {DATA_RATIO}")
    sample_size = int(len(df) * DATA_RATIO)
    df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    print("  ✓ 从Date提取时间特征")

# =========================
# 3) Prepare split (Ames style)
# =========================
print("\n3. 准备 70/10/20 划分（与 Ames 一致）...")
if "Price" not in df.columns:
    raise ValueError("Price column not found")

all_indices = np.arange(len(df))
train_idx, rest_idx = train_test_split(all_indices, test_size=0.3, random_state=RANDOM_SEED)
val_idx, test_idx = train_test_split(rest_idx, test_size=2.0 / 3.0, random_state=RANDOM_SEED)
print(f"  总样本数: {len(df)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# =========================
# 4) Process one split
# =========================
def process_one_fold(df_full, train_idx, val_idx, fold_num):
    """
    处理一折的数据：预处理、特征工程、模型训练和评估
    确保所有预处理步骤只使用训练集的信息，避免数据泄露
    """
    print(f"\n{'='*90}")
    print(f"Split {fold_num + 1} (single 70/10/20 run)")
    print(f"{'='*90}")
    
    # Split data
    df_train = df_full.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_full.iloc[val_idx].copy().reset_index(drop=True)
    print(f"  train={len(df_train)}, val={len(df_val)}")
    
    # =========================
    # 4.1) Missing value imputation
    # =========================
    df_filled_train = df_train.copy()
    df_filled_val = df_val.copy()
    numeric_cols_train = df_filled_train.select_dtypes(include=[np.number]).columns.tolist()

    predictive_impute_cols = ["Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea"]
    imputation_models = {}
    imputation_stats = {}

    for col in predictive_impute_cols:
        if col in df_filled_train.columns:
            train_missing = df_filled_train[col].isnull().sum()
            val_missing = df_filled_val[col].isnull().sum()
            if train_missing > 0 or val_missing > 0:
                feature_cols = [c for c in numeric_cols_train if c != col and c != "Price" and c not in ["Postcode", "Propertycount"]]
                feature_cols = [c for c in feature_cols if df_filled_train[c].notna().sum() > len(df_filled_train) * 0.5]

                if len(feature_cols) > 0 and train_missing > 0:
                    train_data_has_value = df_filled_train[df_filled_train[col].notna()]
                    train_data_missing = df_filled_train[df_filled_train[col].isna()]
                    if len(train_data_has_value) > 10 and len(train_data_missing) > 0:
                        X_train_imp = train_data_has_value[feature_cols].fillna(train_data_has_value[feature_cols].median())
                        y_train_imp = train_data_has_value[col]
                        try:
                            dt_model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED)
                            dt_model.fit(X_train_imp, y_train_imp)
                            imputation_models[col] = dt_model
                            imputation_stats[col] = {"method": "decision_tree", "feature_cols": feature_cols}
                            X_miss_train = train_data_missing[feature_cols].fillna(train_data_has_value[feature_cols].median())
                            df_filled_train.loc[df_filled_train[col].isna(), col] = dt_model.predict(X_miss_train)
                        except Exception:
                            med = df_filled_train[col].median()
                            imputation_stats[col] = {"method": "median", "value": med}
                            df_filled_train[col].fillna(med, inplace=True)
                    else:
                        med = df_filled_train[col].median()
                        imputation_stats[col] = {"method": "median", "value": med}
                        df_filled_train[col].fillna(med, inplace=True)
                else:
                    med = df_filled_train[col].median()
                    imputation_stats[col] = {"method": "median", "value": med}
                    df_filled_train[col].fillna(med, inplace=True)

                if col in imputation_models:
                    dt_model = imputation_models[col]
                    feature_cols = imputation_stats[col]["feature_cols"]
                    val_missing_df = df_filled_val[df_filled_val[col].isna()]
                    if len(val_missing_df) > 0:
                        train_medians = df_filled_train[feature_cols].median()
                        X_miss_val = val_missing_df[feature_cols].fillna(train_medians)
                        df_filled_val.loc[df_filled_val[col].isna(), col] = dt_model.predict(X_miss_val)
                elif col in imputation_stats and imputation_stats[col]["method"] == "median":
                    df_filled_val[col].fillna(imputation_stats[col]["value"], inplace=True)

    for col in numeric_cols_train:
        if col not in predictive_impute_cols and col != "Price":
            if df_filled_train[col].isnull().sum() > 0 or df_filled_val[col].isnull().sum() > 0:
                med = df_filled_train[col].median()
                df_filled_train[col].fillna(med, inplace=True)
                df_filled_val[col].fillna(med, inplace=True)

    if "CouncilArea" in df_filled_train.columns:
        if df_filled_train["CouncilArea"].isnull().sum() > 0 or df_filled_val["CouncilArea"].isnull().sum() > 0:
            mode_val = df_filled_train["CouncilArea"].mode()[0] if len(df_filled_train["CouncilArea"].mode()) > 0 else "Unknown"
            df_filled_train["CouncilArea"].fillna(mode_val, inplace=True)
            df_filled_val["CouncilArea"].fillna(mode_val, inplace=True)

    # =========================
    # 4.2) Prepare text/numeric views
    # =========================
    text_features_train = {}
    text_features_val = {}
    # 弱信号 Method/SellerG 不参与 embedding 文本，仅保留市场相关
    text_cols = ["Suburb", "Type", "CouncilArea", "Regionname"]
    for c in text_cols:
        if c in df_filled_train.columns:
            text_features_train[c] = df_filled_train[c].values
        if c in df_filled_val.columns:
            text_features_val[c] = df_filled_val[c].values

    num_features_train = {}
    num_features_val = {}
    num_cols_for_text = ["Rooms", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea",
                         "YearBuilt", "Distance", "Lattitude", "Longtitude"]
    for c in num_cols_for_text:
        if c in df_filled_train.columns:
            num_features_train[c] = df_filled_train[c].values
        if c in df_filled_val.columns:
            num_features_val[c] = df_filled_val[c].values

    # =========================
    # 4.3) RAG features
    # =========================
    def _fit_rag_space(train_df, val_df):
        if RAG_SPACE == "slim":
            # 地理+面积主导: 更少维度，减少噪声
            use_num = [c for c in ["Lattitude", "Longtitude", "Distance", "Landsize", "BuildingArea", "Rooms", "Bathroom"]
                      if c in train_df.columns]
            use_cat = [c for c in ["Type"] if c in train_df.columns]
        else:
            use_num = [c for c in ["Rooms", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea",
                                   "YearBuilt", "Distance", "Lattitude", "Longtitude", "Postcode", "Propertycount"]
                       if c in train_df.columns]
            use_cat = [c for c in ["Type", "Method", "SellerG", "CouncilArea", "Regionname", "Suburb"]
                       if c in train_df.columns]

        train_num = train_df[use_num].copy() if len(use_num) else pd.DataFrame(index=train_df.index)
        val_num = val_df[use_num].copy() if len(use_num) else pd.DataFrame(index=val_df.index)

        if len(use_num):
            for c in use_num:
                med = train_num[c].median()
                train_num[c] = train_num[c].fillna(med)
                val_num[c] = val_num[c].fillna(med)
            sc = StandardScaler()
            train_num_arr = sc.fit_transform(train_num.values)
            val_num_arr = sc.transform(val_num.values)
        else:
            train_num_arr = np.zeros((len(train_df), 0))
            val_num_arr = np.zeros((len(val_df), 0))

        if len(use_cat):
            train_cat = pd.get_dummies(train_df[use_cat].astype(str), prefix=use_cat)
            val_cat = pd.get_dummies(val_df[use_cat].astype(str), prefix=use_cat)
            val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0)
            train_cat_arr = train_cat.values
            val_cat_arr = val_cat.values
        else:
            train_cat_arr = np.zeros((len(train_df), 0))
            val_cat_arr = np.zeros((len(val_df), 0))

        train_space = np.hstack([train_num_arr, train_cat_arr]).astype(float)
        val_space = np.hstack([val_num_arr, val_cat_arr]).astype(float)
        return train_space, val_space

    def _row_mode(series):
        m = series.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan

    def build_rag_features(train_df, val_df, k=12):
        train_space, val_space = _fit_rag_space(train_df, val_df)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_df)), metric="cosine")
        nn.fit(train_space)

        _, idx_train = nn.kneighbors(train_space)
        _, idx_val = nn.kneighbors(val_space, n_neighbors=min(k, len(train_df)))

        # 仅保留最有用字段
        rag_num_cols = [c for c in ["Rooms", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "Distance", "YearBuilt"] if c in train_df.columns]
        rag_cat_cols = [c for c in ["Suburb", "Type", "CouncilArea", "Regionname"] if c in train_df.columns]

        train_rows = []
        for i in range(len(train_df)):
            neigh = idx_train[i].tolist()
            if i in neigh:
                neigh.remove(i)
            neigh = neigh[:k]
            sub = train_df.iloc[neigh]
            row = {}
            for c in rag_num_cols:
                row[f"rag_mean_{c}"] = float(sub[c].dropna().mean()) if sub[c].notna().sum() else np.nan
            for c in rag_cat_cols:
                row[f"rag_mode_{c}"] = _row_mode(sub[c].astype(str))
            train_rows.append(row)

        val_rows = []
        for i in range(len(val_df)):
            neigh = idx_val[i].tolist()[:k]
            sub = train_df.iloc[neigh]  # 验证集的近邻只能从训练集中找
            row = {}
            for c in rag_num_cols:
                row[f"rag_mean_{c}"] = float(sub[c].dropna().mean()) if sub[c].notna().sum() else np.nan
            for c in rag_cat_cols:
                row[f"rag_mode_{c}"] = _row_mode(sub[c].astype(str))
            val_rows.append(row)

        rag_train = pd.DataFrame(train_rows, index=train_df.index)
        rag_val = pd.DataFrame(val_rows, index=val_df.index)
        return rag_train, rag_val

    rag_train_df, rag_val_df = build_rag_features(
        df_filled_train.drop(columns=["Price"]),
        df_filled_val.drop(columns=["Price"]),
        k=RAG_K
    )

    # =========================
    # 4.4) Text templates (structured/semantic only)
    # =========================
    def generate_structured_description(i, textf, numf):
        parts = []
        if "Suburb" in textf: parts.append(f"Located in {textf['Suburb'][i]}")
        if "Type" in textf: parts.append(f"property type: {textf['Type'][i]}")
        if "CouncilArea" in textf: parts.append(f"council area: {textf['CouncilArea'][i]}")
        if "Regionname" in textf: parts.append(f"region: {textf['Regionname'][i]}")

        if "Rooms" in numf and not pd.isna(numf["Rooms"][i]): parts.append(f"{int(numf['Rooms'][i])} rooms")
        if "Bedroom2" in numf and not pd.isna(numf["Bedroom2"][i]): parts.append(f"{int(numf['Bedroom2'][i])} bedrooms")
        if "Bathroom" in numf and not pd.isna(numf["Bathroom"][i]): parts.append(f"{int(numf['Bathroom'][i])} bathrooms")
        if "Car" in numf and not pd.isna(numf["Car"][i]): parts.append(f"{int(numf['Car'][i])} car spaces")
        if "Landsize" in numf and not pd.isna(numf["Landsize"][i]): parts.append(f"land size: {numf['Landsize'][i]:.0f} sqm")
        if "BuildingArea" in numf and not pd.isna(numf["BuildingArea"][i]) and numf["BuildingArea"][i] > 0:
            parts.append(f"building area: {numf['BuildingArea'][i]:.0f} sqm")
        if "YearBuilt" in numf and not pd.isna(numf["YearBuilt"][i]) and numf["YearBuilt"][i] > 0:
            parts.append(f"built in {int(numf['YearBuilt'][i])}")
        if "Distance" in numf and not pd.isna(numf["Distance"][i]):
            parts.append(f"{numf['Distance'][i]:.1f}km from CBD")
        return ". ".join(parts) + "."

    def _dwelling_size_label(barea):
        if pd.isna(barea) or barea <= 0:
            return "dwelling"
        b = float(barea)
        if b < 100: return "compact dwelling"
        if b < 180: return "medium-sized dwelling"
        if b < 300: return "large dwelling"
        return "very large dwelling"

    def _land_parcel_label(land):
        if pd.isna(land) or land <= 0:
            return "land"
        L = float(land)
        if L < 400: return "limited land parcel"
        if L < 800: return "moderate land parcel"
        if L < 1500: return "larger land parcel"
        return "large land parcel"

    def _type_market_label(ptype):
        if ptype is None or (isinstance(ptype, float) and pd.isna(ptype)):
            return "property"
        s = str(ptype).strip().lower()
        if "house" in s or "h" == s: return "detached house"
        if "town" in s or "t" == s: return "townhouse"
        if "unit" in s or "u" == s: return "apartment"
        return "property"

    def generate_semantic_description(i, textf, numf):
        """Market-style semantic: location + size + age + neighborhood style, 无 Method/SellerG."""
        suburb = textf.get("Suburb", ["unknown"])[i] if "Suburb" in textf else "unknown"
        ptype = textf.get("Type", ["unknown"])[i] if "Type" in textf else "unknown"
        region = textf.get("Regionname", ["unknown"])[i] if "Regionname" in textf else "unknown"
        council = textf.get("CouncilArea", ["unknown"])[i] if "CouncilArea" in textf else "unknown"

        rooms = numf.get("Rooms", [np.nan])[i] if "Rooms" in numf else np.nan
        beds = numf.get("Bedroom2", [np.nan])[i] if "Bedroom2" in numf else np.nan
        baths = numf.get("Bathroom", [np.nan])[i] if "Bathroom" in numf else np.nan
        land = numf.get("Landsize", [np.nan])[i] if "Landsize" in numf else np.nan
        barea = numf.get("BuildingArea", [np.nan])[i] if "BuildingArea" in numf else np.nan
        yb = numf.get("YearBuilt", [np.nan])[i] if "YearBuilt" in numf else np.nan
        dist = numf.get("Distance", [np.nan])[i] if "Distance" in numf else np.nan

        dist_label = qbucket(dist, [(5, "inner"), (15, "middle-ring"), (30, "outer"), (1e18, "outer")])
        age_label = "unknown"
        if not pd.isna(yb) and yb > 0:
            age = max(0, 2026 - int(yb))
            age_label = qbucket(age, [(15, "newer home"), (40, "moderately aged home"), (80, "older home"), (1e18, "old house")])

        dwelling = _dwelling_size_label(barea)
        land_desc = _land_parcel_label(land)
        type_desc = _type_market_label(ptype)

        parts = [
            f"A {type_desc} in {suburb}, {region}, council area {council}. ",
            f"This is a {dwelling} with {int(rooms) if not pd.isna(rooms) else '?'} rooms, "
            f"{int(beds) if not pd.isna(beds) else '?'} bedrooms, {int(baths) if not pd.isna(baths) else '?'} bathrooms. ",
            f"Located in the {dist_label}, built as a {age_label}. ",
            f"Sits on a {land_desc}."
        ]
        return "".join(parts)

    def _rag_mean(rag_row, col):
        key = f"rag_mean_{col}"
        return rag_row.get(key) if key in rag_row else np.nan

    def _rag_mode(rag_row, col):
        key = f"rag_mode_{col}"
        return rag_row.get(key) if key in rag_row else np.nan

    def _compare_num(self_val, rag_val, larger_phrase, similar_phrase, smaller_phrase, rel_tol=0.08):
        if pd.isna(self_val) or pd.isna(rag_val) or rag_val == 0:
            return similar_phrase
        r = float(self_val) / (float(rag_val) + 1e-9)
        if r >= 1 + rel_tol:
            return larger_phrase
        if r <= 1 - rel_tol:
            return smaller_phrase
        return similar_phrase

    def rag_row_to_comparison_text(i, textf, numf, rag_row, k):
        """RAG 写成比较型语义：similar nearby homes..., this home is larger than..., neighborhood characterized by..."""
        sentences = []
        rag = rag_row if isinstance(rag_row, dict) else rag_row.to_dict()

        # 邻居整体画像（用 rag mean 的 bucket 描述）
        mean_barea = _rag_mean(rag, "BuildingArea")
        mean_land = _rag_mean(rag, "Landsize")
        mean_dist = _rag_mean(rag, "Distance")
        mean_yb = _rag_mean(rag, "YearBuilt")
        if pd.notna(mean_barea) and mean_barea > 0:
            dw = _dwelling_size_label(mean_barea)
            sentences.append(f"Similar nearby homes are typically {dw.replace(' dwelling', ' dwellings')}.")
        if pd.notna(mean_land) and mean_land > 0:
            lp = _land_parcel_label(mean_land)
            sentences.append(f"The neighborhood is characterized by {lp}.")
        if pd.notna(mean_yb) and mean_yb > 0:
            age = max(0, 2026 - int(mean_yb))
            age_band = qbucket(age, [(15, "newer housing stock"), (40, "mid-aged housing stock"), (80, "older housing stock"), (1e18, "older housing stock")])
            sentences.append(f"Nearby properties are generally {age_band}.")

        # 是否同 council / region
        council_self = textf["CouncilArea"][i] if "CouncilArea" in textf and i < len(textf["CouncilArea"]) else None
        council_rag = _rag_mode(rag, "CouncilArea")
        if council_self is not None and council_rag is not None and str(council_self).strip() == str(council_rag).strip():
            sentences.append("Neighboring properties are usually located in the same council area.")

        # 本房 vs 邻居 相对比较
        def _self_num(key):
            if key not in numf or i >= len(numf[key]):
                return np.nan
            return numf[key][i]
        self_barea = _self_num("BuildingArea")
        self_land = _self_num("Landsize")
        self_dist = _self_num("Distance")
        self_yb = _self_num("YearBuilt")

        sentences.append(_compare_num(
            self_barea, mean_barea,
            "This home is larger than nearby properties.",
            "This home is similar in size to nearby properties.",
            "This home is smaller than nearby properties."
        ))
        if pd.notna(self_land) and pd.notna(mean_land) and mean_land > 0:
            sentences.append(_compare_num(
                self_land, mean_land,
                "It sits on a larger lot than typical in the area.",
                "Lot size is in line with the neighborhood.",
                "It has a more compact lot than nearby homes."
            ))

        return " ".join(sentences) + f" (based on {k} nearest neighbors)."

    def rag_row_to_text(rag_row, k, prefix="rag"):
        """Legacy numeric dump; prefer rag_row_to_comparison_text for emb_with_rag."""
        parts = [f"{prefix}_evidence_k={k}"]
        num_fields = [
            f"{prefix}_mean_Rooms", f"{prefix}_mean_Bedroom2", f"{prefix}_mean_Bathroom",
            f"{prefix}_mean_Car", f"{prefix}_mean_Landsize", f"{prefix}_mean_BuildingArea",
            f"{prefix}_mean_Distance", f"{prefix}_mean_YearBuilt"
        ]
        cat_fields = [
            f"{prefix}_mode_Suburb", f"{prefix}_mode_Type", f"{prefix}_mode_CouncilArea", f"{prefix}_mode_Regionname"
        ]
        for c in num_fields:
            if c in rag_row and pd.notna(rag_row[c]):
                parts.append(f"{c}={float(rag_row[c]):.3f}")
        for c in cat_fields:
            if c in rag_row and pd.notna(rag_row[c]):
                parts.append(f"{c}={str(rag_row[c])}")
        return " ; ".join(parts) + "."

    def build_embedding_texts(df_part, textf, numf, rag_df=None, template="semantic", use_rag_in_text=False, rag_k=12):
        texts = []
        n = len(df_part)
        for i in range(n):
            if template == "structured":
                base = generate_structured_description(i, textf, numf)
            else:
                base = generate_semantic_description(i, textf, numf)

            if use_rag_in_text and rag_df is not None and USE_RAG_TEXT:
                rag_txt = rag_row_to_comparison_text(i, textf, numf, rag_df.iloc[i], rag_k)
                base = base + " Neighborhood context: " + rag_txt
            texts.append(base)
        return texts

    # =========================
    # 4.5) Embeddings (if available)
    # =========================
    embeddings_plain_train = None
    embeddings_plain_val = None
    embeddings_with_ragtext_train = None
    embeddings_with_ragtext_val = None

    if OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            OPENAI_AVAILABLE_FOLD = False
        else:
            OPENAI_AVAILABLE_FOLD = True
            ensure_cache_dir()
            client = OpenAI(api_key=api_key)

            def get_embeddings_batch(descriptions, batch_size=100):
                embs = []
                for i in range(0, len(descriptions), batch_size):
                    batch = descriptions[i:i + batch_size]
                    try:
                        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
                        embs.extend([item.embedding for item in resp.data])
                    except Exception as e:
                        print(f"    embedding批失败: {e}, 用零向量填充")
                        embs.extend([[0.0] * 3072] * len(batch))
                return np.array(embs)

            # plain texts
            txt_plain_train = build_embedding_texts(
                df_filled_train, text_features_train, num_features_train,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )
            txt_plain_val = build_embedding_texts(
                df_filled_val, text_features_val, num_features_val,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )

            # rag-text texts: 原模板 + RAG证据文本（你定义的emb_with_rag）
            txt_rag_train = build_embedding_texts(
                df_filled_train, text_features_train, num_features_train,
                rag_df=rag_train_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K
            )
            txt_rag_val = build_embedding_texts(
                df_filled_val, text_features_val, num_features_val,
                rag_df=rag_val_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K
            )

            # cache keys (include all relevant parameters to avoid cache conflicts)
            # Note: text_list_hash already captures USE_RAG_TEXT effect (via text content)
            # But we should include DATA_RATIO and RANDOM_SEED to ensure cache validity
            plain_train_key = f"plain_train_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_plain_train)}"
            plain_val_key = f"plain_val_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_plain_val)}"
            rag_train_key = f"ragtxt_train_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_rag_train)}"
            rag_val_key = f"ragtxt_val_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_rag_val)}"

            emb_plain_tr_raw = load_cached_embeddings(plain_train_key)
            emb_plain_val_raw = load_cached_embeddings(plain_val_key)
            emb_rag_tr_raw = load_cached_embeddings(rag_train_key)
            emb_rag_val_raw = load_cached_embeddings(rag_val_key)

            if emb_plain_tr_raw is None:
                emb_plain_tr_raw = get_embeddings_batch(txt_plain_train, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(plain_train_key, emb_plain_tr_raw)
            if emb_plain_val_raw is None:
                emb_plain_val_raw = get_embeddings_batch(txt_plain_val, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(plain_val_key, emb_plain_val_raw)
            if emb_rag_tr_raw is None:
                emb_rag_tr_raw = get_embeddings_batch(txt_rag_train, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(rag_train_key, emb_rag_tr_raw)
            if emb_rag_val_raw is None:
                emb_rag_val_raw = get_embeddings_batch(txt_rag_val, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(rag_val_key, emb_rag_val_raw)

            # plain embedding -> pca_dim (fit on train, transform val)
            embeddings_plain_train, embeddings_plain_val, evr_plain = reduce_embeddings(
                emb_plain_tr_raw, emb_plain_val_raw, PCA_DIM, RANDOM_SEED, DIM_REDUCER
            )

            # rag-text embedding -> rag_pca_dim
            embeddings_with_ragtext_train, embeddings_with_ragtext_val, evr_ragtxt = reduce_embeddings(
                emb_rag_tr_raw, emb_rag_val_raw, RAG_PCA_DIM, RANDOM_SEED, DIM_REDUCER
            )
    else:
        OPENAI_AVAILABLE_FOLD = False

    # =========================
    # 4.6) Box-Cox / skew handling
    # =========================
    boxcox_lambdas = {}
    boxcox_shifts = {}
    PRICE_LAMBDA_FOLD = None
    PRICE_SHIFT_FOLD = None

    for col in ["Price", "Distance"]:
        if col in df_filled_train.columns:
            col_data_train = df_filled_train[col].dropna()
            if len(col_data_train) > 0:
                shift = 0
                if col_data_train.min() <= 0:
                    shift = abs(col_data_train.min()) + 1
                    boxcox_shifts[col] = shift
                    col_data_train = col_data_train + shift
                if col_data_train.min() > 0:
                    try:
                        _, fitted_lambda = stats.boxcox(col_data_train)
                        boxcox_lambdas[col] = fitted_lambda
                        if col == "Price":
                            PRICE_LAMBDA_FOLD = fitted_lambda
                            PRICE_SHIFT_FOLD = shift

                        df_filled_train[col] = boxcox(df_filled_train[col] + shift, fitted_lambda)
                        df_filled_val[col] = boxcox(df_filled_val[col] + boxcox_shifts.get(col, 0), fitted_lambda)
                    except Exception as e:
                        pass

    numeric_cols_for_skew = [
        c for c in df_filled_train.select_dtypes(include=[np.number]).columns.tolist()
        if c not in ["Postcode", "Propertycount", "Year", "Month", "Day", "DayOfWeek"] and c not in boxcox_lambdas
    ]
    for col in numeric_cols_for_skew:
        col_data_train = df_filled_train[col].dropna()
        if len(col_data_train) > 0:
            skewness = stats.skew(col_data_train)
            if abs(skewness) > 1.0:
                q1 = col_data_train.quantile(0.25)
                q3 = col_data_train.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3.5 * iqr
                upper = q3 + 3.5 * iqr
                df_filled_train.loc[df_filled_train[col] < lower, col] = lower
                df_filled_train.loc[df_filled_train[col] > upper, col] = upper
                df_filled_val.loc[df_filled_val[col] < lower, col] = lower
                df_filled_val.loc[df_filled_val[col] > upper, col] = upper

    # =========================
    # 4.7) Tabular preprocess
    # =========================
    if "Address" in df_filled_train.columns:
        df_filled_train = df_filled_train.drop("Address", axis=1)
        df_filled_val = df_filled_val.drop("Address", axis=1)
    if "Date" in df_filled_train.columns:
        df_filled_train = df_filled_train.drop("Date", axis=1)
        df_filled_val = df_filled_val.drop("Date", axis=1)

    categorical_cols = ["Method", "Type", "SellerG"]
    for col in categorical_cols:
        if col in df_filled_train.columns:
            if ENCODE_METHOD == "label" and col == "SellerG":
                le = LabelEncoder()
                train_vals = df_filled_train[col].astype(str)
                df_filled_train[col] = le.fit_transform(train_vals)

                val_vals = df_filled_val[col].astype(str)
                unseen = ~val_vals.isin(le.classes_)
                if unseen.sum() > 0:
                    fallback = train_vals.mode()[0] if len(train_vals.mode()) > 0 else le.classes_[0]
                    val_vals[unseen] = fallback
                df_filled_val[col] = le.transform(val_vals)
            else:
                dtr = pd.get_dummies(df_filled_train[col], prefix=col)
                dte = pd.get_dummies(df_filled_val[col], prefix=col)
                for d in dtr.columns:
                    if d not in dte.columns:
                        dte[d] = 0
                dte = dte[dtr.columns]

                df_filled_train = pd.concat([df_filled_train.drop(columns=[col]), dtr], axis=1)
                df_filled_val = pd.concat([df_filled_val.drop(columns=[col]), dte], axis=1)

    # drop non-numeric leftovers
    non_num_train = df_filled_train.select_dtypes(exclude=[np.number]).columns.tolist()
    non_num_val = df_filled_val.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_num_train) > 0:
        df_filled_train = df_filled_train.drop(columns=non_num_train)
        df_filled_val = df_filled_val.drop(columns=non_num_val)

    df_filled_val = df_filled_val.reindex(columns=df_filled_train.columns, fill_value=0)

    X_train_df = df_filled_train.drop("Price", axis=1).copy()
    y_train = df_filled_train["Price"].copy()
    X_val_df = df_filled_val.drop("Price", axis=1).copy()
    y_val = df_filled_val["Price"].copy()

    # rag df encode numeric
    def encode_rag_df(rag_tr, rag_val):
        rag_tr = rag_tr.copy()
        rag_val = rag_val.copy()
        for c in rag_tr.columns:
            if rag_tr[c].dtype == "object":
                vals = rag_tr[c].astype(str).fillna("None")
                le = LabelEncoder()
                le.fit(vals)
                mp = {k: i for i, k in enumerate(le.classes_)}
                default = mp.get("None", 0)
                rag_tr[c] = vals.map(mp).fillna(default).astype(float)
                rag_val[c] = rag_val[c].astype(str).fillna("None").map(lambda x: mp.get(x, default)).astype(float)
            else:
                med = rag_tr[c].median()
                rag_tr[c] = rag_tr[c].fillna(med)
                rag_val[c] = rag_val[c].fillna(med)
        return rag_tr, rag_val

    rag_train_num, rag_val_num = encode_rag_df(rag_train_df, rag_val_df)

    # minmax scale (fit on train, transform val)
    num_cols_scale = [c for c in X_train_df.select_dtypes(include=[np.number]).columns if c not in ["Postcode", "Propertycount"]]
    for c in num_cols_scale:
        sc = MinMaxScaler()
        sc.fit(X_train_df[[c]])
        X_train_df[c] = sc.transform(X_train_df[[c]])
        X_val_df[c] = sc.transform(X_val_df[[c]])

    # =========================
    # 4.8) Build mode matrices
    # =========================
    X_train_base = X_train_df.values
    X_val_base = X_val_df.values

    X_train_rag = np.hstack([X_train_df.values, rag_train_num.values])
    X_val_rag = np.hstack([X_val_df.values, rag_val_num.values])

    X_train_emb = None
    X_val_emb = None
    X_train_emb_plus_rag = None
    X_val_emb_plus_rag = None
    X_train_emb_with_rag = None
    X_val_emb_with_rag = None
    X_train_emb_with_rag_plus_rag = None
    X_val_emb_with_rag_plus_rag = None

    if OPENAI_AVAILABLE_FOLD and embeddings_plain_train is not None:
        # 2) emb
        X_train_emb = np.hstack([X_train_df.values, embeddings_plain_train])
        X_val_emb = np.hstack([X_val_df.values, embeddings_plain_val])

        # 4) emb+rag
        X_train_emb_plus_rag = np.hstack([X_train_df.values, embeddings_plain_train, rag_train_num.values])
        X_val_emb_plus_rag = np.hstack([X_val_df.values, embeddings_plain_val, rag_val_num.values])

        # 5) emb_with_rag (文本=原模板+RAG证据, 降维=rag_pca_dim)
        X_train_emb_with_rag = np.hstack([X_train_df.values, embeddings_with_ragtext_train])
        X_val_emb_with_rag = np.hstack([X_val_df.values, embeddings_with_ragtext_val])

        # 6) emb_with_rag+rag
        X_train_emb_with_rag_plus_rag = np.hstack([X_train_df.values, embeddings_with_ragtext_train, rag_train_num.values])
        X_val_emb_with_rag_plus_rag = np.hstack([X_val_df.values, embeddings_with_ragtext_val, rag_val_num.values])

    # =========================
    # 4.9) Train/Eval
    # =========================
    def evaluate_model_fold(y_true_transformed, y_pred_transformed, model_name):
        if PRICE_LAMBDA_FOLD is not None:
            y_true_original = inverse_boxcox(y_true_transformed, PRICE_LAMBDA_FOLD, PRICE_SHIFT_FOLD)
            y_pred_original = inverse_boxcox(y_pred_transformed, PRICE_LAMBDA_FOLD, PRICE_SHIFT_FOLD)
        else:
            y_true_original = y_true_transformed
            y_pred_original = y_pred_transformed

        rmse = calculate_rmse(y_true_original, y_pred_original)
        rmsle = calculate_rmsle(y_true_original, y_pred_original)
        mae = calculate_mae(y_true_original, y_pred_original)
        r2 = r2_score(y_true_original, y_pred_original)
        return {"Model": model_name, "RMSLE": rmsle, "RMSE": rmse, "MAE": mae, "R2_Score": r2}

    fold_results = []

    def train_and_evaluate_fold(model_class, model_name, Xtr, Xval, suffix="", is_tabpfn=False):
        try:
            if is_tabpfn:
                # 允许超过 1000 行训练：ignore_pretraining_limits + CPU 大数据集环境变量
                os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
                try:
                    model = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
                    model.fit(Xtr, y_train.values)
                except Exception:
                    model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
                    model.fit(Xtr, y_train.values)
                y_pred = model.predict(Xval)
            elif model_class == xgb.XGBRegressor:
                model = model_class(n_estimators=50, max_depth=4, learning_rate=0.15,
                                    random_state=RANDOM_SEED, n_jobs=-1)
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xval)
            elif model_class == lgb.LGBMRegressor:
                model = model_class(n_estimators=50, max_depth=4, learning_rate=0.15,
                                    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xval)
            elif model_class == cb.CatBoostRegressor:
                model = model_class(iterations=50, depth=4, learning_rate=0.15,
                                    random_seed=RANDOM_SEED, verbose=False)
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xval)
            else:
                return None

            full_name = f"{model_name} ({suffix})"
            return evaluate_model_fold(y_val, y_pred, full_name)
        except Exception as e:
            print(f"  {model_name} ({suffix}) 训练失败: {e}")
            return None

    def run_family_fold(Xtr, Xval, suffix):
        if XGBOOST_AVAILABLE:
            r = train_and_evaluate_fold(xgb.XGBRegressor, "XGBoost", Xtr, Xval, suffix)
            if r:
                fold_results.append(r)
        if LIGHTGBM_AVAILABLE:
            r = train_and_evaluate_fold(lgb.LGBMRegressor, "LightGBM", Xtr, Xval, suffix)
            if r:
                fold_results.append(r)
        if CATBOOST_AVAILABLE:
            r = train_and_evaluate_fold(cb.CatBoostRegressor, "CatBoost", Xtr, Xval, suffix)
            if r:
                fold_results.append(r)
        if TABPFN_AVAILABLE:
            r = train_and_evaluate_fold(None, "TabPFN", Xtr, Xval, suffix, is_tabpfn=True)
            if r:
                fold_results.append(r)

    # Run models based on mode
    if MODE == "baseline":
        run_family_fold(X_train_base, X_val_base, "baseline")
    elif MODE == "emb":
        if X_train_emb is not None:
            run_family_fold(X_train_emb, X_val_emb, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}")
    elif MODE == "rag":
        run_family_fold(X_train_rag, X_val_rag, "rag")
    elif MODE == "emb+rag":
        if X_train_emb_plus_rag is not None:
            run_family_fold(X_train_emb_plus_rag, X_val_emb_plus_rag, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}+rag")
    elif MODE == "emb_with_rag":
        if X_train_emb_with_rag is not None:
            run_family_fold(X_train_emb_with_rag, X_val_emb_with_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}")
    elif MODE == "emb_with_rag+rag":
        if X_train_emb_with_rag_plus_rag is not None:
            run_family_fold(X_train_emb_with_rag_plus_rag, X_val_emb_with_rag_plus_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}+rag")
    elif MODE == "all":
        run_family_fold(X_train_base, X_val_base, "baseline")
        run_family_fold(X_train_rag, X_val_rag, "rag")
        if X_train_emb is not None:
            run_family_fold(X_train_emb, X_val_emb, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}")
            run_family_fold(X_train_emb_plus_rag, X_val_emb_plus_rag, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}+rag")
            run_family_fold(X_train_emb_with_rag, X_val_emb_with_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}")
            run_family_fold(X_train_emb_with_rag_plus_rag, X_val_emb_with_rag_plus_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}+rag")
    
    return fold_results

# =========================
# 5) Run single split (Ames style)
# =========================
print("\n" + "=" * 90)
print("开始单次 70/10/20 划分评估")
print("=" * 90)

all_fold_results = []
df_full = df.copy()
fold_results = process_one_fold(df_full, train_idx, val_idx, 0)
all_fold_results.extend(fold_results)

# =========================
# 6) Summary - single split
# =========================
print("\n" + "=" * 90)
print("单次 70/10/20 划分结果汇总")
print("=" * 90)

if len(all_fold_results) == 0:
    print("No models were successfully trained.")
else:
    results_df = pd.DataFrame(all_fold_results)
    
    # 单次划分：按 model 聚合均值（等价于该次结果）
    summary = results_df.groupby("Model").agg({
        "RMSLE": "mean",
        "RMSE": "mean",
        "MAE": "mean",
        "R2_Score": "mean"
    }).round(6)
    summary.columns = ["RMSLE_mean", "RMSE_mean", "MAE_mean", "R2_mean"]
    summary = summary.reset_index()
    summary = summary.sort_values("RMSLE_mean")
    
    print("\n各模型性能 (按 RMSLE 排序):")
    print(summary.to_string(index=False))
    
    # Save detailed results
    results_df.to_csv(OUTPUT_FILE.replace(".csv", "_detailed.csv"), index=False)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\n详细结果已保存: {OUTPUT_FILE.replace('.csv', '_detailed.csv')}")
    print(f"汇总结果已保存: {OUTPUT_FILE}")

    best = summary.iloc[0]
    print(f"\n最佳模型: {best['Model']}")
    print(f"  RMSLE: {best['RMSLE_mean']:.6f}")
    print(f"  RMSE:  {best['RMSE_mean']:.6f}")
    print(f"  MAE :  {best['MAE_mean']:.6f}")
    print(f"  R²   : {best['R2_mean']:.6f}")

print("\n" + "=" * 90)
print("单次划分评估完成!")
print("=" * 90)
