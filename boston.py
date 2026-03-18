import os
import json
import hashlib
import warnings
import argparse
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
print("Boston Housing — 划分与 Ames 一致 70/10/20；8-MODE: baseline/rag/rag_price/emb/…")
print("=" * 90)

# =========================
# Args
# =========================
parser = argparse.ArgumentParser(description="训练 Boston Housing 预测模型 - 最终整合版")
parser.add_argument("--pca_dim", type=int, default=16, help="embedding降维维度（用于emb/emb+rag）")
parser.add_argument("--rag_pca_dim", type=int, default=0,
                    help="emb_with_rag分支embedding降维维度；0表示与--pca_dim一致")
parser.add_argument("--seed", type=str, default="42", help="随机种子，支持逗号分隔多个如 0,1,2,3,4")
parser.add_argument("--data_ratio", type=float, default=1.0, help="使用数据比例(0,1]")
parser.add_argument("--output_file", type=str, default="boston_housing_results.csv", help="结果输出CSV")
parser.add_argument("--encode", type=str, default="onehot", choices=["onehot", "label"], help="类别编码方式")
parser.add_argument("--embed_template", type=str, default="semantic", choices=["structured", "semantic"], help="embedding文本模板")
parser.add_argument("--dim_reducer", type=str, default="svd", choices=["pca", "svd"], help="embedding降维器")
parser.add_argument("--rag_k", type=int, default=12, help="RAG近邻数")
parser.add_argument("--rag_mode", type=str, default="hybrid", choices=["hybrid", "location"],
                    help="RAG邻居模式: hybrid=仅数值, location=仅CHAS/RAD")
parser.add_argument("--use_rag_text", type=int, default=1, choices=[0, 1], help="是否把RAG证据拼到文本中（对emb_with_rag系列生效）")
parser.add_argument("--rag_template", type=str, default="default",
                    choices=["default", "compare", "delta"],
                    help="emb_with_rag 文本: default=邻居均值描述; compare=与邻居比较(大于/小于/相当); delta=具体差值")
ALL_MODES = ["baseline", "rag", "rag_price", "emb", "emb+rag", "emb_with_rag", "emb_with_rag+rag", "emb_with_rag_price"]
parser.add_argument("--mode", type=str, default="all",
                    help="运行模式，逗号分隔多个如 baseline,rag ；或 all 跑全部")
parser.add_argument("--cache_dir", type=str, default="./emb_cache_boston", help="embedding缓存目录")
parser.add_argument("--emb_batch_size", type=int, default=100, help="embedding批大小")
args = parser.parse_args()


def _parse_seeds(s):
    """Parse comma-separated seeds to list of ints."""
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


SEEDS = _parse_seeds(args.seed)
if not SEEDS:
    SEEDS = [42]
RANDOM_SEED = SEEDS[0]  # used for data sampling and when single seed

PCA_DIM = args.pca_dim
RAG_PCA_DIM = args.rag_pca_dim if args.rag_pca_dim > 0 else args.pca_dim
DATA_RATIO = args.data_ratio
OUTPUT_FILE = args.output_file
ENCODE_METHOD = args.encode
EMBED_TEMPLATE = args.embed_template
DIM_REDUCER = args.dim_reducer
RAG_K = args.rag_k
RAG_MODE = args.rag_mode
USE_RAG_TEXT = bool(args.use_rag_text)
RAG_TEMPLATE = args.rag_template
_modes_str = [m.strip() for m in args.mode.split(",") if m.strip()]
if "all" in _modes_str:
    MODES = ALL_MODES.copy()
else:
    MODES = _modes_str
    for m in MODES:
        if m not in ALL_MODES:
            raise ValueError(f"--mode 无效: '{m}'，可选: {ALL_MODES} 或 all")
CACHE_DIR = args.cache_dir
EMB_BATCH_SIZE = args.emb_batch_size
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

if DATA_RATIO <= 0 or DATA_RATIO > 1.0:
    raise ValueError("--data_ratio 必须在 0.0 到 1.0 之间")

np.random.seed(RANDOM_SEED)

# =========================
# Dataset schema (Boston)
# =========================
TARGET_COL = "MEDV"
TEXT_CAT_COLS = ["CHAS", "RAD"]
NUM_FEATURE_COLS = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS",
    "TAX", "PTRATIO", "B", "LSTAT", "CHAS", "RAD"
]
RAG_NUM_COLS = NUM_FEATURE_COLS.copy()
RAG_CAT_COLS = TEXT_CAT_COLS.copy()

# RAG 邻居模式: hybrid=仅数值列, location=仅类别 CHAS/RAD
RAG_MODE_CONFIG = {
    "hybrid": {"numeric": RAG_NUM_COLS, "cat": []},
    "location": {"numeric": [], "cat": RAG_CAT_COLS},
}

# English labels used by emb_with_rag compare/delta templates
BOSTON_RAG_LABELS = {
    "CRIM": "crime rate", "ZN": "residential zoning", "INDUS": "industrial share", "NOX": "nox concentration",
    "RM": "rooms", "AGE": "building age", "DIS": "distance to jobs", "RAD": "highway access index",
    "TAX": "property tax", "PTRATIO": "pupil-teacher ratio", "B": "B statistic", "LSTAT": "lower status share",
    "CHAS": "charles river access",
}

print("\n配置参数:")
print(f"  mode: {MODES}")
print(f"  pca_dim: {PCA_DIM}")
print(f"  rag_pca_dim: {RAG_PCA_DIM} (for emb_with_rag)")
print(f"  seed: {SEEDS if len(SEEDS) != 1 else RANDOM_SEED}")
print(f"  data_ratio: {DATA_RATIO}")
print(f"  encode: {ENCODE_METHOD}")
print(f"  embed_template: {EMBED_TEMPLATE}")
print(f"  dim_reducer: {DIM_REDUCER}")
print(f"  rag_k: {RAG_K}")
print(f"  rag_mode: {RAG_MODE}")
print(f"  use_rag_text: {USE_RAG_TEXT}")
print(f"  rag_template: {RAG_TEMPLATE}")
print(f"  cache_dir: {CACHE_DIR}")
print(f"  emb_batch_size: {EMB_BATCH_SIZE}")
print(f"  embedding_model: {EMBEDDING_MODEL}")

if any(m in ["emb_with_rag", "emb_with_rag+rag", "emb_with_rag_price"] for m in MODES) and args.rag_pca_dim > 0 and args.pca_dim != RAG_PCA_DIM:
    print("  [提示] 当前模式主要使用 rag_pca_dim；pca_dim 不参与 emb_with_rag 分支降维。")
    print("        若要让 pca_dim 生效，请设置 --rag_pca_dim 0 或令 rag_pca_dim = pca_dim。")

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
        max_dim = min(max(n_train - 1, 1), d_raw)

    used_dim = min(max(1, int(dim)), max_dim)

    if used_dim >= d_raw:
        return train_emb, test_emb, np.nan

    if method == "pca":
        reducer = PCA(n_components=used_dim, random_state=seed)
    else:
        reducer = TruncatedSVD(n_components=used_dim, random_state=seed)

    tr = reducer.fit_transform(train_emb)
    va = reducer.transform(test_emb)
    evr = getattr(reducer, "explained_variance_ratio_", None)
    evr_sum = float(np.sum(evr)) if evr is not None else np.nan
    return tr, va, evr_sum


def reduce_embeddings_3(train_emb, val_emb, test_emb, dim, seed, method="svd"):
    """Fit reducer on train only; transform train, val, test."""
    n_train, d_raw = train_emb.shape
    if method == "pca":
        max_dim = min(n_train, d_raw)
    else:
        max_dim = min(max(n_train - 1, 1), d_raw)
    used_dim = min(max(1, int(dim)), max_dim)
    if used_dim >= d_raw:
        return train_emb, val_emb, test_emb, np.nan
    if method == "pca":
        reducer = PCA(n_components=used_dim, random_state=seed)
    else:
        reducer = TruncatedSVD(n_components=used_dim, random_state=seed)
    tr = reducer.fit_transform(train_emb)
    va = reducer.transform(val_emb)
    te = reducer.transform(test_emb)
    evr = getattr(reducer, "explained_variance_ratio_", None)
    evr_sum = float(np.sum(evr)) if evr is not None else np.nan
    return tr, va, te, evr_sum

def inverse_boxcox(transformed_data, lambda_param, shift=0):
    if lambda_param is None:
        return transformed_data

    arr = np.asarray(transformed_data, dtype=float)
    if lambda_param == 0:
        inverse_data = np.exp(arr)
    else:
        # Keep inverse transform numerically stable when model predictions
        # go outside the strict Box-Cox domain due to regression noise.
        base = arr * lambda_param + 1.0
        base = np.maximum(base, 1e-12)
        inverse_data = np.power(base, 1.0 / lambda_param)

    inverse_data = inverse_data - shift
    inverse_data = np.maximum(inverse_data, 0.0)
    return inverse_data

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_rmsle(y_true, y_pred, scale=1000.0):
    """RMSLE with target scaled by scale (default 1000) before log: sqrt(mean((log1p(pred*scale)-log1p(actual*scale))^2))."""
    y_true = np.asarray(y_true, dtype=float).ravel() * scale
    y_pred = np.asarray(y_pred, dtype=float).ravel() * scale
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# =========================
# 1) Load data
# =========================
print("\n1. 加载数据...")
data_path = "data/boston_housing_dataset.csv"
df = pd.read_csv(data_path)
    
print(f"数据文件: {data_path}")
print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# =========================
# 2) Cleaning duplicates/inconsistency
# =========================
print("\n2. 处理不一致值和重复值...")

original_len = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicates_removed = original_len - len(df)
print(f"  ✓ 删除重复: {duplicates_removed}")

if DATA_RATIO < 1.0:
    print(f"  ✓ 采样数据比例: {DATA_RATIO}")
    sample_size = int(len(df) * DATA_RATIO)
    df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

# =========================
# 3) 与 Ames 相同: 70% train, 10% val, 20% test
# =========================
print("\n3. 划分: 70% train / 10% val / 20% test（与 Ames 一致）")
if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} column not found")
print(f"  总样本数: {len(df)}")

# =========================
# 4) 单次划分 pipeline
# =========================
def process_one_split(df_full, train_idx, val_idx, test_idx, seed=None):
    """
    单次划分：预处理仅在 train 上拟合；RAG 邻居仅来自 train；
    模型在 train 上拟合并评估 val；再在 train+val 上 refit 后评估 test（无 test 泄露）。
    """
    if seed is None:
        seed = RANDOM_SEED
    print(f"\n{'='*90}")
    print(f"seed={seed}  |  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"{'='*90}")

    df_train = df_full.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_full.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df_full.iloc[test_idx].copy().reset_index(drop=True)
    
    # =========================
    # 4.1) Missing value imputation
    # =========================
    df_filled_train = df_train.copy()
    df_filled_val = df_val.copy()
    df_filled_test = df_test.copy()
    numeric_cols_train = df_filled_train.select_dtypes(include=[np.number]).columns.tolist()

    predictive_impute_cols = [c for c in NUM_FEATURE_COLS if c in df_filled_train.columns]
    imputation_models = {}
    imputation_stats = {}

    for col in predictive_impute_cols:
        if col in df_filled_train.columns:
            train_missing = df_filled_train[col].isnull().sum()
            val_missing = df_filled_val[col].isnull().sum()
            test_missing = df_filled_test[col].isnull().sum()
            if train_missing > 0 or val_missing > 0 or test_missing > 0:
                feature_cols = [c for c in numeric_cols_train if c != col and c != TARGET_COL]
                feature_cols = [c for c in feature_cols if df_filled_train[c].notna().sum() > len(df_filled_train) * 0.5]

                if len(feature_cols) > 0 and train_missing > 0:
                    train_data_has_value = df_filled_train[df_filled_train[col].notna()]
                    train_data_missing = df_filled_train[df_filled_train[col].isna()]
                    if len(train_data_has_value) > 10 and len(train_data_missing) > 0:
                        X_train_imp = train_data_has_value[feature_cols].fillna(train_data_has_value[feature_cols].median())
                        y_train_imp = train_data_has_value[col]
                        try:
                            dt_model = DecisionTreeRegressor(max_depth=5, random_state=seed)
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
                if col in imputation_models:
                    test_missing_df = df_filled_test[df_filled_test[col].isna()]
                    if len(test_missing_df) > 0:
                        feature_cols = imputation_stats[col]["feature_cols"]
                        train_medians = df_filled_train[feature_cols].median()
                        X_miss_te = test_missing_df[feature_cols].fillna(train_medians)
                        df_filled_test.loc[df_filled_test[col].isna(), col] = imputation_models[col].predict(X_miss_te)
                if col in imputation_stats and imputation_stats[col]["method"] == "median":
                    df_filled_test[col].fillna(imputation_stats[col]["value"], inplace=True)

    for col in numeric_cols_train:
        if col not in predictive_impute_cols and col != TARGET_COL:
            if df_filled_train[col].isnull().sum() > 0 or df_filled_val[col].isnull().sum() > 0 or df_filled_test[col].isnull().sum() > 0:
                med = df_filled_train[col].median()
                df_filled_train[col].fillna(med, inplace=True)
                df_filled_val[col].fillna(med, inplace=True)
                df_filled_test[col].fillna(med, inplace=True)

    # 处理类别特征的缺失值（Boston: CHAS/RAD）
    for col in TEXT_CAT_COLS:
        if col in df_filled_train.columns:
            if df_filled_train[col].isnull().sum() > 0 or df_filled_val[col].isnull().sum() > 0 or df_filled_test[col].isnull().sum() > 0:
                mode_val = df_filled_train[col].mode()[0] if len(df_filled_train[col].mode()) > 0 else "unknown"
                df_filled_train[col].fillna(mode_val, inplace=True)
                df_filled_val[col].fillna(mode_val, inplace=True)
                df_filled_test[col].fillna(mode_val, inplace=True)

    # =========================
    # 4.2) Prepare text/numeric views
    # =========================
    text_features_train = {}
    text_features_val = {}
    text_features_test = {}
    text_cols = TEXT_CAT_COLS
    for c in text_cols:
        if c in df_filled_train.columns:
            text_features_train[c] = df_filled_train[c].values
        if c in df_filled_val.columns:
            text_features_val[c] = df_filled_val[c].values
        if c in df_filled_test.columns:
            text_features_test[c] = df_filled_test[c].values

    num_features_train = {}
    num_features_val = {}
    num_features_test = {}
    num_cols_for_text = NUM_FEATURE_COLS
    for c in num_cols_for_text:
        if c in df_filled_train.columns:
            num_features_train[c] = df_filled_train[c].values
        if c in df_filled_val.columns:
            num_features_val[c] = df_filled_val[c].values
        if c in df_filled_test.columns:
            num_features_test[c] = df_filled_test[c].values

    # =========================
    # 4.3) RAG features (by rag_mode: hybrid / location)
    # =========================
    def _fit_rag_space(train_df, val_df, use_num, use_cat):
        use_num = [c for c in use_num if c in train_df.columns]
        use_cat = [c for c in use_cat if c in train_df.columns]
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

    def build_rag_features(train_df, val_df, k=12, train_target=None, rag_mode="hybrid"):
        cfg = RAG_MODE_CONFIG.get(rag_mode, RAG_MODE_CONFIG["hybrid"])
        use_num = [c for c in cfg["numeric"] if c in train_df.columns]
        use_cat = [c for c in cfg["cat"] if c in train_df.columns]
        train_space, val_space = _fit_rag_space(train_df, val_df, use_num, use_cat)
        if train_space.shape[1] == 0:
            # 无特征时退化为随机邻居（或全 0 距离）
            train_space = np.zeros((len(train_df), 1))
            val_space = np.zeros((len(val_df), 1))
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_df)), metric="cosine")
        nn.fit(train_space)
        _, idx_train = nn.kneighbors(train_space)
        _, idx_val = nn.kneighbors(val_space, n_neighbors=min(k, len(train_df)))
        rag_num_cols = use_num
        rag_cat_cols = use_cat
        train_target_arr = np.asarray(train_target) if train_target is not None else None
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
            if train_target_arr is not None and len(neigh) > 0:
                row["neighbour_price"] = float(np.nanmean(train_target_arr[neigh]))
            train_rows.append(row)
        val_rows = []
        for i in range(len(val_df)):
            neigh = idx_val[i].tolist()[:k]
            sub = train_df.iloc[neigh]
            row = {}
            for c in rag_num_cols:
                row[f"rag_mean_{c}"] = float(sub[c].dropna().mean()) if sub[c].notna().sum() else np.nan
            for c in rag_cat_cols:
                row[f"rag_mode_{c}"] = _row_mode(sub[c].astype(str))
            if train_target_arr is not None and len(neigh) > 0:
                row["neighbour_price"] = float(np.nanmean(train_target_arr[neigh]))
            val_rows.append(row)
        rag_train = pd.DataFrame(train_rows, index=train_df.index)
        rag_val = pd.DataFrame(val_rows, index=val_df.index)
        return rag_train, rag_val

    def build_rag_features_with_test(train_df, val_df, test_df, k=12, train_target=None, rag_mode="hybrid"):
        cfg = RAG_MODE_CONFIG.get(rag_mode, RAG_MODE_CONFIG["hybrid"])
        use_num = [c for c in cfg["numeric"] if c in train_df.columns]
        use_cat = [c for c in cfg["cat"] if c in train_df.columns]
        train_space, val_space = _fit_rag_space(train_df, val_df, use_num, use_cat)
        _, test_space = _fit_rag_space(train_df, test_df, use_num, use_cat)
        if train_space.shape[1] == 0:
            train_space = np.zeros((len(train_df), 1))
            val_space = np.zeros((len(val_df), 1))
            test_space = np.zeros((len(test_df), 1))
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_df)), metric="cosine")
        nn.fit(train_space)
        _, idx_train = nn.kneighbors(train_space)
        _, idx_val = nn.kneighbors(val_space, n_neighbors=min(k, len(train_df)))
        _, idx_test = nn.kneighbors(test_space, n_neighbors=min(k, len(train_df)))
        rag_num_cols = use_num
        rag_cat_cols = use_cat
        train_target_arr = np.asarray(train_target) if train_target is not None else None

        def rows_for_indices(idxs_matrix, query_len, is_train_self=False):
            out = []
            for i in range(query_len):
                neigh = idxs_matrix[i].tolist() if not is_train_self else idx_train[i].tolist()
                if is_train_self and i in neigh:
                    neigh.remove(i)
                neigh = neigh[:k]
                sub = train_df.iloc[neigh]
                row = {}
                for c in rag_num_cols:
                    row[f"rag_mean_{c}"] = float(sub[c].dropna().mean()) if sub[c].notna().sum() else np.nan
                for c in rag_cat_cols:
                    row[f"rag_mode_{c}"] = _row_mode(sub[c].astype(str))
                if train_target_arr is not None and len(neigh) > 0:
                    row["neighbour_price"] = float(np.nanmean(train_target_arr[neigh]))
                out.append(row)
            return out

        train_rows = rows_for_indices(idx_train, len(train_df), is_train_self=True)
        val_rows = rows_for_indices(idx_val, len(val_df), is_train_self=False)
        test_rows = rows_for_indices(idx_test, len(test_df), is_train_self=False)
        return (
            pd.DataFrame(train_rows, index=train_df.index),
            pd.DataFrame(val_rows, index=val_df.index),
            pd.DataFrame(test_rows, index=test_df.index),
        )

    rag_train_df, rag_val_df, rag_test_df = build_rag_features_with_test(
        df_filled_train.drop(columns=[TARGET_COL]),
        df_filled_val.drop(columns=[TARGET_COL]),
        df_filled_test.drop(columns=[TARGET_COL]),
        k=RAG_K,
        train_target=df_filled_train[TARGET_COL].values,
        rag_mode=RAG_MODE,
    )

    # =========================
    # 4.4) Text templates (structured/semantic only)
    # =========================
    def boston_bucket(feature_name, value):
        if pd.isna(value):
            return "unknown"
        x = float(value)
        if feature_name == "CRIM":
            return qbucket(x, [(1, "low_crime"), (5, "moderate_crime"), (15, "high_crime"), (1e18, "very_high_crime")])
        if feature_name == "ZN":
            return qbucket(x, [(5, "low_zoning"), (20, "moderate_zoning"), (50, "high_zoning"), (1e18, "very_high_zoning")])
        if feature_name == "INDUS":
            return qbucket(x, [(5, "low_industry"), (12, "moderate_industry"), (20, "high_industry"), (1e18, "very_high_industry")])
        if feature_name == "NOX":
            return qbucket(x, [(0.45, "cleaner_air"), (0.55, "moderate_air"), (0.7, "polluted_air"), (1e18, "highly_polluted_air")])
        if feature_name == "RM":
            return qbucket(x, [(5.5, "compact_home"), (6.5, "standard_home"), (7.5, "spacious_home"), (1e18, "very_spacious_home")])
        if feature_name == "AGE":
            return qbucket(x, [(35, "newer_stock"), (65, "mid_age_stock"), (85, "older_stock"), (1e18, "very_old_stock")])
        if feature_name == "DIS":
            return qbucket(x, [(2.5, "very_close_to_jobs"), (5, "close_to_jobs"), (8, "mid_distance_to_jobs"), (1e18, "far_from_jobs")])
        if feature_name == "RAD":
            return qbucket(x, [(4, "low_highway_access"), (8, "moderate_highway_access"), (16, "high_highway_access"), (1e18, "very_high_highway_access")])
        if feature_name == "TAX":
            return qbucket(x, [(300, "low_tax"), (450, "moderate_tax"), (650, "high_tax"), (1e18, "very_high_tax")])
        if feature_name == "PTRATIO":
            return qbucket(x, [(16, "small_class_ratio"), (19, "mid_class_ratio"), (22, "large_class_ratio"), (1e18, "very_large_class_ratio")])
        if feature_name == "B":
            return qbucket(x, [(350, "lower_B_stat"), (390, "mid_B_stat"), (395, "higher_B_stat"), (1e18, "very_high_B_stat")])
        if feature_name == "LSTAT":
            return qbucket(x, [(7, "low_lstat"), (15, "mid_lstat"), (25, "high_lstat"), (1e18, "very_high_lstat")])
        return "unknown"

    def generate_structured_description(i, textf, numf):
        parts = []
        river_bound = "river_bound" if ("CHAS" in numf and not pd.isna(numf["CHAS"][i]) and int(numf["CHAS"][i]) == 1) else "not_river_bound"
        parts.append(f"river_relation: {river_bound}")
        for c in ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]:
            if c in numf:
                parts.append(f"{c.lower()}_band: {boston_bucket(c, numf[c][i])}")
        return ". ".join(parts) + "."

    def generate_semantic_description(i, textf, numf):
        chas = numf.get("CHAS", [np.nan])[i] if "CHAS" in numf else np.nan
        crim = numf.get("CRIM", [np.nan])[i] if "CRIM" in numf else np.nan
        zn = numf.get("ZN", [np.nan])[i] if "ZN" in numf else np.nan
        indus = numf.get("INDUS", [np.nan])[i] if "INDUS" in numf else np.nan
        nox = numf.get("NOX", [np.nan])[i] if "NOX" in numf else np.nan
        rm = numf.get("RM", [np.nan])[i] if "RM" in numf else np.nan
        age = numf.get("AGE", [np.nan])[i] if "AGE" in numf else np.nan
        dis = numf.get("DIS", [np.nan])[i] if "DIS" in numf else np.nan
        rad = numf.get("RAD", [np.nan])[i] if "RAD" in numf else np.nan
        tax = numf.get("TAX", [np.nan])[i] if "TAX" in numf else np.nan
        ptratio = numf.get("PTRATIO", [np.nan])[i] if "PTRATIO" in numf else np.nan
        b_stat = numf.get("B", [np.nan])[i] if "B" in numf else np.nan
        lstat = numf.get("LSTAT", [np.nan])[i] if "LSTAT" in numf else np.nan

        crim_bucket = boston_bucket("CRIM", crim)
        rm_bucket = boston_bucket("RM", rm)
        dis_bucket = boston_bucket("DIS", dis)
        lstat_bucket = boston_bucket("LSTAT", lstat)
        zoning_bucket = boston_bucket("ZN", zn)
        industry_bucket = boston_bucket("INDUS", indus)
        nox_bucket = boston_bucket("NOX", nox)
        tax_bucket = boston_bucket("TAX", tax)
        ptratio_bucket = boston_bucket("PTRATIO", ptratio)
        b_bucket = boston_bucket("B", b_stat)
        age_bucket = boston_bucket("AGE", age)
        rad_bucket = boston_bucket("RAD", rad)

        cues = []
        if not pd.isna(chas) and int(chas) == 1:
            cues.append("charles_river_premium_proxy")
        if not pd.isna(rm) and rm >= 7:
            cues.append("larger_home_signal")
        if not pd.isna(crim) and crim <= 1:
            cues.append("safer_area_signal")
        if not pd.isna(ptratio) and ptratio <= 17:
            cues.append("better_school_ratio_signal")
        if not pd.isna(lstat) and lstat <= 10:
            cues.append("higher_socioeconomic_area_proxy")

        base = (
            f"boston housing profile: river_relation={'river_bound' if (not pd.isna(chas) and int(chas) == 1) else 'not_river_bound'}, "
            f"highway_access_band={rad_bucket}. "
            f"neighborhood: crime_band={crim_bucket}, zoning_band={zoning_bucket}, industry_band={industry_bucket}, "
            f"air_quality_band={nox_bucket}, distance_to_jobs_band={dis_bucket}. "
            f"dwelling: room_band={rm_bucket}, housing_age_band={age_bucket}, "
            f"tax_band={tax_bucket}, school_ratio_band={ptratio_bucket}, B_band={b_bucket}, "
            f"lower_status_band={lstat_bucket}. "
            f"signals: {' '.join(cues) if len(cues) else 'neutral_signals'}."
        )
        return base

    def rag_row_to_text(rag_row, k, prefix="rag"):
        parts = [f"{prefix}_evidence_k={k}"]
        num_fields = [f"{prefix}_mean_{c}" for c in RAG_NUM_COLS]
        cat_fields = [f"{prefix}_mode_{c}" for c in RAG_CAT_COLS]

        for c in num_fields:
            if c in rag_row and pd.notna(rag_row[c]):
                raw_name = c.replace(f"{prefix}_mean_", "")
                parts.append(f"{c}_band={boston_bucket(raw_name, rag_row[c])}")

        for c in cat_fields:
            if c in rag_row and pd.notna(rag_row[c]):
                parts.append(f"{c}={str(rag_row[c])}")

        return " ; ".join(parts) + "."

    def _boston_rag_compare_parts(i, numf, rag_df, rel_tol=0.05):
        """Compare to neighbors: larger/smaller/comparable. Returns English phrases."""
        rag_row = rag_df.iloc[i]
        parts = []
        for c in RAG_NUM_COLS:
            if c not in numf or pd.isna(numf[c][i]):
                continue
            key = f"rag_mean_{c}"
            if key not in rag_row.index or pd.isna(rag_row[key]):
                continue
            self_val = float(numf[c][i])
            nei_val = float(rag_row[key])
            label = BOSTON_RAG_LABELS.get(c, c)
            if nei_val <= 0 or (c in ["CHAS", "RAD"] and nei_val == 0):
                parts.append(f"{label} comparable to neighbors")
                continue
            diff_ratio = (self_val - nei_val) / (nei_val + 1e-12)
            if diff_ratio > rel_tol:
                parts.append(f"{label} higher than neighbors")
            elif diff_ratio < -rel_tol:
                parts.append(f"{label} lower than neighbors")
            else:
                parts.append(f"{label} comparable to neighbors")
        for c in RAG_CAT_COLS:
            key = f"rag_mode_{c}"
            if key in rag_row.index and pd.notna(rag_row[key]):
                parts.append(f"{BOSTON_RAG_LABELS.get(c, c)} aligned with most neighbors")
        return parts

    def _boston_rag_delta_parts(i, numf, rag_df):
        """Concrete delta vs neighbors. Returns English phrases."""
        rag_row = rag_df.iloc[i]
        parts = []
        for c in RAG_NUM_COLS:
            if c not in numf or pd.isna(numf[c][i]):
                continue
            key = f"rag_mean_{c}"
            if key not in rag_row.index or pd.isna(rag_row[key]):
                continue
            self_val = float(numf[c][i])
            nei_val = float(rag_row[key])
            delta = self_val - nei_val
            label = BOSTON_RAG_LABELS.get(c, c)
            if abs(delta) < 1e-6:
                parts.append(f"{label} on par with neighbor mean")
                continue
            if c in ["RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT", "INDUS", "ZN"]:
                d = abs(round(delta, 2))
                if delta > 0:
                    parts.append(f"{label} {d} higher than neighbor mean")
                else:
                    parts.append(f"{label} {d} lower than neighbor mean")
            else:
                d = abs(round(delta, 3))
                if delta > 0:
                    parts.append(f"{label} {d} higher than neighbors")
                else:
                    parts.append(f"{label} {d} lower than neighbors")
        for c in RAG_CAT_COLS:
            key = f"rag_mode_{c}"
            if key in rag_row.index and pd.notna(rag_row[key]):
                parts.append(f"{BOSTON_RAG_LABELS.get(c, c)} consistent with neighbors")
        return parts

    def build_embedding_texts(df_part, textf, numf, rag_df=None, template="semantic", use_rag_in_text=False, rag_k=12, rag_template="default"):
        texts = []
        n = len(df_part)
        for i in range(n):
            if template == "structured":
                base = generate_structured_description(i, textf, numf)
            else:
                base = generate_semantic_description(i, textf, numf)

            if use_rag_in_text and rag_df is not None and USE_RAG_TEXT:
                if rag_template == "default":
                    rag_txt = rag_row_to_text(rag_df.iloc[i].to_dict(), rag_k, prefix="rag")
                    base = base + " neighborhood_context: " + rag_txt
                elif rag_template == "compare":
                    parts = _boston_rag_compare_parts(i, numf, rag_df, rel_tol=0.05)
                    if parts:
                        base = base + " Compared with similar homes: " + ", ".join(parts) + "."
                elif rag_template == "delta":
                    parts = _boston_rag_delta_parts(i, numf, rag_df)
                    if parts:
                        base = base + " Relative to neighbors: " + ", ".join(parts) + "."
                else:
                    rag_txt = rag_row_to_text(rag_df.iloc[i].to_dict(), rag_k, prefix="rag")
                    base = base + " neighborhood_context: " + rag_txt
            texts.append(base)
        return texts

    # =========================
    # 4.5) Embeddings (if available)
    # =========================
    embeddings_plain_train = None
    embeddings_plain_val = None
    embeddings_plain_test = None
    embeddings_with_ragtext_train = None
    embeddings_with_ragtext_val = None
    embeddings_with_ragtext_test = None

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
                failed_batches = 0
                total_batches = 0
                for i in range(0, len(descriptions), batch_size):
                    total_batches += 1
                    batch = descriptions[i:i + batch_size]
                    try:
                        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                        embs.extend([item.embedding for item in resp.data])
                    except Exception as e:
                        failed_batches += 1
                        print(f"    embedding批失败: {e}, 用零向量填充")
                        embs.extend([[0.0] * EMBEDDING_DIM] * len(batch))
                return np.array(embs, dtype=float), failed_batches, total_batches

            # plain texts
            txt_plain_train = build_embedding_texts(
                df_filled_train, text_features_train, num_features_train,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )
            txt_plain_val = build_embedding_texts(
                df_filled_val, text_features_val, num_features_val,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )
            txt_plain_test = build_embedding_texts(
                df_filled_test, text_features_test, num_features_test,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )

            # rag-text texts (emb_with_rag)
            txt_rag_train = build_embedding_texts(
                df_filled_train, text_features_train, num_features_train,
                rag_df=rag_train_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K, rag_template=RAG_TEMPLATE
            )
            txt_rag_val = build_embedding_texts(
                df_filled_val, text_features_val, num_features_val,
                rag_df=rag_val_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K, rag_template=RAG_TEMPLATE
            )
            txt_rag_test = build_embedding_texts(
                df_filled_test, text_features_test, num_features_test,
                rag_df=rag_test_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K, rag_template=RAG_TEMPLATE
            )

            _sk = "split"
            plain_train_key = f"boston_plain_train_{EMBEDDING_MODEL}_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{seed}_{_sk}_{text_list_hash(txt_plain_train)}"
            plain_val_key = f"boston_plain_val_{EMBEDDING_MODEL}_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{seed}_{_sk}_{text_list_hash(txt_plain_val)}"
            plain_test_key = f"boston_plain_test_{EMBEDDING_MODEL}_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{seed}_{_sk}_{text_list_hash(txt_plain_test)}"
            rag_train_key = f"boston_ragtxt_train_{EMBEDDING_MODEL}_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{RAG_TEMPLATE}_{DATA_RATIO}_{seed}_{_sk}_{text_list_hash(txt_rag_train)}"
            rag_val_key = f"boston_ragtxt_val_{EMBEDDING_MODEL}_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{RAG_TEMPLATE}_{DATA_RATIO}_{seed}_{_sk}_{text_list_hash(txt_rag_val)}"
            rag_test_key = f"boston_ragtxt_test_{EMBEDDING_MODEL}_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{RAG_TEMPLATE}_{DATA_RATIO}_{seed}_{_sk}_{text_list_hash(txt_rag_test)}"

            emb_plain_tr_raw = load_cached_embeddings(plain_train_key)
            emb_plain_val_raw = load_cached_embeddings(plain_val_key)
            emb_plain_test_raw = load_cached_embeddings(plain_test_key)
            emb_rag_tr_raw = load_cached_embeddings(rag_train_key)
            emb_rag_val_raw = load_cached_embeddings(rag_val_key)
            emb_rag_test_raw = load_cached_embeddings(rag_test_key)

            if emb_plain_tr_raw is None:
                emb_plain_tr_raw, f1, t1 = get_embeddings_batch(txt_plain_train, batch_size=EMB_BATCH_SIZE)
                if f1 > 0:
                    print(f"    plain_train embedding失败批次: {f1}/{t1}")
                save_cached_embeddings(plain_train_key, emb_plain_tr_raw)
            if emb_plain_val_raw is None:
                emb_plain_val_raw, f2, t2 = get_embeddings_batch(txt_plain_val, batch_size=EMB_BATCH_SIZE)
                if f2 > 0:
                    print(f"    plain_val embedding失败批次: {f2}/{t2}")
                save_cached_embeddings(plain_val_key, emb_plain_val_raw)
            if emb_plain_test_raw is None:
                emb_plain_test_raw, f2b, t2b = get_embeddings_batch(txt_plain_test, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(plain_test_key, emb_plain_test_raw)
            if emb_rag_tr_raw is None:
                emb_rag_tr_raw, f3, t3 = get_embeddings_batch(txt_rag_train, batch_size=EMB_BATCH_SIZE)
                if f3 > 0:
                    print(f"    ragtxt_train embedding失败批次: {f3}/{t3}")
                save_cached_embeddings(rag_train_key, emb_rag_tr_raw)
            if emb_rag_val_raw is None:
                emb_rag_val_raw, f4, t4 = get_embeddings_batch(txt_rag_val, batch_size=EMB_BATCH_SIZE)
                if f4 > 0:
                    print(f"    ragtxt_val embedding失败批次: {f4}/{t4}")
                save_cached_embeddings(rag_val_key, emb_rag_val_raw)
            if emb_rag_test_raw is None:
                emb_rag_test_raw, f4b, t4b = get_embeddings_batch(txt_rag_test, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(rag_test_key, emb_rag_test_raw)

            for name, arr in [
                ("plain_train", emb_plain_tr_raw),
                ("plain_val", emb_plain_val_raw),
                ("plain_test", emb_plain_test_raw),
                ("ragtxt_train", emb_rag_tr_raw),
                ("ragtxt_val", emb_rag_val_raw),
                ("ragtxt_test", emb_rag_test_raw),
            ]:
                if np.allclose(arr, 0.0):
                    print(f"    警告: {name} embedding全为0，emb模式会退化为非emb模式")

            embeddings_plain_train, embeddings_plain_val, embeddings_plain_test, evr_plain = reduce_embeddings_3(
                emb_plain_tr_raw, emb_plain_val_raw, emb_plain_test_raw, PCA_DIM, seed, DIM_REDUCER
            )

            embeddings_with_ragtext_train, embeddings_with_ragtext_val, embeddings_with_ragtext_test, evr_ragtxt = reduce_embeddings_3(
                emb_rag_tr_raw, emb_rag_val_raw, emb_rag_test_raw, RAG_PCA_DIM, seed, DIM_REDUCER
            )
    else:
        OPENAI_AVAILABLE_FOLD = False

    # =========================
    # 4.6) Box-Cox / skew handling
    # =========================
    boxcox_lambdas = {}
    boxcox_shifts = {}
    TOTAL_LAMBDA_FOLD = None
    TOTAL_SHIFT_FOLD = None

    for col in [TARGET_COL, "CRIM", "LSTAT"]:
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
                        if col == TARGET_COL:
                            TOTAL_LAMBDA_FOLD = fitted_lambda
                            TOTAL_SHIFT_FOLD = shift

                        df_filled_train[col] = boxcox(df_filled_train[col] + shift, fitted_lambda)
                        df_filled_val[col] = boxcox(df_filled_val[col] + boxcox_shifts.get(col, 0), fitted_lambda)
                        df_filled_test[col] = boxcox(df_filled_test[col] + boxcox_shifts.get(col, 0), fitted_lambda)
                    except Exception as e:
                        pass

    numeric_cols_for_skew = [
        c for c in df_filled_train.select_dtypes(include=[np.number]).columns.tolist()
        if c not in boxcox_lambdas
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
                df_filled_test.loc[df_filled_test[col] < lower, col] = lower
                df_filled_test.loc[df_filled_test[col] > upper, col] = upper

    # =========================
    # 4.7) Tabular preprocess
    # =========================
    categorical_cols = TEXT_CAT_COLS
    for col in categorical_cols:
        if col in df_filled_train.columns:
            if ENCODE_METHOD == "label":
                le = LabelEncoder()
                train_vals = df_filled_train[col].astype(str)
                df_filled_train[col] = le.fit_transform(train_vals)

                val_vals = df_filled_val[col].astype(str)
                unseen = ~val_vals.isin(le.classes_)
                if unseen.sum() > 0:
                    fallback = train_vals.mode()[0] if len(train_vals.mode()) > 0 else le.classes_[0]
                    val_vals[unseen] = fallback
                df_filled_val[col] = le.transform(val_vals)
                te_vals = df_filled_test[col].astype(str)
                unseen_te = ~te_vals.isin(le.classes_)
                if unseen_te.sum() > 0:
                    te_vals[unseen_te] = fallback
                df_filled_test[col] = le.transform(te_vals)
            else:
                dtr = pd.get_dummies(df_filled_train[col], prefix=col)
                dte = pd.get_dummies(df_filled_val[col], prefix=col)
                dt_te = pd.get_dummies(df_filled_test[col], prefix=col)
                for d in dtr.columns:
                    if d not in dte.columns:
                        dte[d] = 0
                    if d not in dt_te.columns:
                        dt_te[d] = 0
                dte = dte[dtr.columns]
                dt_te = dt_te.reindex(columns=dtr.columns, fill_value=0)

                df_filled_train = pd.concat([df_filled_train.drop(columns=[col]), dtr], axis=1)
                df_filled_val = pd.concat([df_filled_val.drop(columns=[col]), dte], axis=1)
                df_filled_test = pd.concat([df_filled_test.drop(columns=[col]), dt_te], axis=1)

    # drop non-numeric leftovers
    non_num_train = df_filled_train.select_dtypes(exclude=[np.number]).columns.tolist()
    non_num_val = df_filled_val.select_dtypes(exclude=[np.number]).columns.tolist()
    non_num_te = df_filled_test.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_num_train) > 0:
        df_filled_train = df_filled_train.drop(columns=non_num_train)
        df_filled_val = df_filled_val.drop(columns=non_num_val)
        df_filled_test = df_filled_test.drop(columns=non_num_te)

    df_filled_val = df_filled_val.reindex(columns=df_filled_train.columns, fill_value=0)
    df_filled_test = df_filled_test.reindex(columns=df_filled_train.columns, fill_value=0)

    X_train_df = df_filled_train.drop(TARGET_COL, axis=1).copy()
    y_train = df_filled_train[TARGET_COL].copy()
    X_val_df = df_filled_val.drop(TARGET_COL, axis=1).copy()
    y_val = df_filled_val[TARGET_COL].copy()
    X_test_df = df_filled_test.drop(TARGET_COL, axis=1).copy()
    y_test = df_filled_test[TARGET_COL].copy()

    # rag df encode numeric
    def encode_rag_df(rag_tr, rag_val, rag_te=None):
        rag_tr = rag_tr.copy()
        rag_val = rag_val.copy()
        rag_te = rag_te.copy() if rag_te is not None else None
        for c in rag_tr.columns:
            if rag_tr[c].dtype == "object":
                vals = rag_tr[c].astype(str).fillna("None")
                le = LabelEncoder()
                le.fit(vals)
                mp = {k: i for i, k in enumerate(le.classes_)}
                default = mp.get("None", 0)
                rag_tr[c] = vals.map(mp).fillna(default).astype(float)
                rag_val[c] = rag_val[c].astype(str).fillna("None").map(lambda x: mp.get(x, default)).astype(float)
                if rag_te is not None:
                    rag_te[c] = rag_te[c].astype(str).fillna("None").map(lambda x: mp.get(x, default)).astype(float)
            else:
                med = rag_tr[c].median()
                rag_tr[c] = rag_tr[c].fillna(med)
                rag_val[c] = rag_val[c].fillna(med)
                if rag_te is not None:
                    rag_te[c] = rag_te[c].fillna(med)
        return (rag_tr, rag_val, rag_te) if rag_te is not None else (rag_tr, rag_val)

    # Encode RAG features for rag/emb+rag (exclude neighbour_price so only emb_with_rag_price gets it)
    rag_train_for_encode = rag_train_df.drop(columns=["neighbour_price"], errors="ignore")
    rag_val_for_encode = rag_val_df.drop(columns=["neighbour_price"], errors="ignore")
    rag_test_for_encode = rag_test_df.drop(columns=["neighbour_price"], errors="ignore")
    rag_train_num, rag_val_num, rag_test_num = encode_rag_df(
        rag_train_for_encode, rag_val_for_encode, rag_test_for_encode
    )

    # minmax scale
    num_cols_scale = [c for c in X_train_df.select_dtypes(include=[np.number]).columns]
    for c in num_cols_scale:
        sc = MinMaxScaler()
        sc.fit(X_train_df[[c]])
        X_train_df[c] = sc.transform(X_train_df[[c]])
        X_val_df[c] = sc.transform(X_val_df[[c]])
        X_test_df[c] = sc.transform(X_test_df[[c]])

    # =========================
    # 4.8) Build mode matrices
    # =========================
    X_train_base = X_train_df.values
    X_val_base = X_val_df.values
    X_test_base = X_test_df.values

    X_train_rag = np.hstack([X_train_df.values, rag_train_num.values])
    X_val_rag = np.hstack([X_val_df.values, rag_val_num.values])
    X_test_rag = np.hstack([X_test_df.values, rag_test_num.values])

    # rag_price: 仅用 neighbour_price 作为新特征（baseline + 邻居 MEDV 均值）
    X_train_rag_price = None
    X_val_rag_price = None
    X_test_rag_price = None
    if "neighbour_price" in rag_train_df.columns:
        sc_price = MinMaxScaler()
        np_train = sc_price.fit_transform(rag_train_df[["neighbour_price"]].fillna(rag_train_df["neighbour_price"].median()).values)
        np_val = sc_price.transform(rag_val_df[["neighbour_price"]].fillna(rag_train_df["neighbour_price"].median()).values)
        np_te = sc_price.transform(rag_test_df[["neighbour_price"]].fillna(rag_train_df["neighbour_price"].median()).values)
        X_train_rag_price = np.hstack([X_train_df.values, np_train])
        X_val_rag_price = np.hstack([X_val_df.values, np_val])
        X_test_rag_price = np.hstack([X_test_df.values, np_te])

    X_train_emb = None
    X_val_emb = None
    X_test_emb = None
    X_train_emb_plus_rag = None
    X_val_emb_plus_rag = None
    X_test_emb_plus_rag = None
    X_train_emb_with_rag = None
    X_val_emb_with_rag = None
    X_test_emb_with_rag = None
    X_train_emb_with_rag_plus_rag = None
    X_val_emb_with_rag_plus_rag = None
    X_test_emb_with_rag_plus_rag = None
    X_train_emb_with_rag_price = None
    X_val_emb_with_rag_price = None
    X_test_emb_with_rag_price = None

    if OPENAI_AVAILABLE_FOLD and embeddings_plain_train is not None:
        X_train_emb = np.hstack([X_train_df.values, embeddings_plain_train])
        X_val_emb = np.hstack([X_val_df.values, embeddings_plain_val])
        X_test_emb = np.hstack([X_test_df.values, embeddings_plain_test])

        X_train_emb_plus_rag = np.hstack([X_train_df.values, embeddings_plain_train, rag_train_num.values])
        X_val_emb_plus_rag = np.hstack([X_val_df.values, embeddings_plain_val, rag_val_num.values])
        X_test_emb_plus_rag = np.hstack([X_test_df.values, embeddings_plain_test, rag_test_num.values])

        X_train_emb_with_rag = np.hstack([X_train_df.values, embeddings_with_ragtext_train])
        X_val_emb_with_rag = np.hstack([X_val_df.values, embeddings_with_ragtext_val])
        X_test_emb_with_rag = np.hstack([X_test_df.values, embeddings_with_ragtext_test])

        X_train_emb_with_rag_plus_rag = np.hstack([X_train_df.values, embeddings_with_ragtext_train, rag_train_num.values])
        X_val_emb_with_rag_plus_rag = np.hstack([X_val_df.values, embeddings_with_ragtext_val, rag_val_num.values])
        X_test_emb_with_rag_plus_rag = np.hstack([X_test_df.values, embeddings_with_ragtext_test, rag_test_num.values])

        if "neighbour_price" in rag_train_df.columns:
            sc_price = MinMaxScaler()
            np_train = sc_price.fit_transform(rag_train_df[["neighbour_price"]].fillna(rag_train_df["neighbour_price"].median()).values)
            np_val = sc_price.transform(rag_val_df[["neighbour_price"]].fillna(rag_train_df["neighbour_price"].median()).values)
            np_te = sc_price.transform(rag_test_df[["neighbour_price"]].fillna(rag_train_df["neighbour_price"].median()).values)
            X_train_emb_with_rag_price = np.hstack([X_train_df.values, embeddings_with_ragtext_train, np_train])
            X_val_emb_with_rag_price = np.hstack([X_val_df.values, embeddings_with_ragtext_val, np_val])
            X_test_emb_with_rag_price = np.hstack([X_test_df.values, embeddings_with_ragtext_test, np_te])

    # =========================
    # 4.9) Train/Eval
    # =========================
    def evaluate_model_fold(y_true_transformed, y_pred_transformed, model_name, split_label):
        if TOTAL_LAMBDA_FOLD is not None:
            y_true_original = inverse_boxcox(y_true_transformed, TOTAL_LAMBDA_FOLD, TOTAL_SHIFT_FOLD)
            y_pred_original = inverse_boxcox(y_pred_transformed, TOTAL_LAMBDA_FOLD, TOTAL_SHIFT_FOLD)
        else:
            y_true_original = y_true_transformed
            y_pred_original = y_pred_transformed

        y_true_original = np.asarray(y_true_original, dtype=float).ravel()
        y_pred_original = np.asarray(y_pred_original, dtype=float).ravel()
        valid_mask = np.isfinite(y_true_original) & np.isfinite(y_pred_original)
        if not np.all(valid_mask):
            y_true_original = y_true_original[valid_mask]
            y_pred_original = y_pred_original[valid_mask]
        if len(y_true_original) == 0:
            print(f"  {model_name} 评估失败: 无有效样本")
            return None

        rmsle = calculate_rmsle(y_true_original, y_pred_original, scale=1000.0)
        rmse = calculate_rmse(y_true_original, y_pred_original)
        return {"Split": split_label, "Model": model_name, "RMSLE": rmsle, "RMSE": rmse}

    fold_results = []

    def _fit_predict_val_test(model_class, model_name, Xtr, Xval, Xte, suffix, is_tabpfn=False):
        if is_tabpfn:
            try:
                model = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
                model.fit(Xtr, y_train.values)
            except Exception:
                model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
                model.fit(Xtr, y_train.values)
            y_pred_val = model.predict(Xval)
            X_fit = np.vstack([Xtr, Xval])
            y_fit = np.concatenate([y_train.values, y_val.values])
            model.fit(X_fit, y_fit)
            y_pred_test = model.predict(Xte)
        elif model_class == xgb.XGBRegressor:
            model = model_class(
                n_estimators=2373, max_depth=7, learning_rate=0.0018229,
                subsample=0.5154, colsample_bytree=0.4460,
                reg_alpha=0.0020825, reg_lambda=0.0059023,
                random_state=seed, n_jobs=-1,
            )
            model.fit(Xtr, y_train)
            y_pred_val = model.predict(Xval)
            model.fit(np.vstack([Xtr, Xval]), pd.concat([y_train, y_val]))
            y_pred_test = model.predict(Xte)
        elif model_class == lgb.LGBMRegressor:
            model = model_class(
                objective="regression", num_leaves=56, learning_rate=0.03083,
                n_estimators=3570, max_bin=66,
                bagging_fraction=0.9819, bagging_freq=9, feature_fraction=0.2911,
                random_state=seed, n_jobs=-1, verbose=-1,
            )
            model.fit(Xtr, y_train)
            y_pred_val = model.predict(Xval)
            model.fit(np.vstack([Xtr, Xval]), pd.concat([y_train, y_val]))
            y_pred_test = model.predict(Xte)
        elif model_class == cb.CatBoostRegressor:
            model = model_class(
                depth=7, learning_rate=0.03822, l2_leaf_reg=0.005143,
                iterations=779, random_seed=seed, verbose=False,
            )
            model.fit(Xtr, y_train)
            y_pred_val = model.predict(Xval)
            model.fit(np.vstack([Xtr, Xval]), pd.concat([y_train, y_val]))
            y_pred_test = model.predict(Xte)
        else:
            return None, None
        full_name = f"{model_name} ({suffix})"
        r_val = evaluate_model_fold(y_val, y_pred_val, full_name, "val")
        r_test = evaluate_model_fold(y_test, y_pred_test, full_name, "test")
        return r_val, r_test

    def train_and_evaluate_fold(model_class, model_name, Xtr, Xval, Xte, suffix="", is_tabpfn=False):
        try:
            return _fit_predict_val_test(model_class, model_name, Xtr, Xval, Xte, suffix, is_tabpfn=is_tabpfn)
        except Exception as e:
            print(f"  {model_name} ({suffix}) 训练失败: {e}")
            return None, None

    def run_family_fold(Xtr, Xval, Xte, suffix):
        if XGBOOST_AVAILABLE:
            rv, rt = train_and_evaluate_fold(xgb.XGBRegressor, "XGBoost", Xtr, Xval, Xte, suffix)
            if rv:
                fold_results.append(rv)
            if rt:
                fold_results.append(rt)
        if LIGHTGBM_AVAILABLE:
            rv, rt = train_and_evaluate_fold(lgb.LGBMRegressor, "LightGBM", Xtr, Xval, Xte, suffix)
            if rv:
                fold_results.append(rv)
            if rt:
                fold_results.append(rt)
        if CATBOOST_AVAILABLE:
            rv, rt = train_and_evaluate_fold(cb.CatBoostRegressor, "CatBoost", Xtr, Xval, Xte, suffix)
            if rv:
                fold_results.append(rv)
            if rt:
                fold_results.append(rt)
        if TABPFN_AVAILABLE:
            rv, rt = train_and_evaluate_fold(None, "TabPFN", Xtr, Xval, Xte, suffix, is_tabpfn=True)
            if rv:
                fold_results.append(rv)
            if rt:
                fold_results.append(rt)

    if "baseline" in MODES:
        run_family_fold(X_train_base, X_val_base, X_test_base, "baseline")
    if "rag" in MODES:
        run_family_fold(X_train_rag, X_val_rag, X_test_rag, "rag")
    if "rag_price" in MODES:
        if X_train_rag_price is not None:
            run_family_fold(X_train_rag_price, X_val_rag_price, X_test_rag_price, f"rag_price_{RAG_MODE}")
        else:
            print("  [WARN] rag_price 需要 neighbour_price，跳过")
    if "emb" in MODES and X_train_emb is not None:
        run_family_fold(X_train_emb, X_val_emb, X_test_emb, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}")
    if "emb+rag" in MODES and X_train_emb_plus_rag is not None:
        run_family_fold(X_train_emb_plus_rag, X_val_emb_plus_rag, X_test_emb_plus_rag, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}+rag")
    if "emb_with_rag" in MODES and X_train_emb_with_rag is not None:
        run_family_fold(X_train_emb_with_rag, X_val_emb_with_rag, X_test_emb_with_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}")
    if "emb_with_rag+rag" in MODES and X_train_emb_with_rag_plus_rag is not None:
        run_family_fold(X_train_emb_with_rag_plus_rag, X_val_emb_with_rag_plus_rag, X_test_emb_with_rag_plus_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}+rag")
    if "emb_with_rag_price" in MODES and X_train_emb_with_rag_price is not None:
        run_family_fold(X_train_emb_with_rag_price, X_val_emb_with_rag_price, X_test_emb_with_rag_price, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}+price")
    
    return fold_results

# =========================
# 5) 与 Ames 相同划分，按 seed 运行
# =========================
print("\n" + "=" * 90)
print("70%/10%/20% train/val/test（与 Ames 一致）" + (f" — {len(SEEDS)} 个 seed" if len(SEEDS) > 1 else ""))
print("=" * 90)

all_results = []
df_full = df.copy()
all_indices = np.arange(len(df_full))

for seed in SEEDS:
    np.random.seed(seed)
    train_idx, rest = train_test_split(all_indices, test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(rest, test_size=2.0 / 3.0, random_state=seed)
    if len(SEEDS) > 1:
        print(f"\n--- seed={seed} ---")
    split_results = process_one_split(df_full, train_idx, val_idx, test_idx, seed=seed)
    all_results.extend(split_results)

# =========================
# 6) 汇总 val / test
# =========================
print("\n" + "=" * 90)
print("结果汇总（Val = 训练集上拟合后在验证集；Test = train+val refit 后在测试集）")
print("=" * 90)

if len(all_results) == 0:
    print("No models were successfully trained.")
else:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE.replace(".csv", "_detailed.csv"), index=False)

    summary = results_df.groupby(["Model", "Split"], as_index=False).agg(
        {"RMSLE": "mean", "RMSE": "mean"}
    ).round(6)
    summary_val = summary[summary["Split"] == "val"].sort_values("RMSLE")
    summary_test = summary[summary["Split"] == "test"].sort_values("RMSLE")
    summary_out = pd.concat([summary_val, summary_test], ignore_index=True)
    summary_out.to_csv(OUTPUT_FILE, index=False)

    print("\nVal:")
    print(summary_val.to_string(index=False))
    print("\nTest:")
    print(summary_test.to_string(index=False))
    print(f"\n详细: {OUTPUT_FILE.replace('.csv', '_detailed.csv')}")
    print(f"汇总: {OUTPUT_FILE}")

    if len(summary_val) > 0:
        best = summary_val.iloc[0]
        print(f"\nVal 最佳: {best['Model']}  RMSLE={best['RMSLE']:.6f}  RMSE={best['RMSE']:.6f}")
    if len(summary_test) > 0:
        best_t = summary_test.iloc[0]
        print(f"Test 最佳: {best_t['Model']}  RMSLE={best_t['RMSLE']:.6f}  RMSE={best_t['RMSE']:.6f}")

print("\n" + "=" * 90)
print("完成")
print("=" * 90)

