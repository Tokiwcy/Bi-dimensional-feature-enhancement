import os
import json
import hashlib
import warnings
import argparse
import numpy as np
import pandas as pd
from scipy.io import arff

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
print("巴西房屋租金预测模型训练 (6-MODE FINAL: baseline/emb/rag/emb+rag/emb_with_rag/emb_with_rag+rag)")
print("=" * 90)

# =========================
# Args
# =========================
parser = argparse.ArgumentParser(description="训练巴西房屋租金预测模型 - 最终整合版")
parser.add_argument("--pca_dim", type=int, default=16, help="embedding降维维度（用于emb/emb+rag）")
parser.add_argument("--rag_pca_dim", type=int, default=0,
                    help="emb_with_rag分支embedding降维维度；0表示与--pca_dim一致")
parser.add_argument("--seed", type=str, default="42", help="随机种子，支持逗号分隔多个值，如 0,1,2,3,4")
parser.add_argument("--data_ratio", type=float, default=1.0, help="使用数据比例(0,1]")
parser.add_argument("--output_file", type=str, default="brazilian_houses_results.csv", help="结果输出CSV")
parser.add_argument("--encode", type=str, default="onehot", choices=["onehot", "label"], help="类别编码方式")
parser.add_argument("--embed_template", type=str, default="semantic", choices=["structured", "semantic"], help="embedding文本模板")
parser.add_argument("--dim_reducer", type=str, default="svd", choices=["pca", "svd"], help="embedding降维器")
parser.add_argument("--rag_k", type=int, default=12, help="RAG近邻数")
parser.add_argument("--use_rag_text", type=int, default=1, choices=[0, 1], help="是否把RAG证据拼到文本中（对emb_with_rag系列生效）")
parser.add_argument("--mode", type=str, default="all",
                    choices=["all", "baseline", "emb", "rag", "emb+rag", "emb_with_rag", "emb_with_rag+rag"],
                    help="运行模式")
parser.add_argument("--cache_dir", type=str, default="./emb_cache_brazilian", help="embedding缓存目录")
parser.add_argument("--emb_batch_size", type=int, default=100, help="embedding批大小")
args = parser.parse_args()

def _parse_comma_int(value):
    if isinstance(value, int):
        return [value]
    s = str(value).strip()
    return [int(x.strip()) for x in s.split(",") if x.strip()]

PCA_DIM = args.pca_dim
RAG_PCA_DIM = args.rag_pca_dim if args.rag_pca_dim > 0 else args.pca_dim
SEEDS = _parse_comma_int(args.seed)
RANDOM_SEED = SEEDS[0]
DATA_RATIO = args.data_ratio
OUTPUT_FILE = args.output_file
ENCODE_METHOD = args.encode
EMBED_TEMPLATE = args.embed_template
DIM_REDUCER = args.dim_reducer
RAG_K = args.rag_k
USE_RAG_TEXT = bool(args.use_rag_text)
MODE = args.mode
CACHE_DIR = args.cache_dir
EMB_BATCH_SIZE = args.emb_batch_size
TARGET_COL = "total"
CAT_FEATURE_COLS = ["city", "floor", "animal", "furniture"]
NUMERIC_BASE_COLS = ["area", "rooms", "bathroom", "parking_spaces", "hoa", "rent_amount", "property_tax", "fire_insurance", "total", "floor_is_unknown"]
NUM_FEATURE_COLS = [c for c in NUMERIC_BASE_COLS if c != TARGET_COL]
# 固定目标为 total：这些分项与 total 近似确定性关系，作为特征会造成明显泄露
LEAKAGE_DROP_COLS = ["hoa", "rent_amount", "property_tax", "fire_insurance"]

if DATA_RATIO <= 0 or DATA_RATIO > 1.0:
    raise ValueError("--data_ratio 必须在 0.0 到 1.0 之间")

np.random.seed(RANDOM_SEED)

print("\n配置参数:")
print(f"  mode: {MODE}")
print(f"  pca_dim: {PCA_DIM}")
print(f"  rag_pca_dim: {RAG_PCA_DIM} (for emb_with_rag)")
print(f"  seed: {SEEDS if len(SEEDS) > 1 else RANDOM_SEED}")
print(f"  data_ratio: {DATA_RATIO}")
print(f"  encode: {ENCODE_METHOD}")
print(f"  embed_template: {EMBED_TEMPLATE}")
print(f"  dim_reducer: {DIM_REDUCER}")
print(f"  rag_k: {RAG_K}")
print(f"  use_rag_text: {USE_RAG_TEXT}")
print(f"  cache_dir: {CACHE_DIR}")
print(f"  emb_batch_size: {EMB_BATCH_SIZE}")
print(f"  target_col: {TARGET_COL} (log1p)")
print(f"  leakage_drop_cols: {LEAKAGE_DROP_COLS}")

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
    te = reducer.transform(test_emb)
    evr = getattr(reducer, "explained_variance_ratio_", None)
    evr_sum = float(np.sum(evr)) if evr is not None else np.nan
    return tr, te, evr_sum

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
data_path = "brazilian_houses.arff"
try:
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    # 解码字节字符串
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
except Exception as e:
    print(f"使用 arff 读取失败: {e}")
    # 尝试直接读取 CSV（如果文件被转换过）
    df = pd.read_csv(data_path.replace('.arff', '.csv'))
    
print(f"数据文件: {data_path}")
print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# =========================
# 2) Cleaning duplicates/inconsistency
# =========================
print("\n2. 处理不一致值和重复值...")

# 清理字符串列
for col in ['city', 'animal', 'furniture']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# floor 规范化: "-" -> 0，并增加未知标记
if "floor" in df.columns:
    floor_raw = df["floor"]
    floor_str = floor_raw.astype(str).str.strip()
    floor_is_unknown = floor_raw.isna() | floor_str.isin(["-", "", "nan", "None", "null", "NULL"])
    df["floor_is_unknown"] = floor_is_unknown.astype(int)
    floor_str = floor_str.mask(floor_is_unknown, "0")
    df["floor"] = floor_str

original_len = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicates_removed = original_len - len(df)
print(f"  ✓ 删除重复: {duplicates_removed}")

if DATA_RATIO < 1.0:
    print(f"  ✓ 采样数据比例: {DATA_RATIO}")
    sample_size = int(len(df) * DATA_RATIO)
    df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

# 删除会直接泄露目标的信息列
drop_cols_now = [c for c in LEAKAGE_DROP_COLS if c in df.columns and c != TARGET_COL]
if len(drop_cols_now) > 0:
    df = df.drop(columns=drop_cols_now)
    print(f"  ✓ 目标泄露防护: 删除列 {drop_cols_now}")

# =========================
# 3) 70% train / 20% test / 10% val 划分
# =========================
print("\n3. 准备 70/10/20 划分（与 Ames 一致）...")
if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} column not found")

n_total = len(df)
print(f"  总样本数: {n_total}")

# =========================
# 4) Process one fold (function to handle preprocessing and training for one fold)
# =========================
def process_one_fold(df_full, train_sub_idx, es_idx, val_idx, fold_num):
    """
    预处理、特征工程、模型训练和评估（单折）
    评估设计：train_sub 用于所有 fit + 训练，es 仅 early stopping，val 仅最终报告。
    """
    print(f"\n{'='*90}")
    print(f"Split {fold_num + 1} (single 70/10/20 run)")
    print(f"{'='*90}")

    # Split: train_sub (fit + 训练), es (early stopping), test=val (最终报告)
    df_train = df_full.iloc[train_sub_idx].copy().reset_index(drop=True)
    df_es = df_full.iloc[es_idx].copy().reset_index(drop=True)
    df_test = df_full.iloc[val_idx].copy().reset_index(drop=True)
    print(f"  train_sub={len(df_train)}, es={len(df_es)}, test(val)={len(df_test)}")

    # =========================
    # 4.1) Missing value imputation (fit 仅 on train)
    # =========================
    df_filled_train = df_train.copy()
    df_filled_es = df_es.copy()
    df_filled_test = df_test.copy()
    numeric_cols_train = df_filled_train.select_dtypes(include=[np.number]).columns.tolist()

    # 对于巴西房屋数据，预测性填充的列
    predictive_impute_cols = [c for c in NUM_FEATURE_COLS if c in ["area", "rooms", "bathroom", "parking_spaces", "hoa", "rent_amount", "property_tax", "fire_insurance", "total", "floor_is_unknown"]]
    imputation_models = {}
    imputation_stats = {}

    for col in predictive_impute_cols:
        if col in df_filled_train.columns:
            train_missing = df_filled_train[col].isnull().sum()
            es_missing = df_filled_es[col].isnull().sum()
            test_missing = df_filled_test[col].isnull().sum()
            if train_missing > 0 or es_missing > 0 or test_missing > 0:
                feature_cols = [c for c in numeric_cols_train if c != col and c != TARGET_COL]
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
                    train_medians = df_filled_train[feature_cols].median()
                    for _df, _name in [(df_filled_es, "es"), (df_filled_test, "test")]:
                        missing_df = _df[_df[col].isna()]
                        if len(missing_df) > 0:
                            X_miss = missing_df[feature_cols].fillna(train_medians)
                            _df.loc[_df[col].isna(), col] = dt_model.predict(X_miss)
                elif col in imputation_stats and imputation_stats[col]["method"] == "median":
                    fill_val = imputation_stats[col]["value"]
                    df_filled_es[col].fillna(fill_val, inplace=True)
                    df_filled_test[col].fillna(fill_val, inplace=True)

    for col in numeric_cols_train:
        if col not in predictive_impute_cols and col != TARGET_COL:
            if df_filled_train[col].isnull().sum() > 0 or df_filled_es[col].isnull().sum() > 0 or df_filled_test[col].isnull().sum() > 0:
                med = df_filled_train[col].median()
                df_filled_train[col].fillna(med, inplace=True)
                df_filled_es[col].fillna(med, inplace=True)
                df_filled_test[col].fillna(med, inplace=True)

    # 处理类别特征的缺失值
    for col in CAT_FEATURE_COLS:
        if col in df_filled_train.columns:
            if df_filled_train[col].isnull().sum() > 0 or df_filled_es[col].isnull().sum() > 0 or df_filled_test[col].isnull().sum() > 0:
                mode_val = df_filled_train[col].mode()[0] if len(df_filled_train[col].mode()) > 0 else "unknown"
                df_filled_train[col].fillna(mode_val, inplace=True)
                df_filled_es[col].fillna(mode_val, inplace=True)
                df_filled_test[col].fillna(mode_val, inplace=True)

    # =========================
    # 4.2) Prepare text/numeric views (train / es / test)
    # =========================
    text_features_train = {}
    text_features_es = {}
    text_features_test = {}
    text_cols = CAT_FEATURE_COLS
    for c in text_cols:
        if c in df_filled_train.columns:
            text_features_train[c] = df_filled_train[c].values
        if c in df_filled_es.columns:
            text_features_es[c] = df_filled_es[c].values
        if c in df_filled_test.columns:
            text_features_test[c] = df_filled_test[c].values

    num_features_train = {}
    num_features_es = {}
    num_features_test = {}
    num_cols_for_text = NUM_FEATURE_COLS
    for c in num_cols_for_text:
        if c in df_filled_train.columns:
            num_features_train[c] = df_filled_train[c].values
        if c in df_filled_es.columns:
            num_features_es[c] = df_filled_es[c].values
        if c in df_filled_test.columns:
            num_features_test[c] = df_filled_test[c].values

    # =========================
    # 4.3) RAG features
    # =========================
    def _fit_rag_space(train_df, val_df):
        use_num = [c for c in NUM_FEATURE_COLS if c in train_df.columns]
        use_cat = [c for c in CAT_FEATURE_COLS if c in train_df.columns]

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
            train_cat_parts = []
            val_cat_parts = []

            if "city" in use_cat:
                city_freq = train_df["city"].astype(str).value_counts(normalize=True)
                train_city = train_df["city"].astype(str).map(city_freq).fillna(0.0).to_numpy().reshape(-1, 1)
                val_city = val_df["city"].astype(str).map(city_freq).fillna(0.0).to_numpy().reshape(-1, 1)
                train_cat_parts.append(train_city)
                val_cat_parts.append(val_city)

            other_cats = [c for c in use_cat if c != "city"]
            if len(other_cats):
                train_cat = pd.get_dummies(train_df[other_cats].astype(str), prefix=other_cats)
                val_cat = pd.get_dummies(val_df[other_cats].astype(str), prefix=other_cats)
                val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0)
                train_cat_parts.append(train_cat.values)
                val_cat_parts.append(val_cat.values)

            train_cat_arr = np.hstack(train_cat_parts).astype(float) if len(train_cat_parts) else np.zeros((len(train_df), 0))
            val_cat_arr = np.hstack(val_cat_parts).astype(float) if len(val_cat_parts) else np.zeros((len(val_df), 0))
        else:
            train_cat_arr = np.zeros((len(train_df), 0))
            val_cat_arr = np.zeros((len(val_df), 0))

        train_space = np.hstack([train_num_arr, train_cat_arr]).astype(float)
        val_space = np.hstack([val_num_arr, val_cat_arr]).astype(float)
        return train_space, val_space

    def _row_mode(series):
        m = series.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan

    def build_rag_features(train_df, val_df, test_df=None, k=12):
        train_space, val_space = _fit_rag_space(train_df, val_df)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_df)), metric="cosine")
        nn.fit(train_space)

        _, idx_train = nn.kneighbors(train_space)
        _, idx_val = nn.kneighbors(val_space, n_neighbors=min(k, len(train_df)))

        # RAG特征列
        rag_num_cols = [c for c in NUM_FEATURE_COLS if c in train_df.columns]
        rag_cat_cols = [c for c in CAT_FEATURE_COLS if c in train_df.columns]

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
            sub = train_df.iloc[neigh]
            row = {}
            for c in rag_num_cols:
                row[f"rag_mean_{c}"] = float(sub[c].dropna().mean()) if sub[c].notna().sum() else np.nan
            for c in rag_cat_cols:
                row[f"rag_mode_{c}"] = _row_mode(sub[c].astype(str))
            val_rows.append(row)

        rag_train = pd.DataFrame(train_rows, index=train_df.index)
        rag_val = pd.DataFrame(val_rows, index=val_df.index)

        rag_test = None
        if test_df is not None:
            _, test_space = _fit_rag_space(train_df, test_df)
            _, idx_test = nn.kneighbors(test_space, n_neighbors=min(k, len(train_df)))
            test_rows = []
            for i in range(len(test_df)):
                neigh = idx_test[i].tolist()[:k]
                sub = train_df.iloc[neigh]
                row = {}
                for c in rag_num_cols:
                    row[f"rag_mean_{c}"] = float(sub[c].dropna().mean()) if sub[c].notna().sum() else np.nan
                for c in rag_cat_cols:
                    row[f"rag_mode_{c}"] = _row_mode(sub[c].astype(str))
                test_rows.append(row)
            rag_test = pd.DataFrame(test_rows, index=test_df.index)

        return rag_train, rag_val, rag_test

    rag_train_df, rag_es_df, rag_test_df = build_rag_features(
        df_filled_train.drop(columns=[TARGET_COL]),
        df_filled_es.drop(columns=[TARGET_COL]),
        test_df=df_filled_test.drop(columns=[TARGET_COL]),
        k=RAG_K
    )

    # =========================
    # 4.4) Text templates (structured/semantic only)
    # =========================
    def generate_structured_description(i, textf, numf):
        parts = []
        if "city" in textf: parts.append(f"Located in {textf['city'][i]}")
        if "floor" in textf: parts.append(f"floor: {textf['floor'][i]}")
        if "animal" in textf: parts.append(f"animal policy: {textf['animal'][i]}")
        if "furniture" in textf: parts.append(f"furniture: {textf['furniture'][i]}")

        if "area" in numf and not pd.isna(numf["area"][i]): parts.append(f"area: {numf['area'][i]:.0f} sqm")
        if "rooms" in numf and not pd.isna(numf["rooms"][i]): parts.append(f"{int(numf['rooms'][i])} rooms")
        if "bathroom" in numf and not pd.isna(numf["bathroom"][i]): parts.append(f"{int(numf['bathroom'][i])} bathrooms")
        if "parking_spaces" in numf and not pd.isna(numf["parking_spaces"][i]): parts.append(f"{int(numf['parking_spaces'][i])} parking spaces")
        if "hoa" in numf and not pd.isna(numf["hoa"][i]): parts.append(f"HOA: {numf['hoa'][i]:.2f}")
        if "rent_amount" in numf and not pd.isna(numf["rent_amount"][i]): parts.append(f"rent: {numf['rent_amount'][i]:.2f}")
        if "property_tax" in numf and not pd.isna(numf["property_tax"][i]): parts.append(f"property tax: {numf['property_tax'][i]:.2f}")
        if "fire_insurance" in numf and not pd.isna(numf["fire_insurance"][i]): parts.append(f"fire insurance: {numf['fire_insurance'][i]:.2f}")
        return ". ".join(parts) + "."

    def generate_semantic_description(i, textf, numf):
        city = textf.get("city", ["unknown"])[i] if "city" in textf else "unknown"
        floor = textf.get("floor", ["unknown"])[i] if "floor" in textf else "unknown"
        animal = textf.get("animal", ["unknown"])[i] if "animal" in textf else "unknown"
        furniture = textf.get("furniture", ["unknown"])[i] if "furniture" in textf else "unknown"

        area = numf.get("area", [np.nan])[i] if "area" in numf else np.nan
        rooms = numf.get("rooms", [np.nan])[i] if "rooms" in numf else np.nan
        bathroom = numf.get("bathroom", [np.nan])[i] if "bathroom" in numf else np.nan
        parking = numf.get("parking_spaces", [np.nan])[i] if "parking_spaces" in numf else np.nan
        hoa = numf.get("hoa", [np.nan])[i] if "hoa" in numf else np.nan
        rent = numf.get("rent_amount", [np.nan])[i] if "rent_amount" in numf else np.nan
        tax = numf.get("property_tax", [np.nan])[i] if "property_tax" in numf else np.nan
        insurance = numf.get("fire_insurance", [np.nan])[i] if "fire_insurance" in numf else np.nan

        area_bucket = qbucket(area, [(50, "compact"), (100, "medium"), (200, "spacious"), (1e18, "very_spacious")])
        rent_bucket = qbucket(rent, [(2000, "budget"), (5000, "mid_range"), (10000, "premium"), (1e18, "luxury")])
        hoa_bucket = qbucket(hoa, [(500, "low"), (1500, "medium"), (3000, "high"), (1e18, "very_high")])

        cues = []
        if not pd.isna(area) and not pd.isna(rooms) and area > 0 and rooms > 0:
            area_per_room = area / rooms
            if area_per_room > 50:
                cues.append("spacious_per_room")
            elif area_per_room < 25:
                cues.append("compact_per_room")
        if animal == "accept":
            cues.append("pet_friendly")
        if furniture == "furnished":
            cues.append("furnished_property")
        if not pd.isna(parking) and parking >= 2:
            cues.append("multiple_parking")
        if rent_bucket in ["premium", "luxury"]:
            cues.append("premium_rental")
        if hoa_bucket == "low":
            cues.append("low_maintenance_cost")

        # 格式化数值字段
        tax_str = f"{tax:.2f}" if not pd.isna(tax) else "unknown"
        insurance_str = f"{insurance:.2f}" if not pd.isna(insurance) else "unknown"
        
        base = (
            f"rental property profile: city={city}, floor={floor}, "
            f"animal_policy={animal}, furniture={furniture}. "
            f"structure: area={area_bucket}, rooms={int(rooms) if not pd.isna(rooms) else 'unknown'}, "
            f"bathrooms={int(bathroom) if not pd.isna(bathroom) else 'unknown'}, "
            f"parking_spaces={int(parking) if not pd.isna(parking) else 'unknown'}. "
            f"costs: rent_band={rent_bucket}, hoa_band={hoa_bucket}, "
            f"property_tax={tax_str}, "
            f"fire_insurance={insurance_str}. "
            f"signals: {' '.join(cues) if len(cues) else 'neutral_signals'}."
        )
        return base

    def rag_row_to_text(rag_row, k, prefix="rag"):
        parts = [f"{prefix}_evidence_k={k}"]
        num_fields = [
            f"{prefix}_mean_area", f"{prefix}_mean_rooms", f"{prefix}_mean_bathroom",
            f"{prefix}_mean_parking_spaces", f"{prefix}_mean_hoa", f"{prefix}_mean_rent_amount",
            f"{prefix}_mean_property_tax", f"{prefix}_mean_fire_insurance"
        ]
        cat_fields = [
            f"{prefix}_mode_city", f"{prefix}_mode_floor", f"{prefix}_mode_animal", f"{prefix}_mode_furniture"
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
                rag_txt = rag_row_to_text(rag_df.iloc[i].to_dict(), rag_k, prefix="rag")
                base = base + " neighborhood_context: " + rag_txt
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
            txt_plain_es = build_embedding_texts(
                df_filled_es, text_features_es, num_features_es,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )
            txt_plain_test = build_embedding_texts(
                df_filled_test, text_features_test, num_features_test,
                rag_df=None, template=EMBED_TEMPLATE, use_rag_in_text=False, rag_k=RAG_K
            )

            # rag-text texts
            txt_rag_train = build_embedding_texts(
                df_filled_train, text_features_train, num_features_train,
                rag_df=rag_train_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K
            )
            txt_rag_es = build_embedding_texts(
                df_filled_es, text_features_es, num_features_es,
                rag_df=rag_es_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K
            )
            txt_rag_test = build_embedding_texts(
                df_filled_test, text_features_test, num_features_test,
                rag_df=rag_test_df, template=EMBED_TEMPLATE, use_rag_in_text=True, rag_k=RAG_K
            )

            # cache keys (per fold)
            plain_train_key = f"brazilian_plain_train_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_plain_train)}"
            plain_es_key = f"brazilian_plain_es_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_plain_es)}"
            plain_test_key = f"brazilian_plain_test_{EMBED_TEMPLATE}_{DIM_REDUCER}_{PCA_DIM}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_plain_test)}"
            rag_train_key = f"brazilian_ragtxt_train_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_rag_train)}"
            rag_es_key = f"brazilian_ragtxt_es_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_rag_es)}"
            rag_test_key = f"brazilian_ragtxt_test_{EMBED_TEMPLATE}_{DIM_REDUCER}_{RAG_PCA_DIM}_{RAG_K}_{USE_RAG_TEXT}_{DATA_RATIO}_{RANDOM_SEED}_{fold_num}_{text_list_hash(txt_rag_test)}"

            emb_plain_tr_raw = load_cached_embeddings(plain_train_key)
            emb_plain_es_raw = load_cached_embeddings(plain_es_key)
            emb_plain_test_raw = load_cached_embeddings(plain_test_key)
            emb_rag_tr_raw = load_cached_embeddings(rag_train_key)
            emb_rag_es_raw = load_cached_embeddings(rag_es_key)
            emb_rag_test_raw = load_cached_embeddings(rag_test_key)

            if emb_plain_tr_raw is None:
                emb_plain_tr_raw = get_embeddings_batch(txt_plain_train, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(plain_train_key, emb_plain_tr_raw)
            if emb_plain_es_raw is None:
                emb_plain_es_raw = get_embeddings_batch(txt_plain_es, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(plain_es_key, emb_plain_es_raw)
            if emb_plain_test_raw is None:
                emb_plain_test_raw = get_embeddings_batch(txt_plain_test, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(plain_test_key, emb_plain_test_raw)
            if emb_rag_tr_raw is None:
                emb_rag_tr_raw = get_embeddings_batch(txt_rag_train, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(rag_train_key, emb_rag_tr_raw)
            if emb_rag_es_raw is None:
                emb_rag_es_raw = get_embeddings_batch(txt_rag_es, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(rag_es_key, emb_rag_es_raw)
            if emb_rag_test_raw is None:
                emb_rag_test_raw = get_embeddings_batch(txt_rag_test, batch_size=EMB_BATCH_SIZE)
                save_cached_embeddings(rag_test_key, emb_rag_test_raw)

            # plain embedding -> pca_dim (fit on train only)
            embeddings_plain_train, embeddings_plain_es, evr_plain = reduce_embeddings(
                emb_plain_tr_raw, emb_plain_es_raw, PCA_DIM, RANDOM_SEED, DIM_REDUCER
            )
            _, embeddings_plain_test, _ = reduce_embeddings(
                emb_plain_tr_raw, emb_plain_test_raw, PCA_DIM, RANDOM_SEED, DIM_REDUCER
            )

            # rag-text embedding -> rag_pca_dim (fit on train only)
            embeddings_with_ragtext_train, embeddings_with_ragtext_es, evr_ragtxt = reduce_embeddings(
                emb_rag_tr_raw, emb_rag_es_raw, RAG_PCA_DIM, RANDOM_SEED, DIM_REDUCER
            )
            _, embeddings_with_ragtext_test, _ = reduce_embeddings(
                emb_rag_tr_raw, emb_rag_test_raw, RAG_PCA_DIM, RANDOM_SEED, DIM_REDUCER
            )
    else:
        OPENAI_AVAILABLE_FOLD = False

    # =========================
    # 4.6) Box-Cox / skew handling
    # =========================
    numeric_cols_for_skew = [
        c for c in df_filled_train.select_dtypes(include=[np.number]).columns.tolist()
        if c != TARGET_COL
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
                df_filled_es.loc[df_filled_es[col] < lower, col] = lower
                df_filled_es.loc[df_filled_es[col] > upper, col] = upper
                df_filled_test.loc[df_filled_test[col] < lower, col] = lower
                df_filled_test.loc[df_filled_test[col] > upper, col] = upper

    # 对右偏且非负的金额/面积类特征做 log1p，目标列单独处理
    log_cols = [c for c in ["area", "hoa", "property_tax", "fire_insurance", "total", "rent_amount"] if c in df_filled_train.columns and c != TARGET_COL]
    for c in log_cols:
        df_filled_train[c] = np.log1p(df_filled_train[c].clip(lower=0))
        df_filled_es[c] = np.log1p(df_filled_es[c].clip(lower=0))
        df_filled_test[c] = np.log1p(df_filled_test[c].clip(lower=0))

    # =========================
    # 4.7) Tabular preprocess
    # =========================
    categorical_cols = CAT_FEATURE_COLS
    for col in categorical_cols:
        if col in df_filled_train.columns:
            if ENCODE_METHOD == "label":
                le = LabelEncoder()
                train_vals = df_filled_train[col].astype(str)
                df_filled_train[col] = le.fit_transform(train_vals)

                es_vals = df_filled_es[col].astype(str)
                test_vals = df_filled_test[col].astype(str)
                unseen_es = ~es_vals.isin(le.classes_)
                unseen_test = ~test_vals.isin(le.classes_)
                fallback = train_vals.mode()[0] if len(train_vals.mode()) > 0 else le.classes_[0]
                if unseen_es.sum() > 0:
                    es_vals[unseen_es] = fallback
                if unseen_test.sum() > 0:
                    test_vals[unseen_test] = fallback
                df_filled_es[col] = le.transform(es_vals)
                df_filled_test[col] = le.transform(test_vals)
            else:
                dtr = pd.get_dummies(df_filled_train[col], prefix=col)
                dte_es = pd.get_dummies(df_filled_es[col], prefix=col)
                dte_test = pd.get_dummies(df_filled_test[col], prefix=col)
                for d in dtr.columns:
                    if d not in dte_es.columns:
                        dte_es[d] = 0
                    if d not in dte_test.columns:
                        dte_test[d] = 0
                dte_es = dte_es[dtr.columns]
                dte_test = dte_test[dtr.columns]

                df_filled_train = pd.concat([df_filled_train.drop(columns=[col]), dtr], axis=1)
                df_filled_es = pd.concat([df_filled_es.drop(columns=[col]), dte_es], axis=1)
                df_filled_test = pd.concat([df_filled_test.drop(columns=[col]), dte_test], axis=1)

    # drop non-numeric leftovers
    non_num_train = df_filled_train.select_dtypes(exclude=[np.number]).columns.tolist()
    non_num_es = df_filled_es.select_dtypes(exclude=[np.number]).columns.tolist()
    non_num_test = df_filled_test.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_num_train) > 0:
        df_filled_train = df_filled_train.drop(columns=non_num_train)
        df_filled_es = df_filled_es.drop(columns=non_num_es)
        df_filled_test = df_filled_test.drop(columns=non_num_test)

    df_filled_es = df_filled_es.reindex(columns=df_filled_train.columns, fill_value=0)
    df_filled_test = df_filled_test.reindex(columns=df_filled_train.columns, fill_value=0)

    X_train_df = df_filled_train.drop(TARGET_COL, axis=1).copy()
    y_train = np.log1p(pd.to_numeric(df_filled_train[TARGET_COL], errors="coerce").clip(lower=0))
    X_es_df = df_filled_es.drop(TARGET_COL, axis=1).copy()
    y_es = np.log1p(pd.to_numeric(df_filled_es[TARGET_COL], errors="coerce").clip(lower=0))
    X_test_df = df_filled_test.drop(TARGET_COL, axis=1).copy() if TARGET_COL in df_filled_test.columns else df_filled_test.copy()
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)
    y_test = np.log1p(pd.to_numeric(df_filled_test[TARGET_COL], errors="coerce").clip(lower=0))

    # 安全检查：特征矩阵中不应出现目标列
    if TARGET_COL in X_train_df.columns:
        X_train_df = X_train_df.drop(columns=[TARGET_COL])
    if TARGET_COL in X_es_df.columns:
        X_es_df = X_es_df.drop(columns=[TARGET_COL])

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
        if rag_te is not None:
            return rag_tr, rag_val, rag_te
        return rag_tr, rag_val, None

    rag_train_num, rag_es_num, rag_test_num = encode_rag_df(rag_train_df, rag_es_df, rag_test_df)

    # =========================
    # 4.8) Build mode matrices (train / es=early stop / test=final report)
    # =========================
    X_train_base = X_train_df.values
    X_es_base = X_es_df.values
    X_test_base = X_test_df.values

    X_train_rag = np.hstack([X_train_df.values, rag_train_num.values])
    X_es_rag = np.hstack([X_es_df.values, rag_es_num.values])
    X_test_rag = np.hstack([X_test_base, rag_test_num.values])

    X_train_emb = None
    X_es_emb = None
    X_train_emb_plus_rag = None
    X_es_emb_plus_rag = None
    X_train_emb_with_rag = None
    X_es_emb_with_rag = None
    X_train_emb_with_rag_plus_rag = None
    X_es_emb_with_rag_plus_rag = None

    X_test_emb = None
    X_test_emb_plus_rag = None
    X_test_emb_with_rag = None
    X_test_emb_with_rag_plus_rag = None

    if OPENAI_AVAILABLE_FOLD and embeddings_plain_train is not None:
        # 2) emb
        X_train_emb = np.hstack([X_train_df.values, embeddings_plain_train])
        X_es_emb = np.hstack([X_es_df.values, embeddings_plain_es])
        X_test_emb = np.hstack([X_test_base, embeddings_plain_test])

        # 4) emb+rag
        X_train_emb_plus_rag = np.hstack([X_train_df.values, embeddings_plain_train, rag_train_num.values])
        X_es_emb_plus_rag = np.hstack([X_es_df.values, embeddings_plain_es, rag_es_num.values])
        X_test_emb_plus_rag = np.hstack([X_test_base, embeddings_plain_test, rag_test_num.values])

        # 5) emb_with_rag
        X_train_emb_with_rag = np.hstack([X_train_df.values, embeddings_with_ragtext_train])
        X_es_emb_with_rag = np.hstack([X_es_df.values, embeddings_with_ragtext_es])
        X_test_emb_with_rag = np.hstack([X_test_base, embeddings_with_ragtext_test])

        # 6) emb_with_rag+rag
        X_train_emb_with_rag_plus_rag = np.hstack([X_train_df.values, embeddings_with_ragtext_train, rag_train_num.values])
        X_es_emb_with_rag_plus_rag = np.hstack([X_es_df.values, embeddings_with_ragtext_es, rag_es_num.values])
        X_test_emb_with_rag_plus_rag = np.hstack([X_test_base, embeddings_with_ragtext_test, rag_test_num.values])

    # =========================
    # 4.9) Train/Eval
    # =========================
    def evaluate_model_fold(y_true_transformed, y_pred_transformed, model_name, split_label):
        # 统一在 log 空间评估（与训练目标一致），避免价格尺度与 expm1 极端值干扰
        y_true_log = np.asarray(y_true_transformed, dtype=float).ravel()
        pred_log = np.asarray(y_pred_transformed, dtype=float).ravel()
        pred_log = np.clip(pred_log, -20, 20)
        valid = np.isfinite(y_true_log) & np.isfinite(pred_log)
        if not np.all(valid):
            y_true_log = y_true_log[valid]
            pred_log = pred_log[valid]
        if len(y_true_log) == 0:
            print(f"  {model_name} 评估失败: 无有效样本")
            return None

        # Log 空间：与训练目标一致，用于 Log_RMSE 和 R2_Score
        log_rmse = np.sqrt(mean_squared_error(y_true_log, pred_log))
        r2_log = r2_score(y_true_log, pred_log)
        # 价格空间：RMSLE/RMSE/MAE/R2 标准定义
        y_true_orig = np.expm1(y_true_log)
        y_pred_orig = np.maximum(np.expm1(pred_log), 0.0)
        rmsle_price = calculate_rmsle(y_true_orig, y_pred_orig)
        rmse_price = calculate_rmse(y_true_orig, y_pred_orig)
        mae_price = calculate_mae(y_true_orig, y_pred_orig)
        r2_price = r2_score(y_true_orig, y_pred_orig)
        return {
            "Seed": RANDOM_SEED, "Fold": fold_num + 1, "Split": split_label, "Model": model_name,
            "RMSLE": rmsle_price, "Log_RMSE": log_rmse, "R2_Score": r2_log,
            "RMSE": rmse_price, "MAE": mae_price, "R2_price": r2_price,
        }

    fold_results = []

    def train_and_evaluate_fold(model_class, model_name, Xtr, Xes, Xte, suffix="", is_tabpfn=False):
        """训练模型：Xtr 训练，Xes 验证，Xte 测试；同时返回 val/test 两套指标。"""
        try:
            if is_tabpfn:
                # 先尝试使用 GPU，如果失败则回退到 CPU
                try:
                    model = TabPFNRegressor(device='cuda', ignore_pretraining_limits=True)
                    model.fit(Xtr, y_train.values)
                    y_pred_val = model.predict(Xes)
                    y_pred_test = model.predict(Xte)
                    print(f"    TabPFN 使用 GPU 训练成功")
                except Exception as gpu_error:
                    # GPU 失败，回退到 CPU
                    print(f"    TabPFN GPU 训练失败: {str(gpu_error)[:100]}... 回退到 CPU")
                    model = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
                    model.fit(Xtr, y_train.values)
                    y_pred_val = model.predict(Xes)
                    y_pred_test = model.predict(Xte)
                    print(f"    TabPFN 使用 CPU 训练成功")
            elif model_class == xgb.XGBRegressor:
                base_xgb_kwargs = dict(
                    n_estimators=4962, max_depth=8, learning_rate=0.1834132512351,
                    subsample=0.9, colsample_bytree=0.9,
                    reg_alpha=0.0, reg_lambda=1.0,
                    random_state=RANDOM_SEED, n_jobs=-1
                )
                model = model_class(**base_xgb_kwargs)
                try:
                    model.fit(
                        Xtr, y_train,
                        eval_set=[(Xes, y_es)],
                        early_stopping_rounds=60,
                        verbose=False
                    )
                except TypeError as e:
                    if "early_stopping_rounds" not in str(e):
                        raise
                    model = model_class(**base_xgb_kwargs, early_stopping_rounds=60)
                    model.fit(
                        Xtr, y_train,
                        eval_set=[(Xes, y_es)],
                        verbose=False
                    )
                y_pred_val = model.predict(Xes)
                y_pred_test = model.predict(Xte)
            elif model_class == lgb.LGBMRegressor:
                model = model_class(n_estimators=5000, max_depth=-1, num_leaves=127, learning_rate=0.15,
                                    subsample=0.9, colsample_bytree=0.9, min_child_samples=10,
                                    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
                model.fit(
                    Xtr, y_train,
                    eval_set=[(Xes, y_es)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(60, verbose=False)]
                )
                y_pred_val = model.predict(Xes)
                y_pred_test = model.predict(Xte)
            elif model_class == cb.CatBoostRegressor:
                model = model_class(
                    iterations=5000, depth=8, learning_rate=0.15, l2_leaf_reg=1.0,
                    random_seed=RANDOM_SEED, verbose=False,
                    od_type="Iter", od_wait=60
                )
                model.fit(Xtr, y_train, eval_set=(Xes, y_es), use_best_model=True)
                y_pred_val = model.predict(Xes)
                y_pred_test = model.predict(Xte)
            else:
                return None, None

            full_name = f"{model_name} ({suffix})"
            r_val = evaluate_model_fold(y_es, y_pred_val, full_name, "val")
            r_test = evaluate_model_fold(y_test, y_pred_test, full_name, "test")
            return r_val, r_test
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

    # Run models based on mode (es=early stopping only, test=final report)
    if MODE == "baseline":
        run_family_fold(X_train_base, X_es_base, X_test_base, "baseline")
    elif MODE == "rag":
        run_family_fold(X_train_rag, X_es_rag, X_test_rag, "rag")
    elif MODE == "emb":
        if X_train_emb is not None and X_test_emb is not None:
            run_family_fold(X_train_emb, X_es_emb, X_test_emb, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}")
    elif MODE == "emb+rag":
        if X_train_emb_plus_rag is not None and X_test_emb_plus_rag is not None:
            run_family_fold(X_train_emb_plus_rag, X_es_emb_plus_rag, X_test_emb_plus_rag, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}+rag")
    elif MODE == "emb_with_rag":
        if X_train_emb_with_rag is not None and X_test_emb_with_rag is not None:
            run_family_fold(X_train_emb_with_rag, X_es_emb_with_rag, X_test_emb_with_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}")
    elif MODE == "emb_with_rag+rag":
        if X_train_emb_with_rag_plus_rag is not None and X_test_emb_with_rag_plus_rag is not None:
            run_family_fold(X_train_emb_with_rag_plus_rag, X_es_emb_with_rag_plus_rag, X_test_emb_with_rag_plus_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}+rag")
    elif MODE == "all":
        run_family_fold(X_train_base, X_es_base, X_test_base, "baseline")
        run_family_fold(X_train_rag, X_es_rag, X_test_rag, "rag")
        if X_train_emb is not None and X_test_emb is not None:
            run_family_fold(X_train_emb, X_es_emb, X_test_emb, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}")
            run_family_fold(X_train_emb_plus_rag, X_es_emb_plus_rag, X_test_emb_plus_rag, f"emb_{EMBED_TEMPLATE}_d{embeddings_plain_train.shape[1]}+rag")
            run_family_fold(X_train_emb_with_rag, X_es_emb_with_rag, X_test_emb_with_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}")
            run_family_fold(X_train_emb_with_rag_plus_rag, X_es_emb_with_rag_plus_rag, X_test_emb_with_rag_plus_rag, f"emb_with_ragtext_{EMBED_TEMPLATE}_d{embeddings_with_ragtext_train.shape[1]}+rag")
    
    return fold_results

# =========================
# 5) Run single split (Ames style)
# =========================
print("\n" + "=" * 90)
print("开始单次 70/10/20 划分评估")
print("=" * 90)

df_full = df.copy()
all_fold_results = []
for seed in SEEDS:
    RANDOM_SEED = seed
    np.random.seed(seed)
    all_indices = np.arange(len(df_full))
    # train_sub=train(70%), es=val(10%), test=20%
    train_idx, rest_idx = train_test_split(all_indices, test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=2.0 / 3.0, random_state=seed)
    if len(SEEDS) > 1:
        print(f"\n--- seed={seed} ---")
    fold_results = process_one_fold(df_full, train_idx, val_idx, test_idx, 0)
    for r in fold_results:
        all_fold_results.append(r)

# =========================
# 6) Summary (single split)
# =========================
print("\n" + "=" * 90)
print(("多 seed 平均结果汇总" if len(SEEDS) > 1 else "单次 70/10/20 划分结果汇总") + " (按 RMSLE 排序)")
print("=" * 90)

if len(all_fold_results) == 0:
    print("No models were successfully trained.")
else:
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(OUTPUT_FILE.replace(".csv", "_by_seed.csv"), index=False)

    summary = results_df.groupby(["Model", "Split"]).agg({
        "RMSLE": "mean",
        "Log_RMSE": "mean",
        "R2_Score": "mean",
        "RMSE": "mean",
        "MAE": "mean",
        "R2_price": "mean",
    }).reset_index().rename(columns={
        "RMSLE": "RMSLE_mean",
        "Log_RMSE": "Log_RMSE_mean",
        "R2_Score": "R2_mean",
        "RMSE": "RMSE_mean",
        "MAE": "MAE_mean",
        "R2_price": "R2_price_mean",
    })

    summary_val = summary[summary["Split"] == "val"].sort_values("RMSLE_mean")
    summary_test = summary[summary["Split"] == "test"].sort_values("RMSLE_mean")
    summary_out = pd.concat([summary_val, summary_test], ignore_index=True)

    print("\nVal 结果 (按 RMSLE 排序):")
    print(summary_val.to_string(index=False))
    print("\nTest 结果 (按 RMSLE 排序):")
    print(summary_test.to_string(index=False))

    summary_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n按seed明细: {OUTPUT_FILE.replace('.csv', '_by_seed.csv')}")
    print(f"汇总结果: {OUTPUT_FILE}")

    if len(summary_val) > 0:
        best_val = summary_val.iloc[0]
        print(f"\nVal 最佳模型: {best_val['Model']}")
        print(f"  RMSLE:     {best_val['RMSLE_mean']:.6f}")
        print(f"  Log_RMSE:  {best_val['Log_RMSE_mean']:.6f}  (log 空间，与训练一致)")
        print(f"  R² (log):  {best_val['R2_mean']:.6f}")
        print(f"  RMSE(价):  {best_val['RMSE_mean']:.6f}")
        print(f"  R² (价):   {best_val['R2_price_mean']:.6f}")
    if len(summary_test) > 0:
        best_test = summary_test.iloc[0]
        print(f"\nTest 最佳模型: {best_test['Model']}")
        print(f"  RMSLE:     {best_test['RMSLE_mean']:.6f}")
        print(f"  Log_RMSE:  {best_test['Log_RMSE_mean']:.6f}  (log 空间，与训练一致)")
        print(f"  R² (log):  {best_test['R2_mean']:.6f}")
        print(f"  RMSE(价):  {best_test['RMSE_mean']:.6f}")
        print(f"  R² (价):   {best_test['R2_price_mean']:.6f}")

print("\n" + "=" * 90)
print("单次划分评估完成!")
print("=" * 90)

