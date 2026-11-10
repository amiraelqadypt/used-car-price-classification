"""Central configuration for the hybrid clustering-classification pipeline."""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_CSV = DATA_DIR / "car_prices.csv"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
REFERENCE_YEAR = 2024          # used to derive car_age
OUTLIER_LOWER_Q = 0.01         # 1st percentile of sellingprice
OUTLIER_UPPER_Q = 0.99         # 99th percentile of sellingprice
N_PRICE_CLASSES = 4
PRICE_CLASS_LABELS = ["Budget", "Mid-Range", "Premium", "Luxury"]

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["make", "body", "transmission", "state", "color", "interior"]

# Vehicle attributes that are known before a sale takes place.
BASE_NUMERIC_FEATURES = ["year", "condition", "odometer", "mmr", "car_age"]

# Features derived from sellingprice. These are classifier inputs only: they
# are never supplied to K-Means, so cluster assignments carry no information
# about the target. Their availability depends on the deployment context,
# since they require the transaction price to be known.
PRICE_DERIVED_FEATURES = ["price_ratio"]

# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE_SIZE = 50_000
KMEANS_SEARCH_N_INIT = 10
KMEANS_FINAL_N_INIT = 20
KMEANS_SEARCH_MAX_ITER = 300
KMEANS_FINAL_MAX_ITER = 500

# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
TEST_SIZE = 0.2

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="mlogloss",
)

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# LinearSVC is the default and reproduces the accuracy reported in the thesis
# (0.8372) exactly. An RBF-kernel SVC is O(N^2)-O(N^3) on ~438,000 training
# rows, as Section 3.14 notes, and is impractical at this scale.
SVM_PARAMS = dict(
    random_state=RANDOM_STATE,
    max_iter=2000,
    dual=False,
)

# Soft-voting ensemble over the two probabilistic models.
ENSEMBLE_MEMBERS = ["XGBoost", "Random Forest"]

# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------
MODES = {
    # The configuration described in the thesis: price-derived features are
    # classifier inputs, K-Means sees vehicle attributes only, and the scaler
    # and cluster centroids are computed on the full dataset.
    "thesis": dict(
        include_price_derived=True,
        cluster_uses_price_derived=False,
        fit_on_train_only=False,
    ),
    # Deployment-realistic: no features derived from the selling price
    # anywhere, and the scaler and K-Means model are fitted on the training
    # partition only.
    "strict": dict(
        include_price_derived=False,
        cluster_uses_price_derived=False,
        fit_on_train_only=True,
    ),
}
