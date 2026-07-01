"""
src/algorithms.py
=================
Central registry of every ML algorithm Grivora AI supports.

Single source of truth. Every other module (suggester, trainer, UI catalog)
reads from ALGORITHMS below. Each entry is declarative metadata + a builder
lambda that constructs the configured estimator.

Organized by task:
  - regression          (predict a number)
  - classification      (predict a class)
  - time_series         (predict a number indexed by time)
  - clustering          (group similar rows, no target)
  - anomaly             (find weird rows)
  - text                (NLP on unstructured text columns)
  - dim_reduction       (reduce feature count)

Optional-dependency algos register themselves only when the lib is importable.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Always-available (scikit-learn core + scipy + statsmodels)
# ─────────────────────────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor,
    LogisticRegression, RidgeClassifier, SGDClassifier, SGDRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    IsolationForest,
)
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, TruncatedSVD

# ─────────────────────────────────────────────────────────────
# Optional heavy deps
# ─────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Helper: build a metadata record
# ─────────────────────────────────────────────────────────────

def _algo(
    id: str, name: str, family: str, task: str, builder,
    *,
    strengths: list | None = None, weaknesses: list | None = None,
    min_rows: int = 10, max_rows=None,
    handles_missing: bool = False,
    handles_high_cardinality: bool = False,
    interpretable: bool = False,
    speed: str = "medium",          # fast / medium / slow
    default_params: dict | None = None,
    tunable: dict | None = None,
    data_types: list | None = None,
    notes: str = "",
) -> dict:
    return {
        "id": id, "name": name, "family": family, "task": task, "builder": builder,
        "strengths": strengths or [], "weaknesses": weaknesses or [],
        "min_rows": min_rows, "max_rows": max_rows,
        "handles_missing": handles_missing,
        "handles_high_cardinality": handles_high_cardinality,
        "interpretable": interpretable, "speed": speed,
        "default_params": default_params or {},
        "tunable": tunable or {},
        "data_types": data_types or ["tabular"],
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────
# REGRESSION  (predict a number)
# ─────────────────────────────────────────────────────────────
_REGRESSION = [
    _algo("linear_regression", "Linear Regression", "Linear", "regression",
          lambda p: LinearRegression(**p),
          strengths=["very fast", "interpretable", "no tuning needed"],
          weaknesses=["can't model non-linearity", "sensitive to outliers"],
          interpretable=True, speed="fast",
          notes="Great baseline. Read coefficients to see which features matter."),

    _algo("ridge", "Ridge Regression", "Linear", "regression",
          lambda p: Ridge(**p),
          strengths=["handles correlated features", "fast", "stable"],
          weaknesses=["linear only"], interpretable=True, speed="fast",
          default_params={"alpha": 1.0},
          tunable={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),

    _algo("lasso", "Lasso Regression", "Linear", "regression",
          lambda p: Lasso(**p),
          strengths=["performs feature selection", "sparse solutions"],
          weaknesses=["linear only", "can be unstable"],
          interpretable=True, speed="fast",
          default_params={"alpha": 0.1},
          tunable={"alpha": [0.001, 0.01, 0.1, 1.0]}),

    _algo("elasticnet", "ElasticNet", "Linear", "regression",
          lambda p: ElasticNet(**p),
          strengths=["L1+L2 hybrid", "handles correlated features"],
          weaknesses=["two hyperparameters"], interpretable=True, speed="fast",
          default_params={"alpha": 0.1, "l1_ratio": 0.5}),

    _algo("bayesian_ridge", "Bayesian Ridge", "Linear", "regression",
          lambda p: BayesianRidge(**p),
          strengths=["uncertainty estimates", "no alpha tuning"],
          weaknesses=["linear only"], interpretable=True, speed="fast"),

    _algo("huber", "Huber Regressor", "Linear", "regression",
          lambda p: HuberRegressor(**p),
          strengths=["robust to outliers"],
          weaknesses=["linear only", "slower than OLS"],
          interpretable=True, speed="fast"),

    _algo("sgd_regressor", "SGD Regressor", "Linear", "regression",
          lambda p: SGDRegressor(**(p or {"random_state": 42})),
          strengths=["scales to huge data", "online learning"],
          weaknesses=["hyperparameter sensitive"], speed="fast",
          min_rows=1000,
          notes="Stochastic gradient descent linear regressor. Best for >100k rows."),

    _algo("decision_tree_reg", "Decision Tree", "Tree", "regression",
          lambda p: DecisionTreeRegressor(**(p or {"random_state": 42})),
          strengths=["interpretable", "handles mixed data", "no scaling needed"],
          weaknesses=["overfits easily", "high variance"],
          interpretable=True, speed="fast",
          tunable={"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}),

    _algo("random_forest_reg", "Random Forest", "Ensemble", "regression",
          lambda p: RandomForestRegressor(**(p or {"n_estimators": 200, "random_state": 42, "n_jobs": -1})),
          strengths=["robust", "handles non-linearity", "built-in feature importance"],
          weaknesses=["slower to predict", "not sparse-friendly"],
          speed="medium",
          tunable={"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20],
                   "min_samples_split": [2, 5, 10]}),

    _algo("extra_trees_reg", "Extra Trees", "Ensemble", "regression",
          lambda p: ExtraTreesRegressor(**(p or {"n_estimators": 200, "random_state": 42, "n_jobs": -1})),
          strengths=["very fast training", "reduces variance"],
          weaknesses=["slightly worse than RF sometimes"], speed="medium"),

    _algo("gradient_boosting_reg", "Gradient Boosting", "Boosting", "regression",
          lambda p: GradientBoostingRegressor(**(p or {"random_state": 42})),
          strengths=["high accuracy", "tabular champion baseline"],
          weaknesses=["slow to train", "needs tuning"],
          speed="slow",
          tunable={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                   "max_depth": [3, 5, 7]}),

    _algo("hist_gb_reg", "HistGradientBoosting", "Boosting", "regression",
          lambda p: HistGradientBoostingRegressor(**(p or {"random_state": 42})),
          strengths=["fast on large data", "handles missing natively"],
          weaknesses=["newer, less docs"], handles_missing=True, speed="medium",
          min_rows=1000,
          notes="Histogram-based boosting. Faster than GradientBoosting on >10k rows."),

    _algo("adaboost_reg", "AdaBoost", "Boosting", "regression",
          lambda p: AdaBoostRegressor(**(p or {"random_state": 42})),
          strengths=["works well on clean data"],
          weaknesses=["sensitive to noisy data", "slower"], speed="medium"),

    _algo("bagging_reg", "Bagging Regressor", "Ensemble", "regression",
          lambda p: BaggingRegressor(**(p or {"n_estimators": 50, "random_state": 42, "n_jobs": -1})),
          strengths=["reduces variance"], weaknesses=["slower"], speed="medium"),

    _algo("knn_reg", "KNN Regressor", "Instance", "regression",
          lambda p: KNeighborsRegressor(**(p or {"n_neighbors": 5, "n_jobs": -1})),
          strengths=["no training", "interpretable locally"],
          weaknesses=["slow at predict time", "needs scaling", "curse of dimensionality"],
          max_rows=50_000, speed="fast",
          notes="Does not scale past ~50k rows at inference time."),

    _algo("svr_rbf", "SVR (RBF kernel)", "SVM", "regression",
          lambda p: SVR(**(p or {"kernel": "rbf"})),
          strengths=["models non-linearity"],
          weaknesses=["very slow on large data", "hyperparameter sensitive"],
          max_rows=20_000, speed="slow"),

    _algo("svr_linear", "Linear SVR", "SVM", "regression",
          lambda p: LinearSVR(**(p or {"random_state": 42, "max_iter": 5000})),
          strengths=["fast, robust on high-dim"], weaknesses=["linear only"],
          speed="fast"),

    _algo("mlp_reg", "Neural Net (MLP)", "Neural", "regression",
          lambda p: MLPRegressor(**(p or {"hidden_layer_sizes": (64, 32), "max_iter": 300, "random_state": 42})),
          strengths=["captures complex patterns"],
          weaknesses=["needs scaling", "slower", "overfits small data"],
          min_rows=500, speed="slow"),
]

if XGB_AVAILABLE:
    _REGRESSION.append(_algo(
        "xgboost_reg", "XGBoost", "Boosting", "regression",
        lambda p: xgb.XGBRegressor(**(p or {"random_state": 42, "verbosity": 0, "n_jobs": -1})),
        strengths=["state-of-the-art tabular", "handles missing natively", "fast with GPU"],
        weaknesses=["many hyperparameters"], handles_missing=True, speed="medium",
        tunable={"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
                 "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]}))
if LGB_AVAILABLE:
    _REGRESSION.append(_algo(
        "lightgbm_reg", "LightGBM", "Boosting", "regression",
        lambda p: lgb.LGBMRegressor(**(p or {"random_state": 42, "verbose": -1, "n_jobs": -1})),
        strengths=["very fast", "low memory", "handles categorical natively"],
        weaknesses=["can overfit small data"], handles_missing=True,
        handles_high_cardinality=True, speed="fast"))
if CAT_AVAILABLE:
    _REGRESSION.append(_algo(
        "catboost_reg", "CatBoost", "Boosting", "regression",
        lambda p: cb.CatBoostRegressor(**(p or {"random_seed": 42, "verbose": 0, "iterations": 300})),
        strengths=["best-in-class for categorical", "minimal tuning"],
        weaknesses=["slower than LightGBM"], handles_missing=True,
        handles_high_cardinality=True, speed="medium"))


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────
_CLASSIFICATION = [
    _algo("logistic_regression", "Logistic Regression", "Linear", "classification",
          lambda p: LogisticRegression(**(p or {"max_iter": 1000, "random_state": 42, "n_jobs": -1})),
          strengths=["interpretable", "probability outputs", "fast"],
          weaknesses=["linear decision boundary only"],
          interpretable=True, speed="fast",
          tunable={"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}),

    _algo("ridge_classifier", "Ridge Classifier", "Linear", "classification",
          lambda p: RidgeClassifier(**(p or {"random_state": 42})),
          strengths=["fast", "regularized"],
          weaknesses=["no probability outputs"], interpretable=True, speed="fast"),

    _algo("sgd_classifier", "SGD Classifier", "Linear", "classification",
          lambda p: SGDClassifier(**(p or {"random_state": 42, "loss": "log_loss"})),
          strengths=["scales to huge data"], weaknesses=["hyperparameter sensitive"],
          min_rows=1000, speed="fast"),

    _algo("knn_clf", "KNN Classifier", "Instance", "classification",
          lambda p: KNeighborsClassifier(**(p or {"n_neighbors": 5, "n_jobs": -1})),
          strengths=["no training", "intuitive"],
          weaknesses=["slow at predict", "needs scaling"],
          max_rows=50_000, speed="fast",
          tunable={"n_neighbors": [3, 5, 7, 9, 11]}),

    _algo("gaussian_nb", "Gaussian Naive Bayes", "Probabilistic", "classification",
          lambda p: GaussianNB(**(p or {})),
          strengths=["extremely fast", "works on small data"],
          weaknesses=["assumes feature independence"], speed="fast"),

    _algo("multinomial_nb", "Multinomial Naive Bayes", "Probabilistic", "classification",
          lambda p: MultinomialNB(**(p or {})),
          strengths=["great for text / count features"],
          weaknesses=["negative features not allowed"], speed="fast",
          data_types=["text", "tabular"],
          notes="Requires non-negative features; ideal for TF-IDF and count data."),

    _algo("bernoulli_nb", "Bernoulli Naive Bayes", "Probabilistic", "classification",
          lambda p: BernoulliNB(**(p or {})),
          strengths=["binary/boolean features"], speed="fast"),

    _algo("linear_svc", "Linear SVC", "SVM", "classification",
          lambda p: LinearSVC(**(p or {"random_state": 42, "max_iter": 5000})),
          strengths=["fast on high-dim data"], weaknesses=["no probabilities by default"],
          speed="fast"),

    _algo("svc_rbf", "SVM (RBF kernel)", "SVM", "classification",
          lambda p: SVC(**(p or {"kernel": "rbf", "probability": True, "random_state": 42})),
          strengths=["captures non-linearity", "strong on small data"],
          weaknesses=["very slow on big data"],
          max_rows=20_000, speed="slow",
          tunable={"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}),

    _algo("decision_tree_clf", "Decision Tree", "Tree", "classification",
          lambda p: DecisionTreeClassifier(**(p or {"random_state": 42})),
          strengths=["interpretable", "no scaling"], weaknesses=["overfits"],
          interpretable=True, speed="fast"),

    _algo("random_forest_clf", "Random Forest", "Ensemble", "classification",
          lambda p: RandomForestClassifier(**(p or {"n_estimators": 200, "random_state": 42, "n_jobs": -1})),
          strengths=["robust", "handles imbalance well", "feature importance"],
          weaknesses=["slower inference"], speed="medium",
          tunable={"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20]}),

    _algo("extra_trees_clf", "Extra Trees", "Ensemble", "classification",
          lambda p: ExtraTreesClassifier(**(p or {"n_estimators": 200, "random_state": 42, "n_jobs": -1})),
          strengths=["fast training", "reduces variance"], speed="medium"),

    _algo("gradient_boosting_clf", "Gradient Boosting", "Boosting", "classification",
          lambda p: GradientBoostingClassifier(**(p or {"random_state": 42})),
          strengths=["high accuracy"], weaknesses=["slow to train"], speed="slow"),

    _algo("hist_gb_clf", "HistGradientBoosting", "Boosting", "classification",
          lambda p: HistGradientBoostingClassifier(**(p or {"random_state": 42})),
          strengths=["fast on large data", "handles missing natively"],
          handles_missing=True, min_rows=1000, speed="medium"),

    _algo("adaboost_clf", "AdaBoost", "Boosting", "classification",
          lambda p: AdaBoostClassifier(**(p or {"random_state": 42})),
          strengths=["simple, effective"], weaknesses=["sensitive to noise"],
          speed="medium"),

    _algo("bagging_clf", "Bagging Classifier", "Ensemble", "classification",
          lambda p: BaggingClassifier(**(p or {"n_estimators": 50, "random_state": 42, "n_jobs": -1})),
          strengths=["reduces variance"], speed="medium"),

    _algo("lda", "Linear Discriminant Analysis", "Discriminant", "classification",
          lambda p: LinearDiscriminantAnalysis(**(p or {})),
          strengths=["fast", "works with small samples"],
          weaknesses=["assumes normal distribution"], speed="fast"),

    _algo("qda", "Quadratic Discriminant Analysis", "Discriminant", "classification",
          lambda p: QuadraticDiscriminantAnalysis(**(p or {})),
          strengths=["handles non-linear boundaries"],
          weaknesses=["assumes normality per class"], speed="fast"),

    _algo("mlp_clf", "Neural Net (MLP)", "Neural", "classification",
          lambda p: MLPClassifier(**(p or {"hidden_layer_sizes": (64, 32), "max_iter": 300, "random_state": 42})),
          strengths=["captures complex patterns"], weaknesses=["needs scaling", "slower"],
          min_rows=500, speed="slow"),
]

if XGB_AVAILABLE:
    _CLASSIFICATION.append(_algo(
        "xgboost_clf", "XGBoost", "Boosting", "classification",
        lambda p: xgb.XGBClassifier(**(p or {"random_state": 42, "verbosity": 0, "n_jobs": -1,
                                             "eval_metric": "logloss",
                                             "use_label_encoder": False})),
        strengths=["tabular champion", "handles missing natively"],
        weaknesses=["many hyperparameters"], handles_missing=True, speed="medium"))
if LGB_AVAILABLE:
    _CLASSIFICATION.append(_algo(
        "lightgbm_clf", "LightGBM", "Boosting", "classification",
        lambda p: lgb.LGBMClassifier(**(p or {"random_state": 42, "verbose": -1, "n_jobs": -1})),
        strengths=["very fast", "handles categorical natively", "handles missing"],
        handles_missing=True, handles_high_cardinality=True, speed="fast"))
if CAT_AVAILABLE:
    _CLASSIFICATION.append(_algo(
        "catboost_clf", "CatBoost", "Boosting", "classification",
        lambda p: cb.CatBoostClassifier(**(p or {"random_seed": 42, "verbose": 0, "iterations": 300})),
        strengths=["best for categorical-heavy", "minimal tuning"],
        handles_missing=True, handles_high_cardinality=True, speed="medium"))


# ─────────────────────────────────────────────────────────────
# TIME SERIES
# ─────────────────────────────────────────────────────────────
_TIME_SERIES = [
    _algo("naive_last", "Naive (last value)", "Baseline", "time_series",
          lambda p: None,  # handled by custom trainer
          strengths=["trivial baseline", "hard to beat sometimes"],
          weaknesses=["no seasonality"], speed="fast", interpretable=True,
          data_types=["time_series"],
          notes="Forecasts the last observed value forever. Critical baseline."),

    _algo("seasonal_naive", "Seasonal Naive", "Baseline", "time_series",
          lambda p: None,
          strengths=["captures seasonality cheaply"],
          weaknesses=["no trend"], speed="fast", interpretable=True,
          data_types=["time_series"]),

    _algo("ml_lag_rf", "Random Forest on Lags", "ML", "time_series",
          lambda p: RandomForestRegressor(**(p or {"n_estimators": 200, "random_state": 42, "n_jobs": -1})),
          strengths=["captures non-linearity", "no statistical assumptions"],
          weaknesses=["needs many lag features"], speed="medium",
          data_types=["time_series"]),
]

if XGB_AVAILABLE:
    _TIME_SERIES.append(_algo(
        "ml_lag_xgb", "XGBoost on Lags", "ML", "time_series",
        lambda p: xgb.XGBRegressor(**(p or {"random_state": 42, "verbosity": 0, "n_jobs": -1})),
        strengths=["excellent for tabular TS", "handles lag features"],
        speed="medium", data_types=["time_series"]))
if LGB_AVAILABLE:
    _TIME_SERIES.append(_algo(
        "ml_lag_lgb", "LightGBM on Lags", "ML", "time_series",
        lambda p: lgb.LGBMRegressor(**(p or {"random_state": 42, "verbose": -1, "n_jobs": -1})),
        strengths=["fast", "handles categorical TS features"], speed="fast",
        data_types=["time_series"]))
if STATSMODELS_AVAILABLE:
    _TIME_SERIES.append(_algo(
        "arima", "ARIMA", "Statistical", "time_series",
        lambda p: None,  # statsmodels models built inline by trainer
        strengths=["well-studied", "interpretable parameters"],
        weaknesses=["requires stationarity", "univariate"],
        speed="medium", interpretable=True, data_types=["time_series"],
        default_params={"order": (5, 1, 0)}))
    _TIME_SERIES.append(_algo(
        "sarima", "SARIMA", "Statistical", "time_series",
        lambda p: None,
        strengths=["captures seasonality + trend"],
        weaknesses=["slow to fit", "many parameters"],
        speed="slow", interpretable=True, data_types=["time_series"],
        default_params={"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)}))
    _TIME_SERIES.append(_algo(
        "holt_winters", "Holt-Winters", "Statistical", "time_series",
        lambda p: None,
        strengths=["exponential smoothing", "simple"],
        weaknesses=["univariate only"], speed="fast", interpretable=True,
        data_types=["time_series"]))
if PROPHET_AVAILABLE:
    _TIME_SERIES.append(_algo(
        "prophet", "Prophet", "Statistical", "time_series",
        lambda p: None,
        strengths=["handles holidays", "intuitive", "robust to missing"],
        weaknesses=["can over-smooth", "slower"],
        handles_missing=True, speed="slow", data_types=["time_series"]))


# ─────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────
_CLUSTERING = [
    _algo("kmeans", "K-Means", "Partition", "clustering",
          lambda p: KMeans(**(p or {"n_clusters": 4, "random_state": 42, "n_init": 10})),
          strengths=["fast", "scales well"],
          weaknesses=["assumes spherical clusters", "need to pick K"],
          speed="fast",
          tunable={"n_clusters": [2, 3, 4, 5, 6, 8, 10]}),

    _algo("minibatch_kmeans", "MiniBatch KMeans", "Partition", "clustering",
          lambda p: MiniBatchKMeans(**(p or {"n_clusters": 4, "random_state": 42, "n_init": 10})),
          strengths=["very fast", "scales to huge data"],
          weaknesses=["slightly noisier"], min_rows=5000, speed="fast"),

    _algo("dbscan", "DBSCAN", "Density", "clustering",
          lambda p: DBSCAN(**(p or {"eps": 0.5, "min_samples": 5, "n_jobs": -1})),
          strengths=["finds arbitrary shapes", "no K needed", "detects noise"],
          weaknesses=["sensitive to eps"], max_rows=50_000, speed="medium"),

    _algo("agglomerative", "Agglomerative Clustering", "Hierarchical", "clustering",
          lambda p: AgglomerativeClustering(**(p or {"n_clusters": 4})),
          strengths=["hierarchical structure", "dendrogram"],
          weaknesses=["slow on large data"], max_rows=10_000, speed="slow"),

    _algo("spectral", "Spectral Clustering", "Graph", "clustering",
          lambda p: SpectralClustering(**(p or {"n_clusters": 4, "random_state": 42})),
          strengths=["handles non-convex clusters"],
          weaknesses=["very slow"], max_rows=5000, speed="slow"),

    _algo("gmm", "Gaussian Mixture", "Probabilistic", "clustering",
          lambda p: GaussianMixture(**(p or {"n_components": 4, "random_state": 42})),
          strengths=["soft assignments", "probabilistic"],
          weaknesses=["assumes Gaussian"], speed="medium"),
]
if HDBSCAN_AVAILABLE:
    _CLUSTERING.append(_algo(
        "hdbscan", "HDBSCAN", "Density", "clustering",
        lambda p: hdbscan.HDBSCAN(**(p or {"min_cluster_size": 15})),
        strengths=["no K needed", "varied densities", "robust noise handling"],
        speed="medium"))


# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────
_ANOMALY = [
    _algo("isolation_forest", "Isolation Forest", "Ensemble", "anomaly",
          lambda p: IsolationForest(**(p or {"contamination": "auto", "random_state": 42, "n_jobs": -1})),
          strengths=["fast", "scales well", "no distribution assumptions"],
          speed="fast"),

    _algo("local_outlier_factor", "Local Outlier Factor", "Density", "anomaly",
          lambda p: LocalOutlierFactor(**(p or {"novelty": False, "n_jobs": -1})),
          strengths=["finds local anomalies"],
          weaknesses=["slow on large data"], max_rows=50_000, speed="medium"),

    _algo("one_class_svm", "One-Class SVM", "SVM", "anomaly",
          lambda p: OneClassSVM(**(p or {"nu": 0.05})),
          strengths=["good for high-dim data"],
          weaknesses=["very slow"], max_rows=10_000, speed="slow"),

    _algo("elliptic_envelope", "Elliptic Envelope", "Statistical", "anomaly",
          lambda p: EllipticEnvelope(**(p or {"contamination": 0.05, "random_state": 42})),
          strengths=["fast on Gaussian data"],
          weaknesses=["assumes Gaussian"], speed="fast"),
]


# ─────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────
_DIM_REDUCTION = [
    _algo("pca", "PCA", "Linear", "dim_reduction",
          lambda p: PCA(**(p or {"n_components": 2, "random_state": 42})),
          strengths=["fast", "preserves variance"],
          weaknesses=["linear only"], interpretable=True, speed="fast"),

    _algo("truncated_svd", "Truncated SVD", "Linear", "dim_reduction",
          lambda p: TruncatedSVD(**(p or {"n_components": 2, "random_state": 42})),
          strengths=["works on sparse data", "good for TF-IDF"], speed="fast",
          data_types=["tabular", "text"]),
]


# ─────────────────────────────────────────────────────────────
# MASTER REGISTRY
# ─────────────────────────────────────────────────────────────

ALL_ALGORITHMS = {a["id"]: a for a in (
    _REGRESSION + _CLASSIFICATION + _TIME_SERIES +
    _CLUSTERING + _ANOMALY + _DIM_REDUCTION
)}


def algorithms_for_task(task: str) -> list:
    """Return all algos for a given task (regression, classification, etc.)."""
    return [a for a in ALL_ALGORITHMS.values() if a["task"] == task]


def algorithm(algo_id: str) -> dict | None:
    return ALL_ALGORITHMS.get(algo_id)


def catalog_for_llm(task: str) -> list:
    """Compact descriptor list for LLM planners. Drops builder callables."""
    out = []
    for a in algorithms_for_task(task):
        out.append({
            "id": a["id"],
            "name": a["name"],
            "family": a["family"],
            "strengths": a["strengths"],
            "weaknesses": a["weaknesses"],
            "min_rows": a["min_rows"],
            "max_rows": a["max_rows"],
            "handles_missing": a["handles_missing"],
            "handles_high_cardinality": a["handles_high_cardinality"],
            "interpretable": a["interpretable"],
            "speed": a["speed"],
            "notes": a.get("notes", ""),
        })
    return out


def filter_by_data(algos: list, n_rows: int, n_cols: int) -> list:
    """Remove algos that can't handle this dataset size."""
    result = []
    for a in algos:
        if n_rows < a["min_rows"]:
            continue
        if a["max_rows"] is not None and n_rows > a["max_rows"]:
            continue
        result.append(a)
    return result


# Human-readable grouping for the UI's manual catalog tab
TASK_LABELS = {
    "regression":     "Regression (predict a number)",
    "classification": "Classification (predict a category)",
    "time_series":    "Time Series Forecasting",
    "clustering":     "Clustering (group similar rows)",
    "anomaly":        "Anomaly Detection",
    "dim_reduction":  "Dimensionality Reduction",
    "text":           "Text / NLP",
}


def available_deps() -> dict:
    """What optional libs are actually importable right now."""
    return {
        "xgboost": XGB_AVAILABLE,
        "lightgbm": LGB_AVAILABLE,
        "catboost": CAT_AVAILABLE,
        "statsmodels": STATSMODELS_AVAILABLE,
        "prophet": PROPHET_AVAILABLE,
        "hdbscan": HDBSCAN_AVAILABLE,
    }
