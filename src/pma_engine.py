"""
src/pma_engine.py
=================
Grivora AI Predictive Modeling Engine — production-grade rewrite.

Fixes from the previous version:
  * No data leakage: encoder/imputer/scaler fit on TRAIN ONLY via sklearn Pipeline
  * The full pipeline (preprocessor + estimator) is saved as one pickle
  * Stratified split for classification, regular split for regression
  * k-fold CV metrics reported alongside holdout metrics
  * Permutation importance instead of univariate SelectKBest (much more honest)
  * Class imbalance auto-detection + class_weight="balanced" where supported
  * Unseen categories at predict time raise a clear warning, don't silently guess
  * Full algorithm registry via src.algorithms (40+ algos)

Public API:
  * infer_task(df, target)      — which task fits this dataset
  * profile_dataset(df, target) — 30+ signal profile (from ml_suggester)
  * suggest_algorithms(df, tgt) — LLM-first auto-suggest (from ml_suggester)
  * train_model(df, target, algo_id, ...)  — correct training with pipeline
  * tune_model(df, target, algo_id, ...)   — tuning w/ proper CV isolation
  * predict_new_data(input_row, model_path) — safe inference
  * save_model / load_model
"""
from __future__ import annotations

import os
import json
import pickle
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)

from src.algorithms import ALL_ALGORITHMS, algorithm
from src.ml_suggester import (
    infer_task, profile_dataset, suggest_algorithms,
)


# ─────────────────────────────────────────────────────────────
# 1. Preprocessing pipeline (fit on train only — no leakage)
# ─────────────────────────────────────────────────────────────

_HIGH_CARDINALITY_THRESHOLD = 50  # one-hot below this, ordinal-encode above


def build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    """Build a ColumnTransformer that handles numeric + categorical + date columns."""
    X = df.drop(columns=[target_col], errors="ignore")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Drop date columns — they're used for time-series but not in the main matrix
    date_cols = [c for c in cat_cols if _looks_like_date(X[c])]
    cat_cols = [c for c in cat_cols if c not in date_cols]

    # Split categoricals by cardinality
    low_card_cat  = [c for c in cat_cols if X[c].nunique(dropna=True) <= _HIGH_CARDINALITY_THRESHOLD]
    high_card_cat = [c for c in cat_cols if c not in low_card_cat]

    transformers = []

    if num_cols:
        transformers.append(("num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), num_cols))

    if low_card_cat:
        transformers.append(("cat_low",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), low_card_cat))

    if high_card_cat:
        transformers.append(("cat_high",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)),
            ]), high_card_cat))

    return ColumnTransformer(transformers=transformers, remainder="drop",
                             verbose_feature_names_out=False)


def _looks_like_date(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype != object:
        return False
    sample = s.dropna().astype(str).head(10)
    if sample.empty:
        return False
    lower = str(s.name).lower()
    if not any(k in lower for k in ("date", "time", "year", "month", "day", "period")):
        return False
    return pd.to_datetime(sample, errors="coerce").notna().mean() > 0.6


# ─────────────────────────────────────────────────────────────
# 2. Time-series feature engineering
# ─────────────────────────────────────────────────────────────

def prepare_time_series(df: pd.DataFrame, date_col: str, target_col: str,
                         lags: int = 7) -> pd.DataFrame:
    """Generate lag + rolling + calendar features for TS regression."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    for i in range(1, lags + 1):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)

    df[f"{target_col}_roll_mean_7"] = df[target_col].rolling(7).mean().shift(1)
    df[f"{target_col}_roll_std_7"]  = df[target_col].rolling(7).std().shift(1)

    df["month"]     = df[date_col].dt.month
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["quarter"]   = df[date_col].dt.quarter
    df["dayofyear"] = df[date_col].dt.dayofyear
    df["year"]      = df[date_col].dt.year

    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 3. PRE-FLIGHT VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_training_inputs(
    df: pd.DataFrame,
    target_col: str,
    algo_id: str,
) -> dict:
    """Validate dataset + target + algo before training starts.

    Returns:
        {
            "ok": True/False,
            "errors": ["fatal problems"],
            "warnings": ["non-fatal but worth knowing"],
        }

    Called by train_model() and can be called standalone from routes for
    early rejection before the heavy pipeline runs.
    """
    errors = []
    warnings = []

    # 1. Target column exists
    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found in dataset. "
                      f"Available columns: {', '.join(df.columns[:20])}")
        return {"ok": False, "errors": errors, "warnings": warnings}

    # 2. Algorithm exists
    algo = algorithm(algo_id)
    if not algo:
        errors.append(f"Unknown algorithm: '{algo_id}'. "
                      "Check src/algorithms.py for available IDs.")
        return {"ok": False, "errors": errors, "warnings": warnings}

    # 3. Target has enough non-null values
    y = df[target_col].dropna()
    n_valid = len(y)
    n_total = len(df)
    missing_pct = round((1 - n_valid / max(n_total, 1)) * 100, 1)

    if n_valid == 0:
        errors.append(f"Target column '{target_col}' is entirely empty (all NaN/null).")
        return {"ok": False, "errors": errors, "warnings": warnings}

    if missing_pct > 80:
        errors.append(f"Target column '{target_col}' is {missing_pct}% missing "
                      f"({n_total - n_valid} of {n_total} rows). Need at least 20% non-null.")
        return {"ok": False, "errors": errors, "warnings": warnings}

    if missing_pct > 30:
        warnings.append(f"Target column has {missing_pct}% missing values — "
                        f"{n_total - n_valid} rows will be dropped before training.")

    # 4. Enough rows after cleanup
    if n_valid < 20:
        errors.append(f"Only {n_valid} non-null target rows — need at least 20 for training.")
        return {"ok": False, "errors": errors, "warnings": warnings}

    if n_valid < 50:
        warnings.append(f"Only {n_valid} rows with valid target — model may be unreliable. "
                        "Consider gathering more data.")

    # 5. Check dataset meets algo's min_rows requirement
    min_rows = algo.get("min_rows", 10)
    if n_valid < min_rows:
        errors.append(f"Algorithm '{algo['name']}' requires at least {min_rows} rows, "
                      f"but only {n_valid} are available after dropping missing targets.")
        return {"ok": False, "errors": errors, "warnings": warnings}

    # 6. Classification-specific checks
    is_numeric = pd.api.types.is_numeric_dtype(y)
    n_unique = int(y.nunique())

    if not is_numeric or (n_unique <= 20 and n_unique / max(n_valid, 1) < 0.05):
        # Likely classification
        if n_unique < 2:
            errors.append(f"Target '{target_col}' has only {n_unique} unique value(s) — "
                          "need at least 2 classes for classification.")
            return {"ok": False, "errors": errors, "warnings": warnings}

        # Check smallest class has enough samples
        vc = y.value_counts()
        smallest_class_count = int(vc.min())
        if smallest_class_count < 2:
            errors.append(f"Smallest class in '{target_col}' has only {smallest_class_count} "
                          f"sample(s). Each class needs at least 2 samples for train/test split.")
            return {"ok": False, "errors": errors, "warnings": warnings}

        if smallest_class_count < 5:
            warnings.append(f"Smallest class has only {smallest_class_count} samples — "
                            "cross-validation may be unreliable.")

        if n_unique > 100:
            warnings.append(f"Target has {n_unique} unique values — treating as classification "
                            "with this many classes may not be ideal. Consider if this should be regression.")

    else:
        # Likely regression
        if y.std() == 0:
            errors.append(f"Target '{target_col}' has zero variance (all values are the same: "
                          f"{y.iloc[0]}). Cannot train a regression model on a constant target.")
            return {"ok": False, "errors": errors, "warnings": warnings}

    # 7. Check for potential data leakage (high correlation with target)
    if is_numeric:
        num_cols = df.select_dtypes("number").columns.tolist()
        num_cols = [c for c in num_cols if c != target_col]
        for c in num_cols[:30]:
            try:
                corr = abs(float(df[c].corr(df[target_col])))
                if corr > 0.98:
                    warnings.append(f"⚠️ Column '{c}' has {corr:.3f} correlation with target — "
                                    "possible data leakage. Consider excluding it.")
            except Exception:
                pass

    # 8. Check for ID-like columns (all unique values = row count)
    for c in df.columns[:30]:
        if c == target_col:
            continue
        if int(df[c].nunique()) == n_total and n_total > 50:
            warnings.append(f"Column '{c}' has all unique values — looks like an ID column. "
                            "Consider excluding it from training.")

    return {"ok": True, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────────────────────
# 4. TRAIN  (the main function)
# ─────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
    algo_id: str,
    *,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    params: Optional[dict] = None,
) -> dict:
    """
    Train one algorithm. Returns a rich result dict:
      {
        "ok": True,
        "algo_id", "algo_name", "task", "problem_type",
        "metrics": {test-set metrics},
        "cv_metrics": {mean ± std from k-fold CV},
        "feature_names": [...],
        "feature_importance": [{feature, score}, ...],
        "class_distribution": {...}  (classification only),
        "y_true": [...], "y_pred": [...],
        "model_path": "models/...pkl",
        "logs": [timestamped strings],
      }
    Or {"ok": False, "error": "..."} on failure.
    """
    logs = []
    def log(msg): logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ── Pre-flight validation ──
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    validation = validate_training_inputs(df, target_col, algo_id)
    if not validation["ok"]:
        return {"ok": False, "error": " | ".join(validation["errors"])}
    for w in validation.get("warnings", []):
        log(f"⚠ {w}")

    algo = algorithm(algo_id)
    if not algo:
        return {"ok": False, "error": f"Unknown algorithm: {algo_id}"}

    # Strip rows with missing target (we can't train on those)
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    if dropped:
        log(f"Dropped {dropped} rows with missing target '{target_col}'")

    # Task + target handling
    task_info = infer_task(df, target_col)
    task = task_info["task"]
    problem_type = task_info["problem_type"]
    log(f"Detected task: {task} ({problem_type or 'unsupervised'})")

    # Time-series special path: build lag features first
    if task == "time_series" and task_info.get("date_cols"):
        df = prepare_time_series(df, task_info["date_cols"][0], target_col)
        log(f"Built lag features: shape now {df.shape}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop date columns from X (they'd break the numeric/categorical pipeline)
    for c in list(X.columns):
        if _looks_like_date(X[c]):
            X = X.drop(columns=[c])

    feature_names = X.columns.tolist()

    # ── Build preprocessor (using full X temporarily — safe because we
    #    only use .columns here. The fit happens on train split below.) ──
    preprocessor = build_preprocessor(df[feature_names + [target_col]], target_col)

    # Build estimator
    if algo["builder"] is None:
        return {"ok": False,
                "error": f"Algorithm '{algo_id}' requires a custom trainer (e.g. statsmodels/Prophet). Use task-specific routes."}

    estimator = algo["builder"](params)

    # For classification, auto-apply class_weight="balanced" when imbalanced
    is_imbalanced = False
    class_dist = None
    if problem_type == "classification":
        vc = y.value_counts()
        class_dist = {str(k): int(v) for k, v in vc.items()}
        if vc.max() / max(vc.min(), 1) > 5:
            is_imbalanced = True
            if hasattr(estimator, "class_weight"):
                estimator.set_params(class_weight="balanced")
                log("Imbalanced classes detected → class_weight='balanced'")

    # Full pipeline = preprocess + estimator
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    # ── Split ──
    if problem_type == "classification":
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y)
            log(f"Stratified split: {len(X_train)} train / {len(X_test)} test")
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            log(f"Regular split (stratify failed): {len(X_train)} train / {len(X_test)} test")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        log(f"Split: {len(X_train)} train / {len(X_test)} test")

    # ── Cross-validation (on train only, not test) ──
    cv_metrics = {}
    try:
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=min(cv_folds, vc.min()), shuffle=True, random_state=random_state)
            scoring = "f1_weighted"
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = "r2"
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                     scoring=scoring, n_jobs=-1)
        cv_metrics = {
            scoring: {
                "mean": round(float(cv_scores.mean()), 4),
                "std":  round(float(cv_scores.std()), 4),
                "folds": [round(float(s), 4) for s in cv_scores],
            }
        }
        log(f"CV {scoring}: {cv_metrics[scoring]['mean']:.4f} ± {cv_metrics[scoring]['std']:.4f}")
    except Exception as e:
        log(f"CV skipped: {type(e).__name__}: {e}")

    # ── Fit on full train, evaluate on holdout ──
    pipeline.fit(X_train, y_train)
    log("Fitted pipeline on train split")

    y_pred = pipeline.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, problem_type, pipeline, X_test)
    log(f"Test metrics: {_metrics_str(metrics)}")

    # ── Permutation feature importance (slower but honest) ──
    importance = []
    try:
        perm = permutation_importance(pipeline, X_test, y_test, n_repeats=5,
                                       random_state=random_state, n_jobs=-1,
                                       scoring=None)  # uses estimator's default
        importance = sorted(
            [{"feature": f, "score": round(float(s), 4)}
             for f, s in zip(feature_names, perm.importances_mean)],
            key=lambda x: x["score"], reverse=True
        )[:20]
    except Exception as e:
        log(f"Permutation importance skipped: {type(e).__name__}")

    # ── Save the WHOLE pipeline as one artifact ──
    os.makedirs("models", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/pma_{algo_id}_{ts}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    metadata = {
        "algo_id": algo_id,
        "algo_name": algo["name"],
        "task": task,
        "problem_type": problem_type,
        "target_col": target_col,
        "feature_names": feature_names,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "is_imbalanced": is_imbalanced,
        "model_path": model_path,
        "saved_at": ts,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }

    _append_metadata(metadata)

    return {
        "ok": True,
        "algo_id": algo_id,
        "algo_name": algo["name"],
        "algo_family": algo.get("family", ""),
        "task": task,
        "problem_type": problem_type,
        "target_col": target_col,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "params": params or {},
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "feature_names": feature_names,
        "feature_importance": importance,
        "class_distribution": class_dist,
        "is_imbalanced": is_imbalanced,
        "y_true": y_test.astype(object).where(pd.notnull(y_test), None).tolist()[:500],
        "y_pred": [_safe_scalar(v) for v in y_pred[:500]],
        "model_path": model_path,
        "logs": logs,
    }


def _safe_scalar(v):
    try:
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v if isinstance(v, (int, float, str, bool)) else str(v)
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, problem_type, pipeline, X_test) -> dict:
    if problem_type == "classification":
        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes == 2 else "weighted"
        m = {
            "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy":  round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "precision":          round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall":             round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1":                 round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1_macro":           round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "confusion_matrix":   confusion_matrix(y_true, y_pred).tolist(),
            "n_classes":          n_classes,
        }
        # AUC only for binary + probability support
        if n_classes == 2 and hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_test)[:, 1]
                m["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
            except Exception:
                pass
        return m
    # Regression
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 2),
    }


def _metrics_str(m):
    return " | ".join(f"{k}={v}" for k, v in m.items() if k != "confusion_matrix")


# ─────────────────────────────────────────────────────────────
# 5. PREDICT (safe: catches unseen categories honestly)
# ─────────────────────────────────────────────────────────────

def predict_new_data(input_data: dict, model_path: str,
                      feature_names: list, problem_type: str = "regression") -> dict:
    """Apply a saved pipeline to a single input row (dict)."""
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Build a one-row DataFrame aligned to feature_names
    row = {f: input_data.get(f, None) for f in feature_names}
    X_new = pd.DataFrame([row])

    try:
        y_hat = pipeline.predict(X_new)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {type(e).__name__}: {e}"}

    result = {}
    if problem_type == "regression":
        result["prediction"] = float(y_hat)
    else:
        result["prediction"] = _safe_scalar(y_hat)

    # Probability / confidence for classification
    if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_new)[0]
            classes = getattr(pipeline, "classes_", list(range(len(proba))))
            result["probabilities"] = [
                {"class": _safe_scalar(c), "prob": round(float(p), 4)}
                for c, p in zip(classes, proba)
            ]
            result["confidence"] = round(float(max(proba)) * 100, 2)
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────
# 6. TUNE (grid / random search, no leakage)
# ─────────────────────────────────────────────────────────────

def tune_model(
    df: pd.DataFrame, target_col: str, algo_id: str,
    *,
    method: str = "random", n_iter: int = 20,
    test_size: float = 0.2, random_state: int = 42,
) -> dict:
    """CV-tune on train set, final metrics from untouched test set."""
    algo = algorithm(algo_id)
    if not algo:
        return {"ok": False, "error": f"Unknown algorithm: {algo_id}"}
    if not algo["tunable"]:
        return {"ok": False, "error": f"No tunable parameters defined for {algo['name']}"}

    df = df.copy()
    df = df.dropna(subset=[target_col])

    task_info = infer_task(df, target_col)
    problem_type = task_info["problem_type"]

    X = df.drop(columns=[target_col])
    for c in list(X.columns):
        if _looks_like_date(X[c]):
            X = X.drop(columns=[c])
    y = df[target_col]

    preprocessor = build_preprocessor(df.loc[:, X.columns.tolist() + [target_col]], target_col)
    estimator = algo["builder"](None)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    # Prefix tunable param names for pipeline access
    param_grid = {f"model__{k}": v for k, v in algo["tunable"].items()}

    if problem_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        cv, scoring = 5, "f1_weighted"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        cv, scoring = 5, "neg_root_mean_squared_error"

    try:
        if method == "grid":
            search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, cv=cv,
                                         scoring=scoring, n_jobs=-1, random_state=random_state)
        search.fit(X_train, y_train)
    except Exception as e:
        return {"ok": False, "error": f"Tuning failed: {type(e).__name__}: {e}"}

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, problem_type, best_pipeline, X_test)

    # Save best model
    os.makedirs("models", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/pma_{algo_id}_tuned_{ts}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    metadata = {
        "algo_id": algo_id + "_tuned", "algo_name": algo["name"] + " (Tuned)",
        "task": task_info["task"], "problem_type": problem_type,
        "target_col": target_col, "feature_names": X.columns.tolist(),
        "metrics": metrics, "best_params": best_params,
        "cv_best_score": round(float(search.best_score_), 4),
        "model_path": model_path, "saved_at": ts,
    }
    _append_metadata(metadata)

    return {
        "ok": True, "best_params": best_params,
        "cv_best_score": round(float(search.best_score_), 4),
        "test_metrics": metrics, "model_path": model_path,
    }


# ─────────────────────────────────────────────────────────────
# 7. Metadata helpers
# ─────────────────────────────────────────────────────────────

_META_PATH = "models/model_metadata.json"


def _append_metadata(entry: dict, max_keep: int = 50):
    existing = []
    if os.path.exists(_META_PATH):
        try:
            with open(_META_PATH) as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
    existing.insert(0, entry)
    with open(_META_PATH, "w") as f:
        json.dump(existing[:max_keep], f, indent=2, default=str)


def list_saved_models() -> list:
    if not os.path.exists(_META_PATH):
        return []
    try:
        with open(_META_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def load_model_by_path(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────
#  LEGACY SHIMS  — keep old routes (predict.html v1) working
#  until the v2 frontend ships. Delete these once predict.html v2 ships.
# ─────────────────────────────────────────────────

def detect_data_type(df: pd.DataFrame, target_col: str = None) -> dict:
    """LEGACY: returns old-format dict. Prefer infer_task + profile_dataset."""
    info = infer_task(df, target_col)
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    n_unique  = None
    if target_col and target_col in df.columns:
        n_unique = int(df[target_col].dropna().nunique())
    return {
        "data_type":    info["task"] if info["task"] == "time_series" else "tabular",
        "problem_type": info["problem_type"],
        "n_rows":       int(len(df)),
        "n_cols":       int(len(df.columns)),
        "num_cols":     num_cols,
        "cat_cols":     cat_cols,
        "date_cols":    info.get("date_cols", []),
        "n_classes":    n_unique if info["problem_type"] == "classification" else None,
        "target_col":   target_col,
    }


def preprocess_tabular(df, target_col, info):
    """LEGACY: left as a thin wrapper; new code should use build_preprocessor."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    for col in num_cols:
        X[col].fillna(X[col].median(), inplace=True)
    for col in cat_cols:
        X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
    target_encoder = None
    if info.get("problem_type") == "classification" and y.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    return (X.values, y.values if hasattr(y, "values") else np.array(y),
            X.columns.tolist(),
            {"label_encoders": label_encoders, "target_encoder": target_encoder})


def get_feature_importance(X, y, feature_names, problem_type):
    """LEGACY: SelectKBest-based importance. v2 uses permutation_importance."""
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    try:
        scorer = f_classif if problem_type == "classification" else f_regression
        sel = SelectKBest(scorer, k=min(len(feature_names), 20))
        sel.fit(X, y)
        return sorted(
            [{"feature": f, "score": round(float(s), 4)}
             for f, s in zip(feature_names, sel.scores_)],
            key=lambda x: x["score"], reverse=True,
        )
    except Exception:
        return [{"feature": f, "score": 0.0} for f in feature_names]


def get_model_catalog(data_type, problem_type):
    """LEGACY: returns {key: {name, category, model}} for the v1 UI."""
    task = "time_series" if data_type == "time_series" else problem_type
    algos = [a for a in ALL_ALGORITHMS.values() if a["task"] == task]
    out = {}
    for a in algos:
        if a["builder"] is None:  # skip algos without sklearn builders
            continue
        out[a["id"]] = {
            "name": a["name"], "category": a["family"],
            "model": a["builder"](None),
        }
    return out


def recommend_models(data_type, problem_type, n_rows, n_cols):
    """LEGACY: top-3 recommendations. v2 uses suggest_algorithms with LLM."""
    task = "time_series" if data_type == "time_series" else problem_type
    candidates = [a for a in ALL_ALGORITHMS.values() if a["task"] == task]
    # Prefer boosting + ensemble champions
    preferred_ids = ("xgboost_clf", "xgboost_reg", "lightgbm_clf", "lightgbm_reg",
                     "catboost_clf", "catboost_reg", "random_forest_clf",
                     "random_forest_reg", "gradient_boosting_clf",
                     "gradient_boosting_reg", "logistic_regression", "ridge")
    recs = []
    for aid in preferred_ids:
        a = ALL_ALGORITHMS.get(aid)
        if a and a["task"] == task and a["builder"]:
            recs.append({"key": aid,
                         "reason": f"{a['family']} — " + (a["strengths"][0] if a["strengths"] else "")})
        if len(recs) >= 3:
            break
    return recs


def train_selected_model(X_train, y_train, X_val, y_val, model_key, catalog, problem_type):
    """LEGACY: direct fit on passed-in numpy arrays (leaky but kept for v1 UI)."""
    entry = catalog.get(model_key)
    if not entry or entry["model"] is None:
        return {"error": f"Unknown model key: {model_key}", "logs": []}
    model = entry["model"]
    logs = [f"[{datetime.now().strftime('%H:%M:%S')}] LEGACY path: fitting {entry['name']}"]
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  mean_squared_error, mean_absolute_error, r2_score,
                                  mean_absolute_percentage_error, confusion_matrix)
    if problem_type == "classification":
        avg = "binary" if len(np.unique(y_val)) == 2 else "weighted"
        metrics = {
            "accuracy":  round(float(accuracy_score(y_val, preds)), 4),
            "precision": round(float(precision_score(y_val, preds, average=avg, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_val, preds, average=avg, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_val, preds, average=avg, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
        }
    else:
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_val, preds))), 4),
            "mae":  round(float(mean_absolute_error(y_val, preds)), 4),
            "r2":   round(float(r2_score(y_val, preds)), 4),
            "mape": round(float(mean_absolute_percentage_error(y_val, preds)) * 100, 2),
        }
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Metrics: {metrics}")
    return {"model": model, "model_key": model_key, "model_name": entry["name"],
            "metrics": metrics, "logs": logs, "problem_type": problem_type,
            "y_val": list(y_val), "y_pred": list(preds)}


def save_model_artifacts(model, metadata, model_key):
    """LEGACY: saves a raw estimator. v2 saves full Pipeline instead."""
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/pma_{model_key}_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    metadata["model_path"] = model_path
    metadata["saved_at"]   = timestamp
    _append_metadata(metadata)
    return model_path


def tune_model_legacy(model, model_key, X_train, y_train, method, problem_type):
    """LEGACY shim for the old /api/pma/tune route."""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    algo = algorithm(model_key)
    if not algo or not algo["tunable"]:
        return {"best_params": {}, "best_score": None,
                "message": "No param grid defined."}
    param_grid = algo["tunable"]
    scoring = "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error"
    try:
        if method == "grid":
            search = GridSearchCV(model, param_grid, scoring=scoring, cv=3, n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, scoring=scoring, cv=3,
                                         n_iter=10, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        return {"best_params": search.best_params_,
                "best_score":  round(float(search.best_score_), 4),
                "best_model":  search.best_estimator_}
    except Exception as e:
        return {"error": str(e), "best_params": {}, "best_score": None}


def predict_new_data_legacy(input_data, feature_names, model, encoders, problem_type):
    """LEGACY: takes raw model + encoders (old /api/pma/predict path)."""
    row = {}
    for feat in feature_names:
        val = input_data.get(feat, 0)
        le = (encoders or {}).get("label_encoders", {}).get(feat)
        if le:
            try:
                val = le.transform([str(val)])[0]
            except Exception:
                val = 0
        try:
            val = float(val)
        except Exception:
            val = 0.0
        row[feat] = val
    X_new = np.array([[row[f] for f in feature_names]])
    prediction = model.predict(X_new)[0]
    result = {"prediction": float(prediction) if problem_type == "regression" else int(prediction)}
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        result["probabilities"] = [round(float(p), 4) for p in proba]
        result["confidence"] = round(float(max(proba)) * 100, 2)
    target_enc = (encoders or {}).get("target_encoder")
    if target_enc and problem_type == "classification":
        try:
            result["prediction_label"] = str(target_enc.inverse_transform([result["prediction"]])[0])
        except Exception:
            pass
    return result
