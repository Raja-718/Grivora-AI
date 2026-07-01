"""
src/ml_suggester.py
===================
Auto-suggest ML algorithms for a user's dataset.

Flow (LLM-first with rules fallback):
  1. profile_dataset(df, target)  -> compact 30+ signal JSON profile
  2. _llm_suggest(profile, task)   -> asks Gemini to rank 3-5 algos
  3. validate picks against algorithms registry (no hallucinations)
  4. if <3 valid picks came back, top up from rule-based shortlist
  5. return [{"id", "rank", "rationale", "confidence"}]

Also exposes infer_task() which figures out whether the problem is
regression / classification / clustering / anomaly / time_series based
on (a) whether target is provided and (b) its cardinality and type.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.algorithms import (
    ALL_ALGORITHMS, algorithms_for_task, catalog_for_llm, filter_by_data,
)


# ─────────────────────────────────────────────────────────────
# 1. Task inference
# ─────────────────────────────────────────────────────────────

def infer_task(df: pd.DataFrame, target_col: Optional[str]) -> dict:
    """
    Return {task, problem_type, reason} where task is one of:
      regression / classification / time_series / clustering / anomaly
    """
    n_rows, n_cols = df.shape

    # Detect date columns (stronger check than just name matching)
    date_cols = []
    for col in df.columns:
        if _looks_like_date(df[col]):
            date_cols.append(col)

    # No target column → unsupervised
    if not target_col or target_col not in df.columns:
        # If there's a dominant date axis AND a plausible numeric target, we
        # could guess time_series, but without a target we stay unsupervised.
        return {
            "task": "clustering",
            "problem_type": None,
            "reason": "No target column provided → unsupervised (clustering).",
            "date_cols": date_cols,
        }

    y = df[target_col].dropna()
    if y.empty:
        return {"task": "classification", "problem_type": "classification",
                "reason": "Target is entirely missing.", "date_cols": date_cols}

    n_unique = int(y.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(y)

    # Time series if there's a date column AND target is numeric AND n_rows
    # per unique-date ratio suggests sequential data.
    if date_cols and is_numeric:
        date_col = date_cols[0]
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if len(dates) >= 20 and dates.is_monotonic_increasing or len(dates.unique()) / max(len(dates), 1) > 0.5:
                return {
                    "task": "time_series", "problem_type": "regression",
                    "reason": f"Date column '{date_col}' + numeric target '{target_col}'.",
                    "date_cols": date_cols,
                }
        except Exception:
            pass

    # Classification vs Regression
    if not is_numeric or y.dtype == bool:
        return {"task": "classification", "problem_type": "classification",
                "reason": f"Target is categorical ({n_unique} unique values).",
                "date_cols": date_cols}

    # Numeric target but few unique values → likely classification
    if n_unique <= 20 and n_unique / max(n_rows, 1) < 0.05:
        return {"task": "classification", "problem_type": "classification",
                "reason": f"Target is numeric but only {n_unique} unique values → classification.",
                "date_cols": date_cols}

    return {"task": "regression", "problem_type": "regression",
            "reason": f"Target is continuous ({n_unique} unique values).",
            "date_cols": date_cols}


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
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.6


# ─────────────────────────────────────────────────────────────
# 2. Dataset profiling  (feeds both rules and LLM)
# ─────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_col: Optional[str]) -> dict:
    """Compact JSON profile: every signal the suggester or LLM could want."""
    n_rows, n_cols = df.shape

    num_cols  = df.select_dtypes("number").columns.tolist()
    obj_cols  = df.select_dtypes("object").columns.tolist()
    bool_cols = df.select_dtypes("bool").columns.tolist()

    date_cols, text_cols = [], []
    for col in obj_cols:
        if _looks_like_date(df[col]):
            date_cols.append(col)
            continue
        sample = df[col].dropna().astype(str).head(50)
        if not sample.empty and sample.str.len().mean() > 40:
            text_cols.append(col)

    # Keep object cols that aren't dates or long-text as "categorical"
    cat_cols = [c for c in obj_cols if c not in date_cols and c not in text_cols] + bool_cols

    # ── Feature-level stats ────────────────────────────────
    num_info = []
    for c in num_cols[:25]:
        s = df[c].dropna()
        if s.empty:
            continue
        # Skewness and outlier rate, clipped to avoid huge floats in JSON
        try:
            skew = float(s.skew()) if len(s) > 2 else 0.0
        except Exception:
            skew = 0.0
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_rate = float(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean()) if iqr > 0 else 0.0
        num_info.append({
            "name": c,
            "min": _r(s.min()), "max": _r(s.max()),
            "mean": _r(s.mean()), "std": _r(s.std()),
            "missing_pct": round(float(df[c].isna().mean() * 100), 1),
            "skew": round(skew, 2),
            "outlier_rate": round(outlier_rate, 3),
        })

    cat_info = []
    for c in cat_cols[:25]:
        n_unique = int(df[c].nunique(dropna=True))
        cat_info.append({
            "name": c,
            "unique": n_unique,
            "cardinality": "low" if n_unique <= 10 else ("mid" if n_unique <= 50 else "high"),
            "missing_pct": round(float(df[c].isna().mean() * 100), 1),
            "top_value": str(df[c].mode().iloc[0]) if not df[c].mode().empty else None,
        })

    # ── Target stats (the crucial part for model choice) ──
    target_info = None
    if target_col and target_col in df.columns:
        y = df[target_col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(y)
        if is_numeric:
            target_info = {
                "name": target_col, "type": "numeric",
                "n_unique": int(y.nunique()),
                "min": _r(y.min()), "max": _r(y.max()),
                "mean": _r(y.mean()), "std": _r(y.std()),
                "missing_pct": round(float(df[target_col].isna().mean() * 100), 1),
                "skew": round(float(y.skew()) if len(y) > 2 else 0.0, 2),
            }
        else:
            vc = y.value_counts()
            total = int(vc.sum())
            # Class imbalance ratio: majority / minority
            imbalance = round(float(vc.max() / max(vc.min(), 1)), 2)
            target_info = {
                "name": target_col, "type": "categorical",
                "n_classes": int(y.nunique()),
                "imbalance_ratio": imbalance,
                "class_distribution": {str(k): int(v) for k, v in vc.head(10).items()},
                "missing_pct": round(float(df[target_col].isna().mean() * 100), 1),
            }

    # ── Domain hint from column names ────────────────────
    all_colnames = " ".join(str(c).lower() for c in df.columns)
    domain_hints = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(k in all_colnames for k in keywords):
            domain_hints.append(domain)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_date": len(date_cols),
        "n_text": len(text_cols),
        "total_missing_pct": round(float(df.isna().mean().mean() * 100), 1),
        "numeric_cols": num_info,
        "categorical_cols": cat_info,
        "date_cols": date_cols,
        "text_cols": text_cols,
        "target": target_info,
        "domain_hints": domain_hints,
        "sample_rows": _safe_sample(df, 3),
    }


_DOMAIN_KEYWORDS = {
    "finance":     ["price", "cost", "revenue", "profit", "loss", "invest", "return", "stock", "currency"],
    "healthcare":  ["patient", "diagnosis", "disease", "treatment", "clinical", "medical", "hospital", "drug"],
    "retail":      ["customer", "purchase", "order", "cart", "product", "sku", "store"],
    "hr":          ["employee", "salary", "hire", "department", "manager", "attendance"],
    "marketing":   ["campaign", "click", "impression", "ctr", "conversion", "ad_spend"],
    "survey":      ["response", "question", "rating", "satisfaction", "score"],
    "education":   ["student", "course", "grade", "exam", "teacher", "school"],
    "iot":         ["sensor", "device", "temperature", "humidity", "reading", "signal"],
}


def _r(v):
    try:
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 3)
    except Exception:
        return None


def _safe_sample(df: pd.DataFrame, n=3):
    rows = df.head(n).astype(object).where(df.head(n).notna(), None).to_dict(orient="records")
    safe = []
    for row in rows:
        safe.append({
            k: (None if v is None
                else (v if isinstance(v, (int, float, str, bool)) else str(v)))
            for k, v in row.items()
        })
    return safe


# ─────────────────────────────────────────────────────────────
# 3. Rule-based shortlist + fallback
# ─────────────────────────────────────────────────────────────

def _rule_based_shortlist(profile: dict, task: str) -> list[dict]:
    """Deterministic shortlist used as fallback AND to cap the LLM choices."""
    n_rows = profile["n_rows"]
    n_cols = profile["n_cols"]
    tgt = profile.get("target") or {}
    high_card_cats = any(c["cardinality"] == "high" for c in profile["categorical_cols"])
    has_missing = profile["total_missing_pct"] > 5
    imbalanced = tgt.get("imbalance_ratio", 1) > 5 if tgt else False
    has_text = profile["n_text"] > 0

    candidates = filter_by_data(algorithms_for_task(task), n_rows, n_cols)
    scored = []

    for a in candidates:
        score = 50.0  # baseline
        # Dataset-size affinity
        if n_rows < 500 and a["speed"] == "slow": score -= 15
        if n_rows > 50_000 and a["speed"] == "slow": score -= 20
        if n_rows > 100_000 and a["family"] in ("Instance", "SVM"): score -= 30
        # Missing / categorical handling
        if has_missing and a["handles_missing"]: score += 12
        if high_card_cats and a["handles_high_cardinality"]: score += 12
        # Imbalance friendliness (tree ensembles and boosting handle it best)
        if imbalanced and a["family"] in ("Ensemble", "Boosting"): score += 10
        # Text → NB and linear SVM win
        if has_text and a["id"] in ("multinomial_nb", "linear_svc", "logistic_regression"):
            score += 15
        # Small data → simple models
        if n_rows < 200:
            if a["family"] in ("Linear", "Probabilistic", "Discriminant"): score += 10
            if a["family"] == "Neural": score -= 25
        # Reward tabular champions
        if a["id"] in ("xgboost_clf", "xgboost_reg", "lightgbm_clf", "lightgbm_reg",
                       "catboost_clf", "catboost_reg", "hist_gb_clf", "hist_gb_reg",
                       "random_forest_clf", "random_forest_reg"):
            score += 8

        scored.append((score, a))

    scored.sort(key=lambda x: -x[0])
    # Pick diverse families in the top 6
    picked, seen_families = [], set()
    for score, a in scored:
        if a["family"] in seen_families and len(picked) >= 3:
            continue
        picked.append({
            "id": a["id"], "rank": len(picked) + 1,
            "rationale": _rule_rationale(a, profile),
            "confidence": round(score / 100, 2),
        })
        seen_families.add(a["family"])
        if len(picked) >= 6:
            break
    return picked


def _rule_rationale(a: dict, profile: dict) -> str:
    bits = []
    if profile["total_missing_pct"] > 5 and a["handles_missing"]:
        bits.append("handles missing values")
    if any(c["cardinality"] == "high" for c in profile["categorical_cols"]) and a["handles_high_cardinality"]:
        bits.append("good with high-cardinality categoricals")
    if a["interpretable"]:
        bits.append("interpretable")
    tgt = profile.get("target")
    if tgt and tgt.get("imbalance_ratio", 1) > 5 and a["family"] in ("Ensemble", "Boosting"):
        bits.append("robust to class imbalance")
    if a["family"] == "Boosting":
        bits.append("strong tabular baseline")
    elif a["family"] == "Ensemble":
        bits.append("low variance")
    elif a["family"] == "Linear":
        bits.append("fast and simple")
    return "; ".join(bits) or f"reasonable default for {profile['n_rows']} rows"


# ─────────────────────────────────────────────────────────────
# 4. LLM suggester
# ─────────────────────────────────────────────────────────────

_SUGGEST_PROMPT = """You are Grivora AI's ML algorithm picker.

Given a dataset profile, select 3 to 5 algorithms that are most likely to perform well on this specific dataset for the specified task. Pick from the catalog only. Never invent algorithm IDs.

Key considerations:
- Dataset size: tiny (<200) favors Linear / NaiveBayes / LDA; huge (>100k) avoids KNN / SVM-RBF.
- Missing values: prefer algorithms with handles_missing=true if missing_pct > 5%.
- High-cardinality categoricals: CatBoost / LightGBM / HistGradientBoosting handle them natively.
- Class imbalance (imbalance_ratio > 5): favor Ensemble / Boosting, note class_weight support.
- Skewed / outlier-heavy numerics: favor tree-based over Linear.
- Text or count features: MultinomialNB, Linear SVC, Logistic Regression dominate.
- Time series with clear seasonality: Prophet, SARIMA win; also offer a boosted-lags ML model.
- Interpretability matters when domain includes healthcare or finance.

Always include at least one interpretable baseline and at least one high-accuracy champion.

OUTPUT (strict JSON only, no code fences, no commentary):
{
  "picks": [
    {"id": "<catalog id>", "rank": 1, "rationale": "<20 words max>", "confidence": 0.0-1.0}
  ]
}

TASK: <<TASK>>

DATASET PROFILE:
<<PROFILE_JSON>>

ALGORITHM CATALOG (pick from these ids only):
<<CATALOG_JSON>>

Return JSON now.
"""


def _llm_suggest(profile: dict, task: str, timeout: float = 15.0) -> list | None:
    """Call Gemini. Return validated picks, or None if the call fails."""
    try:
        from llm.gemini_client import GeminiClient
    except Exception:
        return None

    prompt = (
        _SUGGEST_PROMPT
        .replace("<<TASK>>", task)
        .replace("<<PROFILE_JSON>>", json.dumps(profile, default=str, indent=2))
        .replace("<<CATALOG_JSON>>", json.dumps(catalog_for_llm(task), indent=2))
    )

    client = GeminiClient()
    try:
        raw = client.generate(prompt, timeout=timeout)
    except Exception:
        return None

    if not raw or raw.startswith("[AI unavailable") or raw.startswith("Error from Gemini"):
        return None

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    picks = data.get("picks") if isinstance(data, dict) else None
    if not isinstance(picks, list):
        return None
    return picks


def _validate_pick(pick: dict, task: str) -> dict | None:
    if not isinstance(pick, dict):
        return None
    algo_id = pick.get("id")
    algo = ALL_ALGORITHMS.get(algo_id)
    if not algo or algo["task"] != task:
        return None
    conf = pick.get("confidence")
    if not isinstance(conf, (int, float)):
        conf = 0.7
    return {
        "id": algo_id,
        "rank": int(pick.get("rank", 1)),
        "rationale": str(pick.get("rationale", ""))[:200],
        "confidence": max(0.0, min(1.0, float(conf))),
    }


# ─────────────────────────────────────────────────────────────
# 5. Public entry point
# ─────────────────────────────────────────────────────────────

def suggest_algorithms(df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
    """
    Full pipeline. Returns:
      {
        "task": "...",
        "problem_type": "...",
        "reason": "...",
        "profile": {...},
        "picks":   [{"id","rank","rationale","confidence","meta"}],
        "source":  "llm" | "rules" | "llm+rules",
      }
    """
    task_info = infer_task(df, target_col)
    profile   = profile_dataset(df, target_col)

    # Enrich profile with task context so the LLM sees it explicitly
    profile["_task"] = task_info["task"]

    # LLM-first
    source = "llm"
    raw_picks = None
    try:
        raw_picks = _llm_suggest(profile, task_info["task"])
    except Exception:
        raw_picks = None

    if not raw_picks:
        raw_picks = _rule_based_shortlist(profile, task_info["task"])
        source = "rules"

    # Validate
    validated = [v for v in (_validate_pick(p, task_info["task"]) for p in raw_picks) if v]

    # Top up with rules if LLM was sparse
    if source == "llm" and len(validated) < 3:
        filler = _rule_based_shortlist(profile, task_info["task"])
        seen = {p["id"] for p in validated}
        for p in filler:
            if p["id"] not in seen:
                validated.append(p)
                seen.add(p["id"])
            if len(validated) >= 5:
                break
        source = "llm+rules"

    # De-duplicate by id, keep first (highest-ranked)
    out, seen = [], set()
    for p in sorted(validated, key=lambda x: x["rank"])[:5]:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        algo = ALL_ALGORITHMS[p["id"]]
        p["meta"] = {
            "name": algo["name"], "family": algo["family"],
            "speed": algo["speed"], "interpretable": algo["interpretable"],
            "strengths": algo["strengths"], "weaknesses": algo["weaknesses"],
        }
        out.append(p)

    # Re-rank based on position (LLM rank may be off)
    for i, p in enumerate(out):
        p["rank"] = i + 1

    return {
        "task":         task_info["task"],
        "problem_type": task_info["problem_type"],
        "reason":       task_info["reason"],
        "profile":      profile,
        "picks":        out,
        "source":       source,
    }
