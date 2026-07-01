"""
src/automl.py
=============
Quick Auto-ML: train the top-N suggested algorithms in parallel, rank them.

Uses ThreadPoolExecutor (not multiprocessing) because:
  - scikit-learn estimators release the GIL during fit for most C-backed code
  - The models write pickles to disk; multiprocessing would complicate this
  - We still run 3-5 trainings concurrently which is plenty for interactive use

Each model is trained via src.pma_engine.train_model, which already handles:
  - No-leakage preprocessing pipeline
  - Stratified split, CV metrics, permutation importance, safe save
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from src.pma_engine import train_model
from src.ml_suggester import suggest_algorithms


# ─────────────────────────────────────────────────────────────
# Leaderboard scoring — one metric per task, higher is better
# ─────────────────────────────────────────────────────────────

def _leaderboard_score(result: dict) -> float:
    """Primary score used to rank the leaderboard. Higher = better."""
    if not result.get("ok"):
        return float("-inf")
    metrics = result.get("metrics", {})
    if result["problem_type"] == "classification":
        # f1_macro is the most honest for imbalanced classes; fall back to accuracy
        return metrics.get("f1_macro", metrics.get("f1", metrics.get("accuracy", 0.0)))
    # Regression: higher R² is better
    return metrics.get("r2", 0.0)


def _primary_metric_name(problem_type: str) -> str:
    return "f1_macro" if problem_type == "classification" else "r2"


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_automl(
    df: pd.DataFrame,
    target_col: str,
    *,
    top_n: int = 5,
    test_size: float = 0.2,
    cv_folds: int = 5,
    max_workers: int = 4,
    pre_picked_algos: Optional[list[str]] = None,
) -> dict:
    """
    End-to-end Auto-ML:
      1. suggest_algorithms(df, target) → picks top-N
         (or use pre_picked_algos if provided, e.g. from manual selection)
      2. train each in parallel via ThreadPoolExecutor
      3. rank by primary metric, return leaderboard

    Returns:
      {
        "ok": True,
        "task": "regression" | "classification" | ...,
        "problem_type": "...",
        "primary_metric": "r2" | "f1_macro",
        "suggestions": [{id, rank, rationale, confidence, meta}, ...],
        "leaderboard": [
          {
            "rank": 1, "algo_id", "algo_name", "family", "score",
            "metrics": {...}, "cv_metrics": {...},
            "model_path": "...", "train_time_sec": 3.4,
            "feature_importance": [...],  # top 5
            "ok": True, "error": None,
          }, ...
        ],
        "best": {"algo_id", "algo_name", "model_path", "score"},
        "source": "llm" | "rules" | "llm+rules",
      }
    """
    t0 = time.time()

    # ── Step 1: get suggestions (LLM-first) ──────────────────
    if pre_picked_algos:
        # Caller specified the list explicitly (used by the manual Auto-ML
        # "compare these 3 models" flow). Skip the suggester entirely.
        from src.ml_suggester import infer_task, profile_dataset
        from src.algorithms import ALL_ALGORITHMS
        task_info = infer_task(df, target_col)
        suggestions = []
        for i, aid in enumerate(pre_picked_algos[:top_n]):
            algo = ALL_ALGORITHMS.get(aid)
            if not algo: continue
            suggestions.append({
                "id": aid, "rank": i + 1,
                "rationale": "User-picked for comparison",
                "confidence": 0.8,
                "meta": {"name": algo["name"], "family": algo["family"],
                         "speed": algo["speed"], "interpretable": algo["interpretable"]},
            })
        task = task_info["task"]
        problem_type = task_info["problem_type"]
        source = "manual"
    else:
        plan = suggest_algorithms(df, target_col)
        suggestions = plan["picks"][:top_n]
        task = plan["task"]
        problem_type = plan["problem_type"]
        source = plan["source"]

    if not suggestions:
        return {"ok": False, "error": "No algorithm suggestions could be generated."}

    if task not in ("regression", "classification", "time_series"):
        return {"ok": False,
                "error": f"Auto-ML currently supports supervised tasks only, not '{task}'."}

    primary_metric = _primary_metric_name(problem_type)

    # ── Step 2: train in parallel ────────────────────────────
    results_by_id = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for s in suggestions:
            future = ex.submit(
                _train_one_safe,
                df.copy(), target_col, s["id"], test_size, cv_folds,
            )
            futures[future] = s["id"]
        for future in as_completed(futures):
            algo_id = futures[future]
            results_by_id[algo_id] = future.result()

    # ── Step 3: build leaderboard ────────────────────────────
    entries = []
    for s in suggestions:
        res = results_by_id.get(s["id"], {})
        meta = s.get("meta", {})
        if res.get("ok"):
            entries.append({
                "algo_id": s["id"],
                "algo_name": meta.get("name") or res.get("algo_name"),
                "family":    meta.get("family", ""),
                "score":     round(_leaderboard_score(res), 4),
                "metrics":   res.get("metrics", {}),
                "cv_metrics": res.get("cv_metrics", {}),
                "feature_importance": res.get("feature_importance", [])[:5],
                "class_distribution": res.get("class_distribution"),
                "is_imbalanced": res.get("is_imbalanced", False),
                "model_path": res.get("model_path"),
                "train_time_sec": res.get("train_time_sec", 0),
                "y_true": res.get("y_true", [])[:200],
                "y_pred": res.get("y_pred", [])[:200],
                "ok": True, "error": None,
            })
        else:
            entries.append({
                "algo_id": s["id"],
                "algo_name": meta.get("name", s["id"]),
                "family":    meta.get("family", ""),
                "score":     None,
                "metrics":   {}, "cv_metrics": {},
                "feature_importance": [],
                "model_path": None,
                "train_time_sec": res.get("train_time_sec", 0),
                "ok": False, "error": res.get("error", "Unknown error"),
            })

    # Sort by score desc (failed runs sink to bottom)
    entries.sort(key=lambda e: e["score"] if e["score"] is not None else float("-inf"),
                 reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    best = next((e for e in entries if e["ok"]), None)

    return {
        "ok": True,
        "task": task,
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "suggestions": suggestions,
        "leaderboard": entries,
        "best": {
            "algo_id":   best["algo_id"],
            "algo_name": best["algo_name"],
            "model_path": best["model_path"],
            "score":     best["score"],
        } if best else None,
        "source": source,
        "total_elapsed_sec": round(time.time() - t0, 2),
    }


# ─────────────────────────────────────────────────────────────
# Worker (runs inside ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────

def _train_one_safe(df, target_col, algo_id, test_size, cv_folds) -> dict:
    """Train one model; catch everything so one failure can't nuke the leaderboard."""
    t0 = time.time()
    try:
        result = train_model(
            df, target_col, algo_id,
            test_size=test_size, cv_folds=cv_folds,
        )
        result["train_time_sec"] = round(time.time() - t0, 2)
        return result
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "train_time_sec": round(time.time() - t0, 2),
        }
