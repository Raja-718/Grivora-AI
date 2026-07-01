"""
app/auto_dashboard_route.py
Two-stage LLM-driven auto dashboard:
  Stage 1: /api/auto-dashboard/plan
    -> loads file, profiles it, asks Gemini (or rules) which charts to build.
       Returns the plan (fast). Frontend can already show "LLM picked N charts."

  Stage 2: /api/auto-dashboard/render
    -> takes the plan, builds every chart's data deterministically in Python.
       Returns KPIs + column profiles + preview rows + all chart payloads.

Both endpoints are bound in app/routes.py.

Back-compat: /api/auto-dashboard (single call) still works; it just runs both
stages in sequence. Used by any older code paths.
"""
import os
import time

import numpy as np
import pandas as pd
from flask import jsonify, request, session


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

_MAX_ROWS_PLAN   = 20_000   # sample for planning/charts to keep it snappy
_MAX_ROWS_FULL   = 50_000   # never exceed this for actual chart computation

def _load_session_df(sample_cap: int | None = _MAX_ROWS_PLAN):
    """Resolve the uploaded file, load, and optionally sample."""
    file_path = session.get("uploaded_file")
    if not file_path or not os.path.exists(file_path):
        return None, "No data uploaded yet. Please upload a file first.", None

    from src.data_loader import load_file
    df_full = load_file(file_path)
    df_full.columns = [str(c).strip() for c in df_full.columns]
    n_total = len(df_full)

    if sample_cap and n_total > sample_cap:
        df = df_full.sample(n=sample_cap, random_state=42).reset_index(drop=True)
        sampled = True
    else:
        df = df_full
        sampled = False

    return df, None, {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "n_total":   n_total,
        "sampled":   sampled,
        "df_full":   df_full,     # needed for accurate KPIs
    }


def _safe_preview(df: pd.DataFrame, n=10):
    """Return first n rows as JSON-safe dicts (NaN/Inf -> None)."""
    view = df.head(n).where(pd.notnull(df.head(n)), None)
    rows = view.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
    return rows


def _col_profiles(df: pd.DataFrame, max_cols=30):
    """Per-column summary stats for the Column Profiles grid."""
    num_cols = df.select_dtypes("number").columns.tolist()
    profiles = []
    for col in df.columns[:max_cols]:
        try:
            miss = int(df[col].isna().sum())
            p = {
                "name":        col,
                "dtype":       "number" if col in num_cols else "text",
                "missing":     miss,
                "missing_pct": round(miss / max(len(df), 1) * 100, 1),
                "unique":      int(df[col].nunique()),
            }
            if col in num_cols:
                s = df[col].dropna()
                if len(s):
                    p.update({
                        "min":    round(float(s.min()), 3),
                        "max":    round(float(s.max()), 3),
                        "mean":   round(float(s.mean()), 3),
                        "median": round(float(s.median()), 3),
                        "std":    round(float(s.std()), 3),
                    })
            else:
                vc = df[col].value_counts()
                p["top_val"]   = str(vc.index[0]) if len(vc) else ""
                p["top_count"] = int(vc.iloc[0])  if len(vc) else 0
            profiles.append(p)
        except Exception:
            continue
    return profiles


# ────────────────────────────────────────────────────────────────
# Stage 1: PLAN
# ────────────────────────────────────────────────────────────────

def _run_plan():
    """POST /api/auto-dashboard/plan  ->  quick profile + chart picks."""
    t0 = time.time()

    df, err, meta = _load_session_df(sample_cap=_MAX_ROWS_PLAN)
    if err:
        return jsonify({"error": err}), 400

    from src.dashboard_planner import plan_dashboard
    try:
        plan = plan_dashboard(df)
    except Exception as e:
        return jsonify({"error": f"Planning failed: {type(e).__name__}: {e}"}), 500

    return jsonify({
        "ok":        True,
        "file_name": meta["file_name"],
        "n_rows":    meta["n_total"],
        "n_cols":    len(df.columns),
        "sampled":   meta["sampled"],
        "picks":     plan["picks"],
        "source":    plan["source"],
        "elapsed":   round(time.time() - t0, 2),
    })


# ────────────────────────────────────────────────────────────────
# Stage 2: RENDER
# ────────────────────────────────────────────────────────────────

def _run_render():
    """POST /api/auto-dashboard/render  ->  builds chart data for the plan."""
    body  = request.get_json(silent=True) or {}
    picks = body.get("picks")
    if not isinstance(picks, list) or not picks:
        return jsonify({"error": "Missing 'picks' in request body."}), 400
    return _render_for_picks(picks)


def _render_for_picks(picks):
    """Shared render path — takes picks list directly, usable without Flask request body."""
    t0 = time.time()

    df, err, meta = _load_session_df(sample_cap=_MAX_ROWS_FULL)
    if err:
        return jsonify({"error": err}), 400

    df_full = meta["df_full"]
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes("object").columns.tolist()

    # ── KPIs (computed on the FULL dataset for accurate counts) ──────
    total_rows  = meta["n_total"]
    n_cols      = len(df.columns)
    total_cells = total_rows * n_cols
    missing_abs = int(df_full.isna().sum().sum())
    missing_pct = round(missing_abs / total_cells * 100, 1) if total_cells else 0

    kpis = {
        "total_rows":   {"label": "Total Records",   "value": total_rows,                "fmt": "number",  "icon": "🗂️"},
        "total_cols":   {"label": "Total Columns",   "value": n_cols,                    "fmt": "number",  "icon": "📋"},
        "completeness": {"label": "Completeness",    "value": round(100 - missing_pct, 1), "fmt": "percent", "icon": "✅"},
        "duplicates":   {"label": "Duplicate Rows",  "value": int(df_full.duplicated().sum()), "fmt": "number", "icon": "⚠️"},
    }
    # Add BIA-style KPIs if detectable
    try:
        from src.bia.bia_engine import compute_kpis
        biakpi = compute_kpis(df)
        for k, v in biakpi.get("kpis", {}).items():
            if k not in kpis:
                kpis[k] = v
    except Exception:
        pass

    # ── Build every chart in the plan ────────────────────────────
    from src.chart_library import build_chart

    charts = []
    for i, pick in enumerate(picks):
        if not isinstance(pick, dict):
            continue
        chart_id = pick.get("chart_id")
        spec     = pick.get("spec", {})
        out = build_chart(df, chart_id, spec)
        charts.append({
            "index":    i,
            "chart_id": chart_id,
            "title":    pick.get("title") or out.get("chart_title") or chart_id,
            "why":      pick.get("why", ""),
            "spec":     spec,
            "data":     out,            # contains Chart.js payload or {"error": "..."}
        })

    # ── Insights (rule-based, fast) ──────────────────────────────
    try:
        from src.bia.bia_engine import generate_auto_insights, compute_kpis
        insights = generate_auto_insights(df, compute_kpis(df))
    except Exception:
        insights = []

    return jsonify({
        "ok":          True,
        "file_name":   meta["file_name"],
        "n_rows":      total_rows,
        "n_cols":      n_cols,
        "sampled":     meta["sampled"],
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "all_cols":    df.columns.tolist(),
        "kpis":        kpis,
        "charts":      charts,
        "insights":    insights,
        "col_profiles": _col_profiles(df),
        "preview_rows": _safe_preview(df, 10),
        "elapsed":     round(time.time() - t0, 2),
    })


# ────────────────────────────────────────────────────────────────
# Back-compat: single-call endpoint = plan + render stitched
# ────────────────────────────────────────────────────────────────

def _run_auto_dashboard():
    """POST /api/auto-dashboard  ->  legacy single call (plan + render)."""
    t0 = time.time()

    df, err, meta = _load_session_df(sample_cap=_MAX_ROWS_PLAN)
    if err:
        return jsonify({"error": err}), 400

    from src.dashboard_planner import plan_dashboard
    try:
        plan = plan_dashboard(df)
    except Exception as e:
        return jsonify({"error": f"Planning failed: {type(e).__name__}: {e}"}), 500

    try:
        response = _render_for_picks(plan["picks"])
    except Exception as e:
        return jsonify({"error": f"Render failed: {type(e).__name__}: {e}"}), 500

    # Attach plan metadata to the render response
    if isinstance(response, tuple):
        return response
    body = response.get_json()
    body["plan_source"] = plan["source"]
    body["picks"]       = plan["picks"]
    body["elapsed"]     = round(time.time() - t0, 2)
    return jsonify(body)
