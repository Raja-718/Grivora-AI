"""
src/chart_library.py
====================
Grivora AI chart library: 25+ chart types with adaptive data generation.

Each chart type is a pure function:
    builder(df, spec) -> dict (Chart.js compatible) OR {"error": str}

A "spec" is a small dict the LLM (or rule-based fallback) fills in to describe
which columns to use, e.g.:
    {"type": "bar", "x": "region", "y": "sales", "agg": "sum"}

The CHART_CATALOG at the bottom is the contract between the LLM planner
and this library — the LLM picks charts BY NAME from this catalog.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
# Safe helpers
# ────────────────────────────────────────────────────────────────

def _num(v):
    """Round floats; preserve None."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    try:
        return round(float(v), 4)
    except Exception:
        return None


def _col_exists(df: pd.DataFrame, name) -> bool:
    return bool(name) and name in df.columns


def _is_numeric(df: pd.DataFrame, col) -> bool:
    return _col_exists(df, col) and pd.api.types.is_numeric_dtype(df[col])


def _agg(series: pd.Series, func: str):
    f = (func or "sum").lower()
    if f == "mean":   return series.mean()
    if f == "median": return series.median()
    if f == "max":    return series.max()
    if f == "min":    return series.min()
    if f == "count":  return series.count()
    return series.sum()


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _top_n_values(series: pd.Series, n=15):
    """Top-n most frequent values in a categorical series."""
    vc = series.astype(str).value_counts().head(n)
    return vc.index.tolist(), vc.values.tolist()


def _group_agg(df, x_col, y_col, agg="sum", top_n=15, sort_desc=True):
    """Group df by x_col, aggregate y_col, return top_n rows."""
    if not _col_exists(df, x_col) or not _col_exists(df, y_col):
        return None
    grp = df.groupby(x_col, dropna=False)[y_col].agg(_agg_name(agg))
    if sort_desc:
        grp = grp.sort_values(ascending=False)
    grp = grp.head(top_n).reset_index()
    return grp


def _agg_name(agg: str) -> str:
    """Map our agg names to pandas-accepted aggregation names."""
    return {"sum":"sum","mean":"mean","median":"median","max":"max",
            "min":"min","count":"count"}.get((agg or "sum").lower(), "sum")


# ────────────────────────────────────────────────────────────────
# CHART BUILDERS  —  25+ types
# Every builder returns a dict with at least:
#   {"type": "<chart.js type>", ...payload...}
# or {"error": "reason"} on failure (so the caller can skip it).
# ────────────────────────────────────────────────────────────────

# 1. BAR ────────────────────────────────────────────────────────
def build_bar(df, spec):
    x, y, agg = spec.get("x"), spec.get("y"), spec.get("agg", "sum")
    top_n = spec.get("top_n", 15)
    if not _col_exists(df, x):
        return {"error": f"column {x!r} missing"}
    if not _is_numeric(df, y):
        return {"error": f"y column {y!r} is not numeric"}
    grp = _group_agg(df, x, y, agg, top_n)
    if grp is None or grp.empty:
        return {"error": "no data after grouping"}
    return {
        "type": "bar",
        "labels": grp[x].astype(str).tolist(),
        "values": [_num(v) for v in grp[y].tolist()],
        "x_label": x, "y_label": y, "agg": agg,
    }


# 2. HORIZONTAL BAR ─────────────────────────────────────────────
def build_hbar(df, spec):
    out = build_bar(df, spec)
    if "error" in out:
        return out
    out["type"] = "bar"
    out["orientation"] = "horizontal"
    return out


# 3. STACKED BAR ────────────────────────────────────────────────
def build_stacked_bar(df, spec):
    x, y, group, agg = spec.get("x"), spec.get("y"), spec.get("group"), spec.get("agg","sum")
    if not (_col_exists(df, x) and _col_exists(df, group) and _is_numeric(df, y)):
        return {"error": "missing x/group/y"}
    top_groups = df[group].astype(str).value_counts().head(6).index.tolist()
    sub = df[df[group].astype(str).isin(top_groups)]
    pivot = sub.pivot_table(index=x, columns=group, values=y, aggfunc=_agg_name(agg), fill_value=0)
    top_x = pivot.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    pivot = pivot.loc[top_x]
    return {
        "type": "bar",
        "stacked": True,
        "labels": [str(v) for v in pivot.index],
        "datasets": [
            {"label": str(g), "values": [_num(v) for v in pivot[g].tolist()]}
            for g in pivot.columns
        ],
        "x_label": x, "y_label": y, "group_label": group,
    }


# 4. GROUPED BAR (side-by-side) ─────────────────────────────────
def build_grouped_bar(df, spec):
    out = build_stacked_bar(df, spec)
    if "error" in out:
        return out
    out["stacked"] = False
    return out


# 5. LINE ──────────────────────────────────────────────────────
def build_line(df, spec):
    x, y, agg = spec.get("x"), spec.get("y"), spec.get("agg", "sum")
    if not _col_exists(df, x) or not _is_numeric(df, y):
        return {"error": "missing x or non-numeric y"}
    # Preserve x order as-is for time series, or sort if clearly numeric
    df2 = df[[x, y]].dropna()
    if pd.api.types.is_datetime64_any_dtype(df2[x]) or _looks_like_date(df2[x]):
        df2[x] = _to_datetime(df2[x])
        df2 = df2.dropna().sort_values(x)
        df2[x] = df2[x].dt.strftime("%Y-%m-%d")
    grp = df2.groupby(x, sort=False)[y].agg(_agg_name(agg)).reset_index()
    return {
        "type": "line",
        "labels": grp[x].astype(str).tolist(),
        "values": [_num(v) for v in grp[y].tolist()],
        "x_label": x, "y_label": y,
    }


# 6. AREA (same as line but filled) ─────────────────────────────
def build_area(df, spec):
    out = build_line(df, spec)
    if "error" in out:
        return out
    out["type"] = "line"
    out["fill"] = True
    return out


# 7. TIME SERIES (auto-resampled) ───────────────────────────────
def build_time_series(df, spec):
    x, y, agg = spec.get("x"), spec.get("y"), spec.get("agg", "sum")
    if not _col_exists(df, x) or not _is_numeric(df, y):
        return {"error": "missing x or non-numeric y"}
    df2 = df[[x, y]].copy()
    df2[x] = _to_datetime(df2[x])
    df2 = df2.dropna()
    if df2.empty:
        return {"error": "no valid dates"}
    span_days = (df2[x].max() - df2[x].min()).days
    freq = "YE" if span_days > 1460 else ("ME" if span_days > 90 else ("W" if span_days > 14 else "D"))
    fmt  = {"YE":"%Y","ME":"%Y-%m","W":"%Y-W%W","D":"%Y-%m-%d"}[freq]
    ts = df2.set_index(x)[y].resample(freq).agg(_agg_name(agg)).dropna()
    return {
        "type": "line",
        "fill": True,
        "labels": ts.index.strftime(fmt).tolist(),
        "values": [_num(v) for v in ts.tolist()],
        "x_label": x, "y_label": y, "frequency": freq,
    }


# 8. MULTI-LINE ─────────────────────────────────────────────────
def build_multi_line(df, spec):
    x, y, group = spec.get("x"), spec.get("y"), spec.get("group")
    if not (_col_exists(df, x) and _is_numeric(df, y) and _col_exists(df, group)):
        return {"error": "need x, y, group"}
    top_groups = df[group].astype(str).value_counts().head(5).index.tolist()
    df2 = df[df[group].astype(str).isin(top_groups)][[x, y, group]].dropna()
    if _looks_like_date(df2[x]):
        df2[x] = _to_datetime(df2[x]).dt.strftime("%Y-%m-%d")
    pivot = df2.pivot_table(index=x, columns=group, values=y, aggfunc="sum", fill_value=0)
    pivot = pivot.sort_index().head(40)
    return {
        "type": "line",
        "labels": [str(v) for v in pivot.index],
        "datasets": [
            {"label": str(g), "values": [_num(v) for v in pivot[g].tolist()]}
            for g in pivot.columns
        ],
        "x_label": x, "y_label": y, "group_label": group,
    }


# 9. PIE ───────────────────────────────────────────────────────
def build_pie(df, spec):
    x, y, agg = spec.get("x"), spec.get("y"), spec.get("agg", "sum")
    top_n = spec.get("top_n", 8)
    if not _col_exists(df, x):
        return {"error": "missing x"}
    if _is_numeric(df, y):
        grp = _group_agg(df, x, y, agg, top_n)
        labels, values = grp[x].astype(str).tolist(), [_num(v) for v in grp[y].tolist()]
    else:
        labels, values = _top_n_values(df[x], top_n)
    return {"type": "pie", "labels": labels, "values": values, "label": x}


# 10. DONUT ─────────────────────────────────────────────────────
def build_donut(df, spec):
    out = build_pie(df, spec)
    if "error" in out:
        return out
    out["type"] = "doughnut"
    out["cutout"] = "58%"
    return out


# 11. POLAR AREA ────────────────────────────────────────────────
def build_polar(df, spec):
    out = build_pie(df, spec)
    if "error" in out:
        return out
    out["type"] = "polarArea"
    return out


# 12. HISTOGRAM ─────────────────────────────────────────────────
def build_histogram(df, spec):
    col  = spec.get("x") or spec.get("col")
    bins = int(spec.get("bins", 20))
    if not _is_numeric(df, col):
        return {"error": "column is not numeric"}
    s = df[col].dropna()
    if s.empty:
        return {"error": "no data"}
    counts, edges = np.histogram(s, bins=min(bins, max(5, len(s)//5 or 5)))
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    return {
        "type": "bar",
        "variant": "histogram",
        "labels": [_num(c) for c in centers],
        "values": [int(v) for v in counts],
        "x_label": col, "y_label": "frequency",
    }


# 13. BOX PLOT (summary-based, since Chart.js has no native box) ──
def build_box(df, spec):
    col = spec.get("x") or spec.get("col")
    if not _is_numeric(df, col):
        return {"error": "column is not numeric"}
    s = df[col].dropna()
    if s.empty:
        return {"error": "no data"}
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    return {
        "type": "bar",
        "variant": "box",
        "col": col,
        "stats": {
            "min":    _num(s.min()),
            "q1":     _num(q1),
            "median": _num(s.median()),
            "mean":   _num(s.mean()),
            "q3":     _num(q3),
            "max":    _num(s.max()),
            "outliers_low":  _num(q1 - 1.5 * iqr),
            "outliers_high": _num(q3 + 1.5 * iqr),
            "n_outliers": int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()),
        },
    }


# 14. SCATTER ───────────────────────────────────────────────────
def build_scatter(df, spec, max_points=500):
    x, y = spec.get("x"), spec.get("y")
    if not (_is_numeric(df, x) and _is_numeric(df, y)):
        return {"error": "need two numeric columns"}
    df2 = df[[x, y]].dropna()
    if df2.empty:
        return {"error": "no data"}
    if len(df2) > max_points:
        df2 = df2.sample(max_points, random_state=42)
    return {
        "type": "scatter",
        "x": [_num(v) for v in df2[x].tolist()],
        "y": [_num(v) for v in df2[y].tolist()],
        "x_label": x, "y_label": y,
    }


# 15. BUBBLE (scatter with size dimension) ──────────────────────
def build_bubble(df, spec, max_points=300):
    x, y, r = spec.get("x"), spec.get("y"), spec.get("r")
    if not (_is_numeric(df, x) and _is_numeric(df, y) and _is_numeric(df, r)):
        return {"error": "need three numeric columns (x, y, r)"}
    df2 = df[[x, y, r]].dropna()
    if df2.empty:
        return {"error": "no data"}
    if len(df2) > max_points:
        df2 = df2.sample(max_points, random_state=42)
    # Normalize r to 4..24 px radius
    rmin, rmax = df2[r].min(), df2[r].max()
    spread = rmax - rmin or 1
    points = [
        {"x": _num(row[x]), "y": _num(row[y]),
         "r": round(4 + 20 * (row[r] - rmin) / spread, 2)}
        for _, row in df2.iterrows()
    ]
    return {"type": "bubble", "points": points,
            "x_label": x, "y_label": y, "r_label": r}


# 16. RADAR ─────────────────────────────────────────────────────
def build_radar(df, spec, max_cols=8):
    cols = spec.get("cols") or df.select_dtypes("number").columns.tolist()[:max_cols]
    cols = [c for c in cols if _is_numeric(df, c)][:max_cols]
    if len(cols) < 3:
        return {"error": "need at least 3 numeric columns"}
    stats = df[cols].describe()
    maxes = df[cols].abs().max().replace(0, 1)
    return {
        "type": "radar",
        "labels": cols,
        "datasets": [
            {"label": "Mean (normalized)", "values": [_num(stats.loc["mean", c] / maxes[c]) for c in cols]},
            {"label": "Median (normalized)", "values": [_num(stats.loc["50%",  c] / maxes[c]) for c in cols]},
        ],
    }


# 17. HEATMAP (correlation matrix, shown as colored grid via scatter) ──
def build_heatmap(df, spec, max_cols=10):
    cols = spec.get("cols") or df.select_dtypes("number").columns.tolist()[:max_cols]
    cols = [c for c in cols if _is_numeric(df, c)][:max_cols]
    if len(cols) < 2:
        return {"error": "need at least 2 numeric columns"}
    corr = df[cols].corr().round(3)
    return {
        "type": "heatmap",
        "cols": cols,
        "matrix": [[_num(v) for v in row] for row in corr.values.tolist()],
    }


# 18. TREEMAP (hierarchical value layout) ───────────────────────
def build_treemap(df, spec):
    x, y = spec.get("x"), spec.get("y")
    if not (_col_exists(df, x) and _is_numeric(df, y)):
        return {"error": "missing x or non-numeric y"}
    grp = df.groupby(x, dropna=False)[y].sum().sort_values(ascending=False).head(20)
    total = float(grp.sum()) or 1.0
    return {
        "type": "treemap",
        "nodes": [
            {"label": str(k), "value": _num(v), "pct": round(float(v)/total*100, 1)}
            for k, v in grp.items()
        ],
        "x_label": x, "y_label": y,
    }


# 19. FUNNEL (ordered categorical → decreasing values) ─────────
def build_funnel(df, spec):
    x, y = spec.get("x"), spec.get("y")
    if not _col_exists(df, x):
        return {"error": "missing x"}
    if _is_numeric(df, y):
        grp = df.groupby(x, dropna=False)[y].sum().sort_values(ascending=False).head(8)
        labels, values = grp.index.astype(str).tolist(), [_num(v) for v in grp.tolist()]
    else:
        labels, values = _top_n_values(df[x], 8)
    return {"type": "funnel", "labels": labels, "values": values, "label": x}


# 20. GAUGE (single metric with target) ────────────────────────
def build_gauge(df, spec):
    col = spec.get("x") or spec.get("col")
    if not _is_numeric(df, col):
        return {"error": "need numeric column"}
    s = df[col].dropna()
    if s.empty:
        return {"error": "no data"}
    return {
        "type": "gauge",
        "label": col,
        "value": _num(s.mean()),
        "min":   _num(s.min()),
        "max":   _num(s.max()),
        "median":_num(s.median()),
    }


# 21. KPI CARD (no chart — just a tile) ────────────────────────
def build_kpi_tile(df, spec):
    col = spec.get("x") or spec.get("col")
    agg = spec.get("agg", "sum")
    if not _is_numeric(df, col):
        return {"error": "need numeric column"}
    s = df[col].dropna()
    return {"type": "kpi_tile", "label": col, "value": _num(_agg(s, agg)), "agg": agg}


# 22. WATERFALL (change contributions) ─────────────────────────
def build_waterfall(df, spec):
    x, y = spec.get("x"), spec.get("y")
    if not (_col_exists(df, x) and _is_numeric(df, y)):
        return {"error": "missing x or non-numeric y"}
    grp = df.groupby(x, dropna=False)[y].sum().head(10)
    vals = grp.tolist()
    running = 0
    steps = []
    for label, v in zip(grp.index.astype(str), vals):
        steps.append({"label": str(label), "start": _num(running),
                      "end": _num(running + v), "delta": _num(v)})
        running += v
    return {"type": "waterfall", "steps": steps, "x_label": x, "y_label": y}


# 23. PARETO (bar + cumulative line) ───────────────────────────
def build_pareto(df, spec):
    x, y, agg = spec.get("x"), spec.get("y"), spec.get("agg","sum")
    if not _col_exists(df, x) or not _is_numeric(df, y):
        return {"error": "missing x or non-numeric y"}
    grp = df.groupby(x)[y].agg(_agg_name(agg)).sort_values(ascending=False).head(15)
    total = float(grp.sum()) or 1.0
    cumulative = grp.cumsum() / total * 100
    return {
        "type": "pareto",
        "labels": grp.index.astype(str).tolist(),
        "values": [_num(v) for v in grp.tolist()],
        "cumulative_pct": [round(float(v), 2) for v in cumulative.tolist()],
        "x_label": x, "y_label": y,
    }


# 24. DENSITY / KDE (approx via smoothed histogram) ────────────
def build_density(df, spec):
    col  = spec.get("x") or spec.get("col")
    bins = int(spec.get("bins", 30))
    if not _is_numeric(df, col):
        return {"error": "column is not numeric"}
    s = df[col].dropna()
    if s.empty:
        return {"error": "no data"}
    counts, edges = np.histogram(s, bins=bins, density=True)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    # Simple 3-point moving average for smoothing
    smoothed = np.convolve(counts, np.ones(3)/3, mode="same")
    return {
        "type": "line",
        "variant": "density",
        "fill": True,
        "labels": [_num(c) for c in centers],
        "values": [_num(v) for v in smoothed.tolist()],
        "x_label": col, "y_label": "density",
    }


# 25. CATEGORY COUNTS (no y_col needed) ────────────────────────
def build_cat_counts(df, spec):
    col, top_n = spec.get("x") or spec.get("col"), int(spec.get("top_n", 12))
    if not _col_exists(df, col):
        return {"error": "missing column"}
    labels, values = _top_n_values(df[col], top_n)
    return {
        "type": "bar",
        "variant": "counts",
        "labels": labels, "values": values,
        "x_label": col, "y_label": "count",
    }


# 26. STATS BAR (min/q1/median/mean/q3/max) ────────────────────
def build_stats_bar(df, spec):
    col = spec.get("x") or spec.get("col")
    if not _is_numeric(df, col):
        return {"error": "column is not numeric"}
    s = df[col].dropna()
    if s.empty:
        return {"error": "no data"}
    half = float(s.std()) * 0.5 if s.std() else 0
    median = float(s.median())
    return {
        "type": "bar",
        "variant": "stats",
        "labels": ["Min", "Q1≈", "Median", "Mean", "Q3≈", "Max"],
        "values": [_num(s.min()), _num(median - half), _num(median),
                   _num(s.mean()), _num(median + half), _num(s.max())],
        "col": col,
    }


# 27. DUAL-AXIS (two y-series) ─────────────────────────────────
def build_dual_axis(df, spec):
    x, y1, y2 = spec.get("x"), spec.get("y"), spec.get("y2")
    if not (_col_exists(df, x) and _is_numeric(df, y1) and _is_numeric(df, y2)):
        return {"error": "need x, y, y2"}
    df2 = df[[x, y1, y2]].dropna()
    if _looks_like_date(df2[x]):
        df2[x] = _to_datetime(df2[x]).dt.strftime("%Y-%m-%d")
    grp = df2.groupby(x, sort=False).sum(numeric_only=True).head(30)
    return {
        "type": "dual_axis",
        "labels": grp.index.astype(str).tolist(),
        "y1": {"label": y1, "values": [_num(v) for v in grp[y1].tolist()]},
        "y2": {"label": y2, "values": [_num(v) for v in grp[y2].tolist()]},
        "x_label": x,
    }


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(5)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.6


# ────────────────────────────────────────────────────────────────
# CATALOG — the contract with the LLM planner.
# Each entry:
#   "id":       machine name (what LLM returns)
#   "title":    display title
#   "needs":    list of required spec keys (x, y, group, ...)
#   "shape":    text hint describing when it's appropriate
#   "builder":  function to call
# ────────────────────────────────────────────────────────────────

CHART_CATALOG = {
    "bar":             {"title": "Bar Chart",              "needs": ["x", "y"],          "shape": "category vs numeric (sum/avg)",           "builder": build_bar},
    "hbar":            {"title": "Horizontal Bar",         "needs": ["x", "y"],          "shape": "many categories (>10) vs numeric",        "builder": build_hbar},
    "stacked_bar":     {"title": "Stacked Bar",            "needs": ["x", "y", "group"], "shape": "category vs numeric broken by group",     "builder": build_stacked_bar},
    "grouped_bar":     {"title": "Grouped Bar",            "needs": ["x", "y", "group"], "shape": "compare group values per category",       "builder": build_grouped_bar},
    "line":            {"title": "Line Chart",             "needs": ["x", "y"],          "shape": "ordered x (date/number) vs numeric",      "builder": build_line},
    "area":            {"title": "Area Chart",             "needs": ["x", "y"],          "shape": "cumulative / filled line",                "builder": build_area},
    "time_series":     {"title": "Time Series",            "needs": ["x", "y"],          "shape": "date vs numeric, auto-resampled",         "builder": build_time_series},
    "multi_line":      {"title": "Multi-Line",             "needs": ["x", "y", "group"], "shape": "time series split by group",              "builder": build_multi_line},
    "pie":             {"title": "Pie Chart",              "needs": ["x"],               "shape": "few categories (≤6) share of whole",      "builder": build_pie},
    "donut":           {"title": "Donut Chart",            "needs": ["x"],               "shape": "few categories share of whole, modern",   "builder": build_donut},
    "polar":           {"title": "Polar Area",             "needs": ["x"],               "shape": "category magnitudes, radial",             "builder": build_polar},
    "histogram":       {"title": "Histogram",              "needs": ["x"],               "shape": "distribution of one numeric column",      "builder": build_histogram},
    "density":         {"title": "Density Plot",           "needs": ["x"],               "shape": "smooth distribution of numeric column",   "builder": build_density},
    "box":             {"title": "Box Plot",               "needs": ["x"],               "shape": "numeric summary with outliers",           "builder": build_box},
    "stats_bar":       {"title": "Statistics Bar",         "needs": ["x"],               "shape": "min/q1/median/mean/q3/max for numeric",   "builder": build_stats_bar},
    "scatter":         {"title": "Scatter Plot",           "needs": ["x", "y"],          "shape": "two numeric columns, correlation",        "builder": build_scatter},
    "bubble":          {"title": "Bubble Chart",           "needs": ["x", "y", "r"],     "shape": "three numerics; r is size",               "builder": build_bubble},
    "radar":           {"title": "Radar Chart",            "needs": ["cols"],            "shape": "compare many numeric features (3-8)",     "builder": build_radar},
    "heatmap":         {"title": "Correlation Heatmap",    "needs": ["cols"],            "shape": "pairwise correlations of numerics",       "builder": build_heatmap},
    "treemap":         {"title": "Treemap",                "needs": ["x", "y"],          "shape": "hierarchical share of value",             "builder": build_treemap},
    "funnel":          {"title": "Funnel Chart",           "needs": ["x"],               "shape": "ordered drop-off / conversion",           "builder": build_funnel},
    "gauge":           {"title": "Gauge",                  "needs": ["x"],               "shape": "single metric with min/max range",        "builder": build_gauge},
    "kpi_tile":        {"title": "KPI Tile",               "needs": ["x"],               "shape": "single aggregated number",                "builder": build_kpi_tile},
    "waterfall":       {"title": "Waterfall Chart",        "needs": ["x", "y"],          "shape": "successive contributions",                "builder": build_waterfall},
    "pareto":          {"title": "Pareto Chart",           "needs": ["x", "y"],          "shape": "80/20 analysis: bars + cum line",         "builder": build_pareto},
    "cat_counts":      {"title": "Category Counts",        "needs": ["x"],               "shape": "frequency of values in a category",       "builder": build_cat_counts},
    "dual_axis":       {"title": "Dual-Axis Line",         "needs": ["x", "y", "y2"],    "shape": "two numerics on same x, different scales","builder": build_dual_axis},
}


def build_chart(df: pd.DataFrame, chart_id: str, spec: dict) -> dict:
    """Dispatch: take a chart_id + spec and return Chart.js-ready data."""
    if chart_id not in CHART_CATALOG:
        return {"error": f"unknown chart type: {chart_id}"}
    try:
        result = CHART_CATALOG[chart_id]["builder"](df, spec or {})
        if isinstance(result, dict) and "error" not in result:
            # Attach metadata for the frontend
            result["chart_id"] = chart_id
            result["chart_title"] = CHART_CATALOG[chart_id]["title"]
        return result
    except Exception as e:
        return {"error": f"build failed: {type(e).__name__}: {e}"}


def catalog_for_llm() -> list:
    """Return a compact list for the LLM planner prompt."""
    return [
        {"id": cid, "title": c["title"], "needs": c["needs"], "shape": c["shape"]}
        for cid, c in CHART_CATALOG.items()
    ]
