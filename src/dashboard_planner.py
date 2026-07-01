"""
src/dashboard_planner.py
========================
Decides *which* charts to put on the auto-dashboard, given a dataset profile.

Uses Gemini as the primary planner (one LLM call per dashboard build).
Falls back to deterministic rules if the LLM is unavailable or returns garbage.

The output is a list of chart specs like:
    [
        {"chart_id": "time_series", "spec": {"x": "date", "y": "revenue", "agg": "sum"},
         "title": "Revenue trend", "why": "daily revenue over 18 months"},
        {"chart_id": "bar", "spec": {"x": "region", "y": "revenue", "agg": "sum"}, ...},
        ...
    ]
Downstream code calls src.chart_library.build_chart(df, chart_id, spec) for each.
"""
from __future__ import annotations

import json
import re
from typing import List, Dict

import pandas as pd

from src.chart_library import CHART_CATALOG, catalog_for_llm


# ────────────────────────────────────────────────────────────────
# Dataset profile — small enough to fit in an LLM prompt easily
# ────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> Dict:
    """Produce a compact JSON-friendly profile for the planner."""
    num_cols = df.select_dtypes("number").columns.tolist()
    obj_cols = df.select_dtypes("object").columns.tolist()

    date_cols = []
    for col in df.columns:
        lower = str(col).lower()
        looks_datey = any(k in lower for k in ("date", "time", "year", "month", "day"))
        if looks_datey:
            try:
                parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                if parsed.notna().mean() > 0.5:
                    date_cols.append(col)
            except Exception:
                pass

    # Ratio of unique values for each categorical (low = good for grouping)
    cat_info = []
    for c in obj_cols[:20]:
        n_unique = int(df[c].nunique(dropna=True))
        cat_info.append({
            "name": c,
            "unique": n_unique,
            "top": str(df[c].mode().iloc[0]) if not df[c].mode().empty else None,
            "cardinality": "low" if n_unique <= 10 else ("mid" if n_unique <= 50 else "high"),
        })

    num_info = []
    for c in num_cols[:20]:
        s = df[c].dropna()
        if s.empty:
            continue
        num_info.append({
            "name": c,
            "min":  _r(s.min()),
            "max":  _r(s.max()),
            "mean": _r(s.mean()),
            "missing_pct": round(float(df[c].isna().mean() * 100), 1),
        })

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "numeric_cols":      num_info,
        "categorical_cols":  cat_info,
        "date_cols":         date_cols,
        "sample_rows":       _json_safe_sample(df, 3),
    }


def _r(v):
    try:
        v = float(v)
        return round(v, 3)
    except Exception:
        return None


def _json_safe_sample(df: pd.DataFrame, n=3):
    rows = df.head(n).astype(object).where(df.head(n).notna(), None).to_dict(orient="records")
    # Coerce non-serializable values (numpy types, timestamps) to strings
    safe = []
    for row in rows:
        safe.append({k: (None if v is None else (v if isinstance(v, (int, float, str, bool)) else str(v))) for k, v in row.items()})
    return safe


# ────────────────────────────────────────────────────────────────
# LLM planner
# ────────────────────────────────────────────────────────────────

_PLANNER_PROMPT = """You are Grivora AI's dashboard planner. Given a dataset profile, your job is to pick between 6 and 10 charts from a fixed catalog that together make the most informative, non-redundant dashboard for THIS specific dataset.

RULES:
1. Output ONLY valid JSON. No preamble, no code fences, no commentary.
2. Pick charts BY ID from the catalog. Never invent a chart_id.
3. Fill spec with EXACT column names from the profile. Never invent column names.
4. Each spec must include exactly the keys in that chart's "needs" list. "agg" is optional (defaults to "sum") and may be one of: sum, mean, median, max, min, count.
5. Match the chart to the data:
   - Date + numeric -> time_series or multi_line
   - Low-cardinality (<=6) category + numeric -> donut or pie
   - Mid-cardinality category + numeric -> bar or pareto
   - High-cardinality category -> hbar or cat_counts
   - Two numerics -> scatter; three -> bubble
   - Multiple numerics -> heatmap (correlation), radar (profile), stats_bar per column
   - Skewed numeric -> histogram or density
   - Single metric summary -> kpi_tile or gauge
6. Avoid redundancy: don't pick bar and pie for the same (x, y). Don't pick two histograms of the same column.
7. Aim for variety across the 6-10 picks: include a trend, a breakdown, a distribution, and a relationship chart whenever possible.
8. Include a short "why" (max 12 words) for each pick.

OUTPUT SCHEMA (exact shape):
{
  "picks": [
    {"chart_id": "<from catalog>", "spec": {...}, "title": "...", "why": "..."}
  ]
}

DATASET PROFILE:
<<PROFILE_JSON>>

CHART CATALOG:
<<CATALOG_JSON>>

Return the JSON now."""


def _call_llm(profile: Dict, timeout: float = 15.0) -> List[Dict] | None:
    """Ask Gemini to plan the dashboard. Returns list of picks or None on failure.

    `timeout` is the hard budget for the LLM call. If the API is slow/hung,
    we want to give up quickly and fall back to rules rather than hang the UI.
    """
    try:
        from llm.gemini_client import GeminiClient
    except Exception:
        return None

    prompt = (
        _PLANNER_PROMPT
        .replace("<<PROFILE_JSON>>", json.dumps(profile, default=str, indent=2))
        .replace("<<CATALOG_JSON>>", json.dumps(catalog_for_llm(), indent=2))
    )

    client = GeminiClient()
    raw = client.generate(prompt, timeout=timeout)
    if not raw or raw.startswith("[AI unavailable") or raw.startswith("Error from Gemini"):
        return None

    # Strip code fences if the model used them despite instructions
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find the first JSON object in the text
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    picks = data.get("picks") if isinstance(data, dict) else None
    if not isinstance(picks, list):
        return None

    return picks


# ────────────────────────────────────────────────────────────────
# Rule-based fallback planner (no LLM)
# ────────────────────────────────────────────────────────────────

def _rule_based_plan(profile: Dict) -> List[Dict]:
    """Deterministic fallback that picks sensible charts from the profile."""
    num_cols = [c["name"] for c in profile["numeric_cols"]]
    cat_cols = [c["name"] for c in profile["categorical_cols"]]
    date_cols = profile["date_cols"]

    low_card_cats = [c["name"] for c in profile["categorical_cols"] if c["cardinality"] == "low"]
    mid_card_cats = [c["name"] for c in profile["categorical_cols"] if c["cardinality"] == "mid"]
    high_card_cats = [c["name"] for c in profile["categorical_cols"] if c["cardinality"] == "high"]

    picks: List[Dict] = []

    value_col = num_cols[0] if num_cols else None
    date_col  = date_cols[0] if date_cols else None

    # 1. Time series if we have a date + value
    if date_col and value_col:
        picks.append({
            "chart_id": "time_series",
            "spec": {"x": date_col, "y": value_col, "agg": "sum"},
            "title": f"{value_col} over time",
            "why": "detected date + numeric column",
        })

    # 2. Bar: primary category vs value
    if value_col and (low_card_cats or mid_card_cats):
        cat = (low_card_cats + mid_card_cats)[0]
        picks.append({
            "chart_id": "bar",
            "spec": {"x": cat, "y": value_col, "agg": "sum"},
            "title": f"{value_col} by {cat}",
            "why": "category vs numeric breakdown",
        })

    # 3. Donut for low-cardinality category share
    if value_col and low_card_cats:
        cat = low_card_cats[0]
        # Don't duplicate with bar
        if not any(p["chart_id"] == "bar" and p["spec"].get("x") == cat for p in picks):
            picks.append({
                "chart_id": "donut",
                "spec": {"x": cat, "y": value_col, "agg": "sum"},
                "title": f"Share of {value_col} by {cat}",
                "why": "few categories, part-to-whole view",
            })
        elif len(low_card_cats) > 1:
            picks.append({
                "chart_id": "donut",
                "spec": {"x": low_card_cats[1], "y": value_col, "agg": "sum"},
                "title": f"Share of {value_col} by {low_card_cats[1]}",
                "why": "second category part-to-whole",
            })

    # 4. Histogram for value distribution
    if value_col:
        picks.append({
            "chart_id": "histogram",
            "spec": {"x": value_col},
            "title": f"{value_col} distribution",
            "why": "shape of the key numeric column",
        })

    # 5. Scatter for two numerics
    if len(num_cols) >= 2:
        picks.append({
            "chart_id": "scatter",
            "spec": {"x": num_cols[1], "y": num_cols[0]},
            "title": f"{num_cols[1]} vs {num_cols[0]}",
            "why": "relationship between two numerics",
        })

    # 6. Correlation heatmap if 3+ numerics
    if len(num_cols) >= 3:
        picks.append({
            "chart_id": "heatmap",
            "spec": {"cols": num_cols[:8]},
            "title": "Correlation between numeric columns",
            "why": "spot related variables at a glance",
        })

    # 7. Horizontal bar for high-cardinality category
    if value_col and high_card_cats:
        picks.append({
            "chart_id": "hbar",
            "spec": {"x": high_card_cats[0], "y": value_col, "agg": "sum", "top_n": 15},
            "title": f"Top 15 {high_card_cats[0]} by {value_col}",
            "why": "many categories need horizontal bars",
        })

    # 8. Cat counts if only categorical data
    if not num_cols and cat_cols:
        picks.append({
            "chart_id": "cat_counts",
            "spec": {"x": cat_cols[0], "top_n": 12},
            "title": f"Frequency of {cat_cols[0]}",
            "why": "no numerics; show category frequency",
        })

    # 9. Pareto if decent mid-card category + value
    if value_col and mid_card_cats:
        picks.append({
            "chart_id": "pareto",
            "spec": {"x": mid_card_cats[0], "y": value_col, "agg": "sum"},
            "title": f"Pareto of {value_col} by {mid_card_cats[0]}",
            "why": "80/20 analysis",
        })

    # 10. Radar if many numerics and no other multi-numeric chart
    if len(num_cols) >= 4 and not any(p["chart_id"] == "radar" for p in picks):
        picks.append({
            "chart_id": "radar",
            "spec": {"cols": num_cols[:6]},
            "title": "Numeric feature profile",
            "why": "compare scales across many numerics",
        })

    return picks[:10]


# ────────────────────────────────────────────────────────────────
# Validation — filter out LLM hallucinations before they crash builders
# ────────────────────────────────────────────────────────────────

def _validate_pick(pick: Dict, profile: Dict) -> Dict | None:
    """Return a clean pick or None to drop it."""
    if not isinstance(pick, dict):
        return None
    chart_id = pick.get("chart_id")
    spec     = pick.get("spec")
    if chart_id not in CHART_CATALOG or not isinstance(spec, dict):
        return None

    all_cols = (
        [c["name"] for c in profile["numeric_cols"]]
        + [c["name"] for c in profile["categorical_cols"]]
        + list(profile["date_cols"])
    )

    # Validate every required key points to a real column
    for key in CHART_CATALOG[chart_id]["needs"]:
        val = spec.get(key)
        if key == "cols":
            if not isinstance(val, list) or not val:
                return None
            spec["cols"] = [c for c in val if c in all_cols]
            if not spec["cols"]:
                return None
        else:
            if not val or val not in all_cols:
                return None

    return {
        "chart_id": chart_id,
        "spec": spec,
        "title": pick.get("title") or CHART_CATALOG[chart_id]["title"],
        "why":   pick.get("why") or "",
    }


def _dedupe(picks: List[Dict]) -> List[Dict]:
    """Remove duplicate (chart_id, x, y) picks while keeping order."""
    seen = set()
    out = []
    for p in picks:
        sig = (p["chart_id"], p["spec"].get("x"), p["spec"].get("y"), p["spec"].get("group"))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(p)
    return out


# ────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────

def plan_dashboard(df: pd.DataFrame) -> Dict:
    """
    Returns:
      {
        "profile": {...},
        "picks":   [validated picks],
        "source":  "llm" | "rules" | "llm+rules",
      }
    """
    profile = profile_dataset(df)

    source = "llm"
    raw_picks = None
    # Never let an LLM / prompt bug kill the dashboard — fall back to rules.
    try:
        raw_picks = _call_llm(profile)
    except Exception as e:
        print(f"[PLANNER] LLM call crashed, using rules. Reason: {type(e).__name__}: {e}",
              flush=True)
        raw_picks = None

    if not raw_picks:
        raw_picks = _rule_based_plan(profile)
        source = "rules"

    validated = [v for v in (_validate_pick(p, profile) for p in raw_picks) if v]

    # If the LLM picks were mostly invalid, top up with rule-based picks
    if source == "llm" and len(validated) < 4:
        filler = _rule_based_plan(profile)
        seen = {(p["chart_id"], p["spec"].get("x"), p["spec"].get("y")) for p in validated}
        for p in filler:
            sig = (p["chart_id"], p["spec"].get("x"), p["spec"].get("y"))
            if sig not in seen:
                validated.append(p)
                seen.add(sig)
            if len(validated) >= 8:
                break
        source = "llm+rules"

    # Last-resort guarantee: if even the rule-based plan came back empty
    # (very unusual dataset), return at least one generic chart so the UI works.
    if not validated:
        num = [c["name"] for c in profile["numeric_cols"]]
        cat = [c["name"] for c in profile["categorical_cols"]]
        if num:
            validated.append({
                "chart_id": "histogram",
                "spec": {"x": num[0]},
                "title": f"{num[0]} distribution",
                "why": "fallback: show distribution of first numeric column",
            })
        if cat:
            validated.append({
                "chart_id": "cat_counts",
                "spec": {"x": cat[0], "top_n": 12},
                "title": f"Frequency of {cat[0]}",
                "why": "fallback: category value counts",
            })
        source = "fallback"

    validated = _dedupe(validated)[:10]

    return {"profile": profile, "picks": validated, "source": source}
