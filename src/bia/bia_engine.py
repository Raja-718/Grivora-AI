# bia_engine.py — Business Intelligence & Analytics Core Engine
# Production-grade: chunk processing, query optimization, MySQL + CSV/Excel fallback
import os, json, warnings, hashlib, time
from datetime import datetime
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# ── Optional MySQL support ──────────────────────────────────────
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

# ── Optional scikit-learn for clustering ───────────────────────
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
BIA_STATE_FILE = "data/bia_state.json"
BIA_CACHE_DIR  = "data/bia_cache"
CHUNK_SIZE     = 10_000   # rows per chunk for large data processing


# ─────────────────────────────────────────────────────────────
#  1. STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────
def save_bia_state(state: dict):
    os.makedirs("data", exist_ok=True)
    try:
        with open(BIA_STATE_FILE, "w") as f:
            json.dump(state, f, default=str, indent=2)
    except Exception:
        pass

def load_bia_state() -> dict:
    if not os.path.exists(BIA_STATE_FILE):
        return {}
    try:
        with open(BIA_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
#  2. MySQL CONNECTION & MANAGEMENT
# ─────────────────────────────────────────────────────────────
def test_mysql_connection(host, port, user, password, database="") -> dict:
    if not MYSQL_AVAILABLE:
        return {"ok": False, "error": "mysql-connector-python not installed. Run: pip install mysql-connector-python"}
    try:
        cfg = dict(host=host, port=int(port), user=user, password=password, connection_timeout=5)
        if database:
            cfg["database"] = database
        conn = mysql.connector.connect(**cfg)
        conn.close()
        return {"ok": True, "message": f"Connected to MySQL at {host}:{port}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_mysql_connection(cfg: dict):
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql-connector-python not installed")
    return mysql.connector.connect(
        host=cfg["host"], port=int(cfg.get("port", 3306)),
        user=cfg["user"], password=cfg["password"],
        database=cfg.get("database", ""),
        connection_timeout=10
    )


def list_mysql_databases(cfg: dict) -> list:
    conn = get_mysql_connection(cfg)
    cur = conn.cursor()
    cur.execute("SHOW DATABASES")
    dbs = [r[0] for r in cur.fetchall()]
    conn.close()
    return dbs


def list_mysql_tables(cfg: dict) -> list:
    conn = get_mysql_connection(cfg)
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return tables


def get_table_schema(cfg: dict, table: str) -> dict:
    conn = get_mysql_connection(cfg)
    cur = conn.cursor()
    cur.execute(f"DESCRIBE `{table}`")
    cols = [{"field": r[0], "type": r[1], "null": r[2], "key": r[3], "default": r[4]} for r in cur.fetchall()]
    cur.execute(f"SELECT COUNT(*) FROM `{table}`")
    row_count = cur.fetchone()[0]
    conn.close()
    return {"columns": cols, "row_count": row_count, "table": table}


# ─────────────────────────────────────────────────────────────
#  3. ETL — EXTRACT / TRANSFORM / LOAD
# ─────────────────────────────────────────────────────────────
def extract_file(file_path: str, chunk_size: int = CHUNK_SIZE) -> dict:
    """Extract data from any supported file using the universal loader."""
    from src.data_loader import load_file as _load_file
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    file_size = os.path.getsize(file_path)
    size_mb = round(file_size / 1024 / 1024, 2)
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

    try:
        # Load first chunk for preview
        df_preview = _load_file(file_path, nrows=chunk_size)
        df_preview.columns = [str(c).strip().replace(" ", "_").lower() for c in df_preview.columns]

        # Estimate total rows cheaply
        total_rows = len(df_preview)
        try:
            if ext == "csv":
                with open(file_path, 'rb') as f:
                    total_rows = sum(1 for _ in f) - 1
            elif ext in ('xlsx', 'xls', 'xlsb', 'xlsm', 'ods'):
                df_full = _load_file(file_path, nrows=None)
                total_rows = len(df_full)
        except Exception:
            pass

        # Clean NaN in preview for JSON safety
        preview_records = df_preview.head(10).where(pd.notnull(df_preview.head(10)), None).to_dict(orient="records")

        return {
            "ok": True,
            "rows_loaded": len(df_preview),
            "total_rows": total_rows,
            "columns": df_preview.columns.tolist(),
            "size_mb": size_mb,
            "preview": preview_records,
            "dtypes": df_preview.dtypes.astype(str).to_dict(),
        }
    except Exception as e:
        return {"error": str(e)}


def transform_data(df: pd.DataFrame, options: dict) -> dict:
    """Apply transformations: clean, normalize, aggregate."""
    log = []
    original_rows = len(df)

    # 1. Drop duplicates
    if options.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            log.append(f"Removed {removed} duplicate rows")

    # 2. Handle missing values
    fill_strategy = options.get("fill_missing", "auto")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in num_cols:
        if df[col].isnull().any():
            if fill_strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif fill_strategy == "median" or fill_strategy == "auto":
                df[col].fillna(df[col].median(), inplace=True)
            elif fill_strategy == "zero":
                df[col].fillna(0, inplace=True)

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown", inplace=True)

    # 3. Normalize numeric columns
    if options.get("normalize", False):
        for col in num_cols:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
        log.append("Normalized numeric columns to [0,1]")

    # 4. Date parsing
    date_cols_found = []
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "time", "created", "updated", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                date_cols_found.append(col)
            except Exception:
                pass

    if date_cols_found:
        log.append(f"Parsed date columns: {date_cols_found}")

    log.append(f"Transformation complete: {len(df)} rows remaining (removed {original_rows - len(df)})")
    return {"df": df, "log": log, "num_cols": num_cols, "cat_cols": cat_cols, "date_cols": date_cols_found}


def load_to_mysql(df: pd.DataFrame, cfg: dict, table_name: str, if_exists: str = "replace") -> dict:
    """Load DataFrame into MySQL using chunked inserts."""
    if not MYSQL_AVAILABLE:
        return {"ok": False, "error": "MySQL not available"}
    try:
        from sqlalchemy import create_engine
        conn_str = f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg.get('port',3306)}/{cfg['database']}"
        engine = create_engine(conn_str)
        df.to_sql(table_name, engine, if_exists=if_exists, index=False, chunksize=1000)
        return {"ok": True, "rows": len(df), "table": table_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────
#  4. KPI ENGINE
# ─────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    """Auto-detect and compute business KPIs from dataframe."""
    kpis = {}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    VALUE_KEYS = ["sales","revenue","amount","total","profit","income","price","cost","value","score"]
    QTY_KEYS   = ["quantity","qty","count","units","orders","volume","num","sold"]
    DATE_KEYS  = ["date","time","created","month","year","period","timestamp"]

    value_col = next((c for c in num_cols if any(k in c.lower() for k in VALUE_KEYS)), num_cols[0] if num_cols else None)
    qty_col   = next((c for c in num_cols if any(k in c.lower() for k in QTY_KEYS) and c != value_col), None)
    date_col  = next((c for c in df.columns if any(k in c.lower() for k in DATE_KEYS)), None)
    cat_col   = cat_cols[0] if cat_cols else None

    if value_col:
        total = float(df[value_col].sum())
        avg   = float(df[value_col].mean())
        kpis["total_value"]   = {"label": f"Total {value_col.replace('_',' ').title()}", "value": round(total, 2), "fmt": "currency"}
        kpis["avg_value"]     = {"label": f"Avg {value_col.replace('_',' ').title()}", "value": round(avg, 2), "fmt": "currency"}
        kpis["max_value"]     = {"label": f"Peak {value_col.replace('_',' ').title()}", "value": round(float(df[value_col].max()), 2), "fmt": "currency"}
        kpis["min_value"]     = {"label": f"Min {value_col.replace('_',' ').title()}", "value": round(float(df[value_col].min()), 2), "fmt": "number"}

        # Growth (if time-series)
        if date_col:
            try:
                df2 = df.copy()
                df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
                df2 = df2.dropna(subset=[date_col]).sort_values(date_col)
                mid = len(df2) // 2
                first_half = float(df2.iloc[:mid][value_col].sum())
                second_half = float(df2.iloc[mid:][value_col].sum())
                if first_half > 0:
                    growth = round((second_half - first_half) / first_half * 100, 2)
                    kpis["growth_rate"] = {"label": "Period Growth", "value": growth, "fmt": "percent", "trend": "up" if growth > 0 else "down"}
            except Exception:
                pass

    kpis["total_records"] = {"label": "Total Records", "value": len(df), "fmt": "number"}
    kpis["total_columns"] = {"label": "Features", "value": len(df.columns), "fmt": "number"}

    if qty_col:
        kpis["total_qty"] = {"label": f"Total {qty_col.replace('_',' ').title()}", "value": int(df[qty_col].sum()), "fmt": "number"}

    if cat_col:
        kpis["unique_categories"] = {"label": f"Unique {cat_col.replace('_',' ').title()}", "value": int(df[cat_col].nunique()), "fmt": "number"}

    return {"kpis": kpis, "value_col": value_col, "qty_col": qty_col, "date_col": date_col, "cat_col": cat_col}


# ─────────────────────────────────────────────────────────────
#  5. AGGREGATION ENGINE
# ─────────────────────────────────────────────────────────────
def aggregate_data(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """Compute daily, weekly, monthly aggregations."""
    result = {}
    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.dropna(subset=[date_col])

        for freq, label in [("D", "daily"), ("W", "weekly"), ("ME", "monthly"), ("QE", "quarterly"), ("YE", "yearly")]:
            try:
                agg = df2.set_index(date_col)[value_col].resample(freq).sum().reset_index()
                fmt = {"D": "%Y-%m-%d", "W": "%Y-W%W", "ME": "%Y-%m", "QE": "%Y-Q%q", "YE": "%Y"}[freq]
                try:
                    labels = agg[date_col].dt.strftime(fmt).tolist()
                except Exception:
                    labels = agg[date_col].astype(str).tolist()
                result[label] = {"labels": labels, "values": [round(float(v), 2) for v in agg[value_col].tolist()]}
            except Exception:
                pass
    except Exception as e:
        result["error"] = str(e)
    return result


# ─────────────────────────────────────────────────────────────
#  6. EDA ENGINE
# ─────────────────────────────────────────────────────────────
def compute_eda(df: pd.DataFrame) -> dict:
    """Full EDA: stats, correlations, outliers, distributions."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Summary stats
    stats = {}
    for col in num_cols[:12]:
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        outlier_count = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        stats[col] = {
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "min":    round(float(s.min()), 4),
            "max":    round(float(s.max()), 4),
            "q1":     round(q1, 4),
            "q3":     round(q3, 4),
            "skew":   round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "missing": int(df[col].isnull().sum()),
            "outliers": outlier_count,
        }

    # Correlation matrix
    corr = {}
    if len(num_cols) >= 2:
        c = df[num_cols[:10]].corr().round(4)
        corr = {"cols": c.columns.tolist(), "matrix": c.values.tolist()}

    # Category distributions
    cat_dist = {}
    for col in cat_cols[:6]:
        vc = df[col].value_counts().head(10)
        cat_dist[col] = {"labels": vc.index.astype(str).tolist(), "counts": vc.values.tolist()}

    # Completeness
    total_cells = df.shape[0] * df.shape[1]
    missing_total = int(df.isnull().sum().sum())
    completeness = round((1 - missing_total / total_cells) * 100, 1) if total_cells else 100

    return {
        "stats": stats,
        "correlation": corr,
        "cat_distributions": cat_dist,
        "completeness": completeness,
        "missing_total": missing_total,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


# ─────────────────────────────────────────────────────────────
#  7. VISUALIZATION DATA ENGINE
# ─────────────────────────────────────────────────────────────
def build_chart_data(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str,
                     color_col: str = None, agg_func: str = "sum") -> dict:
    """Build chart-ready data for any chart type."""
    try:
        df2 = df[[c for c in [x_col, y_col, color_col] if c and c in df.columns]].copy()
        df2 = df2.dropna(subset=[x_col, y_col])

        AGG = {"sum": "sum", "mean": "mean", "count": "count", "max": "max", "min": "min"}
        fn = AGG.get(agg_func, "sum")

        if chart_type in ("bar", "line", "area"):
            if color_col and color_col in df2.columns:
                pivot = df2.groupby([x_col, color_col])[y_col].agg(fn).reset_index()
                series = {}
                for val in pivot[color_col].unique():
                    sub = pivot[pivot[color_col] == val]
                    series[str(val)] = sub[y_col].round(2).tolist()
                labels = pivot[x_col].unique().astype(str).tolist()
                return {"labels": labels, "series": series, "type": chart_type}
            else:
                agg = df2.groupby(x_col)[y_col].agg(fn).reset_index()
                agg = agg.sort_values(y_col, ascending=False).head(20)
                return {"labels": agg[x_col].astype(str).tolist(), "values": agg[y_col].round(2).tolist(), "type": chart_type}

        elif chart_type == "pie":
            agg = df2.groupby(x_col)[y_col].agg(fn).reset_index()
            agg = agg.sort_values(y_col, ascending=False).head(8)
            return {"labels": agg[x_col].astype(str).tolist(), "values": agg[y_col].round(2).tolist(), "type": "pie"}

        elif chart_type == "scatter":
            sample = df2.sample(min(500, len(df2)))
            return {"x": sample[x_col].tolist(), "y": sample[y_col].round(2).tolist(), "type": "scatter"}

        elif chart_type == "histogram":
            counts, edges = np.histogram(df2[y_col].dropna(), bins=20)
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
            return {"labels": [round(c, 2) for c in centers], "values": counts.tolist(), "type": "histogram"}

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include="number")
            cols = num_df.columns[:8].tolist()
            corr = num_df[cols].corr().round(2)
            return {"cols": cols, "matrix": corr.values.tolist(), "type": "heatmap"}

        else:
            return {"error": f"Unknown chart type: {chart_type}"}

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
#  8. ADVANCED ANALYTICS
# ─────────────────────────────────────────────────────────────
def customer_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """K-Means clustering on numeric features."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}
    try:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return {"error": "Need at least 2 numeric columns for clustering"}

        features = df[num_cols[:8]].dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)

        cluster_stats = []
        for i in range(n_clusters):
            mask = labels == i
            stats = {"cluster": i, "size": int(mask.sum()), "pct": round(float(mask.mean()) * 100, 1)}
            for col in num_cols[:4]:
                stats[col + "_mean"] = round(float(features[col][mask].mean()), 2)
            cluster_stats.append(stats)

        return {
            "ok": True,
            "n_clusters": n_clusters,
            "clusters": cluster_stats,
            "scatter_x": coords[:, 0].round(3).tolist()[:500],
            "scatter_y": coords[:, 1].round(3).tolist()[:500],
            "labels": labels.tolist()[:500],
            "explained_variance": round(float(pca.explained_variance_ratio_.sum()) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def detect_anomalies(df: pd.DataFrame, value_col: str) -> dict:
    """IQR-based anomaly detection."""
    try:
        s = df[value_col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        anomaly_mask = (df[value_col] < lower) | (df[value_col] > upper)
        anomaly_rows = df[anomaly_mask][[value_col]].head(50)

        return {
            "ok": True,
            "total_anomalies": int(anomaly_mask.sum()),
            "anomaly_pct": round(float(anomaly_mask.mean()) * 100, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "anomaly_values": anomaly_rows[value_col].round(2).tolist(),
            "normal_values": df[~anomaly_mask][value_col].sample(min(100, (~anomaly_mask).sum())).round(2).tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


def time_series_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 12) -> dict:
    """Simple trend-based time series forecast."""
    try:
        df2 = df[[date_col, value_col]].copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.dropna().sort_values(date_col)

        # Compute monthly aggregation
        monthly = df2.set_index(date_col)[value_col].resample("ME").mean().reset_index()
        monthly.columns = ["ds", "y"]

        if len(monthly) < 4:
            return {"error": "Need at least 4 time periods for forecasting"}

        # Linear trend + seasonal component
        n = len(monthly)
        x = np.arange(n)
        y = monthly["y"].values

        # Fit linear trend
        coeffs = np.polyfit(x, y, 1)
        trend_slope, trend_intercept = coeffs

        # Forecast future periods
        future_x = np.arange(n, n + periods)
        forecast_values = trend_slope * future_x + trend_intercept

        # Generate future dates
        last_date = monthly["ds"].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="ME")[1:]

        return {
            "ok": True,
            "history_dates": monthly["ds"].dt.strftime("%Y-%m").tolist(),
            "history_values": [round(float(v), 2) for v in y.tolist()],
            "forecast_dates": future_dates.strftime("%Y-%m").tolist(),
            "forecast_values": [round(float(v), 2) for v in forecast_values.tolist()],
            "trend_slope": round(float(trend_slope), 4),
            "periods": periods,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
#  9. INSIGHT GENERATOR (no LLM — rule-based fast insights)
# ─────────────────────────────────────────────────────────────
def generate_auto_insights(df: pd.DataFrame, kpi_data: dict) -> list:
    """Generate automatic business insights from data patterns."""
    insights = []
    kpis = kpi_data.get("kpis", {})
    value_col = kpi_data.get("value_col")
    date_col  = kpi_data.get("date_col")
    cat_col   = kpi_data.get("cat_col")

    # 1. Growth insight
    if "growth_rate" in kpis:
        g = kpis["growth_rate"]["value"]
        direction = "📈 increased" if g > 0 else "📉 decreased"
        insights.append({
            "type": "trend",
            "icon": "📈" if g > 0 else "📉",
            "title": f"{'Positive' if g > 0 else 'Negative'} Growth Detected",
            "body": f"{value_col} has {direction} by {abs(g):.1f}% comparing the first and second half of the dataset.",
            "severity": "success" if g > 0 else "warning",
        })

    # 2. Top category
    if cat_col and value_col and cat_col in df.columns and value_col in df.columns:
        top = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False)
        if len(top) > 0:
            top_name = str(top.index[0])
            top_pct = round(float(top.iloc[0] / top.sum() * 100), 1)
            insights.append({
                "type": "top_performer",
                "icon": "🏆",
                "title": f"Top Performer: {top_name}",
                "body": f"'{top_name}' accounts for {top_pct}% of total {value_col}. Consider prioritizing this segment.",
                "severity": "info",
            })

    # 3. Missing data warning
    missing_pct = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 1)
    if missing_pct > 5:
        insights.append({
            "type": "quality",
            "icon": "⚠️",
            "title": f"{missing_pct}% Missing Data Detected",
            "body": f"Your dataset has {missing_pct}% missing values. Consider imputation or data collection improvements.",
            "severity": "warning",
        })

    # 4. Outlier warning
    if value_col and value_col in df.columns:
        s = df[value_col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        if outliers > 0:
            insights.append({
                "type": "anomaly",
                "icon": "🔍",
                "title": f"{outliers} Outliers in {value_col}",
                "body": f"Found {outliers} statistical outliers ({round(outliers/len(s)*100,1)}% of data). These may skew averages.",
                "severity": "warning",
            })

    # 5. Dataset size insight
    if len(df) > 100_000:
        insights.append({"type": "scale", "icon": "🚀", "title": "Large Dataset", "body": f"Dataset has {len(df):,} rows — optimized chunk processing recommended.", "severity": "info"})

    # 6. Correlation insight
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) >= 2 and value_col:
        corr = df[num_cols].corr()[value_col].drop(value_col).abs().sort_values(ascending=False)
        if len(corr) > 0:
            top_corr_col = corr.index[0]
            top_corr_val = round(float(corr.iloc[0]), 3)
            if top_corr_val > 0.5:
                insights.append({
                    "type": "correlation",
                    "icon": "🔗",
                    "title": f"Strong Correlation Found",
                    "body": f"'{top_corr_col}' has a {top_corr_val:.2f} correlation with '{value_col}'. These metrics move together.",
                    "severity": "success",
                })

    return insights


# ─────────────────────────────────────────────────────────────
#  10. MONITORING & LOGGING
# ─────────────────────────────────────────────────────────────
def log_event(event_type: str, details: dict):
    """Append event to monitoring log."""
    os.makedirs("data", exist_ok=True)
    log_path = "data/bia_monitor.json"
    entry = {"ts": datetime.now().isoformat(), "type": event_type, **details}
    records = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                records = json.load(f)
        except Exception:
            records = []
    records.insert(0, entry)
    records = records[:500]  # keep last 500 events
    with open(log_path, "w") as f:
        json.dump(records, f, default=str)


def get_monitor_logs(limit: int = 50) -> list:
    log_path = "data/bia_monitor.json"
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path) as f:
            records = json.load(f)
        return records[:limit]
    except Exception:
        return []


def get_system_health() -> dict:
    """Return system health metrics."""
    import shutil
    disk = shutil.disk_usage(".")
    return {
        "disk_total_gb": round(disk.total / 1e9, 1),
        "disk_used_gb":  round(disk.used / 1e9, 1),
        "disk_free_gb":  round(disk.free / 1e9, 1),
        "disk_pct":      round(disk.used / disk.total * 100, 1),
        "mysql_available": MYSQL_AVAILABLE,
        "sklearn_available": SKLEARN_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  11. EXPORT ENGINE
# ─────────────────────────────────────────────────────────────
def export_to_csv(df: pd.DataFrame, filename: str) -> str:
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{filename}.csv"
    df.to_csv(path, index=False)
    return path


def build_pdf_report(kpis: dict, insights: list, filename: str = "bia_report") -> dict:
    """Build a text-based report (PDF generation requires reportlab)."""
    os.makedirs("uploads", exist_ok=True)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
        from reportlab.lib import colors

        path = f"uploads/{filename}.pdf"
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Grivora AI — Business Intelligence Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Key Performance Indicators", styles["Heading1"]))
        kpi_data = [[k.replace("_"," ").title(), str(v.get("value",""))] for k, v in kpis.items()]
        t = Table([["KPI", "Value"]] + kpi_data, colWidths=[300, 150])
        t.setStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#7c3aed")),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                    ("GRID", (0,0), (-1,-1), 0.5, colors.grey)])
        story.append(t)
        story.append(Spacer(1, 20))

        story.append(Paragraph("AI Insights", styles["Heading1"]))
        for ins in insights:
            story.append(Paragraph(f"• {ins['title']}: {ins['body']}", styles["Normal"]))
            story.append(Spacer(1, 6))

        doc.build(story)
        return {"ok": True, "path": path}
    except ImportError:
        # Fallback: plain text report
        path = f"uploads/{filename}.txt"
        with open(path, "w") as f:
            f.write("Grivora AI — Business Intelligence Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("=== KPIs ===\n")
            for k, v in kpis.items():
                f.write(f"{v.get('label','')}: {v.get('value','')}\n")
            f.write("\n=== Insights ===\n")
            for ins in insights:
                f.write(f"• {ins['title']}: {ins['body']}\n")
        return {"ok": True, "path": path, "note": "PDF not available, exported as TXT. Install: pip install reportlab"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
