"""
src/experiments.py
==================
SQLite-backed registry of every ML training run.

Every time a user trains a model (manual, auto-suggest, or Auto-ML) we store
a full record of what happened. This lets users:
  * browse their recent experiments in the sidebar
  * star the best ones
  * open a past model card to see all its details
  * compare two or more models side-by-side
  * jump back to a past result card without re-training

Schema-first: one SQLite DB file at `experiments.db`, WAL mode, single table.
No ORM. Stdlib only.

Public API:
  save_experiment(...)     -> int id
  list_experiments(user_id, limit=50, filters=...) -> list[dict]
  get_experiment(exp_id, user_id) -> dict | None
  delete_experiment(exp_id, user_id) -> bool
  toggle_star(exp_id, user_id) -> bool
  compare_experiments(exp_ids, user_id) -> list[dict]
  experiments_summary(user_id) -> dict  (counts for the sidebar header)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────
# Connection management
# ─────────────────────────────────────────────────────────────
_DB_PATH = os.environ.get("GRIVORA_EXPERIMENTS_DB", "experiments.db")
_LOCK = threading.RLock()
_INIT_DONE = False


def _conn() -> sqlite3.Connection:
    """Thread-safe SQLite connection in WAL mode."""
    c = sqlite3.connect(_DB_PATH, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


@contextmanager
def _cursor():
    with _LOCK:
        c = _conn()
        try:
            yield c
        finally:
            c.close()


def _init_db():
    """Create schema on first call. Idempotent."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    with _cursor() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL DEFAULT 'anonymous',
            created_at      TEXT    NOT NULL,
            dataset_name    TEXT    NOT NULL,
            n_rows          INTEGER NOT NULL DEFAULT 0,
            n_cols          INTEGER NOT NULL DEFAULT 0,
            target_col      TEXT    NOT NULL,
            task            TEXT    NOT NULL,
            problem_type    TEXT,
            algo_id         TEXT    NOT NULL,
            algo_name       TEXT    NOT NULL,
            algo_family     TEXT,
            primary_metric  TEXT    NOT NULL,
            primary_score   REAL,
            metrics_json    TEXT    NOT NULL,
            cv_metrics_json TEXT,
            feat_imp_json   TEXT,
            params_json     TEXT,
            logs_json       TEXT,
            model_path      TEXT,
            training_time_s REAL    DEFAULT 0,
            is_imbalanced   INTEGER DEFAULT 0,
            starred         INTEGER DEFAULT 0,
            notes           TEXT    DEFAULT '',
            source          TEXT    DEFAULT 'manual',
            y_true_json     TEXT    DEFAULT '',
            y_pred_json     TEXT    DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_user_created
            ON experiments(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_user_starred
            ON experiments(user_id, starred DESC, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_user_algo
            ON experiments(user_id, algo_id);
        """)
        # ── Schema migration: add y_true_json / y_pred_json to existing DBs ──
        # Safe: ALTER TABLE ADD COLUMN is a no-op if the column already exists
        # (we catch the "duplicate column name" error).
        for col in ("y_true_json", "y_pred_json"):
            try:
                c.execute(f"ALTER TABLE experiments ADD COLUMN {col} TEXT DEFAULT ''")
            except Exception:
                pass  # Column already exists — fine
    _INIT_DONE = True


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _json_dump(v: Any) -> str:
    if v is None:
        return ""
    try:
        return json.dumps(v, default=str)
    except Exception:
        return ""


def _json_load(s: Optional[str], default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _primary_metric(problem_type: Optional[str], metrics: dict) -> tuple[str, Optional[float]]:
    """Pick the one number that ranks this model."""
    if problem_type == "classification":
        for key in ("f1_macro", "f1", "balanced_accuracy", "accuracy"):
            if key in metrics:
                return key, float(metrics[key])
    else:
        for key in ("r2", "rmse", "mae"):
            if key in metrics:
                return key, float(metrics[key])
    return "score", None


def _row_to_dict(row: sqlite3.Row, *, light: bool = False) -> dict:
    """Convert a DB row into the wire format the frontend expects.

    light=True skips the heavy JSON columns (metrics, logs, etc.) for list views.
    """
    out = dict(row)
    # Common cheap fields
    out["starred"] = bool(out.get("starred", 0))
    out["is_imbalanced"] = bool(out.get("is_imbalanced", 0))
    if light:
        # Drop heavy columns for list views
        for k in ("metrics_json", "cv_metrics_json", "feat_imp_json", "params_json", "logs_json"):
            out.pop(k, None)
        # Pull summary metrics the sidebar shows
        metrics = _json_load(row["metrics_json"], {})
        out["metrics_preview"] = {k: v for k, v in metrics.items()
                                   if k != "confusion_matrix"}
        return out
    # Full detail view: expand JSONs
    out["metrics"]            = _json_load(row["metrics_json"], {})
    out["cv_metrics"]         = _json_load(row["cv_metrics_json"], {})
    out["feature_importance"] = _json_load(row["feat_imp_json"], [])
    out["params"]             = _json_load(row["params_json"], {})
    out["logs"]               = _json_load(row["logs_json"], [])
    # y_true / y_pred for reconstructing diagnostic charts on past experiments
    out["y_true"]             = _json_load(out.get("y_true_json", ""), [])
    out["y_pred"]             = _json_load(out.get("y_pred_json", ""), [])
    # Drop the raw json column names from the output
    for k in ("metrics_json", "cv_metrics_json", "feat_imp_json", "params_json",
              "logs_json", "y_true_json", "y_pred_json"):
        out.pop(k, None)
    return out


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def save_experiment(
    *,
    user_id: str,
    dataset_name: str,
    n_rows: int,
    n_cols: int,
    target_col: str,
    task: str,
    problem_type: Optional[str],
    algo_id: str,
    algo_name: str,
    algo_family: str = "",
    metrics: dict,
    cv_metrics: Optional[dict] = None,
    feature_importance: Optional[list] = None,
    params: Optional[dict] = None,
    logs: Optional[list] = None,
    model_path: Optional[str] = None,
    training_time_s: float = 0,
    is_imbalanced: bool = False,
    source: str = "manual",
    y_true: Optional[list] = None,
    y_pred: Optional[list] = None,
) -> int:
    """Insert a new experiment row. Returns the new id.

    y_true / y_pred are capped at 500 entries to prevent DB bloat while
    still providing enough data for diagnostic charts (confusion matrix,
    predicted-vs-actual scatter, residual histograms).
    """
    _init_db()
    primary_name, primary_value = _primary_metric(problem_type, metrics or {})

    # Cap y_true / y_pred at 500 rows to avoid oversized DB rows
    _MAX_DIAG_ROWS = 500
    y_true_capped = (y_true or [])[:_MAX_DIAG_ROWS]
    y_pred_capped = (y_pred or [])[:_MAX_DIAG_ROWS]

    row = (
        user_id or "anonymous",
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        dataset_name or "(unknown)",
        int(n_rows or 0), int(n_cols or 0),
        target_col or "",
        task or "",
        problem_type,
        algo_id, algo_name, algo_family,
        primary_name, primary_value,
        _json_dump(metrics or {}),
        _json_dump(cv_metrics or {}),
        _json_dump(feature_importance or []),
        _json_dump(params or {}),
        _json_dump(logs or []),
        model_path or "",
        float(training_time_s or 0),
        1 if is_imbalanced else 0,
        0,  # starred
        "",  # notes
        source,
        _json_dump(y_true_capped),
        _json_dump(y_pred_capped),
    )
    with _cursor() as c:
        cur = c.execute("""
            INSERT INTO experiments (
                user_id, created_at, dataset_name, n_rows, n_cols,
                target_col, task, problem_type,
                algo_id, algo_name, algo_family,
                primary_metric, primary_score,
                metrics_json, cv_metrics_json, feat_imp_json, params_json, logs_json,
                model_path, training_time_s, is_imbalanced, starred, notes, source,
                y_true_json, y_pred_json
            ) VALUES (?, ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?,
                      ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?,
                      ?, ?)
        """, row)
        return int(cur.lastrowid)


def list_experiments(
    user_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    task: Optional[str] = None,
    algo_id: Optional[str] = None,
    starred_only: bool = False,
    search: Optional[str] = None,
) -> list[dict]:
    """Return light-weight experiment rows for the sidebar."""
    _init_db()
    clauses = ["user_id = ?"]
    params: list[Any] = [user_id or "anonymous"]
    if task:
        clauses.append("task = ?"); params.append(task)
    if algo_id:
        clauses.append("algo_id = ?"); params.append(algo_id)
    if starred_only:
        clauses.append("starred = 1")
    if search:
        clauses.append("(dataset_name LIKE ? OR algo_name LIKE ? OR target_col LIKE ?)")
        s = f"%{search}%"
        params.extend([s, s, s])
    params.extend([limit, offset])
    sql = f"""
        SELECT * FROM experiments
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    with _cursor() as c:
        rows = c.execute(sql, params).fetchall()
    return [_row_to_dict(r, light=True) for r in rows]


def get_experiment(exp_id: int, user_id: str) -> Optional[dict]:
    """Return full detail for a single experiment owned by this user."""
    _init_db()
    with _cursor() as c:
        row = c.execute(
            "SELECT * FROM experiments WHERE id = ? AND user_id = ?",
            (int(exp_id), user_id or "anonymous"),
        ).fetchone()
    return _row_to_dict(row, light=False) if row else None


def delete_experiment(exp_id: int, user_id: str, *, delete_pickle: bool = False) -> bool:
    """Delete an experiment row. Optionally also unlink the .pkl file."""
    _init_db()
    with _cursor() as c:
        row = c.execute(
            "SELECT model_path FROM experiments WHERE id = ? AND user_id = ?",
            (int(exp_id), user_id or "anonymous"),
        ).fetchone()
        if not row:
            return False
        c.execute(
            "DELETE FROM experiments WHERE id = ? AND user_id = ?",
            (int(exp_id), user_id or "anonymous"),
        )
        if delete_pickle and row["model_path"]:
            try:
                if os.path.exists(row["model_path"]):
                    os.remove(row["model_path"])
            except Exception:
                pass  # Don't fail the whole delete over a stray pickle
    return True


def toggle_star(exp_id: int, user_id: str) -> bool:
    """Flip the starred flag. Returns the new value."""
    _init_db()
    with _cursor() as c:
        row = c.execute(
            "SELECT starred FROM experiments WHERE id = ? AND user_id = ?",
            (int(exp_id), user_id or "anonymous"),
        ).fetchone()
        if not row:
            return False
        new_val = 0 if row["starred"] else 1
        c.execute(
            "UPDATE experiments SET starred = ? WHERE id = ? AND user_id = ?",
            (new_val, int(exp_id), user_id or "anonymous"),
        )
        return bool(new_val)


def update_notes(exp_id: int, user_id: str, notes: str) -> bool:
    _init_db()
    with _cursor() as c:
        cur = c.execute(
            "UPDATE experiments SET notes = ? WHERE id = ? AND user_id = ?",
            (notes or "", int(exp_id), user_id or "anonymous"),
        )
        return cur.rowcount > 0


def compare_experiments(exp_ids: list[int], user_id: str) -> list[dict]:
    """Full detail for each experiment id, in the order given. Skips ids not found."""
    _init_db()
    if not exp_ids:
        return []
    placeholders = ",".join("?" for _ in exp_ids)
    params = [user_id or "anonymous", *[int(i) for i in exp_ids]]
    with _cursor() as c:
        rows = c.execute(
            f"SELECT * FROM experiments WHERE user_id = ? AND id IN ({placeholders})",
            params,
        ).fetchall()
    by_id = {r["id"]: _row_to_dict(r, light=False) for r in rows}
    # Preserve the caller's requested order
    return [by_id[int(i)] for i in exp_ids if int(i) in by_id]


def experiments_summary(user_id: str) -> dict:
    """Lightweight counts + best-score for the sidebar header."""
    _init_db()
    with _cursor() as c:
        total = c.execute(
            "SELECT COUNT(*) AS n FROM experiments WHERE user_id = ?",
            (user_id or "anonymous",),
        ).fetchone()["n"]
        starred = c.execute(
            "SELECT COUNT(*) AS n FROM experiments WHERE user_id = ? AND starred = 1",
            (user_id or "anonymous",),
        ).fetchone()["n"]
        # Best by primary score (classification ranks high, regression ranks high for r2)
        best_row = c.execute(
            """SELECT algo_name, primary_metric, primary_score
               FROM experiments
               WHERE user_id = ? AND primary_score IS NOT NULL
               ORDER BY primary_score DESC LIMIT 1""",
            (user_id or "anonymous",),
        ).fetchone()
        recent_algos = c.execute(
            """SELECT DISTINCT algo_name FROM experiments
               WHERE user_id = ? ORDER BY created_at DESC LIMIT 5""",
            (user_id or "anonymous",),
        ).fetchall()
    return {
        "total":   int(total),
        "starred": int(starred),
        "best":    dict(best_row) if best_row else None,
        "recent_algos": [r["algo_name"] for r in recent_algos],
    }


# ─────────────────────────────────────────────────────────────
# Back-compat helper used by pma_engine — returns the saved id
# so callers can include it in API responses.
# ─────────────────────────────────────────────────────────────

def save_from_training_result(
    training_result: dict,
    *,
    user_id: str,
    dataset_name: str,
    source: str = "manual",
    training_time_s: float = 0,
    params: Optional[dict] = None,
) -> int:
    """Convenience: take the dict returned by pma.train_model → persist it.

    Automatically captures y_true / y_pred from the training result so
    past experiment model cards can reconstruct diagnostic charts.
    """
    from src.algorithms import algorithm
    algo = algorithm(training_result.get("algo_id")) or {}
    return save_experiment(
        user_id=user_id,
        dataset_name=dataset_name,
        n_rows=training_result.get("n_rows", 0),
        n_cols=training_result.get("n_cols", 0),
        target_col=training_result.get("target_col", ""),
        task=training_result.get("task", ""),
        problem_type=training_result.get("problem_type"),
        algo_id=training_result.get("algo_id", ""),
        algo_name=training_result.get("algo_name", ""),
        algo_family=algo.get("family", ""),
        metrics=training_result.get("metrics", {}),
        cv_metrics=training_result.get("cv_metrics", {}),
        feature_importance=training_result.get("feature_importance", []),
        params=params or {},
        logs=training_result.get("logs", []),
        model_path=training_result.get("model_path", ""),
        training_time_s=training_time_s,
        is_imbalanced=training_result.get("is_imbalanced", False),
        source=source,
        y_true=training_result.get("y_true", []),
        y_pred=training_result.get("y_pred", []),
    )
