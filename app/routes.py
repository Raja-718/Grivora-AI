# coding: utf-8
"""
app/routes.py  --  Grivora AI main route blueprint.
The /api/auto-dashboard endpoint delegates to app/auto_dashboard_route.py
for a fast, clean implementation.
"""
from flask import Blueprint, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
from agents.orchestrator import Orchestrator
from src.data_loader import load_file
from src import pma_engine as pma
from src.ml_suggester import suggest_algorithms, profile_dataset, infer_task
from src.algorithms import (
    ALL_ALGORITHMS, algorithms_for_task, TASK_LABELS, available_deps,
    filter_by_data,
)
from src.automl import run_automl
from src import experiments as expt
from src.bia import bia_engine as bia
from agents.ml_agent import MLAgent
from app import limiter
import os, json, pickle
import pandas as pd
import numpy as np

_ml_agent = MLAgent()

main = Blueprint('main', __name__)
orchestrator = Orchestrator()


def _user_id() -> str:
    """Return a stable per-user id for experiment ownership. Falls back to
    'anonymous' when no auth is in effect (single-user dev mode)."""
    return (session.get('user_id') or session.get('user_email') or 'anonymous')


def _dataset_name() -> str:
    """Friendly dataset name from the uploaded file path, for experiment rows."""
    fp = session.get('pma_file') or session.get('uploaded_file') or ''
    return os.path.basename(fp) if fp else '(unknown)'

# ---------- keyword map -------------------------------------------------
KEYWORD_MAP = {
    'value':    ['sales','amount','revenue','price','total','income','value','profit',
                 'earn','turnover','gross','net','cost','expense','budget','salary',
                 'wage','fee','marks','score','grade','gpa','percentage','percent',
                 'rating','points','likes','followers','views','clicks','impressions',
                 'reach','engagement','cases','deaths','patients','count','number',
                 'qty','quantity','units','attendance','hours','duration','age',
                 'weight','height','temperature','population','votes','donations',
                 'fund','investment','return','loss','orders','tickets','calls',
                 'downloads','installs','purchases','transactions'],
    'qty':      ['qty','quantity','units','count','volume','num','sold','orders',
                 'items','pieces','students','employees','patients','users','members',
                 'followers','likes','views','clicks','cases','records','entries',
                 'rows','responses'],
    'date':     ['date','time','day','month','year','period','week','quarter',
                 'timestamp','created','updated','joined','enrolled','admitted',
                 'posted','published','submitted','recorded','reported','born',
                 'dob','start','end','deadline'],
    'category': ['category','cat','type','product','class','group','segment','kind',
                 'line','brand','dept','department','division','subject','course',
                 'stream','major','field','sector','industry','domain','topic','tag',
                 'label','status','stage','gender','grade','level','rank','tier',
                 'plan','package','model','series','platform','channel','source',
                 'medium','campaign','post_type','content_type'],
    'region':   ['region','area','zone','location','city','state','country',
                 'territory','district','branch','market','store','campus','school',
                 'college','hospital','office','site','address','place','village',
                 'town','province','county','ward','block','cluster'],
    'person':   ['rep','agent','employee','staff','person','salesperson','seller',
                 'assigned','owner','handler','manager','teacher','professor',
                 'doctor','nurse','student','user','author','creator','by','name',
                 'handled_by','assigned_to','reported_by'],
    'channel':  ['channel','platform','source','medium','store','outlet','mode',
                 'method','via','network','social','site','app','device','browser',
                 'os','carrier'],
    'segment':  ['customer_type','customer','client','buyer','segment','tier',
                 'membership','student_type','employee_type','patient_type',
                 'user_type','account_type','subscription','plan'],
    'rate':     ['discount','disc','rebate','reduction','off','rate','ratio','pct',
                 'percent','percentage','tax','commission','margin','growth','change',
                 'pass_rate','fail_rate','attendance_rate','conversion','ctr','bounce'],
}


def detect_col(columns, key):
    keywords = KEYWORD_MAP.get(key, [])
    for col in columns:
        if any(k in col.lower().replace(' ', '_') for k in keywords):
            return col
    return None


def smart_detect_all(df):
    num_cols  = df.select_dtypes(include='number').columns.tolist()
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    value_col = detect_col(num_cols, 'value') or (num_cols[0] if num_cols else None)
    qty_col   = detect_col(num_cols, 'qty')
    if qty_col == value_col:
        qty_col = next((c for c in num_cols if c != value_col), None)
    rate_col    = detect_col(num_cols, 'rate')
    date_col    = detect_col(df.columns.tolist(), 'date')
    cat_col     = detect_col(cat_cols, 'category') or (cat_cols[0] if cat_cols else None)
    region_col  = detect_col(cat_cols, 'region')
    person_col  = detect_col(cat_cols, 'person')
    channel_col = detect_col(cat_cols, 'channel')
    segment_col = detect_col(cat_cols, 'segment')
    cat2_col    = next(
        (c for c in [segment_col, channel_col, person_col] if c and c != cat_col), None
    )
    return {
        'value': value_col, 'qty': qty_col, 'rate': rate_col, 'date': date_col,
        'category': cat_col, 'region': region_col, 'person': person_col,
        'channel': channel_col, 'segment': segment_col, 'cat2': cat2_col,
        'num_cols': num_cols, 'cat_cols': cat_cols, 'all_cols': df.columns.tolist(),
    }


# ========================================================================
#  PAGE ROUTES
# ========================================================================

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analysis')
def analysis():
    return render_template('analysis.html')

@main.route('/predict')
def predict():
    return render_template('predict.html')

@main.route('/bi')
def bi():
    return render_template('bi.html')

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/upload')
def upload():
    return render_template('upload.html')

@main.route('/chat')
def chat():
    return render_template('chat.html')

@main.route('/data-preview')
def data_preview():
    return render_template('data_preview.html')

@main.route('/analysis-dashboard')
def analysis_dashboard():
    return render_template('analysis_dashboard.html')

@main.route('/auto-dashboard')
def auto_dashboard():
    return render_template('auto_dashboard.html')


# ========================================================================
#  CORE API ENDPOINTS
# ========================================================================

@main.route('/api/upload', methods=['POST'])
@limiter.limit("20 per minute")
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # ── Sanitize filename to prevent path traversal ────────────────
    safe_name = secure_filename(file.filename or '')
    if not safe_name:
        return jsonify({'error': 'Invalid filename'}), 400

    # ── Enforce extension allowlist (universal loader formats) ─────
    ALLOWED_EXTS = {'csv', 'tsv', 'tab', 'txt', 'dat',
                    'xlsx', 'xls', 'xlsb', 'xlsm', 'ods',
                    'json', 'parquet', 'pq', 'xml'}
    ext = safe_name.rsplit('.', 1)[-1].lower() if '.' in safe_name else ''
    if ext not in ALLOWED_EXTS:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', safe_name)
    file.save(filepath)
    session['uploaded_file'] = filepath
    return jsonify({'message': 'File uploaded successfully', 'path': filepath,
                    'filename': safe_name})


@main.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour")
def chat_api():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    file_path = session.get('uploaded_file') or data.get('file_path')
    response = orchestrator.run(user_message, file_path)
    return jsonify({'response': response})


@main.route('/api/session-file', methods=['GET'])
def get_session_file():
    return jsonify({'file_path': session.get('uploaded_file', None)})


# ========================================================================
#  DATA PREVIEW & EDIT
# ========================================================================

@main.route('/api/preview-data', methods=['POST'])
def preview_data():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = df.select_dtypes(include='number').columns.tolist()
        total_cells   = df.shape[0] * df.shape[1]
        total_missing = int(df.isnull().sum().sum())
        completeness  = round((1 - total_missing / total_cells) * 100, 1) if total_cells else 100
        duplicates    = int(df.duplicated().sum())

        col_profiles = []
        for col in df.columns:
            miss_count = int(df[col].isnull().sum())
            miss_pct   = round(miss_count / len(df) * 100, 1)
            is_num     = col in num_cols
            p = {
                'name': col, 'dtype': 'number' if is_num else 'text',
                'missing_count': miss_count, 'missing_pct': miss_pct,
                'unique': int(df[col].nunique()),
            }
            if is_num:
                not_all_null = not df[col].isnull().all()
                p.update({
                    'min':    round(float(df[col].min()),    2) if not_all_null else None,
                    'max':    round(float(df[col].max()),    2) if not_all_null else None,
                    'mean':   round(float(df[col].mean()),   2) if not_all_null else None,
                    'median': round(float(df[col].median()), 2) if not_all_null else None,
                    'std':    round(float(df[col].std()),    2) if not_all_null else None,
                })
            else:
                top_val = (df[col].value_counts().index[0]
                           if not df[col].isnull().all() and len(df[col].dropna()) > 0
                           else None)
                p['top'] = str(top_val) if top_val is not None else None
            col_profiles.append(p)

        profile = {'completeness': completeness, 'total_missing': total_missing,
                   'duplicates': duplicates, 'columns': col_profiles}
        df_safe = df.where(pd.notnull(df), None)
        rows = df_safe.head(500).to_dict(orient='records')
        for row in rows:
            for k, v in row.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row[k] = None
        return jsonify({'columns': df.columns.tolist(), 'rows': rows,
                        'total_rows': len(df), 'profile': profile})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/edit-data', methods=['POST'])
def edit_data():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    action = data.get('action')
    column = data.get('column', '')
    value  = data.get('value', '')
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        if action == 'rename':
            if not value:
                return jsonify({'error': 'New name is required'}), 400
            df = df.rename(columns={column: value})
            msg = f'Column "{column}" renamed to "{value}"'
        elif action == 'drop-col':
            if column not in df.columns:
                return jsonify({'error': f'Column "{column}" not found'}), 400
            df = df.drop(columns=[column])
            msg = f'Column "{column}" dropped'
        elif action == 'fill-missing':
            if column not in df.columns:
                return jsonify({'error': f'Column "{column}" not found'}), 400
            if value == 'mean':     df[column].fillna(df[column].mean(),    inplace=True)
            elif value == 'median': df[column].fillna(df[column].median(),  inplace=True)
            elif value == 'mode':   df[column].fillna(df[column].mode()[0], inplace=True)
            elif value == '0':      df[column].fillna(0,                    inplace=True)
            else:                   df[column].fillna(value,                inplace=True)
            msg = f'Missing values in "{column}" filled with {value}'
        elif action == 'drop-duplicates':
            before = len(df)
            df = df.drop_duplicates()
            msg = f'Removed {before - len(df)} duplicate rows'
        elif action == 'sort':
            if column not in df.columns:
                return jsonify({'error': f'Column "{column}" not found'}), 400
            df = df.sort_values(by=column, ascending=(value == 'asc'))
            msg = f'Sorted by "{column}" {"ascending" if value == "asc" else "descending"}'
        elif action == 'filter-rows':
            before = len(df)
            try:
                df = df.query(f'`{column}` {value}')
                msg = f'Filtered: kept {len(df)} of {before} rows'
            except Exception as fe:
                return jsonify({'error': f'Filter error: {str(fe)}'}), 400
        else:
            return jsonify({'error': 'Unknown action'}), 400
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)
        return jsonify({'message': msg, 'rows': len(df), 'columns': len(df.columns)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/chart-data', methods=['POST'])
def chart_data():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session.'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        cols = smart_detect_all(df)
        result = {'detected_columns': {k: v for k, v in cols.items()
                                        if v and k not in ['num_cols', 'cat_cols', 'all_cols']}}
        value_col  = cols['value']
        qty_col    = cols['qty']
        rate_col   = cols['rate']
        date_col   = cols['date']
        cat_col    = cols['category']
        region_col = cols['region']
        person_col = cols['person']
        num_cols   = cols['num_cols']

        if date_col and value_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_t = df.dropna(subset=[date_col]).copy()
                dr   = (df_t[date_col].max() - df_t[date_col].min()).days
                freq = 'Y' if dr > 730 else ('M' if dr > 60 else 'W')
                m    = df_t.set_index(date_col)[value_col].resample(freq).sum().reset_index()
                fmt  = '%Y' if freq == 'Y' else ('%Y-%m' if freq == 'M' else '%Y-%m-%d')
                result['time_series'] = {
                    'x': m[date_col].dt.strftime(fmt).tolist(),
                    'y': m[value_col].round(2).tolist(),
                    'x_label': date_col, 'y_label': value_col,
                }
            except Exception:
                pass

        if cat_col and value_col:
            top = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(15).reset_index()
            result['bar_chart'] = {
                'x': top[cat_col].astype(str).tolist(), 'y': top[value_col].round(2).tolist(),
                'x_label': cat_col, 'y_label': value_col,
            }
        if value_col:
            cv = df[value_col].dropna()
            counts, edges = np.histogram(cv, bins=20)
            result['histogram'] = {
                'x': [round((edges[i] + edges[i+1]) / 2, 2) for i in range(len(counts))],
                'y': counts.tolist(), 'label': value_col,
            }
        if cat_col and value_col:
            pie = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(8).reset_index()
            result['pie_chart'] = {
                'labels': pie[cat_col].astype(str).tolist(),
                'values': pie[value_col].round(2).tolist(),
                'label': f'{value_col} by {cat_col}',
            }
        x_col = qty_col or (num_cols[1] if len(num_cols) > 1 else None)
        if x_col and value_col and x_col != value_col:
            sc_cols = [x_col, value_col]
            if rate_col: sc_cols.append(rate_col)
            if cat_col:  sc_cols.append(cat_col)
            sc = df[sc_cols].dropna()
            result['scatter'] = {
                'x': sc[x_col].tolist(), 'y': sc[value_col].round(2).tolist(),
                'color': sc[rate_col].tolist() if rate_col else [0] * len(sc),
                'labels': sc[cat_col].astype(str).tolist() if cat_col else [''] * len(sc),
                'x_label': x_col, 'y_label': value_col, 'color_label': rate_col or '',
            }
        if region_col and value_col:
            if person_col:
                agg = df.groupby([region_col, person_col])[value_col].sum().reset_index()
                ur  = agg[region_col].unique().tolist()
                rt  = agg.groupby(region_col)[value_col].sum().to_dict()
                result['treemap'] = {
                    'labels': ['All'] + ur + agg[person_col].astype(str).tolist(),
                    'parents': [''] + ['All'] * len(ur) + agg[region_col].tolist(),
                    'values': [round(float(agg[value_col].sum()), 2)]
                               + [round(float(rt[r]), 2) for r in ur]
                               + agg[value_col].round(2).tolist(),
                }
            else:
                r_df = df.groupby(region_col)[value_col].sum().reset_index()
                result['treemap'] = {
                    'labels': ['All'] + r_df[region_col].astype(str).tolist(),
                    'parents': [''] + ['All'] * len(r_df),
                    'values': [round(float(r_df[value_col].sum()), 2)] + r_df[value_col].round(2).tolist(),
                }
        if len(num_cols) >= 3:
            hc = num_cols[:8]
            corr = df[hc].corr().round(2)
            result['heatmap'] = {'x': hc, 'y': hc, 'z': corr.values.tolist()}
        if len(num_cols) >= 2:
            bc = num_cols[:6]
            result['box_plot'] = {'cols': bc, 'data': {c: df[c].dropna().tolist() for c in bc}}
        if value_col:
            result['summary'] = {
                'total': round(float(df[value_col].sum()), 2),
                'records': len(df),
                'avg': round(float(df[value_col].mean()), 2),
                'max': round(float(df[value_col].max()), 2),
                'columns': len(df.columns),
                'value_label': value_col,
                'qty_total': int(df[qty_col].sum()) if qty_col else 0,
                'qty_label': qty_col or 'N/A',
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
#  PMA -- PREDICTIVE MODELING & ANALYSIS ROUTES
# ========================================================================

@main.route('/api/pma/detect', methods=['POST'])
def pma_detect():
    data = request.get_json()
    file_path  = session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        # Store only small scalars/paths in session, not the whole info blob.
        # Full info is recomputed on train — it's cheap.
        session['pma_target']  = target_col
        session['pma_file']    = file_path
        session['pma_data_type']    = info.get('data_type')
        session['pma_problem_type'] = info.get('problem_type')
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/columns', methods=['POST'])
def pma_columns():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        return jsonify({
            'all_cols': df.columns.tolist(),
            'columns': df.columns.tolist(),   # v2 UI alias
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'n_rows':   len(df),
            'file_name': os.path.basename(file_path),
            'sample':   df.head(5).to_dict(orient='records'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/feature-importance', methods=['POST'])
def pma_feature_importance():
    data = request.get_json()
    file_path  = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_info', {}).get('target_col')
    if not file_path or not target_col:
        return jsonify({'error': 'Missing file or target column'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        X, y, feature_names, encoders = pma.preprocess_tabular(df, target_col, info)
        importance = pma.get_feature_importance(X, y, feature_names, info['problem_type'])
        session['pma_target'] = target_col
        session['pma_data_type']    = info.get('data_type')
        session['pma_problem_type'] = info.get('problem_type')
        return jsonify({'features': importance, 'problem_type': info['problem_type'],
                        'data_type': info['data_type']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/models', methods=['POST'])
@limiter.limit("20 per minute; 120 per hour")
def pma_models():
    data = request.get_json()
    # Prefer request body; fall back to small session scalars.
    data_type    = data.get('data_type')    or session.get('pma_data_type',    'tabular')
    problem_type = data.get('problem_type') or session.get('pma_problem_type', 'classification')
    n_rows = data.get('n_rows', 1000)
    n_cols = data.get('n_cols', 10)
    try:
        catalog         = pma.get_model_catalog(data_type, problem_type)
        recommendations = pma.recommend_models(data_type, problem_type, n_rows, n_cols)
        ai_note = ''
        try:
            ai_note = _ml_agent.explain_model_selection(
                {'data_type': data_type, 'problem_type': problem_type,
                 'n_rows': n_rows, 'n_cols': n_cols,
                 'target_col': session.get('pma_target'),
                 'num_cols': [], 'cat_cols': []},
                recommendations
            )
        except Exception:
            ai_note = 'AI explanation unavailable.'
        models_list = [{'key': k, 'name': v['name'], 'category': v['category']}
                       for k, v in catalog.items()]
        return jsonify({'models': models_list, 'recommendations': recommendations,
                        'ai_note': ai_note})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/train', methods=['POST'])
@limiter.limit("10 per minute; 40 per hour")
def pma_train():
    data = request.get_json()
    file_path  = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    model_key  = data.get('model_key', 'random_forest')
    test_size  = float(data.get('test_size', 0.2))
    if not file_path or not target_col:
        return jsonify({'error': 'Missing file or target column'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        if info['data_type'] == 'time_series' and info['date_cols']:
            df = pma.prepare_time_series(df, info['date_cols'][0], target_col)
        X, y, feature_names, encoders = pma.preprocess_tabular(df, target_col, info)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        catalog = pma.get_model_catalog(info['data_type'], info['problem_type'])
        if model_key not in catalog:
            return jsonify({'error': f'Unknown model key: {model_key}'}), 400
        result = pma.train_selected_model(X_train, y_train, X_test, y_test,
                                          model_key, catalog, info['problem_type'])
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        ai_note = ''
        try:
            ai_note = _ml_agent.explain_metrics(result['metrics'], info['problem_type'],
                                                 result['model_name'])
        except Exception:
            ai_note = 'AI explanation unavailable.'
        metadata = {'model_key': model_key, 'model_name': result['model_name'],
                    'target_col': target_col, 'feature_names': feature_names,
                    'problem_type': info['problem_type'], 'data_type': info['data_type'],
                    'metrics': result['metrics']}
        model_path = pma.save_model_artifacts(result['model'], metadata, model_key)
        enc_path   = model_path.replace('.pkl', '_encoders.pkl')
        with open(enc_path, 'wb') as f:
            pickle.dump(encoders, f)
        session['pma_model_path']    = model_path
        session['pma_enc_path']      = enc_path
        session['pma_feature_names'] = feature_names
        session['pma_problem_type']  = info['problem_type']
        session['pma_target']        = target_col
        return jsonify({
            'metrics': result['metrics'], 'logs': result.get('logs', []),
            'model_name': result['model_name'], 'model_path': model_path,
            'feature_names': feature_names, 'problem_type': info['problem_type'],
            'y_val': result.get('y_val', [])[:100], 'y_pred': result.get('y_pred', [])[:100],
            'ai_note': ai_note,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/tune', methods=['POST'])
@limiter.limit("5 per minute; 20 per hour")
def pma_tune():
    data = request.get_json()
    file_path  = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    model_key  = data.get('model_key', 'random_forest')
    method     = data.get('method', 'random')
    if not file_path or not target_col:
        return jsonify({'error': 'Missing file or target column'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        X, y, feature_names, encoders = pma.preprocess_tabular(df, target_col, info)
        catalog = pma.get_model_catalog(info['data_type'], info['problem_type'])
        if model_key not in catalog or catalog[model_key]['model'] is None:
            return jsonify({'error': 'Cannot tune this model type'}), 400
        base_model  = catalog[model_key]['model']
        tune_result = pma.tune_model_legacy(base_model, model_key, X, y, method, info['problem_type'])
        if 'error' in tune_result:
            return jsonify({'error': tune_result['error']}), 500
        best_model = tune_result.get('best_model')
        if best_model:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            retrain = pma.train_selected_model(
                X_train, y_train, X_test, y_test, model_key,
                {model_key: {'name': catalog[model_key]['name'], 'model': best_model}},
                info['problem_type'])
            if 'metrics' in retrain:
                tune_result['tuned_metrics'] = retrain['metrics']
                metadata = {
                    'model_key': model_key + '_tuned',
                    'model_name': catalog[model_key]['name'] + ' (Tuned)',
                    'target_col': target_col, 'feature_names': feature_names,
                    'problem_type': info['problem_type'], 'data_type': info['data_type'],
                    'metrics': retrain['metrics'], 'best_params': tune_result['best_params'],
                }
                model_path = pma.save_model_artifacts(best_model, metadata, model_key + '_tuned')
                enc_path   = model_path.replace('.pkl', '_encoders.pkl')
                with open(enc_path, 'wb') as f:
                    pickle.dump(encoders, f)
                session['pma_model_path'] = model_path
                session['pma_enc_path']   = enc_path
                tune_result['model_path'] = model_path
        return jsonify({'best_params': tune_result.get('best_params', {}),
                        'best_score': tune_result.get('best_score'),
                        'tuned_metrics': tune_result.get('tuned_metrics', {})})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/predict', methods=['POST'])
@limiter.limit("30 per minute")
def pma_predict():
    data          = request.get_json()
    input_data    = data.get('input_data', {})
    model_path    = data.get('model_path')    or session.get('pma_model_path')
    enc_path      = data.get('enc_path')      or session.get('pma_enc_path')
    feature_names = data.get('feature_names') or session.get('pma_feature_names', [])
    problem_type  = data.get('problem_type')  or session.get('pma_problem_type', 'regression')
    if not model_path:
        return jsonify({'error': 'No model trained yet. Please train a model first.'}), 400
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        encoders = {}
        if enc_path and os.path.exists(enc_path):
            with open(enc_path, 'rb') as f:
                encoders = pickle.load(f)
        result   = pma.predict_new_data_legacy(input_data, feature_names, model, encoders, problem_type)
        ai_note  = ''
        try:
            context = f'Prediction: {result}, Input: {input_data}, Type: {problem_type}'
            ai_note = _ml_agent.run(
                'Explain this prediction result to a non-technical user.', context)
        except Exception:
            pass
        result['ai_note'] = ai_note
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/saved-models', methods=['GET'])
def pma_saved_models():
    meta_path = 'models/model_metadata.json'
    if not os.path.exists(meta_path):
        return jsonify({'models': []})
    try:
        with open(meta_path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            records = []
        safe = [{'model_key': r.get('model_key'), 'model_name': r.get('model_name'),
                 'target_col': r.get('target_col'), 'problem_type': r.get('problem_type'),
                 'data_type': r.get('data_type'), 'metrics': r.get('metrics', {}),
                 'saved_at': r.get('saved_at'), 'model_path': r.get('model_path')}
                for r in records]
        return jsonify({'models': safe})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/load-model', methods=['POST'])
def pma_load_model():
    data       = request.get_json()
    model_path = data.get('model_path')
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    try:
        meta_path = 'models/model_metadata.json'
        metadata  = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                records = json.load(f)
            for r in (records if isinstance(records, list) else []):
                if r.get('model_path') == model_path:
                    metadata = r
                    break
        enc_path = model_path.replace('.pkl', '_encoders.pkl')
        session['pma_model_path']    = model_path
        session['pma_enc_path']      = enc_path if os.path.exists(enc_path) else None
        session['pma_feature_names'] = metadata.get('feature_names', [])
        session['pma_problem_type']  = metadata.get('problem_type', 'regression')
        session['pma_target']        = metadata.get('target_col')
        return jsonify({'message': 'Model loaded into session', 'metadata': metadata})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/suggestions', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour")
def pma_suggestions():
    data      = request.get_json()
    metrics   = data.get('metrics', {})
    model_key = data.get('model_key', '')
    try:
        info = {
            'data_type':    session.get('pma_data_type'),
            'problem_type': session.get('pma_problem_type'),
            'target_col':   session.get('pma_target'),
        }
        suggestion = _ml_agent.suggest_improvements(metrics, model_key, info)
        return jsonify({'suggestion': suggestion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/export', methods=['POST'])
def pma_export():
    data       = request.get_json()
    y_val      = data.get('y_val', [])
    y_pred     = data.get('y_pred', [])
    model_name = data.get('model_name', 'model')
    try:
        rows   = [{'actual': a, 'predicted': p} for a, p in zip(y_val, y_pred)]
        out_df = pd.DataFrame(rows)
        # Sanitize model_name — comes from client, could contain path separators
        safe_model = secure_filename(str(model_name)) or 'model'
        path   = f'uploads/predictions_{safe_model}.csv'
        out_df.to_csv(path, index=False)
        return jsonify({'download_url': f'/{path}', 'rows': len(out_df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
#  NEW PMA ENDPOINTS  (v2 — uses src.algorithms + src.ml_suggester + src.automl)
# ========================================================================

@main.route('/api/pma/v2/profile', methods=['POST'])
def pma_v2_profile():
    """Return the full dataset profile + inferred task, for the suggest UI."""
    data = request.get_json() or {}
    file_path  = session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        profile = profile_dataset(df, target_col)
        task    = infer_task(df, target_col)
        session['pma_file']         = file_path
        session['pma_target']       = target_col
        session['pma_task']         = task['task']
        session['pma_problem_type'] = task['problem_type']
        return jsonify({
            'profile': profile, 'task': task['task'],
            'problem_type': task['problem_type'],
            'reason': task['reason'],
            'available_deps': available_deps(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/suggest', methods=['POST'])
@limiter.limit("20 per minute; 120 per hour")
def pma_v2_suggest():
    """LLM-first algorithm suggestion. Returns top 3-5 with rationales."""
    data = request.get_json() or {}
    file_path  = session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        result = suggest_algorithms(df, target_col)
        session['pma_target'] = target_col
        session['pma_task']   = result['task']
        session['pma_problem_type'] = result['problem_type']
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/algorithms', methods=['POST'])
def pma_v2_algorithms():
    """Full catalog for the Manual tab, filtered by task + dataset size."""
    data = request.get_json() or {}
    task   = data.get('task') or session.get('pma_task')
    n_rows = int(data.get('n_rows', 1000))
    n_cols = int(data.get('n_cols', 10))
    if not task:
        return jsonify({'error': 'No task specified'}), 400
    try:
        all_algos = algorithms_for_task(task)
        feasible  = filter_by_data(all_algos, n_rows, n_cols)
        # Group by family for the UI
        families = {}
        for a in feasible:
            fam = a['family']
            families.setdefault(fam, []).append({
                'id': a['id'], 'name': a['name'],
                'strengths': a['strengths'], 'weaknesses': a['weaknesses'],
                'speed': a['speed'], 'interpretable': a['interpretable'],
                'handles_missing': a['handles_missing'],
                'notes': a.get('notes', ''),
            })
        # Excluded (too small/large) algos so user can see why
        excluded = []
        feasible_ids = {a['id'] for a in feasible}
        for a in all_algos:
            if a['id'] not in feasible_ids:
                reason = None
                if n_rows < a['min_rows']:
                    reason = f"needs ≥ {a['min_rows']} rows"
                elif a['max_rows'] and n_rows > a['max_rows']:
                    reason = f"too slow past {a['max_rows']:,} rows"
                excluded.append({'id': a['id'], 'name': a['name'], 'reason': reason})
        return jsonify({
            'task': task,
            'task_label': TASK_LABELS.get(task, task),
            'families': families,
            'excluded': excluded,
            'total': len(feasible),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/train', methods=['POST'])
@limiter.limit("10 per minute; 40 per hour")
def pma_v2_train():
    """Train ONE algorithm via the new leakage-free pipeline.

    On success, persists the run as an Experiment row so the user can find it
    again in their sidebar / compare view.
    """
    import time
    data = request.get_json() or {}
    file_path  = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    algo_id    = data.get('algo_id') or data.get('model_key')
    test_size  = float(data.get('test_size', 0.2))
    cv_folds   = int(data.get('cv_folds', 5))
    params     = data.get('params') or None
    source     = data.get('source', 'manual')   # 'manual' | 'suggest' | 'automl'
    if not file_path or not target_col or not algo_id:
        return jsonify({'error': 'Need file_path, target_col, and algo_id'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        t0 = time.time()
        result = pma.train_model(df, target_col, algo_id,
                                  test_size=test_size, cv_folds=cv_folds,
                                  params=params)
        train_time = round(time.time() - t0, 2)
        if not result.get('ok'):
            return jsonify({'error': result.get('error', 'Training failed')}), 400
        # Store model path in session for the predict endpoint
        session['pma_model_path']    = result['model_path']
        session['pma_feature_names'] = result['feature_names']
        session['pma_problem_type']  = result['problem_type']
        # AI analysis
        ai_note = ''
        try:
            ai_note = _ml_agent.explain_metrics(result['metrics'], result['problem_type'],
                                                 result['algo_name'])
        except Exception:
            ai_note = 'AI explanation unavailable.'
        result['ai_note'] = ai_note
        result['training_time_s'] = train_time

        # Persist as an experiment row — best-effort, don't fail the response
        try:
            exp_id = expt.save_from_training_result(
                result, user_id=_user_id(), dataset_name=_dataset_name(),
                source=source, training_time_s=train_time, params=params or {},
            )
            result['experiment_id'] = exp_id
            session['pma_last_experiment_id'] = exp_id
        except Exception as e:
            result['experiment_save_error'] = str(e)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/automl', methods=['POST'])
@limiter.limit("3 per minute; 15 per hour")
def pma_v2_automl():
    """Quick Auto-ML: train top 5 in parallel, return leaderboard.

    Each successful model in the leaderboard is also persisted as a separate
    experiment row (source='automl') so the user can find any of them later.
    """
    data = request.get_json() or {}
    file_path  = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    top_n      = int(data.get('top_n', 5))
    test_size  = float(data.get('test_size', 0.2))
    pre_picked = data.get('algo_ids')   # optional: for manual comparison mode
    if not file_path or not target_col:
        return jsonify({'error': 'Need file_path and target_col'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        result = run_automl(df, target_col, top_n=top_n, test_size=test_size,
                            pre_picked_algos=pre_picked)
        if result.get('best'):
            session['pma_model_path'] = result['best']['model_path']

        # Persist each leaderboard entry as a separate experiment row
        try:
            ds_name = _dataset_name()
            uid = _user_id()
            for entry in (result.get('leaderboard') or []):
                if not entry.get('ok') or not entry.get('model_path'):
                    continue
                # Build a training-result-shaped dict for the saver
                training_result = {
                    'algo_id':            entry['algo_id'],
                    'algo_name':          entry['algo_name'],
                    'task':               result['task'],
                    'problem_type':       result['problem_type'],
                    'target_col':         target_col,
                    'n_rows':             entry.get('n_rows', 0),
                    'n_cols':             entry.get('n_cols', 0),
                    'metrics':            entry.get('metrics', {}),
                    'cv_metrics':         entry.get('cv_metrics', {}),
                    'feature_importance': entry.get('feature_importance', []),
                    'is_imbalanced':      entry.get('is_imbalanced', False),
                    'model_path':         entry['model_path'],
                    'logs':               entry.get('logs', []),
                }
                exp_id = expt.save_from_training_result(
                    training_result, user_id=uid, dataset_name=ds_name,
                    source='automl',
                    training_time_s=entry.get('train_time_sec', 0),
                )
                entry['experiment_id'] = exp_id
                if entry['rank'] == 1:
                    session['pma_last_experiment_id'] = exp_id
        except Exception as e:
            result['experiment_save_error'] = str(e)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
#  EXPERIMENTS  — history, model cards, compare, deploy
# ========================================================================

@main.route('/api/pma/v2/experiments', methods=['GET'])
def pma_v2_list_experiments():
    """List recent experiments for the current user. Sidebar uses this.

    Query params: limit, offset, task, algo_id, starred (0/1), q (search).
    """
    try:
        limit       = min(int(request.args.get('limit', 50)), 200)
        offset      = max(int(request.args.get('offset', 0)), 0)
        task        = request.args.get('task') or None
        algo_id     = request.args.get('algo_id') or None
        starred_only = request.args.get('starred') in ('1', 'true', 'yes')
        q           = request.args.get('q') or None
        rows = expt.list_experiments(
            _user_id(), limit=limit, offset=offset,
            task=task, algo_id=algo_id, starred_only=starred_only, search=q,
        )
        summary = expt.experiments_summary(_user_id())
        return jsonify({'experiments': rows, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/experiments/<int:exp_id>', methods=['GET'])
def pma_v2_get_experiment(exp_id):
    """Full detail for one experiment — powers the model card view."""
    try:
        row = expt.get_experiment(exp_id, _user_id())
        if not row:
            return jsonify({'error': 'Experiment not found'}), 404
        # Convenience: also load it into the session so the deploy panel works
        if row.get('model_path'):
            session['pma_model_path']    = row['model_path']
            session['pma_feature_names'] = row.get('params', {}).get('feature_names') or []
            session['pma_problem_type']  = row.get('problem_type', 'regression')
            session['pma_target']        = row.get('target_col')
        return jsonify({'experiment': row})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/experiments/<int:exp_id>/star', methods=['POST'])
def pma_v2_star_experiment(exp_id):
    """Toggle the starred flag on an experiment."""
    try:
        new_val = expt.toggle_star(exp_id, _user_id())
        return jsonify({'starred': new_val})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/experiments/<int:exp_id>', methods=['DELETE'])
def pma_v2_delete_experiment(exp_id):
    """Delete an experiment row. Optionally pass ?delete_pickle=1 to also
    remove the .pkl model file from disk."""
    try:
        also_pkl = request.args.get('delete_pickle') in ('1', 'true', 'yes')
        ok = expt.delete_experiment(exp_id, _user_id(), delete_pickle=also_pkl)
        if not ok:
            return jsonify({'error': 'Experiment not found'}), 404
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/experiments/compare', methods=['POST'])
def pma_v2_compare_experiments():
    """Side-by-side detail for 2+ experiments. Body: {'ids': [1, 2, 3]}."""
    data = request.get_json() or {}
    ids = data.get('ids') or []
    if not isinstance(ids, list) or len(ids) < 2:
        return jsonify({'error': 'Pass at least 2 experiment ids in {"ids": [...]}'}), 400
    try:
        rows = expt.compare_experiments([int(i) for i in ids], _user_id())
        return jsonify({'experiments': rows, 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/deploy', methods=['POST'])
@limiter.limit("60 per minute")
def pma_v2_deploy():
    """Live single-row prediction against a saved experiment's pipeline.

    Body: {'experiment_id': 123, 'input_data': {col: val, ...}}
    Falls back to the last-trained pipeline in session if no id given.
    """
    data = request.get_json() or {}
    exp_id     = data.get('experiment_id')
    input_data = data.get('input_data', {})

    model_path = None
    feature_names = []
    problem_type = 'regression'

    if exp_id:
        row = expt.get_experiment(int(exp_id), _user_id())
        if not row:
            return jsonify({'error': 'Experiment not found'}), 404
        model_path    = row.get('model_path')
        problem_type  = row.get('problem_type') or 'regression'
        feature_names = (row.get('params', {}).get('feature_names')
                          if isinstance(row.get('params'), dict) else []) or []
    else:
        model_path    = session.get('pma_model_path')
        feature_names = session.get('pma_feature_names', [])
        problem_type  = session.get('pma_problem_type', 'regression')

    if not model_path:
        return jsonify({'error': 'No trained model available'}), 400

    try:
        result = pma.predict_new_data(input_data, model_path, feature_names, problem_type)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/tune', methods=['POST'])
@limiter.limit("5 per minute; 20 per hour")
def pma_v2_tune():
    """Hyperparameter tuning on the new pipeline."""
    data = request.get_json() or {}
    file_path  = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    algo_id    = data.get('algo_id') or data.get('model_key')
    method     = data.get('method', 'random')
    n_iter     = int(data.get('n_iter', 20))
    if not file_path or not target_col or not algo_id:
        return jsonify({'error': 'Need file_path, target_col, and algo_id'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        result = pma.tune_model(df, target_col, algo_id, method=method, n_iter=n_iter)
        if not result.get('ok'):
            return jsonify({'error': result.get('error', 'Tuning failed')}), 400
        session['pma_model_path'] = result['model_path']
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/pma/v2/predict', methods=['POST'])
@limiter.limit("30 per minute")
def pma_v2_predict():
    """Run a saved pipeline on a single input row."""
    data = request.get_json() or {}
    input_data    = data.get('input_data', {})
    model_path    = data.get('model_path') or session.get('pma_model_path')
    feature_names = data.get('feature_names') or session.get('pma_feature_names', [])
    problem_type  = data.get('problem_type') or session.get('pma_problem_type', 'regression')
    if not model_path:
        return jsonify({'error': 'No model trained yet.'}), 400
    try:
        result = pma.predict_new_data(input_data, model_path, feature_names, problem_type)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================================================
#  BIA -- BUSINESS INTELLIGENCE & ANALYTICS ROUTES
# ========================================================================

def _bia_df():
    fp = session.get('bia_file') or session.get('uploaded_file')
    if not fp or not os.path.exists(fp):
        return None, 'No data file in session'
    try:
        df = load_file(fp)
        df.columns = [str(c).strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, str(e)


@main.route('/api/bia/extract', methods=['POST'])
def bia_extract():
    data      = request.get_json() or {}
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session. Please upload data first.'}), 400
    try:
        result = bia.extract_file(file_path)
        if 'error' in result:
            return jsonify(result), 500
        session['bia_file'] = file_path
        bia.log_event('extract', {'file': os.path.basename(file_path),
                                   'rows': result['total_rows']})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/mysql-test', methods=['POST'])
def bia_mysql_test():
    data   = request.get_json() or {}
    result = bia.test_mysql_connection(
        data.get('host', 'localhost'), data.get('port', 3306),
        data.get('user', 'root'),      data.get('password', ''),
        data.get('database', ''))
    if result['ok']:
        session['bia_mysql_cfg'] = data
    return jsonify(result)


@main.route('/api/bia/mysql-tables', methods=['POST'])
def bia_mysql_tables():
    cfg = session.get('bia_mysql_cfg') or request.get_json() or {}
    try:
        return jsonify({'tables': bia.list_mysql_tables(cfg)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/transform', methods=['POST'])
def bia_transform():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    opts = {'drop_duplicates': data.get('drop_duplicates', True),
            'fill_missing': data.get('fill_missing', 'auto'),
            'normalize': data.get('normalize', False)}
    try:
        result   = bia.transform_data(df, opts)
        clean_df = result['df']
        fp = session.get('bia_file')
        if fp and fp.endswith('.csv'):
            clean_df.to_csv(fp, index=False)
        bia.log_event('transform', {'rows': len(clean_df), 'log': result['log']})
        return jsonify({'ok': True, 'rows': len(clean_df), 'columns': len(clean_df.columns),
                        'log': result['log'], 'num_cols': result['num_cols'],
                        'cat_cols': result['cat_cols'], 'date_cols': result['date_cols'],
                        'preview': clean_df.head(8).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/mysql-load', methods=['POST'])
def bia_mysql_load():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data   = request.get_json() or {}
    cfg    = session.get('bia_mysql_cfg') or data.get('cfg', {})
    table  = data.get('table_name', 'bia_data')
    result = bia.load_to_mysql(df, cfg, table)
    if result['ok']:
        bia.log_event('mysql_load', {'table': table, 'rows': result['rows']})
    return jsonify(result)


@main.route('/api/bia/kpis', methods=['POST'])
def bia_kpis():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        result = bia.compute_kpis(df)
        aggs   = {}
        if result.get('date_col') and result.get('value_col'):
            aggs = bia.aggregate_data(df, result['date_col'], result['value_col'])
        result['aggregations'] = aggs
        # Store only small scalar column names — not the whole KPI blob.
        # Downstream endpoints only need value_col/date_col/cat_col/qty_col.
        session['bia_value_col'] = result.get('value_col')
        session['bia_date_col']  = result.get('date_col')
        session['bia_cat_col']   = result.get('cat_col')
        session['bia_qty_col']   = result.get('qty_col')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/eda', methods=['POST'])
def bia_eda():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        return jsonify(bia.compute_eda(df))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/chart', methods=['POST'])
def bia_chart():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    try:
        result = bia.build_chart_data(df,
            chart_type=data.get('chart_type', 'bar'),
            x_col=data.get('x_col', ''),
            y_col=data.get('y_col', ''),
            color_col=data.get('color_col'),
            agg_func=data.get('agg_func', 'sum'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/dashboard', methods=['POST'])
def bia_dashboard():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        eda_data = bia.compute_eda(df)
        insights = bia.generate_auto_insights(df, kpi_data)
        charts   = {}
        value_col = kpi_data.get('value_col')
        date_col  = kpi_data.get('date_col')
        cat_col   = kpi_data.get('cat_col')
        num_cols  = eda_data.get('num_cols', [])
        if date_col and value_col:
            agg = bia.aggregate_data(df, date_col, value_col)
            if 'monthly' in agg:
                charts['time_series'] = agg['monthly']
        if cat_col and value_col:
            charts['bar'] = bia.build_chart_data(df, 'bar', cat_col, value_col)
            charts['pie'] = bia.build_chart_data(df, 'pie', cat_col, value_col)
        if len(num_cols) >= 2:
            charts['heatmap'] = bia.build_chart_data(df, 'heatmap', num_cols[0], num_cols[1])
        if value_col:
            charts['histogram'] = bia.build_chart_data(df, 'histogram', value_col, value_col)
        return jsonify({'kpis': kpi_data['kpis'], 'insights': insights, 'charts': charts,
                        'eda_summary': {'completeness': eda_data['completeness'],
                                        'n_rows': eda_data['n_rows'],
                                        'n_cols': eda_data['n_cols']}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/segment', methods=['POST'])
def bia_segment():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data       = request.get_json() or {}
    n_clusters = int(data.get('n_clusters', 4))
    result     = bia.customer_segmentation(df, n_clusters)
    if result.get('ok'):
        bia.log_event('segmentation', {'clusters': n_clusters})
    return jsonify(result)


@main.route('/api/bia/anomalies', methods=['POST'])
def bia_anomalies():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data      = request.get_json() or {}
    value_col = data.get('value_col') or session.get('bia_value_col')
    if not value_col:
        num_cols  = df.select_dtypes(include='number').columns.tolist()
        value_col = num_cols[0] if num_cols else None
    if not value_col:
        return jsonify({'error': 'No numeric column found'}), 400
    return jsonify(bia.detect_anomalies(df, value_col))


@main.route('/api/bia/forecast', methods=['POST'])
def bia_forecast():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data      = request.get_json() or {}
    date_col  = data.get('date_col')  or session.get('bia_date_col')
    value_col = data.get('value_col') or session.get('bia_value_col')
    periods   = int(data.get('periods', 12))
    if not date_col or not value_col:
        return jsonify({'error': 'Need date_col and value_col for forecasting'}), 400
    return jsonify(bia.time_series_forecast(df, date_col, value_col, periods))


@main.route('/api/bia/insights', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour")
def bia_insights():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data      = bia.compute_kpis(df)
        auto_insights = bia.generate_auto_insights(df, kpi_data)
        ai_insight    = ''
        try:
            from agents.ml_agent import MLAgent
            agent   = MLAgent()
            context = (f"Dataset: {len(df)} rows x {len(df.columns)} cols\n"
                       f"KPIs: {json.dumps({k: v.get('value') for k,v in kpi_data['kpis'].items()})}\n"
                       f"Top insights: {[i['title'] for i in auto_insights]}")
            ai_insight = agent.run(
                'You are a BI analyst. Give 3 specific, actionable business insights in plain English.',
                context)
        except Exception:
            pass
        return jsonify({'insights': auto_insights, 'ai_insight': ai_insight})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/ask', methods=['POST'])
@limiter.limit("10 per minute; 60 per hour")
def bia_ask():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data     = request.get_json() or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        response = orchestrator.run(question, session.get('bia_file'))
        bia.log_event('ask', {'question': question[:100]})
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/refresh', methods=['POST'])
def bia_refresh():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        import datetime
        return jsonify({'kpis': kpi_data['kpis'], 'rows': len(df),
                        'refreshed_at': datetime.datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/export-csv', methods=['POST'])
def bia_export_csv():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        path = bia.export_to_csv(df, 'bia_export')
        return jsonify({'ok': True, 'download_url': f'/{path}', 'rows': len(df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/export-report', methods=['POST'])
def bia_export_report():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        insights = bia.generate_auto_insights(df, kpi_data)
        result   = bia.build_pdf_report(kpi_data['kpis'], insights)
        if result['ok']:
            return jsonify({'ok': True, 'download_url': f'/{result["path"]}',
                            'note': result.get('note', '')})
        return jsonify({'error': result.get('error', 'Export failed')}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/monitor', methods=['GET'])
def bia_monitor():
    return jsonify({'logs': bia.get_monitor_logs(50), 'health': bia.get_system_health()})


@main.route('/api/bia/columns', methods=['POST'])
def bia_columns():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return jsonify({'all_cols': df.columns.tolist(), 'num_cols': num_cols,
                    'cat_cols': cat_cols, 'n_rows': len(df)})


# ========================================================================
#  AUTO DASHBOARD  --  delegates to app/auto_dashboard_route.py
# ========================================================================

@main.route('/api/auto-dashboard', methods=['POST'])
def api_auto_dashboard():
    """Legacy single-call: plan + render in one request."""
    from app.auto_dashboard_route import _run_auto_dashboard
    return _run_auto_dashboard()


@main.route('/api/auto-dashboard/plan', methods=['POST'])
@limiter.limit("15 per minute; 80 per hour")
def api_auto_dashboard_plan():
    """Stage 1: profile dataset + LLM picks charts. Fast, rate-limited (hits LLM)."""
    from app.auto_dashboard_route import _run_plan
    return _run_plan()


@main.route('/api/auto-dashboard/render', methods=['POST'])
@limiter.limit("30 per minute")
def api_auto_dashboard_render():
    """Stage 2: take plan, compute all chart data. No LLM call."""
    from app.auto_dashboard_route import _run_render
    return _run_render()
