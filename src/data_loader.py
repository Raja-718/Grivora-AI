# data_loader.py — Universal Data Loader
# Handles: CSV (any encoding), Excel (.xlsx, .xls, .xlsb, .ods),
#          JSON, TSV, Parquet, XML, plain text — with auto-fallbacks
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ── Encoding candidates tried in order ────────────────────────────
_ENCODINGS = [
    'utf-8', 'utf-8-sig',          # UTF-8 with/without BOM
    'latin-1', 'iso-8859-1',       # Western European (covers 0x80–0xFF)
    'cp1252', 'windows-1252',      # Windows Western — covers 0x92 apostrophe etc.
    'cp1250',                      # Windows Central European
    'utf-16', 'utf-16-le', 'utf-16-be',
    'ascii',
    'mac_roman',
    'cp437',                       # IBM PC
]

# ── Separators tried for plain-text CSV-like files ─────────────────
_SEPARATORS = [',', ';', '\t', '|', ':']


def _try_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Try to read a CSV/text file with multiple encoding + separator combos."""
    last_err = None

    # First: try chardet auto-detect if available
    try:
        import chardet
        raw = open(file_path, 'rb').read(min(200_000, os.path.getsize(file_path)))
        detected = chardet.detect(raw)
        enc = detected.get('encoding') or 'utf-8'
        try:
            df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip',
                             low_memory=False, **kwargs)
            if len(df.columns) > 0:
                return df
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback: try encoding list
    sep = kwargs.pop('sep', None)
    for enc in _ENCODINGS:
        try:
            if sep:
                df = pd.read_csv(file_path, encoding=enc, sep=sep,
                                 on_bad_lines='skip', low_memory=False, **kwargs)
            else:
                df = pd.read_csv(file_path, encoding=enc,
                                 on_bad_lines='skip', low_memory=False,
                                 sep=None, engine='python', **kwargs)
            if len(df.columns) > 0:
                return df
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Could not read file with any encoding. Last error: {last_err}")


def _try_excel(file_path: str, **kwargs) -> pd.DataFrame:
    """Try to read an Excel file with multiple engine fallbacks."""
    ext = file_path.rsplit('.', 1)[-1].lower()
    last_err = None

    # Engine priority map
    engine_map = {
        'xlsx':  ['openpyxl', 'xlrd'],
        'xls':   ['xlrd', 'openpyxl'],
        'xlsb':  ['pyxlsb', 'openpyxl'],
        'xlsm':  ['openpyxl'],
        'ods':   ['odf'],
    }
    engines = engine_map.get(ext, ['openpyxl', 'xlrd', 'pyxlsb'])

    for engine in engines:
        try:
            df = pd.read_excel(file_path, engine=engine, **kwargs)
            return df
        except ImportError:
            # Engine not installed — try next
            last_err = ImportError(
                f"Engine '{engine}' not installed. "
                f"Fix: pip install {'xlrd>=2.0.1' if engine=='xlrd' else engine}"
            )
            continue
        except Exception as e:
            last_err = e
            continue

    # Build helpful error message
    if isinstance(last_err, ImportError):
        if ext == 'xls':
            raise ImportError(
                "Reading .xls files requires xlrd>=2.0.1. "
                "Run: pip install xlrd>=2.0.1"
            )
        raise last_err

    raise ValueError(f"Could not read Excel file. Last error: {last_err}")


def load_file(file_path: str, nrows: int = None) -> pd.DataFrame:
    """
    Universal file loader. Supports:
      CSV / TSV / TXT  — auto-encoding detection, bad-line skipping
      XLSX / XLS / XLSB / XLSM / ODS  — engine fallbacks
      JSON — records, split, index, columns, values orientations
      Parquet — via pyarrow or fastparquet
      XML — pandas read_xml fallback
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    kwargs = {}
    if nrows is not None:
        kwargs['nrows'] = nrows

    # ── CSV / TSV / TXT ───────────────────────────────────────────
    if ext in ('csv', 'txt', 'tsv', 'tab', 'dat'):
        sep = '\t' if ext in ('tsv', 'tab') else None
        if sep:
            kwargs['sep'] = sep
        return _try_csv(file_path, **kwargs)

    # ── Excel ─────────────────────────────────────────────────────
    if ext in ('xlsx', 'xls', 'xlsb', 'xlsm', 'ods'):
        return _try_excel(file_path, **kwargs)

    # ── JSON ──────────────────────────────────────────────────────
    if ext == 'json':
        for orient in ('records', 'split', 'index', 'columns', 'values', None):
            try:
                df = pd.read_json(file_path, orient=orient)
                if len(df.columns) > 0:
                    return df
            except Exception:
                continue
        raise ValueError("Could not parse JSON file with any known orientation.")

    # ── Parquet ───────────────────────────────────────────────────
    if ext in ('parquet', 'pq'):
        try:
            return pd.read_parquet(file_path)
        except ImportError:
            raise ImportError("Parquet support requires pyarrow. Run: pip install pyarrow")

    # ── XML ───────────────────────────────────────────────────────
    if ext == 'xml':
        try:
            return pd.read_xml(file_path)
        except Exception as e:
            raise ValueError(f"Could not parse XML: {e}")

    # ── Fallback: try as CSV ───────────────────────────────────────
    try:
        return _try_csv(file_path, **kwargs)
    except Exception:
        pass

    raise ValueError(
        f"Unsupported or unreadable file format: .{ext}\n"
        f"Supported: CSV, TSV, TXT, XLSX, XLS, XLSB, ODS, JSON, Parquet, XML"
    )


def get_summary(df: pd.DataFrame) -> dict:
    return {
        "shape":          df.shape,
        "columns":        list(df.columns),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample":         df.head(5).to_dict(),
    }
