"""What is in this file -- a terminal answer for a saved parquet table.

Point it at a file or a `saved_analysis_results/` directory. Three views, narrowing:

* **a directory** -> one line per table: rows, columns, size on disk.
* **a table** -> one line per *column*: dtype, non-null count, distinct count, value
  summary. Never the frame -- `trial_data` is 58-70 columns wide, so any row-shaped
  view is unreadable in a terminal.
* **a column** -> that column alone, with real values.

- A reader and nothing else: read-only, computing nothing any metric consumes. Not a
  route to loading a metric -- `metrics_*.json` is an export, never an input
  (DECISIONS.md section 5).
- These functions take a **path**; only `scripts/parquet_peek.py` resolves a session.
  Resolution is the expensive half, so a caller already holding a directory must not
  re-pay the walk.
- Standard library and pandas only, `pyarrow` optional at every site. **No schema check
  here** -- `io/load_results.py` already does it, and a second spelling of that
  comparison can drift from the first. Ask the loader.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import pandas as pd

__all__ = ["peek", "DEFAULT_ROWS"]

# How many values `--column` lists. `--rows` overrides, and `--rows 0` prints the statistics alone.
DEFAULT_ROWS = 20

# Longest value rendered in the one-line-per-column view, and in the per-value listing
# of a column. The first keeps the summary table aligned; the second stops a single
# JSON blob from filling the screen.
_SUMMARY_VALUE_CHARS = 38
_DETAIL_VALUE_CHARS = 200


# --------------------------------------------------------------------------- helpers

def _fmt_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _fmt_num(value) -> str:
    """A number at roughly four significant figures, without scientific noise.

    Integral values print without a decimal point so a count column does not read as a
    measurement.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "nan"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.6g}"


def _fmt_pct(non_null: int, n_rows: int) -> str:
    """How full a column is -- and `100%` only when it is genuinely full.

    Rounding is **floored**, so a colum with 99.9% data does not print '100%'.
    """
    if not n_rows:
        return "   -"
    if non_null == n_rows:
        return "100%"
    return f"{int(100 * non_null / n_rows):3d}%"


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[: limit - 2] + ".."


def _parquet_files(results_dir: Path) -> list[Path]:
    """Every parquet in a directory, minus the AppleDouble shadows an SMB mount leaves.

    `._trial_data.parquet` is not a parquet; reading one raises, and listing it as a
    table would be a lie about what the directory holds. `utils/helpers.find_tracking_file`
    filters the same prefix for the same reason.
    """
    return sorted(p for p in results_dir.glob("*.parquet") if not p.name.startswith("._"))


def _shape_from_footer(path: Path) -> tuple[int, int]:
    """Rows and columns without deserialising a single data page.

    Worth the fallback branch: an inventory over a VPN-mounted share then costs one
    footer read per table instead of a full read of every one. `pyarrow` is what pandas
    reads parquet *with*, so it is present wherever this works at all -- but it is not
    declared in `pyproject.toml`, so this does not assume it.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        frame = pd.read_parquet(path)
        return len(frame), frame.shape[1]
    meta = pq.ParquetFile(path).metadata
    return meta.num_rows, meta.num_columns


def _column_names(path: Path) -> list[str]:
    """The column names, from the footer where possible."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return list(pd.read_parquet(path).columns)
    return list(pq.ParquetFile(path).schema_arrow.names)


def _stamp(manifest: dict, key: str) -> str:
    """One manifest field, distinguishing an absent key from a null one.

    Section 19 writes `commit` and `version` **always, even as `None`**, precisely so a
    reader can tell "written before provenance existed" (no key) from "written by code
    whose commit could not be resolved" (`null`). Not collapsing both here. 
    """
    if key not in manifest:
        return "absent"
    value = manifest.get(key)
    return "null" if value is None else str(value)


def _manifest_header(results_dir: Path) -> list[str]:
    """One line naming what wrote this directory, or nothing if there is no manifest.
    Absent for a stray parquet outside a `saved_analysis_results/`, which is a normal
    thing to peek at -- so its absence is silent rather than an error.
    """
    path = Path(results_dir) / "manifest.json"
    if not path.exists():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"manifest.json present but unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return ["manifest.json present but not an object"]

    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    session = " ".join(str(paths[k]) for k in ("sub_folder", "ses_folder") if paths.get(k))
    parts = [f"manifest: {session}" if session else "manifest:"]
    parts.append(f"protocol_mode={_stamp(manifest, 'protocol_mode')}")
    parts.append(f"commit={_stamp(manifest, 'commit')}")
    parts.append(f"version={_stamp(manifest, 'version')}")
    created = manifest.get("created_at")
    if created:
        parts.append(f"written {created}")
    return ["  ".join(parts)]


def _maybe_text(series: pd.Series) -> bool:
    """Whether this column could hold text at all.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    )


def _is_json_column(series: pd.Series) -> bool:
    """True when the column holds JSON *text* rather than the values it encodes.
    """
    values = series.dropna()
    if values.empty:
        return False
    for value in values.head(20):
        if not isinstance(value, str):
            return False
        text = value.strip()
        if not text or text[0] not in "[{":
            return False
        try:
            json.loads(text)
        except ValueError:
            return False
    return True


def _dtype_label(series: pd.Series, is_json: bool) -> str:
    return f"{series.dtype} (JSON)" if is_json else str(series.dtype)


def _describe_values(series: pd.Series, n_unique: int) -> str:
    """The value summary for one line of the per-column view, chosen by dtype."""
    values = series.dropna()
    if values.empty:
        return "-"
    if pd.api.types.is_bool_dtype(series):
        counts = values.value_counts()
        return " / ".join(f"{key} {int(count)}" for key, count in counts.items())
    if pd.api.types.is_numeric_dtype(series):
        return (f"{_fmt_num(values.min())} .. {_fmt_num(values.max())}"
                f"   mean {_fmt_num(values.mean())}")
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return f"{values.min()} .. {values.max()}"
    shown = [_truncate(str(v), _SUMMARY_VALUE_CHARS) for v in values.unique()[:2]]
    text = " | ".join(shown)
    return f"{text} .." if n_unique > len(shown) else text


# --------------------------------------------------------------------------- reports

def _directory_report(results_dir: Path) -> str:
    lines = _manifest_header(results_dir)
    files = _parquet_files(results_dir)
    if not files:
        lines.append(f"no parquet tables in {results_dir}")
        return "\n".join(lines)

    rows = [(p.stem, *_shape_from_footer(p), p.stat().st_size) for p in files]
    width = max(len(name) for name, _, _, _ in rows)
    lines.append("")
    lines.append(f"{'table'.ljust(width)}  {'rows':>7}  {'cols':>5}  {'size':>9}")
    for name, n_rows, n_cols, size in rows:
        lines.append(f"{name.ljust(width)}  {n_rows:>7}  {n_cols:>5}  {_fmt_size(size):>9}")
    names = [name for name, _, _, _ in rows]
    lines.append("")
    lines.append(f"one table in detail:  --table "
                 f"{'trial_data' if 'trial_data' in names else names[0]}")
    return "\n".join(lines)


def _table_report(path: Path, max_columns: int | None = None) -> str:
    frame = pd.read_parquet(path)
    lines = _manifest_header(path.parent)
    lines.append(f"{path.name}  {_fmt_size(path.stat().st_size)}  "
                 f"{len(frame)} rows x {frame.shape[1]} columns")

    if frame.shape[1] == 0:
        lines.append("")
        lines.append("the table has no columns")
        return "\n".join(lines)

    n_rows = len(frame)
    stats = []
    for name in frame.columns:
        series = frame[name]
        is_json = _is_json_column(series) if _maybe_text(series) else False
        non_null = int(series.notna().sum())
        n_unique = int(series.nunique(dropna=True))
        stats.append({
            "name": str(name),
            "dtype": _dtype_label(series, is_json),
            "non_null": non_null,
            "n_unique": n_unique,
            "values": _describe_values(series, n_unique),
        })

    if n_rows:
        empty = [s["name"] for s in stats if s["non_null"] == 0]
        if empty:
            lines.append(f"{len(empty)} column(s) entirely null: {', '.join(empty)}")
    else:
        lines.append("the table has no rows -- column statistics below are all empty")

    shown = stats if max_columns is None else stats[:max_columns]
    name_w = max(len(s["name"]) for s in shown)
    dtype_w = max(len(s["dtype"]) for s in shown)
    count_w = max(len(str(s["non_null"])) for s in shown)
    uniq_w = max(4, max(len(str(s["n_unique"])) for s in shown))

    lines.append("")
    lines.append(f"{'column'.ljust(name_w)}  {'dtype'.ljust(dtype_w)}  "
                 f"{'non-null'.rjust(count_w + 5)}  {'uniq'.rjust(uniq_w)}  values")
    for s in shown:
        pct = _fmt_pct(s["non_null"], n_rows)
        lines.append(
            f"{s['name'].ljust(name_w)}  {s['dtype'].ljust(dtype_w)}  "
            f"{str(s['non_null']).rjust(count_w)} {pct}  "
            f"{str(s['n_unique']).rjust(uniq_w)}  {s['values']}"
        )

    if len(shown) < len(stats):
        lines.append(f"... {len(stats) - len(shown)} more column(s); drop --max-columns for all")
    lines.append("")
    lines.append(f"one column in detail:  --column {shown[0]['name']}")
    return "\n".join(lines)


def _column_report(path: Path, column: str, rows: int = DEFAULT_ROWS) -> str:
    names = _column_names(path)
    if column not in names:
        near = difflib.get_close_matches(column, names, n=3, cutoff=0.6)
        if near:
            hint = (f" Did you mean: {', '.join(near)}? "
                    f"Drop --column to list all {len(names)}.")
        else:
            shown = ", ".join(names[:20])
            more = f", ... and {len(names) - 20} more" if len(names) > 20 else ""
            hint = f" It has {len(names)}: {shown}{more}"
        raise KeyError(f"{path.name} has no column {column!r}.{hint}")

    series = pd.read_parquet(path, columns=[column])[column]
    is_json = _is_json_column(series) if _maybe_text(series) else False
    non_null = int(series.notna().sum())
    n_rows = len(series)
    pct = _fmt_pct(non_null, n_rows).strip()

    lines = _manifest_header(path.parent)
    lines.append(f"{path.name} -> {column}")
    lines.append(f"dtype {_dtype_label(series, is_json)}   {n_rows} rows   "
                 f"{non_null} non-null ({pct})   {n_rows - non_null} null   "
                 f"{int(series.nunique(dropna=True))} distinct")

    values = series.dropna()
    lines.append("")
    if values.empty:
        lines.append("no values")
    elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        lines.append(f"min {_fmt_num(values.min())}   max {_fmt_num(values.max())}   "
                     f"mean {_fmt_num(values.mean())}   median {_fmt_num(values.median())}   "
                     f"sd {_fmt_num(values.std())}")
    elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        lines.append(f"first {values.min()}   last {values.max()}")
    else:
        counts = values.value_counts()
        lines.append(f"most common of {len(counts)} distinct value(s):")
        for key, count in counts.head(10).items():
            lines.append(f"  {count:>7}  {_truncate(str(key), _DETAIL_VALUE_CHARS)}")
        if len(counts) > 10:
            lines.append(f"  ... {len(counts) - 10} more")

    if rows > 0:
        head = series.head(rows)
        lines.append("")
        lines.append(f"first {min(rows, n_rows)} of {n_rows} row(s):")
        for index, value in head.items():
            text = "<null>" if pd.isna(value) else _truncate(str(value), _DETAIL_VALUE_CHARS)
            lines.append(f"  {str(index):>7}  {text}")
        if non_null and not head.notna().any():
            lines.append(f"  (all of the first {len(head)} rows are null; "
                         f"{non_null} non-null value(s) appear later)")
    return "\n".join(lines)


# ------------------------------------------------------------------------ entry point

def peek(path, *, table: str | None = None, column: str | None = None,
         rows: int = DEFAULT_ROWS, max_columns: int | None = None) -> str:
    """The report for `path`, as text. Print it; the caller decides where it goes.

    `path` is a `.parquet` file or a directory holding some (a session's
    `saved_analysis_results/`). Naming a `table` selects one inside a directory; naming
    a `column` narrows to that column alone.

    Returns the text rather than printing it so a notebook can hold it, a test can
    assert on it, and stdout stays the caller's to control.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")

    if path.is_dir():
        if table is None:
            if column is not None:
                raise ValueError(f"{path} is a directory: name a table (--table / table=) "
                                 f"before asking for a column")
            return _directory_report(path)
        target = path / f"{table}.parquet"
        if not target.exists():
            available = ", ".join(p.stem for p in _parquet_files(path)) or "none"
            raise FileNotFoundError(f"no table {table!r} in {path}. Available: {available}")
    else:
        if table is not None:
            raise ValueError("table= applies only when the path is a directory")
        target = path

    if column is not None:
        return _column_report(target, column, rows)
    return _table_report(target, max_columns)
