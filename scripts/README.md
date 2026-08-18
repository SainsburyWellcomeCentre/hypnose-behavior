# scripts

Terminal entry points for running the pipeline. They are thin wrappers over
functions in `src/hypnose_behavior/` and contain no analysis logic. Run from the repo
root in the project conda environment; the scripts add `src/` to the path, so no
install is required.

| Script | What it does |
| --- | --- |
| `run_trial_classification.py` | Trial classification → writes `trial_data` (+ summary) to derivatives |
| `run_metrics_analysis.py` | Behavioural metric analysis (reads saved classification results) |
| `batch_process.py` | Runs trial classification, then metric analysis |
| `parquet_peek.py` | Read-only: what is in a saved parquet table |

Run trial classification **before** metric analysis — metrics read the saved
classification results from the derivatives tree.

`parquet_peek.py` takes the same `--subjids`/`--dates`, but reads the *derivatives*
tree rather than rawdata: it writes nothing and runs no analysis, so it shows what
the other three already wrote. See [its own section](#parquet_peekpy) below.

## Arguments (shared)

| Argument | Meaning |
| --- | --- |
| `--subjids ID [ID ...]` | One or more subject ids. Omit to run **all** subjects found in rawdata. |
| `--dates D [D ...]` | One or more specific dates `YYYYMMDD`. |
| `--date-range START END` | Inclusive date range `YYYYMMDD YYYYMMDD` (alternative to `--dates`). |

`--dates` and `--date-range` are mutually exclusive; omit both to run **all dates**
for the selected subjects. Subjects/dates with no data are validated and skipped
with a clear message (`hypnose_behavior.qc.validate.validate_subject`).

Script-specific:

| Argument | Script(s) | Meaning |
| --- | --- | --- |
| `--no-save` | all | Do not write derivatives / metrics output |
| `--no-summary` | run_trial_classification | Suppress the merged per-session summary |
| `--verbose` | run_trial_classification, batch_process | Verbose per-run logging |
| `--quiet` | run_metrics_analysis | Suppress per-session logging |
| `--protocol STR` | run_metrics_analysis, batch_process | Only sessions whose stage name contains `STR` |

## Examples

```bash
# one subject, one date
python scripts/run_trial_classification.py --subjids 53 --dates 20260528

# one subject, several specific dates
python scripts/run_trial_classification.py --subjids 53 --dates 20260520 20260528

# multiple subjects, a date range
python scripts/run_trial_classification.py --subjids 53 58 --date-range 20260501 20260531

# all subjects, all dates
python scripts/run_trial_classification.py

# metrics for a protocol only (run after classification)
python scripts/run_metrics_analysis.py --subjids 53 --date-range 20260501 20260531 --protocol singrew

# classification + metrics in one go
python scripts/batch_process.py --subjids 53 58 --dates 20260528
```

## `parquet_peek.py`

Parquet is the format and CSV is off by default, so "open the CSV and look" stopped
being an answer for anything written after Phase 7b.3. This is the replacement.
Three views, narrowing:

```bash
# every table in the session: rows, columns, size on disk
python scripts/parquet_peek.py --subjids 57 --dates 20260709

# one table: ONE LINE PER COLUMN -- dtype, non-null count, distinct count, value range
python scripts/parquet_peek.py --subjids 57 --dates 20260709 --table trial_data

# one column, with its actual values
python scripts/parquet_peek.py --subjids 57 --dates 20260709 \
    --table trial_data --column response_time_ms
python scripts/parquet_peek.py --subjids 57 --dates 20260709 \
    --table trial_data --column odor_sequence --rows 50
```

It never prints the frame: `trial_data` is 58–73 columns wide and hundreds of rows
long, so a row-shaped view is unreadable in a terminal. One line per column is, and
`--max-columns N` trims it further on a narrow screen.

| Argument | Meaning |
| --- | --- |
| `--subjids ID [ID ...]` | **required** — a peek is scoped to a subject, never the whole tree |
| `--dates` / `--date-range` | as the other scripts; omit for every analysed session of that subject |
| `--table NAME` | which table; omit for an inventory of all of them |
| `--column NAME` | that column alone, with values. A name that is not there **raises**, and suggests near matches |
| `--rows N` | values to list for `--column` (default 20; `0` = statistics only) |
| `--max-columns N` | show only the first N columns (default: all) |

Two things it shows that a bare `pd.read_parquet` does not:

- **`(JSON)` on a column's dtype.** Object columns are JSON-encoded on the way into
  the parquet as well as the CSV, so `odor_sequence` is on disk as the *string*
  `'["OdorG", "OdorE", "OdorB"]'`. Reporting it as a plain string column would be
  misleading about exactly the columns worth inspecting.
- **A manifest header** naming the protocol mode and the commit that wrote the file —
  which is how you find out that a saved session predates the code you are reading.

It reads only; it writes nothing anywhere, including under a read-only mount.
