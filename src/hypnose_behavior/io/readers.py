"""Low-level harp/aeon reader classes and the file-reading primitives built on them.

Single definition site for these eight names. ``io/loaders.py`` imports them from here
rather than redefining them, and this module deliberately imports nothing from
``hypnose_behavior`` -- it is a leaf, so ``detect_settings`` / ``detect_stage`` can use it without
creating a cycle back through ``loaders``.

The ``load*`` functions are tolerant by design: an unreadable or empty chunk is skipped
rather than propagated, and a root with no matching files yields an empty DataFrame
instead of raising ``ValueError: No objects to concatenate``. Callers detect "nothing
here" from the empty frame.
"""
import os
import json
from dotmap import DotMap
import pandas as pd
from pathlib import Path
from glob import glob
from aeon.io.reader import Reader, Csv
import aeon.io.api as api


class SessionData(Reader):
    """Extracts metadata information from a settings .jsonl file."""

    def __init__(self, pattern="Metadata"):
        super().__init__(pattern, columns=["metadata"], extension="jsonl")

    def read(self, file):
        """Returns metadata for the specified epoch."""
        with open(file) as fp:
            metadata = [json.loads(line) for line in fp]

        data = {
            "metadata": [DotMap(entry['value']) for entry in metadata]
        }
        timestamps = [api.aeon(entry['seconds']) for entry in metadata]

        return pd.DataFrame(data, index=timestamps, columns=self.columns)


class Video(Csv):
    """Extracts video frame metadata."""

    def __init__(self, pattern="VideoData"):
        super().__init__(pattern, columns=["hw_counter", "hw_timestamp", "_frame", "_path", "_epoch"])
        self._rawcolumns = ["Time"] + self.columns[0:2]

    def read(self, file):
        """Reads video metadata from the specified file."""
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["_frame"] = data.index
        data["_path"] = os.path.splitext(file)[0] + ".avi"
        data["_epoch"] = file.parts[-3]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data


class TimestampedCsvReader(Csv):
    def __init__(self, pattern, columns):
        super().__init__(pattern, columns, extension="csv")
        self._rawcolumns = ["Time"] + columns

    def read(self, file):
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["Seconds"] = data["Time"]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data


def load_json(reader: SessionData, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_*.{reader.extension}"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(Path(file))
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out


def load(reader: Reader, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_{reader.register.address}_*.bin"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(file)
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out


def load_video(reader: Video, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_*.csv"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(Path(file))
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out


def concat_digi_events(series_low: pd.DataFrame, series_high: pd.DataFrame) -> pd.DataFrame:
    """Concatenate seperate high and low dataframes to produce on/off vector"""
    data_off = ~series_low[series_low==True]
    data_on = series_high[series_high==True]
    return pd.concat([data_off, data_on]).sort_index()


def load_csv(reader: Csv, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(reader.pattern).joinpath(reader.pattern)}_*.{reader.extension}"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(Path(file))
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out
