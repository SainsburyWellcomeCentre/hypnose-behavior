"""A loader's frames for one cohort selection, kept on disk and reused while still current.

    frames = load_or_build("discrimination_learning", key, sources, code, build)

- One folder per selection: ``<derivatives>/cache/<name>/<digest of key>/``, a parquet file
  per frame and ``manifest.json`` (the selection, a stamp and each frame's dtypes).
- **Current** means the stamp still matches: ``sources`` (size and mtime of every file the
  loader reads, per matched session, from `fingerprint`) and ``code`` (`code_digest` of the
  modules that build the frames). A re-analysed, added or removed session, or an edit to
  those modules, rebuilds.
- Paths in the stamp are relative to the derivatives tree and mtimes whole seconds, so one
  cache serves every machine that mounts the same server.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import derivatives

__all__ = ["CACHE_DIRNAME", "cache_folder", "code_digest", "fingerprint", "load_or_build"]

CACHE_DIRNAME = "cache"

_MANIFEST = "manifest.json"


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def cache_folder(name: str, key, root=None) -> Path:
    """Where the selection ``key`` of loader ``name`` is kept; ``root`` replaces
    ``<derivatives>/cache``."""
    base = Path(root) if root is not None else derivatives.root / CACHE_DIRNAME
    return base / name / _digest(key)


def code_digest(*modules) -> str:
    """Digest of the modules' source, line endings normalized."""
    h = hashlib.sha1()
    for module in modules:
        h.update(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def _scan(folder: Path, names) -> dict:
    """``{file: [size, mtime]}`` of the named files directly in ``folder``."""
    try:
        with os.scandir(folder) as entries:
            stats = {e.name: e.stat() for e in entries if e.name in names and e.is_file()}
    except OSError:
        return {}
    return {n: [s.st_size, int(s.st_mtime)] for n, s in stats.items()}


def fingerprint(results_dirs, names) -> dict:
    """``{session: {file: [size, mtime]}}`` of the files named ``names`` in each results
    directory, in either layout; a session with none of them maps to ``{}``.

    One directory listing per folder, so no file is opened.
    """
    names = set(names)
    subfolders = sorted({layout.results_subfolder(n) for n in names} - {None})
    out = {}
    for results_dir in map(Path, results_dirs):
        files = _scan(results_dir, names)
        for sub in subfolders:
            files.update({f"{sub}/{n}": v for n, v in _scan(results_dir / sub, names).items()})
        out[f"{results_dir.parent.parent.name}/{results_dir.parent.name}"] = files
    return out


def _changed(old: dict, new: dict) -> int:
    """Sessions added, removed or with a file changed."""
    return sum(old.get(s) != new.get(s) for s in set(old) | set(new))


def _restore(frame: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """The frame's columns back in the dtypes they were written from (an all-bool object
    column reads back as bool, a missing value in one as None)."""
    for column, dtype in dtypes.items():
        if str(frame[column].dtype) == dtype:
            continue
        if dtype == "object":
            values = frame[column].astype(object)
            frame[column] = values.where(values.notna(), np.nan)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame


def _read(folder: Path, stamp: dict, label: str, verbose: bool):
    """The cached frames if the folder holds a current copy, else None."""
    try:
        manifest = json.loads((folder / _MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        warnings.warn(f"{label}: unreadable cache manifest in {folder} ({error}); rebuilding.",
                      RuntimeWarning, stacklevel=3)
        return None
    if manifest.get("code") != stamp["code"]:
        if verbose:
            print(f"[{label}] cache stale (loader code changed): rebuilding")
        return None
    if manifest.get("sources") != stamp["sources"]:
        if verbose:
            n = _changed(manifest.get("sources") or {}, stamp["sources"])
            print(f"[{label}] cache stale ({n} session(s) added, removed or re-analysed): "
                  f"rebuilding")
        return None
    try:
        frames = {name: _restore(pd.read_parquet(folder / f"{name}.parquet"), dtypes)
                  for name, dtypes in manifest["dtypes"].items()}
    except (OSError, KeyError, ValueError) as error:
        warnings.warn(f"{label}: unreadable cache in {folder} ({error}); rebuilding.",
                      RuntimeWarning, stacklevel=3)
        return None
    if verbose:
        print(f"[{label}] cache read: {len(stamp['sources'])} session(s) from {folder}")
    return frames


def _replace(path: Path, write) -> None:
    """``write(tmp)``, then move it over ``path``, so a reader never sees half a file."""
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def _write(folder: Path, key, stamp: dict, frames: dict, label: str, verbose: bool) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            _replace(folder / f"{name}.parquet", lambda p, f=frame: f.to_parquet(p, index=False))
        manifest = {"selection": key, "created": datetime.now().isoformat(timespec="seconds"),
                    **stamp, "rows": {n: len(f) for n, f in frames.items()},
                    "dtypes": {n: {c: str(t) for c, t in f.dtypes.items()}
                               for n, f in frames.items()}}
        _replace(folder / _MANIFEST, lambda p: p.write_text(
            json.dumps(manifest, indent=1, default=str), encoding="utf-8"))
    except OSError as error:
        warnings.warn(f"{label}: could not write the cache to {folder} ({error}).",
                      RuntimeWarning, stacklevel=3)
        return
    if verbose:
        print(f"[{label}] cache written: {folder}")


def load_or_build(name: str, key, sources: dict, code: str, build, *, root=None,
                  refresh: bool = False, label=None, verbose: bool = True) -> dict:
    """``build()``'s frames for the selection ``key``, read from the cache while it is current.

    - ``sources``: `fingerprint` of the matched sessions; ``code``: `code_digest`.
    - ``refresh`` rebuilds and rewrites whatever the cache holds.
    - A cache that cannot be written warns and returns the built frames.
    """
    label = label or name
    folder = cache_folder(name, key, root)
    stamp = {"code": code, "sources": sources}
    frames = None if refresh else _read(folder, stamp, label, verbose)
    if frames is None:
        frames = build()
        _write(folder, key, stamp, frames, label, verbose)
    return frames
