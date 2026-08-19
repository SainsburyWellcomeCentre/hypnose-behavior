from __future__ import annotations

from pathlib import Path

from hypnose_helpers.io.paths import DataLocations, RAW_SUBDIR, DERIV_SUBDIR, env_path

# Resolution order for the data roots (highest priority first):
#   1. HYPNOSE_* environment variables  (deliberate override: CI, the QC sandbox, one-offs)
#   2. the active data-location profile  (configs/data_locations.yml + the per-machine
#      git-ignored configs/data_locations.local.yml selecting `active`)
#   3. legacy fallback: the data/rawdata symlink (and server/derivatives) under the repo
# So everyday use is driven by the config; env vars override it temporarily without
# touching the config file; the symlink remains a fallback.


def get_repo_root() -> Path:
    """
    Returns the root of the hypnose-behavior repository,
    assuming standard src/ layout.
    """
    return Path(__file__).resolve().parents[3]
    # paths.py → io → hypnose_behavior → src → hypnose-behavior


_locations = DataLocations(
    config_dir=get_repo_root() / "configs",
    data_root=get_repo_root() / "data",
)

# Bound methods, so `get_rawdata_root()` and friends keep working unchanged at ~17 call
# sites -- and still expose `.cache_clear()`, which qc/_common.py relies on to redirect
# derivatives into a temp dir per session.
load_profiles = _locations.load_profiles
get_active = _locations.get_active
set_active = _locations.set_active
reload = _locations.reload
get_data_root = _locations.get_data_root
get_rawdata_root = _locations.get_rawdata_root
get_server_root = _locations.get_server_root
get_derivatives_root = _locations.get_derivatives_root

# Used by scripts/set_data_location.py to report where the selection was written.
_local_path = _locations._local_path
_profiles_path = _locations._profiles_path
_active_profile = _locations._active_profile
_env_path = env_path

__all__ = [
    "get_repo_root", "get_data_root", "get_rawdata_root", "get_server_root",
    "get_derivatives_root", "load_profiles", "get_active", "set_active", "reload",
    "RAW_SUBDIR", "DERIV_SUBDIR",
]
