"""Configuration loading and project-relative path handling."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


def _find_project_root() -> Path:
    """Locate the repo root.

    Order: LOAN_RECOVERY_ROOT env var -> the folder two levels above src/loan_recovery
    (editable install / running from the repo) -> current working directory.
    """
    env_root = os.environ.get("LOAN_RECOVERY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "config.yaml").exists():
        return repo_root
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    config_path = Path(path or os.environ.get("LOAN_RECOVERY_CONFIG", PROJECT_ROOT / "config.yaml"))
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path: str | Path) -> Path:
    """Turn a config path (relative to the repo root) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
