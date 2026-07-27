"""Shared pytest helpers for the ED triage test suite. Not a test file itself."""

import sys
from pathlib import Path

# Allow `from src import ...` when pytest is run from the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data
from src.utils import load_config

CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_real_sample(n=50, seed=0):
    """Load a small random sample of the triage CSV.

    Requires the dataset at config.yaml's data.raw_path 
    """
    cfg = load_config(CONFIG_PATH)
    raw = data.load_raw(cfg["data"]["raw_path"])
    return raw.sample(n=n, random_state=seed)
