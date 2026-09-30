import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Isolated workspace so tests never touch real experiments/models/selected.yaml."""
    ws = tmp_path_factory.mktemp("ws")
    os.environ["ENTIQ_WORKSPACE"] = str(ws)
    return ws


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory, workspace):
    from src.data.synthetic import write_synthetic

    d = tmp_path_factory.mktemp("synth")
    write_synthetic(d, n_train=400, n_test=250, seed=3)
    return d


@pytest.fixture(scope="session")
def small_cfg(workspace, synth_dir):
    from src.utils.config import load_config

    return load_config(overrides={
        "paths.dataset_dir": str(synth_dir), "model.cv_folds": 3,
        "model.params.hgb": {"max_iter": 60, "learning_rate": 0.1, "max_leaf_nodes": 15},
        "decision.grid.threshold": [0.3, 0.5, 0.7], "decision.grid.singleton_delta": [0.0, 0.1],
        "decision.grid.min_gap": [0.0], "decision.grid.max_per_source": [None], "decision.grid.exclusive_assignment": [False],
    })
