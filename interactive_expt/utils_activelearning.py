"""Helpers for the live experiment app (``main_run.py``), on abbot.

The dataset CSV holds the whole candidate grid of reaction conditions; rows not
yet measured carry a negative ``Yield``. Each proposal rebuilds a campaign from
the measured rows, refits the GPs and asks SwitchBAX for the next design(s).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from pandas import DataFrame

from baxstics import XCOLS, YCOLS, TopKPerBin, grid_index, make_campaign, raw_mean_std

DATASETS_DIR = Path("./data_expt")
DATASETS_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("./models_expt")
MODELS_DIR.mkdir(exist_ok=True)

# The goal: the highest-yield condition in each band of median carbon number.
GOAL = TopKPerBin([(9, 11), (11, 13), (13, 15), (15, 17), (17, 19)], k=1, epsilon=0.0)
STRATEGY = "SwitchBAX"
N_POSTERIOR_SAMPLES = 10


def fetch_data(df: DataFrame):
    """The grid ``X``, outputs ``Y`` (physical units) and which rows are measured."""
    X = df[XCOLS].to_numpy(dtype=float)
    Y = df[YCOLS].to_numpy(dtype=float)
    measured = Y[:, 0] >= 0
    return X, Y, measured


def propose(df: DataFrame, batch_size: int = 1, seed: int = 0):
    """Next grid rows to measure, plus the GP's mean and std ``(N, 2)`` over the
    grid and the campaign. A batch of 2 is chosen greedily: the second pick
    assumes the first returns the GP's predicted value (kriging believer)."""
    torch.manual_seed(seed)
    X, Y, measured = fetch_data(df)
    grid = torch.as_tensor(X, dtype=torch.double)
    campaign = make_campaign(grid, GOAL, strategy=STRATEGY, n_posterior_samples=N_POSTERIOR_SAMPLES)
    for i in np.flatnonzero(measured):
        campaign.record(grid[i], torch.as_tensor(Y[i]))
    next_ids = [grid_index(grid, cand) for cand in campaign.suggest(batch_size)]
    mean, std = raw_mean_std(campaign, grid)
    return next_ids, mean.numpy(), std.numpy(), campaign


def save_model(campaign, paths) -> None:
    """Write the fitted GPs' hyperparameters as JSON to each of ``paths``."""
    model = campaign.current_model()
    hypers = {
        name: {
            "lengthscale": sub.covar_module.lengthscale.detach().reshape(-1).tolist(),
            "noise": float(sub.likelihood.noise.detach().reshape(-1)[0]),
            "mean": float(sub.mean_module.constant.detach()),
        }
        for name, sub in zip(YCOLS, model.models)
    }
    for path in paths:
        with open(path, "w") as f:
            json.dump(hypers, f, indent=2)
