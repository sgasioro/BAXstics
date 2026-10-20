"""In-silico benchmark of BAX strategies on a hydrogenolysis surrogate, on abbot.

For each strategy, start from a few random grid points and let the policy pick
the rest (in batches of 1 or 2), tracking how many points of the true target set
it has collected and how well the GP's estimate of the target set matches it.
Saves a plot of the maximum acquired yield per selectivity bin for SwitchBAX.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warnings

import numpy as np
import pandas as pd
import torch
from abbot import MockDatasetExperiment
from matplotlib import pyplot as plt
from tqdm import tqdm

from baxstics import XCOLS, YCOLS, TopKPerBin, make_campaign, raw_mean_std

warnings.filterwarnings("ignore")

_style = "../multibax-sklearn/src/matplotlib.rc"  # the paper's plot style, if checked out
if os.path.exists(_style):
    plt.style.use(_style)

seed = 0


def build_batch_array(Nmax, addtarget):
    """Random batch sizes of 1 or 2 summing to ``addtarget``."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        intarrs = rng.integers(1, 3, size=Nmax)
        sumarrs = np.cumsum(intarrs)
        if np.any(sumarrs == addtarget):
            Nid = np.where(sumarrs == addtarget)[0]
            return intarrs[: Nid[0] + 1]
    raise ValueError("no batch sequence sums to the target")


def max_in_bin(y_here, bins_start, bins_end):
    max_y = []
    for i in range(len(bins_start)):
        bin_mask = (y_here[:, 1] >= bins_start[i]) & (y_here[:, 1] <= bins_end[i])
        y_in_bin = y_here[bin_mask, 0]
        max_y.append(np.amax(y_in_bin) if len(y_in_bin) else 0)

    return max_y


def add_noise_fn(X, Y, noise_kwargs):
    rng = np.random.default_rng(seed)
    if noise_kwargs["type"] == "standard":
        sigma = noise_kwargs["scale"]
        # Scaling noise by dataset standard deviation
        noise = sigma * rng.normal(size=(Y.shape), scale=np.std(Y, axis=0))
        Y = Y + noise
    elif noise_kwargs["type"] == "yield_skew":
        sigma = noise_kwargs["scale"]
        noise = sigma * rng.normal(size=(Y.shape), scale=np.std(Y, axis=0))
        # Add some skewness so that higher yields are noisier
        noise[:, 0] *= 1 + noise_kwargs["skew_size"] * Y[:, 0] / Y[:, 0].max(axis=0)
        Y = Y + noise

    Y[Y < 0] = 0.0  # can't have negative yields
    Y[:, 1] = np.round(Y[:, 1], 0)  # Need Median to be an integer

    return X, Y


def fetch_data(simulator_type, add_noise, noise_kwargs):
    if simulator_type == "parabola":
        datapath = "../data/3D_data1A1.csv"
    else:
        datapath = "../data_sathya_edits/nn_fitted_interpolation.csv"
    df = pd.read_csv(datapath)
    X = df[XCOLS].to_numpy(dtype=float)
    Y = df[YCOLS].to_numpy(dtype=float)
    if add_noise:
        X, Y = add_noise_fn(X, Y, noise_kwargs)
    return X, Y


def get_n_obtained(collected_X, target_X):
    """How many target designs are among the collected ones."""
    hits = torch.isclose(target_X.unsqueeze(1), collected_X.unsqueeze(0)).all(dim=-1)
    return int(hits.any(dim=1).sum())


def get_jaccard(predicted_ids, true_target_ids):
    a, b = set(predicted_ids), set(true_target_ids)
    return len(a & b) / len(a | b) if a | b else 1.0


def run_strategy(strategy, X, Y, goal, n_initial, batches_arr, n_posterior_samples, repeat):
    """One campaign replaying the grid's known outputs; returns the history and
    per-shot metrics."""
    grid = torch.as_tensor(X, dtype=torch.double)
    bounds = {name: (X[:, j].min(), X[:, j].max()) for j, name in enumerate(XCOLS)}
    campaign = make_campaign(
        grid, goal, strategy=strategy, n_posterior_samples=n_posterior_samples, bounds=bounds
    )
    experiment = MockDatasetExperiment(campaign.spec, grid, torch.as_tensor(Y))
    true_target_ids = goal.identify_subspace(torch.as_tensor(Y), grid)
    target_X = grid[true_target_ids]

    np.random.seed(repeat)  # same initial points for every strategy
    torch.manual_seed(repeat)
    for i in np.random.choice(len(X), n_initial):
        campaign.record(grid[i], torch.as_tensor(Y[i]))

    n_obtained, jaccard, switched = [], [], []
    for batch_size in tqdm(batches_arr):
        campaign.step(experiment, lambda y: y, n=int(batch_size))
        mean, _ = raw_mean_std(campaign, grid)
        predicted = goal.identify_subspace(mean, grid)
        collected_X = campaign.history.designs
        # For metrics, one entry per acquired point in the batch
        for i_batch in range(batch_size - 1, -1, -1):
            n_obtained.append(get_n_obtained(collected_X[: len(collected_X) - i_batch], target_X))
            jaccard.append(get_jaccard(predicted, true_target_ids))
            switched.append(campaign.policy.switched)
    metrics = {"n_obtained": n_obtained, "jaccard_posterior_index": jaccard, "switch_strategy": switched}
    return campaign.history, metrics


if __name__ == "__main__":
    X, Y = fetch_data("parabola", True, {"type": "yield_skew", "scale": 0.01, "skew_size": 20})

    bins_start = [9, 11, 13, 15, 17]
    bins_end = [10, 12, 14, 16, 18]
    goal = TopKPerBin(list(zip(bins_start, bins_end)), k=5, epsilon=0.2)

    n_posterior_samples = 5  # relevant for InfoBAX / SwitchBAX's fallback
    n_initial = 5  # Number of initial datapoints
    n_iters = 50  # Number of measurements to be performed
    batches_arr = build_batch_array(n_iters, n_iters)
    n_repeats = 1  # Repeats with different dataset initializations

    strategies = ["US", "MeanBAX", "SwitchBAX"]
    metrics = {s: {"n_obtained": [], "jaccard_posterior_index": [], "switch_strategy": []} for s in strategies}
    collected_collection = {}

    for strategy in strategies:
        for j in range(n_repeats):  # to see variance w.r.t initial datapoint choice
            print(f"Using strategy: {strategy}, repeat number: {j}")
            history, m = run_strategy(
                strategy, X, Y, goal, n_initial, batches_arr, n_posterior_samples, j
            )
            for key, values in m.items():
                metrics[strategy][key].append(values)
        collected_collection[strategy] = history.outputs.numpy()

    collected_Y = collected_collection["SwitchBAX"]
    acq_bin_maxes = [max_in_bin(collected_Y[:i], bins_start, bins_end) for i in range(len(collected_Y))]
    actual_max = max_in_bin(Y, bins_start, bins_end)
    for i in range(len(bins_start)):
        plt.plot(np.asarray(acq_bin_maxes)[:, i], c=f"C{i}", label=f"{bins_start[i]}-{bins_end[i]}")

    plt.hlines(
        actual_max,
        0,
        len(collected_Y),
        colors=[f"C{i}" for i in range(len(bins_start))],
        ls="dashed",
    )
    plt.ylabel("Max Acquired Yield")
    plt.xlabel("Number Acquired Points")
    plt.legend()
    os.makedirs("../plot_outputs", exist_ok=True)
    plt.savefig("../plot_outputs/sim_SKEW_20_1%_PB_MIX.png")
