import sys

sys.path.extend(["../multibax-sklearn"])
from src.models import MGPR, fit_matern_hypers
import warnings


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import MinMaxScaler


from src.utils import XY_from_csv, random_sampling_no_replace
from src.acquisition import run_acquisition

from src.models import MGPR, fit_matern_hypers
from src.metrics import get_n_obtained, get_jaccard_posterior


import warnings

warnings.filterwarnings("ignore")

plt.style.use("../multibax-sklearn/src/matplotlib.rc")
seed = 0


def build_batch_array(Nmax, addtarget):
    rng = np.random.default_rng(seed)
    exitcond = False
    ct = 0
    ct += 1
    while not exitcond:
        intarrs = rng.integers(1, 3, size=Nmax)
        sumarrs = np.cumsum(intarrs)
        if np.any(sumarrs - addtarget == 0):
            Nid = np.where(sumarrs == addtarget)[0]

            return intarrs[: Nid[0] + 1]
        ct += 1
        if ct > 1000:
            raise ValueError("Hmmm")


def max_in_bin(Y, collected_ids, bins_start, bins_end):
    y_here = Y[collected_ids]

    max_y = []
    for i in range(len(bins_start)):
        # Should bins be edge inclusive?
        bin_mask = (y_here[:, 1] >= bins_start[i]) & (y_here[:, 1] <= bins_end[i])
        y_in_bin = y_here[bin_mask, 0]

        if len(y_in_bin) == 0:
            max_y.append(0)
        else:
            max_y.append(np.amax(y_in_bin))

    return max_y


def run_hall_acquisition(
    x_train,
    y_train,
    X,
    Y,
    strategy,
    algorithm,
    model,
    collected_ids,
    n_posterior_samples,
    prevent_requery=True,
    batch_strategy=None,
):
    # Ideally this would just be handled in an argument to run_acquisition, but to keep the repo
    _, _, trained_model, collected_ids, acquisition_function, switch_strategy = (
        run_acquisition(
            x_train.copy(),
            y_train.copy(),
            X.copy(),
            Y.copy(),
            strategy,
            algorithm,
            model,
            collected_ids.copy(),
            n_posterior_samples,
            prevent_requery=prevent_requery,
        )
    )
    # Get what would be the next acquired index
    next_id = collected_ids[-1]
    x_next = X[next_id]

    # Swap out next y with posterior mean prediction
    if batch_strategy == "post_mean":
        posterior_mean, posterior_std = trained_model.predict(x_next.reshape(1, -1))
        y_next = posterior_mean
    else:
        y_next = Y[next_id]

    # Add to training set
    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))

    return (
        x_train,
        y_train,
        trained_model,
        collected_ids,
        acquisition_function,
        switch_strategy,
    )


def batch_acquisition(
    x_train,
    y_train,
    X,
    Y,
    strategy,
    algorithm,
    model,
    collected_ids,
    n_posterior_samples,
    batch_size,
    prevent_requery=True,
    batch_strategy="post_mean",
):
    # Copy so we don't mix up real vs hallucinated
    x_train_here = x_train.copy()
    y_train_here = y_train.copy()

    X_here = X.copy()
    Y_here = Y.copy()

    collected_ids_here = collected_ids.copy()

    # Loop through hallucination and get next idxs
    hall_idxs = []
    for i_hall in range(batch_size):
        (
            x_train_here,
            y_train_here,
            trained_model_here,
            collected_ids_here,
            acquisition_function_here,
            switch_strategy,
        ) = run_hall_acquisition(
            x_train_here,
            y_train_here,
            X_here,
            Y_here,
            strategy,
            algorithm,
            model,
            collected_ids_here,
            n_posterior_samples,
            prevent_requery=True,
            batch_strategy=batch_strategy,
        )
        hall_idxs.append(collected_ids_here[-1])

    # Return batch
    x_next = X[hall_idxs]
    y_next = Y[hall_idxs]

    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))

    return (
        x_train,
        y_train,
        trained_model_here,
        collected_ids_here,
        acquisition_function_here,
        switch_strategy,
    )


def add_noise_fn(X, Y, noise_kwargs):
    rng = np.random.default_rng(seed)
    if noise_kwargs["type"] == "standard":
        sigma = noise_kwargs["scale"]
        # Scaling noise by Dataset standard Deviation
        noise = sigma * rng.normal(size=(Y.shape), scale=np.std(Y, axis=0))
        Y = Y + noise
    else:
        Y = Y
    Y[Y < 0] = 0.0  # can't have negative yields
    Y[:, 1] = np.round(Y[:, 1], 0)  # Need Median to be an integer

    return X, Y


def fetch_data(simulator_type, add_noise, noise_kwargs):
    if simulator_type == "parabola":
        datapath = "../data/3D_data1A1.csv"
    else:
        datapath = "../data_sathya_edits/nn_fitted_interpolation.csv"
    xcols = ["Time", "Temperature", "Pressure"]
    ycols = ["Yield", "Median"]
    X, Y = XY_from_csv(
        datapath,
        columns_x=xcols,
        columns_y=ycols,
    )
    if add_noise:
        X, Y = add_noise_fn(X, Y, noise_kwargs)

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    X_norm = x_scaler.fit_transform(X)
    # Y_norm = y_scaler.fit_transform(Y) # Ideal Normalization

    # Normalization based on parameters
    yield_min = 0
    yield_max = 100
    selectivity_min = 8
    selectivity_max = 20

    y_scaler.fit(np.array([[yield_min, selectivity_min], [yield_max, selectivity_max]]))
    Y_norm = y_scaler.fit_transform(Y)
    scalers = [x_scaler, y_scaler]

    return X, Y, X_norm, Y_norm, scalers


from src.algorithms import SubsetAlgorithm


class TopKBinAlgorithm(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        bins_start = self.user_algo_params["bins_start"]
        bins_end = self.user_algo_params["bins_end"]
        bin_epsilon = self.user_algo_params["bin_epsilon"]  # Overflow

        topk_num = int(self.user_algo_params["num_points_per_bin"])  # Target topk

        list_of_max_ids = []

        f_1 = f_x[:, 0]  # only take the first property column, yield
        f_2 = f_x[:, 1]  # only take the first property column, median

        for i in range(0, len(bins_start)):
            ids1 = set(np.where(f_2 >= bins_start[i] - bin_epsilon)[0])
            ids2 = set(np.where(f_2 <= bins_end[i] + bin_epsilon)[0])
            ids3 = list(ids1.intersection(ids2))
            vals = f_1[ids3]
            ids3 = np.array(ids3)
            if len(ids3) != 0:
                list_of_max_ids = list_of_max_ids + list(
                    ids3[np.flip(np.argsort(vals))[:topk_num]]
                )

        return list(set(list_of_max_ids))


if __name__ == "__main__":
    X, Y, X_norm, Y_norm, scalers = fetch_data(
        "parabola", True, {"type": "standard", "scale": 0.01}
    )

    bins_start = [9, 11, 13, 15, 17]
    bins_end = [10, 12, 14, 16, 18]
    algorithm = TopKBinAlgorithm(
        user_algo_params={
            "scalers": scalers,
            "bins_start": bins_start,
            "bins_end": bins_end,
            "num_points_per_bin": 5,
            "bin_epsilon": 0.2,
        }
    )

    from logging import StreamHandler

    n_features = 3  # change to 3
    n_properties = 2

    kernel_initial = ConstantKernel(
        constant_value=1.0, constant_value_bounds=[0.01, 3.0]
    ) * Matern(
        nu=5 / 2,
        length_scale=n_features * [1.0],
        length_scale_bounds=n_features * [[0.01, 3.0]],
    ) + WhiteKernel(
        noise_level=0.1, noise_level_bounds="fixed"
    )
    kernel_initial_list = n_properties * [kernel_initial]
    plotting = True
    prevent_requery = False  # (default this somewhere else)
    n_posterior_samples = (
        5  # relevant for InfoBAX and mixedBAX ~ generally 10-15 is a good number
    )
    n_initial = 5  # Number of initial datapoints
    n_iters = 50  # Number of measurements to be performed
    batches_arr = build_batch_array(n_iters, n_iters)
    n_loops = len(batches_arr)

    plot_frequency = 50
    n_repeats = 1  # Repeats with different dataset initializations
    fixed_hypers = False  # We should really test this # true uses pre-fixed hypers fit to the entire dataset
    adaptive_fit_freq = (
        2  # Frequency to fit hypers; only applies if fixed_hypers = False
    )

    metrics = {
        "SwitchBAX": {
            "n_obtained": [],
            "jaccard_posterior_index": [],
            "switch_strategy": [],
        },
        "US": {"n_obtained": [], "jaccard_posterior_index": [], "switch_strategy": []},
        "MeanBAX": {
            "n_obtained": [],
            "jaccard_posterior_index": [],
            "switch_strategy": [],
        },
        "InfoBAX": {
            "n_obtained": [],
            "jaccard_posterior_index": [],
            "switch_strategy": [],
        },
    }
    all_ids = list(np.arange(0, len(X_norm)))
    true_target_ids = algorithm.identify_subspace(f_x=Y_norm, x=X_norm)
    # Baseling for random sampling without replacement (expectation of hypergeometric distribution)
    random_sampling = [
        random_sampling_no_replace(len(X_norm), len(true_target_ids), n)
        for n in range(n_initial, n_iters)
    ]

    # Baseline for best possible acquisition (i.e. acquire a target point at each iteration; need an "oracle" to do this)
    if n_iters <= len(true_target_ids):
        best_possible_n_obtained = np.arange(n_initial, n_iters + n_initial)
    else:
        best_possible_n_obtained = list(
            np.arange(n_initial, len(true_target_ids))
        ) + list(
            len(true_target_ids) * np.ones(n_iters + n_initial - len(true_target_ids))
        )

    # Acquisition functions that use BAX for subset estimation
    strategies = ["US", "MeanBAX", "SwitchBAX"]

    collected_collection = {}
    for strat in strategies:
        collected_collection.update({strat: None})
    # Calculate hypers based on the entire dataset; this is not possible in a real experiment but allows us to compare acquisition fn to acquisition fn
    if fixed_hypers:
        kernel_list = fit_matern_hypers(
            x_train=X_norm,
            y_train=Y_norm,
            kernel_list=kernel_initial_list,
            n_restarts_optimizer=1,
        )

    for strategy in strategies:
        for j in range(n_repeats):  # to see variance w.r.t initial datapoint choice

            print("Using strategy: {}, repeat number: {}".format(strategy, j))

            np.random.seed(j)  # make sure all strategies get same initial points
            train_indices = list(np.random.choice(all_ids, n_initial))
            x_train = X_norm[train_indices]
            y_train = Y_norm[train_indices]

            collected_ids = list(train_indices)
            n_obtained_list = []
            jaccard_posterior_list = []
            switch_list = []

            for i in tqdm(range(n_loops)):
                # Make batch align with total number

                batch_size_here = batches_arr[i]

                # Adaptive hyperparameter fitting
                if (i % adaptive_fit_freq == 0) and (fixed_hypers == False):
                    kernel_list = fit_matern_hypers(
                        x_train=x_train,
                        y_train=y_train,
                        kernel_list=kernel_initial_list,
                    )

                # Define GP model with fixed, fitted hypers. Note, we need this so that all the n_posterior models for InfoBAX have the same kernel
                multi_gpr = MGPR(kernel_list=kernel_list)

                # Acquire next index
                (
                    x_train,
                    y_train,
                    model,
                    collected_ids,
                    acquisition_function,
                    switch_strategy,
                ) = batch_acquisition(
                    x_train,
                    y_train,
                    X_norm,
                    Y_norm,
                    strategy,
                    algorithm,
                    multi_gpr,
                    collected_ids,
                    n_posterior_samples,
                    batch_size_here,
                )

                # Calculate metrics
                posterior_mean, posterior_std = model.predict(X_norm)

                predicted_target_ids = algorithm.identify_subspace(
                    f_x=posterior_mean, x=X_norm
                )

                # For metrics, duplicate and/or keep track for each batch point
                for i_batch in range(batch_size_here - 1, -1, -1):
                    n_obtained_list.append(
                        get_n_obtained(
                            collected_ids[: len(collected_ids) - i_batch],
                            true_target_ids,
                        )
                    )
                    jaccard_posterior_list.append(
                        get_jaccard_posterior(predicted_target_ids, true_target_ids)
                    )
                    switch_list.append(switch_strategy)

            metrics[strategy]["n_obtained"].append(n_obtained_list)
            metrics[strategy]["jaccard_posterior_index"].append(jaccard_posterior_list)
            metrics[strategy]["switch_strategy"].append(switch_list)

        collected_collection.update({strat: collected_ids})

    collected_ids = collected_collection["SwitchBAX"]
    acq_bin_maxes = []
    for i in range(len(collected_ids)):
        acq_bin_maxes.append(max_in_bin(Y, collected_ids[:i], bins_start, bins_end))
    actual_max = max_in_bin(Y, np.arange(len(Y)), bins_start, bins_end)
    for i in range(len(bins_start)):
        plt.plot(
            np.asarray(acq_bin_maxes)[:, i],
            c=f"C{i}",
            label=f"{bins_start[i]}-{bins_end[i]}",
        )

    plt.hlines(
        actual_max,
        0,
        len(collected_ids),
        colors=[f"C{i}" for i in range(len(bins_start))],
        ls="dashed",
    )
    plt.ylabel("Max Acquired Yield")
    plt.xlabel("Number Acquired Points")
    plt.legend()
    plt.savefig("../plot_outputs/sim_STDN_1%_PB_MIX.png")
