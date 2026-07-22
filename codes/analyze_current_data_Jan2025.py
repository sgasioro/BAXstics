import sys

sys.path.extend(["../multibax-sklearn"])

from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    ConstantKernel,
    Matern,
)
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from src.utils import XY_from_csv, random_sampling_no_replace, load_ternary_data
from typing import List
from src.models import MGPR, fit_matern_hypers
import warnings
from sklearn.preprocessing import MinMaxScaler
from src.acquisition import (
    multiproperty_infobax,
    multiproperty_meanbax,
    multiproperty_switchbax,
    run_acquisition,
)
from src.algorithms import SubsetAlgorithm
from matplotlib.animation import ArtistAnimation, PillowWriter
from pandas import DataFrame, read_csv

XSCALEARR = np.array([[1, 200, 18], [24, 250, 66]])
YSCALEARR = np.array([[0, 8], [100, 20]])

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaler.fit(XSCALEARR)

y_scaler.fit(YSCALEARR)

SCALERS = [x_scaler, y_scaler]
warnings.filterwarnings("ignore")

plt.style.use("../multibax-sklearn/src/matplotlib.rc")

SEED = 0
import random


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


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
    print(next_id, "!")
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

    return hall_idxs, trained_model_here


# Create a custom colormap function
def custom_bin_colormap(value, bins_start, bins_end, colors):
    for start, end, color in zip(bins_start, bins_end, colors):
        if start <= value <= end:
            return color
    return (0.7, 0.7, 0.7, 1.0)


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
        # print(f_1[list(set(list_of_max_ids))])
        return list(set(list_of_max_ids))


if __name__ == "__main__":
    set_all_seeds(SEED)
    datapath: str = "/pscratch/sd/a/akashr97/BAXstics/codes/expt_data_jan2025.csv"
    xcols: List[str] = ["Time", "Temperature", "Pressure"]
    ycols: List[str] = ["Yield", "Median", "Gas Yield"]

    # Read data from most recent CSV file
    X, Yall = XY_from_csv(
        datapath,
        columns_x=xcols,
        columns_y=ycols,
    )

    print(X.shape)

    # Seperate the gas yield seperately, probably don't
    # need to do this
    Y = Yall[:, :-1]
    GYield = Yall[:, -1].reshape(-1, 1)

    n_properties: int = 2
    n_features: int = 3

    nl_param: float = 0.01

    X_norm = SCALERS[0].transform(X)
    Y_norm = SCALERS[1].transform(Y)

    set_hypers_from_scratch = True
    if set_hypers_from_scratch:
        kernel_initial = ConstantKernel(
            constant_value=1.0, constant_value_bounds=[0.01, 3.0]
        ) * Matern(
            nu=5 / 2,
            length_scale=n_features * [1.0],
            length_scale_bounds=n_features * [[0.01, 3.0]],
        ) + WhiteKernel(
            noise_level=nl_param, noise_level_bounds="fixed"
        )

        kernel_initial_list = n_properties * [kernel_initial]

        kernel_list = fit_matern_hypers(
            x_train=X_norm,
            y_train=Y_norm,
            kernel_list=kernel_initial_list,
            n_restarts_optimizer=5,
        )
    else:
        raise NotImplementedError("Not implemented for now")

    multi_gpr = MGPR(kernel_list=kernel_list)
    multi_gpr.fit(X_norm, Y_norm)

    bins_start = [9, 11, 13, 15, 17]
    bins_end = [11, 13, 15, 17, 19]
    algorithm = TopKBinAlgorithm(
        user_algo_params={
            "scalers": SCALERS,
            "bins_start": bins_start,
            "bins_end": bins_end,
            "num_points_per_bin": 1,
            "bin_epsilon": 0.0,
        }
    )

    valid_times = np.arange(1.0, 25.0, 1.0)
    valid_pressures = np.arange(18.0, 66.0, 4)

    # Create meshgrid
    times_grid, pressures_grid = np.meshgrid(valid_times, valid_pressures)

    # Reshape the grids into N×2 array
    npts = pressures_grid.ravel().shape[0]

    X_predict = np.column_stack(
        (times_grid.ravel(), 200 * np.ones(npts), pressures_grid.ravel())
    )

    indices = []

    # Convert the X1 array to a set of tuples
    set_X1 = set(map(tuple, X))

    # Iterate over each row in X2 and check if it is in the set of X1
    for idx, row in enumerate(X_predict):
        if tuple(row) in set_X1:
            indices.append(idx)

    X_predict_norm = SCALERS[0].transform(X_predict)
    plot_things = True

    acquisition_function_meanbax, trained_model, switch_strategy = (
        multiproperty_meanbax(
            x_domain=X_predict_norm,
            x_train=X_norm,
            y_train=Y_norm,
            model=multi_gpr,
            algorithm=algorithm,
            collected_ids=[],
        )
    )

    print(X_predict[np.argmax(acquisition_function_meanbax)])

    acquisition_function_info_bax, trained_model_2, term1, term2 = (
        multiproperty_infobax(
            x_domain=X_predict_norm,
            x_train=X_norm,
            y_train=Y_norm,
            model=multi_gpr,
            algorithm=algorithm,
            n_posterior_samples=10,
            verbose=False,
        )
    )
    print(X_predict[np.argmax(acquisition_function_info_bax)])

    Y_pred_mu, Y_pred_sigma = trained_model_2.predict(X_predict_norm)

    Y_pred_mu_tr, Y_pred_sigma_tr = trained_model_2.predict(X_norm)
    Y_predict_mu_t2r = SCALERS[1].inverse_transform(Y_pred_mu_tr)
    # for i in range(0, Y_pred_mu_tr.shape[0]):
    #   print(X[i], Y_predict_mu_t2r[i])

    Y_predict = SCALERS[1].inverse_transform(Y_pred_mu)

    strategy = "SwitchBAX"
    n_posterior_samples = 10
    batch_size = 2
    add_ids = [148, 85, 199, 128]
    indices = indices + add_ids
    next_ids, model = batch_acquisition(
        X_norm,
        Y_norm,
        X_predict_norm,
        Y_predict,
        strategy,
        algorithm,
        multi_gpr,
        indices,
        n_posterior_samples,
        batch_size,
    )
    ct = 1
    for idi in next_ids:
        xid = X_predict[idi]
        yid = Y_predict[idi]
        print(
            f"Prediction {ct} | ID {idi}: {xid[0]} h | {xid[1]} C | {xid[2]}bar : {yid[0]:.2f}% yield | {yid[1]:.2f} median"
        )
        ct += 1
    print(X_predict[next_ids])
    print(Y_predict[next_ids])
    print("Hi")
    ss = algorithm.identify_subspace(Y_pred_mu, X_predict_norm)
    print(Y_predict[ss])
    print("Bye")

    if plot_things:
        fig, axs = plt.subplots(1, 3, figsize=(8, 3), dpi=600)
        fig: Figure
        axs: List[Axes]

        yield_plt = axs[0].scatter(
            X_predict[:, 0],
            X_predict[:, 2],
            c=Y_predict[:, 0],
            s=10,
            alpha=0.7,
            cmap="Blues",
        )
        axs[0].scatter(
            X_predict[acquisition_function_meanbax > 0, 0],
            X_predict[acquisition_function_meanbax > 0, 2],
            s=0.5,
            marker="x",
            color="k",
        )
        # Method 1: Using Tab20 colormap
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
        colors = tab20[: len(bins_start)]

        bounds = bins_start + [bins_end[-1]]
        print(bounds)
        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
        c1 = Y_predict[:, 1] < bins_end[-1]
        c2 = Y_predict[:, 1] > bins_start[0]
        cond = (c1) & (c2)
        median_plot = axs[1].scatter(
            X_predict[cond, 0],
            X_predict[cond, 2],
            c=Y_predict[cond, 1],
            s=10,
            alpha=0.7,
            cmap=custom_cmap,
        )
        r = (~cond) & c1
        axs[1].scatter(
            X_predict[r, 0],
            X_predict[r, 2],
            c="gray",
            s=10,
            alpha=0.7,
        )

        r = (~cond) & c2
        axs[1].scatter(
            X_predict[r, 0],
            X_predict[r, 2],
            c="k",
            s=10,
            alpha=0.7,
        )

        # axs[1].scatter(
        #    X_predict[acquisition_function_meanbax > 0, 0],
        #    X_predict[acquisition_function_meanbax > 0, 2],
        #    s=0.5,
        #    marker="x",
        #    color="k",
        # )

        axs[1].scatter(
            X_predict[indices, 0],
            X_predict[indices, 2],
            s=0.5,
            marker="s",
            color="b",
        )

        axs[1].scatter(
            X_predict[next_ids, 0],
            X_predict[next_ids, 2],
            s=2,
            marker="*",
            color="g",
        )

        mbax_acqfn_plt = axs[2].scatter(
            X_predict[:, 0],
            X_predict[:, 2],
            c=acquisition_function_info_bax,
            s=10,
            alpha=0.7,
            cmap="Greens",
        )

        for ax in axs:
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Pressure (bar)")

        cbar_1 = fig.colorbar(yield_plt, ax=axs[0])
        cbar_1.set_label("Liquid Yield (%)")
        cbar_2 = fig.colorbar(median_plot, ax=axs[1])
        cbar_2.set_label("Median C")
        cbar_3 = fig.colorbar(mbax_acqfn_plt, ax=axs[2])
        cbar_3.set_label("Info BAX acqfn")
        fig.tight_layout()
        fig.savefig("acqnvis.png")

    plot_posterior = False
    if plot_posterior:
        Y_samples = trained_model_2.sample_y(X_predict_norm, 10)
        print(Y_samples.shape[2])
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=600)
        fig: Figure
        axs: List[Axes]

        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
        colors = tab20[: len(bins_start)]

        bounds = bins_start + [bins_end[-1]]
        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
        frames = []
        for i in range(0, Y_samples.shape[2]):
            Y_predict_posterior = SCALERS[1].inverse_transform(Y_samples[:, :, i])
            yield_plt = axs[0].scatter(
                X_predict[:, 0],
                X_predict[:, 2],
                c=Y_predict_posterior[:, 0],
                s=0.5,
                alpha=0.7,
                cmap="viridis",
                vmax=100,
                vmin=0,
            )

            # Method 1: Using Tab20 colormap

            median_plot = axs[1].scatter(
                X_predict[:, 0],
                X_predict[:, 2],
                c=Y_predict_posterior[:, 1],
                s=0.5,
                alpha=0.7,
                cmap=custom_cmap,
            )

            title1 = axs[0].text(
                0.5, 1.05, f"Posterior {i}", ha="center", transform=axs[0].transAxes
            )

            title2 = axs[1].text(
                0.5, 1.05, f"Posterior {i}", ha="center", transform=axs[1].transAxes
            )
            frames.append([yield_plt, median_plot, title1, title2])

        cbar_1 = fig.colorbar(yield_plt, ax=axs[0])
        cbar_1.set_label("Liquid Yield (%)")
        cbar_2 = fig.colorbar(median_plot, ax=axs[1])
        cbar_2.set_label("Median C")
        fig.tight_layout()

        anim = ArtistAnimation(
            fig, frames, interval=500, blit=True, repeat=True  # 500ms between frames
        )

        # Save as GIF
        writer = PillowWriter(fps=2)  # 2 frames per second
        anim.save("./posterior_samples.gif", writer=writer)
