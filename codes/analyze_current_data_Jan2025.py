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
from src.acquisition import multiproperty_infobax, multiproperty_meanbax
from src.algorithms import SubsetAlgorithm


XSCALEARR = np.array([[1, 200, 18], [24, 250, 50]])
YSCALEARR = np.array([[0, 8], [100, 20]])

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaler.fit(XSCALEARR)

y_scaler.fit(YSCALEARR)

SCALERS = [x_scaler, y_scaler]
warnings.filterwarnings("ignore")

plt.style.use("../multibax-sklearn/src/matplotlib.rc")


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

        return list(set(list_of_max_ids))


if __name__ == "__main__":
    datapath: str = "/pscratch/sd/a/akashr97/BAXstics/codes/expt_data_jan2025.csv"
    xcols: List[str] = ["Time", "Temperature", "Pressure"]
    ycols: List[str] = ["Yield", "Median", "Gas Yield"]

    # Read data from most recent CSV file
    X, Yall = XY_from_csv(
        datapath,
        columns_x=xcols,
        columns_y=ycols,
    )

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

    valid_times = np.arange(1.0, 24.0, 0.5)
    valid_pressures = np.arange(18.0, 50.0, 2)

    # Create meshgrid
    times_grid, pressures_grid = np.meshgrid(valid_times, valid_pressures)

    # Reshape the grids into N×2 array
    npts = pressures_grid.ravel().shape[0]

    X_predict = np.column_stack(
        (times_grid.ravel(), 200 * np.ones(npts), pressures_grid.ravel())
    )
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

    Y_pred_mu, Y_pred_sigma = trained_model.predict(X_predict_norm)
    Y_predict = SCALERS[1].inverse_transform(Y_pred_mu)

    if plot_things:
        fig, axs = plt.subplots(1, 3, figsize=(8, 3), dpi=400)
        fig: Figure
        axs: List[Axes]

        yield_plt = axs[0].scatter(
            X_predict[:, 0],
            X_predict[:, 2],
            c=Y_predict[:, 0],
            s=0.5,
            alpha=0.7,
            cmap="viridis",
        )
        # Method 1: Using Tab20 colormap
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
        colors = tab20[: len(bins_start)]

        bounds = bins_start + [bins_end[-1]]
        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
        median_plot = axs[1].scatter(
            X_predict[:, 0],
            X_predict[:, 2],
            c=Y_predict[:, 1],
            s=0.5,
            alpha=0.7,
            cmap=custom_cmap,
        )

        mbax_acqfn_plt = axs[2].scatter(
            X_predict[:, 0],
            X_predict[:, 2],
            c=acquisition_function_meanbax,
            s=0.5,
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
        cbar_3.set_label("Mean BAX acqfn")
        fig.tight_layout()
        fig.savefig("acqnvis.png")
