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


def max_in_bin(Y, collected_ids, bins_start, bins_end):
    y_here = Y[collected_ids]

    max_y = []
    for i in range(len(bins_start)):
        # Should bins be edge inclusive?
        bin_mask = (y_here[:, 1] >= bins_start[i]) & (y_here[:, 1] < bins_end[i])
        y_in_bin = y_here[bin_mask, 0]

        if len(y_in_bin) == 0:
            max_y.append(0)
        else:
            max_y.append(np.amax(y_in_bin))

    return max_y


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


from analyze_current_data_Jan2025 import TopKBinAlgorithm

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
    Y = Yall[:, :-1]
    acq_bin_maxes = []
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
    colors = tab20[: len(bins_start)]

    bounds = bins_start + [bins_end[-1]]
    print(bounds)
    custom_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    print(custom_cmap.colors[0])
    for i in range(18, -1, -1):
        collected_ids = np.arange(0, X.shape[0] - i)

        bin_maxes = max_in_bin(Y, collected_ids, bins_start, bins_end)
        # print(bin_maxes)
        acq_bin_maxes.append(bin_maxes)
    acq_bin_maxes = np.array(acq_bin_maxes)
    for i in range(len(bins_start)):
        plt.plot(
            np.asarray(acq_bin_maxes)[:, i],
            "o-",
            c=custom_cmap.colors[i],
            label=f"{bins_start[i]}-{bins_end[i]}",
        )

    plt.ylabel("Max Acquired Yield")
    plt.xlabel("Number Acquired Points")
    plt.legend()
    plt.savefig("hi.png")
