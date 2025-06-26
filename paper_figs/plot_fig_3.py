import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import os
from matplotlib.patheffects import Stroke, Normal
import matplotlib as mpl

import random
import matplotlib.colors as mcolors
from typing import List
from matplotlib.animation import FuncAnimation, PillowWriter


def XY_from_csv(path_to_csv: str, columns_x: List[str], columns_y: List[str]):
    # Load the data from the CSV file
    df = pd.read_csv(path_to_csv)

    # assert that there are no NaNs in the dataframe (chatgpt)
    assert df.isna().sum().sum() == 0, "Should not have NANs in the dataset"

    # assert that df does not have any repeated rows (chatgpt)
    assert df.duplicated().sum() == 0, "Dataset should not have repeated rows"

    X = np.array(df[columns_x])
    y = np.array(df[columns_y])

    assert X.shape[1] >= 2, "Input X must be at least two dimensional"
    return X, y


def user_algorithm(f_x, bins_start, bins_end, bin_epsilon, topk_num=1):

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


def max_in_bin(Y, collected_ids, bins_start, bins_end):
    y_here = Y[collected_ids]
    max_y = []
    for i in range(len(bins_start)):

        bin_mask = (y_here[:, 1] >= bins_start[i]) & (y_here[:, 1] < bins_end[i])
        y_in_bin = y_here[bin_mask, 0]

        if len(y_in_bin) == 0:
            max_y.append(0)
        else:
            max_y.append(np.amax(y_in_bin))

    return max_y


home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:
    fm.fontManager.addfont(fnt)
plt.style.use("./matplotlib.rc")


FIGWIDTH = 4
FIGHEIGHT = 4.1


FIG2WIDTH = 4
FIG2HEIGHT = 6.2

if __name__ == "__main__":
    datapath: str = "./expt_data.csv"
    xcols: List[str] = ["Time", "Temperature", "Pressure"]
    ycols: List[str] = ["Yield", "Median", "Gas Yield"]

    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT), layout="constrained")
    fig: Figure
    ax: Axes

    X, Yall = XY_from_csv(
        datapath,
        columns_x=xcols,
        columns_y=ycols,
    )

    bins_start = [9, 11, 13, 15, 17]
    bins_end = [11, 13, 15, 17, 19]

    Y = Yall[:, :-1]
    acq_bin_maxes = []
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
    colors = tab20[: len(bins_start)]

    bounds = bins_start + [bins_end[-1]]
    custom_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    print(Y.shape, Y[-26])
    for i in range(26, -1, -1):

        collected_ids = np.arange(0, X.shape[0] - i)

        bin_maxes = max_in_bin(Y, collected_ids, bins_start, bins_end)
        # print(bin_maxes)
        acq_bin_maxes.append(bin_maxes)
    acq_bin_maxes = np.array(acq_bin_maxes)
    ax.vlines([6, 22], 50, 90, linestyles="--", color="k")
    lines = []
    for i in range(len(bins_start)):
        print(np.asarray(acq_bin_maxes)[:, i])
        ax.plot(
            np.asarray(acq_bin_maxes)[:, i],
            "o-",
            c=custom_cmap.colors[i],
            label=f"{bins_start[i]}-{bins_end[i]}",
        )

    ax.set_xticks(np.arange(0, 27, 2))
    ax.set_ylabel("Max. Acquired Yield (%)")
    ax.set_xlabel("No. of Acquired Points")

    print(acq_bin_maxes[np.array([0, -1]), :].T)

    fig.savefig("fig_3_a.png")
    fig.savefig("fig_3_a.pdf")
