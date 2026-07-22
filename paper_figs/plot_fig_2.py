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
from scipy.interpolate import CubicSpline
import random
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
from typing import List
from plot_fig_3 import XY_from_csv

# Set all seeds for reproducibility

SEED = 0
np.random.seed(SEED)
random.seed(SEED)


def user_algorithm(f_x, bins_start, bins_end, bin_epsilon, topk_num=1):

    list_of_max_ids = []

    f_1 = f_x[:, 0]  # only take the first property column, yield
    f_2 = f_x[:, 1]  # only take the first property column, median
    print(len(bins_start))
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


home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:
    fm.fontManager.addfont(fnt)
plt.style.use("./matplotlib.rc")


FIGWIDTH = 4
FIGHEIGHT = 4


def generate_background(color_hex, width, height, alpha=8):
    # Convert hex color to RGBA (ignore alpha, we'll set it manually)
    rgba_color = mcolors.to_rgba(color_hex)

    # Create an empty 3D array for the gradient (RGBA)
    gradient = np.zeros((height, width, 4))

    # Set the color for R, G, B channels
    gradient[:, :, 0] = rgba_color[0]  # Red
    gradient[:, :, 1] = rgba_color[1]  # Green
    gradient[:, :, 2] = rgba_color[2]  # Blue

    # Vary the alpha channel from 0.5 to 0
    gradient[:, :, 3] = alpha

    return gradient


if __name__ == "__main__":

    datapath: str = "./expt_data.csv"
    xcols: List[str] = ["Time", "Temperature", "Pressure"]
    ycols: List[str] = ["Yield", "Median", "Gas Yield"]

    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT), layout="constrained")
    fig: Figure
    ax: Axes

    fig2, ax2 = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT), layout="constrained")
    fig2: Figure
    ax2: Axes

    valid_times = np.arange(1.0, 25.0, 1)
    valid_pressures = np.arange(18.0, 66.0, 4)

    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
    colors = tab20[:5]

    # Create meshgrid
    times_grid, pressures_grid = np.meshgrid(valid_times, valid_pressures)
    X, Yall = XY_from_csv(
        datapath,
        columns_x=xcols,
        columns_y=ycols,
    )

    bins_start = [9, 11, 13, 15, 17]
    bins_end = [11, 13, 15, 17, 19]

    Y_predict = Yall[:, :-1]
    N = pressures_grid.ravel().shape[0]  # Number of random numbers
    X_predict = np.column_stack((times_grid.ravel(), pressures_grid.ravel()))
    liqyield = np.random.normal(loc=62, scale=15, size=N)
    median = np.random.normal(loc=14, scale=2, size=N)

    # Y_predict = np.column_stack((liqyield, median))
    bins_start = [9, 11, 13, 15, 17]
    bins_end = [11, 13, 15, 17, 19]
    ssid = np.array(user_algorithm(Y_predict, bins_start, bins_end, 0.0))
    print(ssid)
    # color_all_pts = "#5F9AFF"
    for i in range(0, 5):
        bg = generate_background(colors[i], 100, 256, 0.4)
        extent = [9 + 2 * i, 9 + 2 * (i + 1), 0, 100]

        ax.imshow(bg, aspect="auto", extent=extent)
    ax.scatter(
        Y_predict[:, 1],
        Y_predict[:, 0],
        c="#5F9AFF",
        s=20,
        alpha=0.5,
        linewidth=0.3,
        edgecolor="k",
    )

    ax.scatter(
        Y_predict[ssid, 1],
        Y_predict[ssid, 0],
        c="#FFD864",
        s=25,
        marker="D",
        linewidth=0.6,
        edgecolor="k",
    )

    ax.set_ylabel("Liquid Alkane Yield (%)")
    ax.set_xlabel("Median Alkane Number (n)")

    fig.savefig("fig_2_a_v2.png")
    fig.savefig("fig_2_a_nd_v2.pdf")

    ax2.scatter(
        X_predict[:, 0],
        X_predict[:, 1],
        c="#5F9AFF",
        s=20,
        alpha=0.5,
        linewidth=0.3,
        edgecolor="k",
    )

    ax2.scatter(
        X_predict[ssid, 0],
        X_predict[ssid, 1],
        c="#FFD864",
        s=0,
        marker="D",
        linewidth=0.6,
        edgecolor="k",
    )
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Pressure (bar)")

    fig2.savefig("fig_2_b_v2.png")
    fig2.savefig("fig_2_b_nd_v2.pdf")
