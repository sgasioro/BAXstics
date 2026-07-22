from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import os
import matplotlib.colors as mcolors


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


FIGWIDTH = 4
FIGHEIGHT = 4.1


FIG2WIDTH = 4
FIG2HEIGHT = 6.2

home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:
    fm.fontManager.addfont(fnt)
plt.style.use("./matplotlib.rc")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

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

    # Generate data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT), layout="constrained")

    bins_start = [9, 11, 13, 15, 17]
    bins_end = [11, 13, 15, 17, 19]

    Y = Yall[:, :-1]
    acq_bin_maxes = []
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
    colors = tab20[: len(bins_start)]

    bounds = bins_start + [bins_end[-1]]
    custom_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    (line,) = ax.plot([], [], "o-", color=colors[0])
    (line2,) = ax.plot([], [], "o-", color=colors[1])
    (line3,) = ax.plot([], [], "o-", color=colors[2])
    (line4,) = ax.plot([], [], "o-", color=colors[3])
    (line5,) = ax.plot([], [], "o-", color=colors[4])

    for i in range(26, -1, -1):

        collected_ids = np.arange(0, X.shape[0] - i)

        bin_maxes = max_in_bin(Y, collected_ids, bins_start, bins_end)
        # print(bin_maxes)
        acq_bin_maxes.append(bin_maxes)
    acq_bin_maxes = np.array(acq_bin_maxes)

    xs = []
    ys = []
    for i in range(len(bins_start)):
        ys.append(np.asarray(acq_bin_maxes)[:, i])
        xs.append(np.arange(1, np.asarray(acq_bin_maxes).shape[0] + 1))

    x = xs[0]
    y = ys[0]

    x2 = xs[1]
    y2 = ys[1]

    x3 = xs[2]
    y3 = ys[2]

    x4 = xs[3]
    y4 = ys[3]

    x5 = xs[4]
    y5 = ys[4]

    # Initialize the plot
    def init():
        ax.set_xticks(np.arange(0, 27, 2))
        ax.set_ylabel("Max. Acquired Yield (%)")
        ax.set_xlabel("No. of Acquired Points")
        ax.set_ylim(50, 90)
        ax.set_xlim(0, 27)
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        return (line, line2, line3, line4, line5)

    # Update the plot
    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line2.set_data(x2[:frame], y2[:frame])
        line3.set_data(x3[:frame], y3[:frame])
        line4.set_data(x4[:frame], y4[:frame])
        line5.set_data(x5[:frame], y5[:frame])
        return (line, line2, line3, line4, line5)

    # Create the animation
    # print
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True)

    # Save the animation as a gif
    ani.save("animated_plot.gif", writer="pillow")
