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
import matplotlib.colors as mcolors
from typing import List

home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:
    fm.fontManager.addfont(fnt)
plt.style.use("./matplotlib.rc")

FIGWIDTH = 2.5 * 3
FIGHEIGHT = 2 * 2
# Assuming df is already loaded
if __name__ == "__main__":
    df = pd.read_csv("./fig_X_plots.csv")

    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
    colors = tab20[:5]
    print(colors)
    for rgba in colors:
        hex_color = mcolors.to_hex(rgba, keep_alpha=False)
        print(hex_color)

    # Extracting the time and numeric data columns
    print(df.head())
    time_points = df.index
    numeric_data = df.iloc[:, :]  # Numeric data starts from column 7 (index 3)
    # Set up the plot
    fig, axs = plt.subplots(2, 3, figsize=(FIGWIDTH, FIGHEIGHT), layout="constrained")
    fig: Figure
    axs = axs.flatten()
    axs: List[Axes]

    bar_width = 1
    window_size = 6

    x = np.arange(len(numeric_data.columns))
    colors = list(colors) + ["k"]

    x_ticks = np.arange(7, 37, 1)
    bincolors = [4, 2, 5, 2, 1, 4]
    # Plot each timepoint individually
    for i, time in enumerate(time_points):
        ax = axs[i]
        # Bar plot
        ax.bar(
            x,
            numeric_data.iloc[i],
            alpha=0.7,
            width=bar_width,
            edgecolor="black",
            color=colors[bincolors[i]],
        )
        # Optional smoothed line (commented out)
        # smoothed_line = savgol_filter(numeric_data.iloc[i], window_length=window_size, polyorder=2)
        # plt.plot(x, smoothed_line, linewidth=3, color=darker_pastels[i])
        # Axes and labels
        ax.set_xlabel("Alkane number (n)")
        ax.set_ylabel("Alkane Yield (%)")
        # Skip every other tick to avoid crowding
        reduced_x_ticks = x_ticks[::4]
        reduced_x_indices = np.arange(0, len(x_ticks), 4)
        ax.set_xticks(reduced_x_indices, reduced_x_ticks.astype(str))
        ax.set_xlim(-0.5, len(x_ticks) - 0.5)
        ax.set_ylim(0, 10)
        # No legend
    fig.savefig("fig_X_a.png")
    fig.savefig("fig_X_a.pdf")
