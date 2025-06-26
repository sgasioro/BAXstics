import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter  # Savitzky-Golay filter for smoother curves

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

# Assuming the dataframe is already loaded into df


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


home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:
    fm.fontManager.addfont(fnt)
plt.style.use("./matplotlib.rc")

FIGWIDTH = 8
FIGHEIGHT = 4
if __name__ == "__main__":
    df = pd.read_csv("./bin_winner_distributions_conversion.csv")

    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))  # Get all 20 colors
    colors = tab20[:5]
    print(colors)
    for rgba in colors:
        hex_color = mcolors.to_hex(rgba, keep_alpha=False)
        print(hex_color)

    # Extracting the time and numeric data columns
    time_points = df["Bin"]
    numeric_data = df.iloc[:, 3:]  # Numeric data starts from column 7 (index 3)
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT), layout="constrained")
    fig: Figure
    ax: Axes
    # Define the number of bars (corresponding to columns 7, 8, 9, ..., up to the last one)
    x = np.arange(len(numeric_data.columns))  # Indices of the measurements
    # Bar width (for spacing between groups of bars)
    bar_width = 1 / (
        len(colors) + 1
    )  # Bar width (adjusted to fit bars at same positions)
    # Smoothing window size (can be adjusted)
    window_size = 6  # Smoothing window size (use odd numbers like 5, 7, etc.)
    # Beautiful pastel color palette with more variation and elegance
    pastel_colors = [
        "#BFD8B8",
        "#F4A5A1",
        "#F9D7A5",
        "#A9C5D3",
        "#C3A1D3",
    ]  # Soft greens, coral, mustard, soft blue, lavender
    # Darker shade for smooth lines (we'll darken the original pastel colors)
    darker_pastels = [
        "#80B78C",
        "#D77D76",
        "#D19D57",
        "#7A9EBD",
        "#9A81B8",
    ]  # Darker versions of the pastel colors
    darker_pastels = colors
    # Loop through the data and plot the bars for each time point
    default_linewidth = mpl.rcParams["lines.linewidth"]
    print(default_linewidth)
    linewidth_out = default_linewidth + 2
    pe = [
        Stroke(linewidth=linewidth_out, foreground="w"),
        Normal(),
    ]

    for i, time in enumerate(time_points):
        # Plot bars with pastel colors at the same x positions for each time point (no shifting)
        offset = bar_width * i
        print(offset, x)
        # rects = ax.bar(
        #    (x + offset),
        #    numeric_data.iloc[i],
        #    alpha=0.7,
        #    width=bar_width,
        #    label=f"{time} Bin",
        #    edgecolor="k",
        #    facecolor=colors[i],
        # )
        # Apply Savitzky-Golay filter for smoother curves
        # smoothed_line = savgol_filter(
        #    numeric_data.iloc[i], window_length=window_size, polyorder=2
        # )

        # x_fine = np.linspace(x.min(), x.max(), num=10 * len(x))

        # Apply Cubic Spline interpolation to the smoothed data
        # cs = CubicSpline(x, smoothed_line)
        # smoothed_interp_line = cs(x_fine)
        x_fine = x  # + offset
        smoothed_interp_line = numeric_data.iloc[i]
        # Plot smoothed line with corresponding darker pastel color for contrast
        ax.plot(
            x_fine,
            smoothed_interp_line,
            color=darker_pastels[i],
            path_effects=pe,
        )
    for i in range(0, 5):
        bg = generate_background(colors[i], 100, 256, 0.4)
        extent = [2 - bar_width / 2 + i * 2, 2 + (2) * (i + 1) - 1 / 6, 0, 10]
        print(extent)

        ax.imshow(bg, aspect="auto", extent=extent)

    # Adding labels (title removed) and equal padding
    label_fontsize = 18
    tick_fontsize = 14
    label_pad = 100

    ax.set_xlabel("Alkane number (n)")
    ax.set_ylabel("Alkane Yield (%)")
    # Adjust the x-axis to show the correct ticks (measurement indices from 7 to 36)
    x_ticks = np.arange(7, 37, 1)  # Alkane numbers from 7 to 36
    print(x_ticks)
    ax.set_xticks(x_ticks - 7 + 2 / 6, x_ticks)
    ax.tick_params(length=0)
    # ax.set_xticks(xscale * np.arange(len(x_ticks)), x_ticks.astype(str))
    twidth = 0.5
    ticklocs = np.sort(
        list(x_ticks - 7 - bar_width / 2) + list(x_ticks - 7 + 4 / 6 + bar_width / 2)
    )
    print(ticklocs)
    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks(ticklocs, labels=[])
    sec2.tick_params("x", length=20, direction="out", width=twidth)
    ax.set_xlim(-0.5, 37 - 7)
    # Set the x-axis limits to start at 7 and go to 36

    # Add a legend with larger font
    ax.legend(title="", fontsize=14)
    print(max)
    # Show plot
    fig.savefig("fig_4_v2_lonly.png")
    fig.savefig("fig_4_v2_lonly.pdf")
