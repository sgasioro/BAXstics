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

FIGWIDTH = 2.5
FIGHEIGHT = 2

if __name__ == "__main__":
    # Create your main plot
    marker_bounds_cols = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
    ]
    labels = [
        "Best C9-10",
        "Best C11-12",
        "Best C13-14",
        "Best C15-16",
        "Best C17-18",
        "ML Experiments",
        "Prior Data",
    ]

    fig, ax = plt.subplots()
    all_handles = []
    for ctssi in range(0, 7):
        if ctssi < 5:
            si = ax.scatter(
                [0, 1],
                [0, 1],
                s=30,
                color="#FFD864",
                marker="D",
                linewidth=0.6,
                edgecolor=marker_bounds_cols[ctssi],
                label=labels[ctssi],
            )
        elif ctssi == 5:

            si = ax.scatter(
                [0, 1], [0, 1], s=30, marker="x", linewidth=4, color="maroon"
            )
        elif ctssi == 6:
            si = ax.scatter(
                [0, 1], [0, 1], s=30, marker="x", linewidth=4, color="black"
            )
        all_handles += [si]

    legend = ax.legend()

    # Create a new figure just for the legend
    fig_legend = plt.figure(figsize=(2, 1))
    # Add the legend to the new figure
    fig_legend.legend(handles=all_handles, labels=labels, loc="center")
    # Turn off axes
    fig_legend.gca().axis("off")

    # Save the legend-only figure
    fig_legend.savefig("legend_only.png", bbox_inches="tight")
