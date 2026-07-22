# BAXstics

**Goal-aware Bayesian algorithm execution for polyethylene hydrogenolysis.**

This repository accompanies the paper *"Enhancing selectivity and yield in
polyethylene hydrogenolysis using goal-aware Bayesian algorithm execution."*
It uses **Bayesian algorithm execution (BAX)** — specifically the
**SwitchBAX** acquisition strategy — to decide which reaction conditions
(pressure, temperature, time) to measure next when upcycling single-use
plastics via catalyzed hydrogenolysis.

Rather than optimizing a single objective, BAX lets us state a rich
experimental *goal* as a simple algorithm and then acquires data that is
maximally informative for satisfying that goal. Here the goal is to find, for
each range of product chain lengths, the reaction conditions that give the
highest yield — enabling selective control over the product distribution while
spending as few experiments as possible.

## Background

Catalyzed hydrogenolysis breaks down polyethylene into shorter hydrocarbons.
The value of the process depends on both **how much** product is formed
(*yield*) and **what** is formed (*selectivity*, summarized by the median
number of carbon atoms in the product). We treat the problem as a
multi-property active-learning task:

- **Design space (inputs):** reaction `Time`, `Temperature`, `Pressure`.
- **Measured properties (outputs):** `Yield` (%) and `Median` — the median of
  the liquid-yield distribution over alkane carbon number for a given
  measurement.

The experimental goal is expressed as a `TopKBinAlgorithm`: partition the
`Median` (carbon-number) axis into bins — C9–10, C11–12, C13–14, C15–16, and
C17–18 — and, within each bin, identify the conditions that maximize `Yield`. BAX then chooses measurements
that most quickly reveal this set of "best conditions per selectivity band,"
using multi-output Gaussian process surrogates.

## Method

The acquisition strategies are provided by the
[`multibax-sklearn`](https://github.com/src47/multibax-sklearn) library
(included here as a git submodule), which implements InfoBAX, MeanBAX,
SwitchBAX, and uncertainty sampling (US). See Chitturi et al., *Targeted
materials discovery using Bayesian algorithm execution*, npj Computational
Materials (2024). This repo layers the hydrogenolysis-specific goal, data,
simulations, and an interactive experiment interface on top of that library.

## Repository layout

| Path | Description |
| --- | --- |
| `multibax-sklearn/` | Core BAX library (git submodule): acquisition strategies, GP models, metrics, tutorials. |
| `codes/` | Simulation and analysis scripts. `run_algo.py` benchmarks acquisition strategies on the hydrogenolysis surrogate; `loocv_data.py` runs leave-one-out cross-validation; `analyze_current_data_Jan2025.py` and `maxyield_plot.py` analyze experimental data. `expt_data_jan2025.csv` holds real measurements. |
| `interactive_expt/` | Solara web app (`main_run.py`) that runs the live experiment loop: propose the next condition, record the measured yield/median, retrain, and save. |
| `data/` | Simulator datasets used as ground truth for in-silico benchmarks (`3D_data*.csv`, `1D_data.csv`). |
| `data_sathya_edits/` | Neural-network-fitted interpolation of the experimental data (`nn_fitted_interpolation.csv`) used as a smooth ground-truth surrogate. |
| `paper_figs/` | Scripts and CSVs to regenerate the figures in the paper. |
| `notebooks/` | Exploratory Jupyter notebooks (simulations, next-point selection, simulator development). |

## Installation

Clone the repository **with submodules** (the `multibax-sklearn` core is a
submodule):

```bash
git clone --recurse-submodules <repo-url>
cd BAXstics
# if you already cloned without submodules:
git submodule update --init --recursive
```

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r multibax-sklearn/requirements.txt
```

The interactive app additionally requires `solara` and `plotly`:

```bash
pip install solara plotly
```

> Tested with Python 3.8 / scikit-learn 1.5. See
> `multibax-sklearn/requirements.txt` for pinned versions.

## Usage

### Run an in-silico benchmark

Compare acquisition strategies (US, MeanBAX, SwitchBAX) on a surrogate of the
hydrogenolysis response surface:

```bash
cd codes
python run_algo.py
```

This fits multi-output GPs, runs the BAX acquisition loop, and saves a plot of
the maximum acquired yield per selectivity bin versus the number of
experiments.

### Run the interactive experiment interface

The Solara app proposes the next reaction condition to try, lets you enter the
measured yield and median, and saves the updated dataset and model:

```bash
cd interactive_expt
solara run main_run.py
```

Workflow inside the app: load a dataset (or the latest saved one) → **Get next
experiment** → run that condition in the lab → enter the measured `Yield` and
`Median` → **Save experiment**. The app supports single- or two-point batches.

### Reproduce paper figures

```bash
cd paper_figs
python plot_fig_2.py   # etc.
```

## Citation

The paper describing this work is forthcoming. In the meantime, please cite the
underlying BAX method:

```bibtex
@article{chitturi2024targeted,
  title={Targeted materials discovery using Bayesian algorithm execution},
  author={Chitturi, Sathya R and Ramdas, Akash and Wu, Yue and Rohr, Brian and Ermon, Stefano and Dionne, Jennifer and Jornada, Felipe H da and Dunne, Mike and Tassone, Christopher and Neiswanger, Willie and Ratner, Daniel},
  journal={npj Computational Materials},
  volume={10},
  number={1},
  pages={156},
  year={2024}
}
```

## License

Released under the BSD 2-Clause License, matching the bundled
[`multibax-sklearn`](https://github.com/src47/multibax-sklearn) library.
