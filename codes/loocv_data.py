# Thanks ChatGPT
def check_within_error(y_true, y_pred, y_err):
    """
    Returns an array of size n where each element is 1 if the corresponding
    row in y_true is entirely within the range of y_pred +/- y_err, otherwise 0.

    Parameters:
    y_true : np.array of shape (n, m)
    y_pred : np.array of shape (n, m)
    y_err  : np.array of shape (n, m)

    Returns:
    result : np.array of size (n) where each element is 1 or 0.
    """
    # Check if y_true is within the bounds of y_pred ± y_err
    within_bounds = (y_true >= (y_pred - y_err)) & (y_true <= (y_pred + y_err))

    # For each row, check if all values are within the bounds
    result = np.all(within_bounds, axis=1).astype(int)

    return result


if __name__ == "__main__":
    import sys

    sys.path.extend(["../multibax-sklearn"])
    from src.models import MGPR, fit_matern_hypers

    import numpy as np
    from sklearn.model_selection import LeaveOneOut
    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from sklearn.gaussian_process.kernels import (
        WhiteKernel,
        ConstantKernel,
        Matern,
        RBF,
    )
    from sklearn.preprocessing import MinMaxScaler
    import ipywidgets as widgets
    from IPython.display import display

    from src.utils import XY_from_csv, random_sampling_no_replace, load_ternary_data

    from src.models import MGPR, fit_matern_hypers
    import warnings

    warnings.filterwarnings("ignore")

    plt.style.use("../multibax-sklearn/src/matplotlib.rc")

    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    datapath = "../data/exptdata.csv"
    xcols = ["Time", "Temperature", "Pressure"]
    ycols = ["Yield", "Median"]
    X, Y = XY_from_csv(
        datapath,
        columns_x=xcols,
        columns_y=ycols,
    )

    visualize_data = True

    if visualize_data:
        fig, axs = plt.subplots(2, 3, figsize=(6 * 3, 6 * 2))
        for i in range(0, 2):
            for j in range(0, 3):
                axs[i][j].plot(X[:, j], Y[:, i], "o")
                axs[i][j].set_xlabel(xcols[j])
                axs[i][j].set_ylabel(ycols[i])
        fig.tight_layout()
        fig.savefig("../plot_outputs/expt_dat_xy.png")

        fig, ax = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1))
        ax.plot(Y[:, 1], Y[:, 0], "o")
        ax.set_xlabel(ycols[1])
        ax.set_ylabel(ycols[0])
        bins_start = [9, 11, 13, 15, 17]
        bins_end = [10, 12, 14, 16, 18]
        for i in range(0, len(bins_start)):
            ax.fill_between(
                [bins_start[i], bins_end[i]], 0, 100, alpha=0.1, color="orange"
            )
        fig.tight_layout()
        fig.savefig("../plot_outputs/expt_dat_yield_median.png")

    n_properties = 2
    n_features = 3
    kernel_initial = ConstantKernel(
        constant_value=1.0, constant_value_bounds=[0.01, 3.0]
    ) * Matern(
        nu=5 / 2,
        length_scale=n_features * [1.0],
        length_scale_bounds=n_features * [[0.01, 3.0]],
    ) + WhiteKernel(
        noise_level=0.1, noise_level_bounds="fixed"
    )
    kernel_initial_list = n_properties * [kernel_initial]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler_median = MinMaxScaler(feature_range=(-1, 1))

    X_norm = x_scaler.fit_transform(X)

    # Y_norm = y_scaler.fit_transform(Y) # Ideal Normalization

    # Normalization based on parameters
    yield_min = 0
    yield_max = 100
    selectivity_min = 8
    selectivity_max = 20

    y_scaler.fit(np.array([[yield_min, selectivity_min], [yield_max, selectivity_max]]))
    y_scaler_median.fit(np.array([[selectivity_min], [selectivity_max]]))
    bins_start_norm = y_scaler_median.transform(np.array(bins_start).reshape(-1, 1))
    bins_end_norm = y_scaler_median.transform(np.array(bins_end).reshape(-1, 1))
    Y_norm = y_scaler.transform(Y)
    stack_outs = None
    stack_model_mu = None
    stack_model_sigma = None
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        kernel_list = fit_matern_hypers(
            x_train=X_norm[train_index],
            y_train=Y_norm[train_index],
            kernel_list=kernel_initial_list,
            n_restarts_optimizer=5,
        )
        multi_gpr = MGPR(kernel_list=kernel_list)
        multi_gpr.fit(X_norm[train_index], Y_norm[train_index])
        Y_pred_mu, Y_pred_sigma = multi_gpr.predict(X_norm[test_index])
        if i == 0:
            stack_outs = Y_norm[test_index]
            stack_model_mu = Y_pred_mu
            stack_model_sigma = Y_pred_sigma
        else:
            stack_outs = np.vstack((stack_outs, Y_norm[test_index]))
            stack_model_mu = np.vstack((stack_model_mu, Y_pred_mu))
            stack_model_sigma = np.vstack((stack_model_sigma, Y_pred_sigma))
    metric = check_within_error(stack_outs, stack_model_mu, stack_model_sigma)
    print(metric)
    print(metric.sum() / metric.shape[0])
    if visualize_data:
        fig, axs = plt.subplots(1, 2, figsize=(6 * 1, 6 * 1), width_ratios=[0.98, 0.02])
        ax = axs[0]
        cax = axs[1]
        sm = ax.scatter(
            stack_outs[:, 1],
            stack_outs[:, 0],
            marker="x",
            label="Data From Expt.",
            c=metric,
            cmap="RdBu",
        )
        ax.errorbar(
            stack_model_mu[:, 1],
            stack_model_mu[:, 0],
            fmt=" ",
            xerr=stack_model_sigma[:, 1],
            yerr=stack_model_sigma[:, 0],
            label="LOO Prediction",
            # color=np.arange(0, stack_outs.shape[0]),
        )
        ax.scatter(
            stack_model_mu[:, 1],
            stack_model_mu[:, 0],
            marker="o",
            label="Data From Expt.",
            # c=np.arange(0, stack_outs.shape[0], dtype=float),
        )

        cb = fig.colorbar(sm, cax)
        cb.set_label("In Stddev?")
        ax.set_xlabel(f"Normalized_{ycols[1]}")
        ax.set_ylabel(f"Normalized_{ycols[0]}")

        for i in range(0, len(bins_start)):
            ax.fill_between(
                [bins_start_norm[i, 0], bins_end_norm[i, 0]],
                -1,
                1,
                alpha=0.1,
                color="orange",
            )
        ax.legend()
        fig.tight_layout()
        fig.savefig("../plot_outputs/expt_dat_yield_loocv.png")

        fig, axs = plt.subplots(1, 2, figsize=(6 * 1, 6 * 1))
        ax = axs[0]
        idx = 1
        resid = (stack_outs[:, idx] - stack_model_mu[:, idx]) / stack_model_sigma[
            :, idx
        ]

        ax.plot(np.arange(1, stack_outs.shape[0] + 1), resid, "o")
        ax.set_xlabel("Dataset ID")
        ax.set_ylabel(
            r"$\frac{(y -\hat{y}_\mu)}{\hat{y}_\sigma}$" f" of Normalized_{ycols[idx]}"
        )

        ax = axs[1]
        idx = 0
        resid = (stack_outs[:, idx] - stack_model_mu[:, idx]) / stack_model_sigma[
            :, idx
        ]

        ax.plot(np.arange(1, stack_outs.shape[0] + 1), resid, "o")
        ax.set_xlabel("Dataset ID")
        ax.set_ylabel(
            r"$\frac{(y -\hat{y}_\mu)}{\hat{y}_\sigma}$" f"of Normalized_{ycols[idx]}"
        )
        fig.tight_layout()
        fig.savefig("../plot_outputs/expt_data_resid.png")
