# This is going to have all the utilities we need 
# to make the main code easy to read

import sys
import numpy as np
from pathlib import Path
from pandas import DataFrame, read_csv
sys.path.extend(["../multibax-sklearn"])

from src.utils import XY_from_csv
from src.models import MGPR, fit_matern_hypers
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import MinMaxScaler
from src.algorithms import SubsetAlgorithm
from src.acquisition import run_acquisition

# Defaults
XCOLS = ["Time", "Temperature", "Pressure"]
YCOLS = ["Yield", "Median"]
XSCALEARR = np.array([[1, 200, 18], [24, 250, 50]])
YSCALEARR = np.array([[0, 8], [100, 20]])
DATASETS_DIR = Path("./data_expt")
DATASETS_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("./models_expt")
MODELS_DIR.mkdir(exist_ok=True)

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

tmp = 1
x_scaler.fit(XSCALEARR)


y_scaler.fit(YSCALEARR)
    
SCALERS = [x_scaler, y_scaler]


def fetch_data(df: DataFrame):
    """
    Fetches relevant data from a dataframe
    """
    xcols = XCOLS
    ycols = YCOLS
    x_scaler= SCALERS[0]
    y_scaler = SCALERS[1]

    X = np.array(df[xcols])
    Y = np.array(df[ycols])

    X_norm = x_scaler.transform(X)
    Y_norm = y_scaler.transform(Y)


    return X, Y, X_norm, Y_norm



def run_hall_acquisition(
    x_train,
    y_train,
    X,
    Y,
    strategy,
    algorithm,
    model,
    collected_ids,
    n_posterior_samples,
    prevent_requery=True,
    batch_strategy=None,
):
    # Ideally this would just be handled in an argument to run_acquisition, but to keep the repo
    _, _, trained_model, collected_ids, acquisition_function, switch_strategy = (
        run_acquisition(
            x_train.copy(),
            y_train.copy(),
            X.copy(),
            Y.copy(),
            strategy,
            algorithm,
            model,
            collected_ids.copy(),
            n_posterior_samples,
            prevent_requery=prevent_requery,
        )
    )
    # Get what would be the next acquired index
    next_id = collected_ids[-1]
    x_next = X[next_id]

    # Swap out next y with posterior mean prediction
    if batch_strategy == "post_mean":
        posterior_mean, posterior_std = trained_model.predict(x_next.reshape(1, -1))
        y_next = posterior_mean
    else:
        y_next = Y[next_id]

    # Add to training set
    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))

    return (
        x_train,
        y_train,
        trained_model,
        collected_ids,
        acquisition_function,
        switch_strategy,
    )


def batch_acquisition(
    x_train,
    y_train,
    X,
    Y,
    strategy,
    algorithm,
    model,
    collected_ids,
    n_posterior_samples,
    batch_size,
    prevent_requery=True,
    batch_strategy="post_mean",
):
    # Copy so we don't mix up real vs hallucinated
    x_train_here = x_train.copy()
    y_train_here = y_train.copy()

    X_here = X.copy()
    Y_here = Y.copy()

    collected_ids_here = collected_ids.copy()

    # Loop through hallucination and get next idxs
    hall_idxs = []
    for i_hall in range(batch_size):
        (
            x_train_here,
            y_train_here,
            trained_model_here,
            collected_ids_here,
            acquisition_function_here,
            switch_strategy,
        ) = run_hall_acquisition(
            x_train_here,
            y_train_here,
            X_here,
            Y_here,
            strategy,
            algorithm,
            model,
            collected_ids_here,
            n_posterior_samples,
            prevent_requery=True,
            batch_strategy=batch_strategy,
        )
        hall_idxs.append(collected_ids_here[-1])

    # Return batch

    return hall_idxs, trained_model_here




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
