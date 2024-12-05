from pandas import DataFrame, read_csv

from utils_activelearning import fetch_data, MODELS_DIR, DATASETS_DIR, MGPR, XCOLS, YCOLS, fit_matern_hypers, TopKBinAlgorithm, SCALERS, batch_acquisition
import solara
from plotly.subplots import make_subplots
from solara.components.columns import Columns
from solara.components.file_drop import FileDrop
from typing import Optional, cast
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern
import sys
import numpy as np
import plotly.graph_objects as go
import json
# Additions for Interface
# UQ Plot
# 3D Grid (of design space)
# Slider | Box
# Dataset Update Trigger save -> Button Pop up
# Hypers chosen iteration -- append |
# Throw in unit tests
# Seeding (SKlearn-BAX)
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

def kernel_hyperparams_to_dict(kernel):
    params = kernel.get_params()
    hyperparams = {}
    for key, value in params.items():
        if isinstance(value, (float, int, str, tuple)):  # Simple types
            hyperparams[key] = value
        elif hasattr(value, 'bounds') and hasattr(value, 'length_scale'):  # Example: RBF, ConstantKernel
            hyperparams[key] = {
                "bounds": np.asarray(value.bounds).tolist(),  # Convert numpy array to list
                "hyperparameter": np.asarray(value.length_scale).tolist() if hasattr(value, "length_scale") else None
            }
        elif hasattr(value, 'k1') and hasattr(value, 'k2'):  # For Sum or Product kernels
            hyperparams[key] = {
                "k1": kernel_hyperparams_to_dict(value.k1),
                "k2": kernel_hyperparams_to_dict(value.k2),
            }
        else:
            hyperparams[key] = str(value)  # Catch-all for unexpected types
    return hyperparams

class ProgramState:

    df = solara.reactive(cast(Optional[DataFrame], None))
    text_model_used = solara.reactive(cast(Optional[str], None))
    all_models = None
    latest_ct = solara.reactive(cast(Optional[int], None))
    nnext_batch = solara.reactive(False)
    got_next_measurement = solara.reactive(False)
    disable_save_button = solara.reactive(False)
    next_data = None
    median_new = solara.reactive(14)
    yield_new = solara.reactive(63.2)

    median_new2 = solara.reactive(14)
    yield_new2 = solara.reactive(63.2)


    @staticmethod
    def load_from_file(file):
        df = read_csv(file["file_obj"])
        ProgramState.df.value = df
        X, Y, X_norm, Y_norm = fetch_data(df)
        ProgramState.df.value = df
        collected_ids = list(np.arange(0, X_norm.shape[0])[Y[:, 0] >= 0])
        ct = len(collected_ids)
        ProgramState.latest_ct.value = ct
        ProgramState.setup_model()
    

    @staticmethod
    def load_latest_dset():
        fpath = DATASETS_DIR / f"exptdata_latest.csv"
        df = read_csv(fpath)
        X, Y, X_norm, Y_norm = fetch_data(df)
        ProgramState.df.value = df
        collected_ids = list(np.arange(0, X_norm.shape[0])[Y[:, 0] >= 0])
        ct = len(collected_ids)
        ProgramState.latest_ct.value = ct
        ProgramState.setup_model()
        ProgramState.disable_save_button.value = True
        
    
    @staticmethod
    def setup_model():
        df = ProgramState.df.value
        X, Y, X_norm, Y_norm = fetch_data(df)

        x_train = X_norm[Y[:, 0] >= 0]
        y_train = Y_norm[Y[:, 0] >= 0]

        collected_ids = list(np.arange(0, X_norm.shape[0])[Y[:, 0] >= 0])
        ct = len(collected_ids)
        model_file = MODELS_DIR / f"model_latest.pkl"

        n_features = len(XCOLS)
        n_properties = len(YCOLS)
                
        if model_file.exists():
            models = MGPR.load(model_file)
            
        else:
            kernel_initial = ConstantKernel(
            constant_value=1.0, constant_value_bounds=[0.01, 3.0]
            ) * Matern(
                nu=5 / 2,
                length_scale=n_features * [1.0],
                length_scale_bounds=n_features * [[0.01, 3.0]],
            ) + WhiteKernel(
                noise_level=0.01, noise_level_bounds="fixed"
            )
            kernel_initial_list = n_properties * [kernel_initial]
            models = MGPR(kernel_list=kernel_initial_list)

        kernel_initial_list = models.kernel_list
        kernel_list = fit_matern_hypers(x_train=x_train,y_train=y_train,kernel_list=kernel_initial_list)

        ProgramState.all_models = MGPR(kernel_list=kernel_list)

    @staticmethod   
    def update_model_get_next_measurement():
        df = ProgramState.df.value
        if ProgramState.nnext_batch.value:
            batch_size = 2
        else:
            batch_size = 1
        multi_gpr = ProgramState.all_models
        X, Y, X_norm, Y_norm = fetch_data(df)

        x_train = X_norm[Y[:, 0] >= 0]
        y_train = Y_norm[Y[:, 0] >= 0]

        collected_ids = list(np.arange(0, X_norm.shape[0])[Y[:, 0] >= 0])
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
        strategy = "SwitchBAX"
        n_posterior_samples = 10

        multi_gpr.fit(x_train, y_train)
        next_ids, model = batch_acquisition(
            x_train,
            y_train,
            X_norm,
            Y_norm,
            strategy,
            algorithm,
            multi_gpr,
            collected_ids,
            n_posterior_samples,
            batch_size,
        )
        posterior_mean, posterior_std = model.predict(X_norm)

        plot_means = SCALERS[1].inverse_transform(posterior_mean)
        df["Predicted Norm Median"] = posterior_mean[:, 1]
        df["Predicted Norm Yield"] = posterior_mean[:, 0]
        
        df["Std Norm Median"] = posterior_std[:, 1]
        df["Std Norm Yield"] = posterior_std[:, 0]
        df["Predicted Median"] = plot_means[:, 1]
        df["Predicted Yield"] = plot_means[:, 0]
        # ProgramState.df.value = df

        ProgramState.next_data = next_ids
        ProgramState.all_models = model
        ProgramState.latest_ct.value = None
        ProgramState.got_next_measurement.value = True
        ProgramState.disable_save_button.value = False

    @staticmethod
    def reset():
        ProgramState.df.value = None

        



    @staticmethod
    def save_dataset_and_model():
        df = ProgramState.df.value
        
        X, Y, X_norm, Y_norm = fetch_data(df)
        collected_ids = list(np.arange(0, X_norm.shape[0])[Y[:, 0] >= 0])
        ct = len(collected_ids)
        df.to_csv(DATASETS_DIR / f"exptdata_{ct}.csv")
        df.to_csv(DATASETS_DIR / f"exptdata_latest.csv")
        model_file_1 = MODELS_DIR / f"model_latest.pkl"
        model_file_2 = MODELS_DIR / f"model_{ct}.pkl"
        ProgramState.all_models.save(model_file_1)
        ProgramState.all_models.save(model_file_2)

        hypers_file_1 = MODELS_DIR / f"hypers_latest.json"
        hypers_file_2 = MODELS_DIR / f"hypers_{ct}.json"

        with open(hypers_file_1, "w") as f:
            kl = ProgramState.all_models.kernel_list
            print(kernel_hyperparams_to_dict(kl[0]))
            hp_list = [kernel_hyperparams_to_dict(k) for k in kl]
            print(hp_list)
            json.dump(hp_list, f)

        with open(hypers_file_2, "w") as f:
            json.dump(ProgramState.all_models.kernel_list, f)
        
        
        ProgramState.latest_ct.value = ct
        ProgramState.disable_save_button.value = True

    @staticmethod
    def add_data_point_to_df():
        df = ProgramState.df.value
        df[YCOLS[0]][ProgramState.next_data[0]] = ProgramState.yield_new.value
        df[YCOLS[1]][ProgramState.next_data[0]] = ProgramState.median_new.value
    
    @staticmethod
    def add_2data_point_to_df():
        df = ProgramState.df.value
        df[YCOLS[0]][ProgramState.next_data[1]] = ProgramState.yield_new2.value
        df[YCOLS[1]][ProgramState.next_data[1]] = ProgramState.median_new2.value

@solara.component
def Page():
    df = ProgramState.df.value
    ct_val = ProgramState.latest_ct.value

    with solara.AppBar():
        solara.lab.ThemeToggle()

    with solara.Sidebar():
        with solara.Card("Run BAXStics!", margin=0, elevation=0):
            with solara.Column():
                #with solara.Row():
                    #solara.Button("Load dataset", color="primary", text=True, outlined=True, on_click=ProgramState.load_from_file, disabled=df is not None)
                
                if sys.platform != "emscripten":
                    FileDrop(on_file=ProgramState.load_from_file, on_total_progress=lambda *args: None, label="Drag file here to load CSV")
                else:
                    solara.Info("File upload not supported in this environment")
                solara.Button("Load latest dataset", color="primary", text=True, outlined=True, on_click=ProgramState.load_latest_dset)
                # solara.Button("Save Iteration", color="primary", text=True, outlined=True, on_click=ProgramState.save_dataset_and_model)
                solara.Button("Clear dataset", color="primary", text=True, outlined=True, on_click=ProgramState.reset)
    
    if df is not None:
        with solara.Column():
            if ct_val is not None:
                fname1 = DATASETS_DIR / f"exptdata_latest.csv"
                fname2 = DATASETS_DIR / f"exptdata_{ct_val}.csv"
                model_file_1 = MODELS_DIR / f"model_latest.pkl"
                model_file_2 = MODELS_DIR / f"model_{ct_val}.pkl"
                if fname1.exists():
                    solara.Success(f"Saved csv to {fname1.absolute()} and {fname2.absolute()}")
                else:
                    solara.Warning("CSVs not saved")
                if model_file_1.exists():
                    solara.Success(f"Saved models to {model_file_1.absolute()} and {model_file_2.absolute()}")
                else:
                    solara.Warning("Models not saved")

            
            dfplt = df[df[YCOLS[0]] >= 0.0]
            scat_1 = go.Scatter(x=dfplt[YCOLS[1]], y=dfplt[YCOLS[0]],
                    mode='markers',
                    name='Experimental Data',
                    marker_line_color="midnightblue", marker_color="lightskyblue",
                    marker_line_width=2, marker_size=15,)
            fig = go.Figure()
            fig.add_trace(scat_1)
            
            fig.update_layout(
                        title=dict(text="Experimental Data"),
                        template="plotly_white", 
                        xaxis=dict(
                            title=dict(
                                text="Median no. of C atoms"
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                text="Yield (%)"
                            )
                        ),
                    )
            solara.FigurePlotly(fig)

            solara.Checkbox(label="Get 2 datapoints instead of 1", value=ProgramState.nnext_batch)
            #fig = 
            solara.Button("Get next experiment", color="primary", text=True, outlined=True, on_click=ProgramState.update_model_get_next_measurement)
            if ProgramState.got_next_measurement:
                fig_2 = go.Figure()
                if f"Predicted {YCOLS[1]}" in df.columns:
                    scat_2 = go.Scatter(x=df[f"Predicted {YCOLS[1]}"], y=df[f"Predicted {YCOLS[0]}"],
                        mode='markers',
                        name='Predicted Data',
                        marker_line_color="midnightblue", marker_color="lightskyblue",
                        marker_line_width=2, marker_size=15, visible='legendonly')
                    fig_2.add_trace(scat_2)
                    if ProgramState.next_data is not None:
                        dfnext =df.loc[ProgramState.next_data]
                        scat_n = go.Scatter(x=dfnext[f"Predicted {YCOLS[1]}"], y=dfnext[f"Predicted {YCOLS[0]}"],
                        mode='markers',
                        marker_symbol="x",
                        name='Next Measurement',
                        marker_line_color="Red", marker_color="Red",
                        marker_line_width=2, marker_size=15,)
                        fig_2.add_trace(scat_n)
    
                    scat_4 = go.Scatter(x=dfplt[f"Predicted {YCOLS[1]}"], y=dfplt[f"Predicted {YCOLS[0]}"],
                        mode='markers',
                        name='Expt Data',
                        marker_symbol="x-thin",
                        marker_line_color="green", marker_color="green",
                        marker_line_width=2, marker_size=15,)
                    fig_2.add_trace(scat_4)
                #


                scat_3 = go.Scatter(x=dfplt[YCOLS[1]], y=dfplt[YCOLS[0]],
                    mode='markers',
                    name='Expt Data',
                    marker_symbol="x-thin",
                    marker_line_color="black", marker_color="black",
                    marker_line_width=2, marker_size=15,)
                
                fig_2.add_trace(scat_3)
                fig_2.update_layout(
                        title=dict(text="Predicted Measurement"),
                        template="plotly_white", 
                        xaxis=dict(
                            title=dict(
                                text="Median no. of C atoms"
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                text="Yield (%)"
                            )
                        ),
                    )
                solara.FigurePlotly(fig_2)
            
            disabled_save_button = True
            if ProgramState.got_next_measurement:
                disabled_save_button = False
                fig_3 = make_subplots(rows=1, cols=2)
                if f"Predicted {YCOLS[1]}" in df.columns:
                    fig_3.add_trace(
                    go.Scatter(x=dfplt[f"Predicted {YCOLS[1]}"], y=dfplt[f"{YCOLS[1]}"],mode='markers',),
                    row=1, col=1
                    )

                    fig_3.add_trace(
                    go.Scatter(x=dfplt[f"{YCOLS[1]}"], y=dfplt[f"{YCOLS[1]}"],mode='lines',),
                    row=1, col=1
                    )

                    fig_3.add_trace(
                    go.Scatter(x=dfplt[f"Predicted {YCOLS[0]}"], y=dfplt[f"{YCOLS[0]}"],mode='markers',),
                    row=1, col=2
                    )
                    fig_3.add_trace(
                    go.Scatter(x=dfplt[f"{YCOLS[0]}"], y=dfplt[f"{YCOLS[0]}"],mode='lines',),
                    row=1, col=2
                    )

                    #fig_3.update_layout(
                    #    title=dict(text="Predicted Measurement"),
                    #)

                    #solara.FigurePlotly(fig_3)
            if ProgramState.got_next_measurement:
                if ProgramState.next_data is not None:
                    ndatas = len(ProgramState.next_data)
                   
                    nexttime = df["Time"][ProgramState.next_data[0]]
                    nexttemp = df["Temperature"][ProgramState.next_data[0]]
                    nextpress = df["Pressure"][ProgramState.next_data[0]]
                    
                    solara.Success(f"Next query point is {nexttime}h | {nexttemp} C | {nextpress} bar")
                    solara.SliderInt("Median of new point", value=ProgramState.median_new, min=8, max=20)
                    solara.Markdown(f"**Median value**: {ProgramState.median_new.value}")
                    solara.SliderFloat("Yield of new point", value=ProgramState.yield_new, min=0, max=100)
                    solara.Markdown(f"**Yield value**: {ProgramState.yield_new.value}")
                    solara.Button("Update Data", color="primary", text=True, outlined=True, on_click=ProgramState.add_data_point_to_df)

                    if ndatas == 2:
                        nexttime2 = df["Time"][ProgramState.next_data[1]]
                        nexttemp2 = df["Temperature"][ProgramState.next_data[1]]
                        nextpress2 = df["Pressure"][ProgramState.next_data[1]]
                        
                        solara.Success(f"Next query point is {nexttime2}h | {nexttemp2} C | {nextpress2} bar")
                        solara.SliderInt("Median of new point", value=ProgramState.median_new2, min=8, max=20)
                        solara.Markdown(f"**Median value**: {ProgramState.median_new2.value}")
                        solara.SliderFloat("Yield of new point", value=ProgramState.yield_new2, min=0, max=100)
                        solara.Markdown(f"**Yield value**: {ProgramState.yield_new2.value}")
                        solara.Button("Update Data", color="primary", text=True, outlined=True, on_click=ProgramState.add_2data_point_to_df)


            solara.Button("Save experiment", color="primary", text=True, outlined=True, on_click=ProgramState.save_dataset_and_model, disabled=ProgramState.disable_save_button.value)