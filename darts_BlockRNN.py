import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
import mlflow
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import BlockRNNModel
from darts.utils.likelihood_models import GaussianLikelihood
from optuna.integration import MLflowCallback
import pandas as pd
from darts import TimeSeries

# https://github.com/unit8co/darts/blob/master/examples/17-hyperparameter-optimization.ipynb

dataset = pd.read_parquet(r"D:\Git\darts-pipeline\data\dataset\rossmann.parquet")

features = [
    "DayOfWeek",
    "Customers",
    "Open",
    "Promo",
    "SchoolHoliday",
    "month",
    "week",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "hour",
    "weekend",
    "Sales_lag_1D",
    "Customers_lag_1D",
    "Open_lag_1D",
    "Promo_lag_1D",
    "Sales_lag_2D",
    "Customers_lag_2D",
    "Open_lag_2D",
    "Promo_lag_2D",
    "Sales_lag_3D",
    "Customers_lag_3D",
    "Open_lag_3D",
    "Promo_lag_3D",
    "Sales_lag_4D",
    "Customers_lag_4D",
    "Open_lag_4D",
    "Promo_lag_4D",
    "Sales_lag_5D",
    "Customers_lag_5D",
    "Open_lag_5D",
    "Promo_lag_5D",
    "Sales_lag_6D",
    "Customers_lag_6D",
    "Open_lag_6D",
    "Promo_lag_6D",
    "Sales_lag_7D",
    "Customers_lag_7D",
    "Open_lag_7D",
    "Promo_lag_7D",
    "Sales_lag_8D",
    "Customers_lag_8D",
    "Open_lag_8D",
    "Promo_lag_8D",
    "Sales_lag_9D",
    "Customers_lag_9D",
    "Open_lag_9D",
    "Promo_lag_9D",
    "Sales_lag_10D",
    "Customers_lag_10D",
    "Open_lag_10D",
    "Promo_lag_10D",
    "Sales_lag_11D",
    "Customers_lag_11D",
    "Open_lag_11D",
    "Promo_lag_11D",
    "Sales_lag_12D",
    "Customers_lag_12D",
    "Open_lag_12D",
    "Promo_lag_12D",
    "Sales_lag_13D",
    "Customers_lag_13D",
    "Open_lag_13D",
    "Promo_lag_13D",
    "Sales_lag_14D",
    "Customers_lag_14D",
    "Open_lag_14D",
    "Promo_lag_14D",
    "Sales_lag_15D",
    "Customers_lag_15D",
    "Open_lag_15D",
    "Promo_lag_15D",
    "Sales_lag_16D",
    "Customers_lag_16D",
    "Open_lag_16D",
    "Promo_lag_16D",
    "Sales_lag_17D",
    "Customers_lag_17D",
    "Open_lag_17D",
    "Promo_lag_17D",
    "Sales_lag_18D",
    "Customers_lag_18D",
    "Open_lag_18D",
    "Promo_lag_18D",
    "Sales_lag_19D",
    "Customers_lag_19D",
    "Open_lag_19D",
    "Promo_lag_19D",
    "Sales_window_7D_mean",
    "Customers_window_7D_mean",
    "Open_window_7D_mean",
    "Promo_window_7D_mean",
    "Sales_window_14D_mean",
    "Customers_window_14D_mean",
    "Open_window_14D_mean",
    "Promo_window_14D_mean",
]

VAL_LEN = 36


def get_ts(df, features):
    y = TimeSeries.from_group_dataframe(
        df, time_col="Date", group_cols=["Store"], value_cols=["Sales"], freq="D"
    )

    covariates = TimeSeries.from_group_dataframe(
        df, time_col="Date", group_cols=["Store"], value_cols=features, freq="D"
    )

    return y, covariates


def split_data(VAL_LEN):
    y, covariates = get_ts(dataset, features)

    y_train_list = [ts[:-VAL_LEN] for ts in y]
    y_test_list = [ts[-VAL_LEN:] for ts in y]

    y_scaler = Scaler(scaler=MaxAbsScaler(), global_fit=True)
    covariates_scaler = Scaler(scaler=MaxAbsScaler(), global_fit=True)

    train = y_scaler.fit_transform(y_train_list)
    test = y_scaler.transform(y_test_list)

    covariates = covariates_scaler.fit_transform(covariates)

    return y, y_scaler, train, test, covariates


y, y_scaler, train, test, covariates = split_data(VAL_LEN=VAL_LEN)


def objective(trial):
    try:
        in_len = trial.suggest_int("in_len", 12, 36)
        out_len = trial.suggest_int("out_len", 1, in_len - 1)
        n_rnn_layers = trial.suggest_int("n_rnn_layers ", 1, 5)
        hidden_dim = trial.suggest_int("hidden_dim  ", 25, 50)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        model = trial.suggest_categorical("model", ["RNN", "LSTM", "GRU"])

        pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stopper = EarlyStopping(
            "val_loss", min_delta=0.005, patience=3, verbose=True
        )
        callbacks = [pruner, early_stopper]

        if torch.cuda.is_available():
            num_workers = 20
        else:
            num_workers = 0

        pl_trainer_kwargs = {
            "accelerator": "auto",
            "callbacks": callbacks,
        }

        torch.manual_seed(42)

        model = BlockRNNModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            model=model,
            n_rnn_layers=n_rnn_layers,
            hidden_dim=hidden_dim,
            batch_size=30,
            n_epochs=50,
            nr_epochs_val_period=1,
            loss_fn=torch.nn.MSELoss(),
            optimizer_kwargs={"lr": lr},
            # likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="BlockRNNModel",
            force_reset=True,
            save_checkpoints=True,
        )

        val_series_list = [ts[-(VAL_LEN + in_len) :] for ts in y]

        model_val_set = y_scaler.transform(val_series_list)

        model.fit(
            series=train,
            val_series=model_val_set,
            past_covariates=covariates,
            val_past_covariates=covariates,
            num_loader_workers=num_workers,
        )

        model = BlockRNNModel.load_from_checkpoint("BlockRNNModel")

        preds = model.predict(series=train, past_covariates=covariates, n=VAL_LEN)
        smapes = smape(test, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        if not np.isfinite(smape_val):
            smape_val = float("inf")

        return smape_val

    except Exception as e:
        print(f"Exception in trial: {e}")
        return float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    mlflow_callback = MLflowCallback(
        tracking_uri="http://127.0.0.1:5000", metric_name="smape"
    )
    study.optimize(objective, n_trials=50, callbacks=[print_callback, mlflow_callback])
