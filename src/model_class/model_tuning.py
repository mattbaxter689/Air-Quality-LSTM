import optuna
from optuna import Trial
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader
from src.model_class.weather_model import WeatherLSTM
from src.model_class.early_stopper import EarlyStopping
from src.datasets.weather_dataset import WeatherDataset
from pandas import DataFrame, Series
import pandas as pd
from src.utils.mlflow_manager import MLFlowLogger
from typing import Callable
import logging

logger = logging.getLogger("torch_weather")


class AirQualityFitHelper:
    """
    Helper class to perform hyperparameter optimization and model fit
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        past_input_size: int = 9,
        future_input_size: int = 6,
        device: str = "cpu",
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.past_input_size = past_input_size
        self.future_input_size = future_input_size
        self.device = device

    def train_model(
        self,
        model: WeatherLSTM,
        early_stopper: EarlyStopping,
        num_epochs: int = 20,
        lr: float = 1e-3,
        trial: Trial | None = None,
        use_validation: bool = True,
    ) -> tuple[WeatherLSTM, list[float], list[float] | None]:
        """
        Train the LSTM model, using the appropriate Trial object is it is part of
        hyperparameter tuning, or the validation dataset to perform final
        model fit

        Args:
            model (WeatherLSTM): The WeatherLSTM PyTorch model
            early_stopper (EarlyStopping): The early stopper helper class
            num_epochs (int, optional): The number of epochs to run model fit for. Defaults to 20.
            lr (float, optional): The learning rate to use for model fit. Defaults to 1e-3.
            trial (Trial | None, optional): The optuna Trial object for hyperparameter tuning. Defaults to None.
            use_validation (bool, optional): Use the validation set to assess fit?. Defaults to True.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned from Optuna seeing Trial is poor

        Returns:
            tuple[WeatherLSTM, list[float], list[float] | None]: Return the model fit, training, and validation loss results of the model fit
        """
        model = model.to(self.device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        train_epoch_losses = []
        val_epoch_losses = []

        for epoch in range(num_epochs):
            model.train()
            train_losses = []

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
            ):
                X_past, X_future, y_batch = (
                    batch["x_past"].to(self.device),
                    batch["x_future"].to(self.device),
                    batch["target"].to(self.device),
                )

                optimizer.zero_grad()
                outputs = model(X_past, X_future)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            train_epoch_losses.append(avg_train_loss)

            # Validation
            if use_validation:
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch in self.val_loader:
                        X_val_past = batch["x_past"].to(self.device)
                        X_val_future = batch["x_future"].to(self.device)
                        y_val = batch["target"].to(self.device)

                        val_outputs = model(X_val_past, X_val_future)
                        val_loss = criterion(val_outputs, y_val)
                        val_losses.append(val_loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                val_epoch_losses.append(avg_val_loss)

                if trial is not None:
                    trial.report(avg_val_loss, epoch)
                    if trial.should_prune():
                        logger.info("Optuna pruning trial")
                        raise optuna.exceptions.TrialPruned()

                if early_stopper(avg_val_loss):
                    logger.info(
                        f"Early stopping triggered at epoch: {epoch+1}"
                    )
                    logger.info(
                        f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}"
                    )
                    break

                logger.info(
                    f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}"
                )
        return model, train_epoch_losses, val_epoch_losses

    def test_model(self, model: WeatherLSTM) -> tuple[float, float]:
        """
        Assess the model on the testing set

        Args:
            model (WeatherLSTM): The LSTM PyTorch model

        Returns:
            tuple[float, float]: Return the RMSE and MSE on the test set
        """
        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                X_test_past = batch["x_past"].to(self.device)
                X_test_future = batch["x_future"].to(self.device)
                y_test = batch["target"].to(self.device)

                outputs = model(X_test_past, X_test_future)
                preds.append(outputs.numpy())
                targets.append(y_test.numpy())

        y_pred_all = np.concatenate(preds)
        y_true_all = np.concatenate(targets)

        # Inverse transform
        y_pred_real = np.expm1(y_pred_all)
        y_true_real = np.expm1(y_true_all)

        rmse = root_mean_squared_error(
            y_true_real.flatten(), y_pred_real.flatten()
        )
        mae = mean_absolute_error(y_true_real.flatten(), y_pred_real.flatten())

        logger.info(f"Test RMSE (original scale): {rmse:.4f}")
        logger.info(f"Test MAE  (original scale): {mae:.4f}")

        return rmse, mae

    def train_on_best(
        self,
        train_data: DataFrame,
        val_data: DataFrame,
        train_target: Series,
        val_target: Series,
        best_params: dict[str, int | float],
        num_epochs: int = 20,
    ) -> tuple[WeatherLSTM, list[float]]:
        """
        Train the final model on the best hyperparameter combination found
        during Optuna training

        Args:
            train_data (DataFrame): The training data as a pandas dataframe
            val_data (DataFrame): The validation data as a pandas dataframe
            train_target (Series): The training target
            val_target (Series): The validation target
            best_params (dict[str, int  |  float]): Dictionary containing the best hyperparameter values
            num_epochs (int, optional): The number of epochs to train for. Defaults to 20.

        Returns:
            WeatherLSTM: The final LSTM model fit
        """
        combined_data = pd.concat([train_data, val_data]).sort_index(
            ascending=True
        )
        combined_target = pd.concat([train_target, val_target]).sort_index(
            ascending=True
        )
        combined_dataset = WeatherDataset(
            weather=combined_data, target=combined_target, window_size=12
        )
        combined_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True,
        )

        model = WeatherLSTM(
            past_input_size=self.past_input_size,
            future_input_size=self.future_input_size,
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
        )

        early_stopper = EarlyStopping(patience=5)

        # overwrite the class instance of the loader and then revert
        original_loader = self.train_loader
        self.train_loader = combined_loader
        model, train_losses, _ = self.train_model(
            model=model,
            early_stopper=early_stopper,
            num_epochs=num_epochs,
            lr=best_params["lr"],
            use_validation=False,
        )
        self.train_loader = original_loader
        return model, train_losses

    def predict_with_timestamps(
        self,
        model: WeatherLSTM,
        test_df: DataFrame,
        time_index: pd.Index,
        window_size: int,
        forecast_len: int = 4,
    ) -> pd.DataFrame:
        """
        Helper function to plot air quality predictions with timestamps

        Args:
            model (WeatherLSTM): THe LSTM model to use
            test_df (DataFrame): The test data to perform predictions on. No target should be present
            time_index (pd.Index): The timestamps for the test data
            window_size (int): The size of the sliding window. This should match the window size of the data loader

        Returns:
            Series: Predicted air quality values with associated timestamps as index
        """
        model.eval()
        preds = []

        known_future_cols = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]

        past_value_cols = [
            col for col in test_df.columns if col not in known_future_cols
        ]
        test_past = test_df[past_value_cols].values.astype(np.float32)
        test_future = test_df[known_future_cols].values.astype(np.float32)

        with torch.no_grad():
            # Convert entire test_df to numpy for windowing
            data_np = test_df.values.astype(np.float32)

            for i in range(len(test_df) - window_size - forecast_len + 1):
                x_past = (
                    torch.tensor(test_past[i : i + window_size])
                    .unsqueeze(0)
                    .to(self.device)
                )
                x_future = (
                    torch.tensor(
                        test_future[
                            i + window_size : i + window_size + forecast_len
                        ]
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

                output = model(x_past, x_future).cpu().numpy()
                preds.append(output.squeeze())

        # Construct timestamps starting from window_size offset
        pred_index = time_index[window_size : window_size + len(preds)]

        # Convert predictions back from log scale
        columns = [f"t+{i+1}" for i in range(forecast_len)]
        preds_df = pd.DataFrame(
            data=np.expm1(preds), index=pred_index, columns=columns
        )

        return preds_df


def create_objective(
    fit_helper: AirQualityFitHelper, ml_logger: MLFlowLogger
) -> Callable[[Trial], float]:
    """
    Create the optuna study objective to execute for the hyperparameter
    tuning experiment

    Args:
        fit_helper (AirQualityFitHelper): The class that has functionality to handle model fit
        ml_logger (MLFlowLogger): The logger that will assist with logging to mlflow

    Returns:
        Callable[[Trial], float]: The objective function for the optuna study
    """

    def objective(trial: Trial) -> float:
        hidden_size = trial.suggest_int("hidden_size", 16, 64)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
        }

        model = WeatherLSTM(
            past_input_size=fit_helper.past_input_size,
            future_input_size=fit_helper.future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        early_stopper = EarlyStopping(patience=3)

        try:
            model, train_losses, val_losses = fit_helper.train_model(
                model=model,
                lr=lr,
                num_epochs=20,
                early_stopper=early_stopper,
                trial=trial,
            )
        except optuna.TrialPruned:
            raise

        ml_logger.log_trial(
            trial_number=trial.number,
            params=params,
            train_loss=train_losses,
            val_loss=val_losses,
        )

        return val_losses[-1]

    return objective
