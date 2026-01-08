import optuna
from optuna import Trial
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataclasses import dataclass

logger = logging.getLogger("torch_weather")


@dataclass
class TrainingConfig:
    """Training parameter configurations"""

    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if not 0 <= self.max_grad_norm <= 10:
            raise ValueError("max_grad_norm should be between 0 and 10")


@dataclass
class TrainingResult:
    """Results of model training"""

    model: WeatherLSTM
    train_losses: list[float]
    val_losses: list[float] | None = None

    @property
    def best_val_loss(self) -> float | None:
        """Return best validation loss"""
        return min(self.val_losses) if self.val_losses else None

    @property
    def final_train_loss(self) -> float:
        """Get the final training loss"""
        return self.train_losses[-1] if self.train_losses else float("inf")


class AirQualityFitHelper:
    """
    Helper class to perform hyperparameter optimization and model fit
    """

    KNOWN_FUTURE_COLS = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
    ]

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        past_input_size: int = 9,
        future_input_size: int = 6,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the Helper fit class

        Args:
            train_loader (DataLoader): The Pytorch training data loader
            val_loader (DataLoader): The PyTorch validation data loader
            test_loader (DataLoader): The PyTorch test data loader
            past_input_size (int, optional): The size of known past covariates. Defaults to 9.
            future_input_size (int, optional): The size of known future-time covariates. Defaults to 6.
            device (str, optional): Device type for. Defaults to "cpu".
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.past_input_size = past_input_size
        self.future_input_size = future_input_size
        self.device = device

    def _create_optimizer_and_scheduler(
        self,
        model: WeatherLSTM,
        config: TrainingConfig,
        use_scheduler: bool = True,
    ) -> tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau | None]:
        """
        Helper to create the optimizer and scheduler if needed

        Args:
            model (WeatherLSTM): The weather model parameters to create the optimizer
            config (TrainingConfig): Training configuration variables
            use_scheduler (bool): Boolean determining if scheduler should be used or not

        Returns:
            tuple(optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau | None): Tuple containing the optimizer and scheduler if scheduler needed
        """
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )

        return optimizer, scheduler

    def _train_epoch(
        self,
        model: WeatherLSTM,
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        epoch_num: int,
    ) -> float:
        """

        Args:
            model (WeatherLSTM): The weather model
            optimizer (optim.Optimizer): The optimizer to be used
            config (TrainingConfig): Training config containing training parameters
            epoch_num (int): The epoch number being run

        Returns:
            float: The average loss of the epoch
        """
        model.train()
        epoch_losses = []

        for batch in self.train_loader:
            X_past, X_future, y_batch = (
                batch["x_past"].to(self.device),
                batch["x_future"].to(self.device),
                batch["target"].to(self.device),
            )

            optimizer.zero_grad()
            outputs = model(X_past, X_future)
            loss = F.smooth_l1_loss(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            optimizer.step()

            epoch_losses.append(loss.item())

        return sum(epoch_losses) / len(epoch_losses)

    def _validate_epoch(self, model: WeatherLSTM) -> float:
        """
        Check model metrics on validation data

        Args:
            model (WeatherLSTM): The weather model

        Returns:
            float: The average validation loss of the epoch
        """
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                X_val_past = batch["x_past"].to(self.device)
                X_val_future = batch["x_future"].to(self.device)
                y_val = batch["target"].to(self.device)

                val_outputs = model(X_val_past, X_val_future)
                val_loss = F.smooth_l1_loss(val_outputs, y_val)
                val_losses.append(val_loss.item())

        return sum(val_losses) / len(val_losses)

    def _should_stop_early(
        self, trial: Trial | None, val_loss: float, epoch: int
    ) -> bool:
        """
        Check if a trial should be pruned during optuna Trials

        Args:
            trial (Trial | None): The trial being run
            val_loss (float): The validation loss of the model
            epoch (int): The epoch number being run

        Returns:
            bool: True if the trail should be pruned
        """
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.info("Optuna pruning trial")
                return True
        return False

    def train_model(
        self,
        model: WeatherLSTM,
        early_stopper: EarlyStopping,
        config: TrainingConfig | None = None,
        trial: Trial | None = None,
        use_validation: bool = True,
    ) -> TrainingResult:
        """
        Train the LSTM model, using the appropriate Trial object is it is part of
        hyperparameter tuning, or the validation dataset to perform final
        model fit

        Args:
            model (WeatherLSTM): The WeatherLSTM PyTorch model
            early_stopper (EarlyStopping): The early stopper helper class
            config (TrainingConfig | None, optional): The training configuration. Defaults to None
            trial (Trial | None, optional): The optuna Trial object for hyperparameter tuning. Defaults to None.
            use_validation (bool, optional): Use the validation set to assess fit?. Defaults to True.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned from Optuna seeing Trial is poor

        Returns:
            TrainingResult: Dataclass holding final model git and best parameters
        """

        if config is None:
            config = TrainingConfig()

        model = model.to(self.device)
        optimizer, scheduler = self._create_optimizer_and_scheduler(
            model, config, use_scheduler=not use_validation
        )

        train_epoch_losses = []
        val_epoch_losses = [] if use_validation else None

        for epoch in tqdm(range(config.num_epochs), desc="Training progress"):

            # Training portion
            avg_train_loss = self._train_epoch(model, optimizer, config, epoch)
            train_epoch_losses.append(avg_train_loss)

            if use_validation:
                avg_val_loss = self._validate_epoch(model)
                val_epoch_losses.append(avg_val_loss)

                if self._should_stop_early(trial, avg_val_loss, epoch):
                    raise optuna.exceptions.TrialPruned()

                if early_stopper(val_loss=avg_val_loss):
                    logger.info(
                        f"Early stopping triggered at epoch: {epoch + 1}"
                    )
                    break

                logger.info(
                    f"Epoch {epoch+1}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}"
                )
            else:
                if scheduler is not None:
                    scheduler.step(avg_train_loss)
                logger.info(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}")

        return TrainingResult(model, train_epoch_losses, val_epoch_losses)

    def _get_predictions_and_targets(
        self, model: WeatherLSTM, data_loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to generate predicted values and associate with target value for
        model scoring

        Args:
            model (WeatherLSTM): Model to be used to generate predictions
            data_loader (DataLoader): The torch data loader to use

        Returns:
            tuple(np.ndarray, np.ndarray): Tuple of numpy arrays containing the predicted and target values
        """
        
        preds, targets = [], []
        model.eval()

        with torch.no_grad():
            for batch in data_loader:
                X_test_past = batch["x_past"].to(self.device)
                X_test_future = batch["x_future"].to(self.device)
                y_test = batch["target"].to(self.device)

                outputs = model(X_test_past, X_test_future)
                preds.append(outputs.cpu().numpy())
                targets.append(y_test.cpu().numpy())

        return np.concatenate(preds), np.concatenate(targets)

    def test_model(self, model: WeatherLSTM) -> tuple[float, float]:
        """
        Assess the model on the testing set

        Args:
            model (WeatherLSTM): The LSTM PyTorch model

        Returns:
            tuple[float, float]: Return the RMSE and MSE on the test set
        """
        predictions, targets = self._get_predictions_and_targets(
            model, self.test_loader
        )

        # Inverse transform
        y_pred_real = np.expm1(predictions)
        y_true_real = np.expm1(targets)

        rmse = root_mean_squared_error(
            y_true_real.flatten(), y_pred_real.flatten()
        )
        mae = mean_absolute_error(y_true_real.flatten(), y_pred_real.flatten())

        logger.info(f"Test RMSE (original scale): {rmse:.4f}")
        logger.info(f"Test MAE  (original scale): {mae:.4f}")

        return rmse, mae

    def _create_combined_dataset(
        self,
        train_data: DataFrame,
        val_data: DataFrame,
        train_target: Series,
        val_target: Series,
        window_size: int = 12,
        batch_size: int = 32,
    ) -> DataLoader:
        """
        Create a data loader combining the training and validation data

        Args:
            train_data (DataFrame): The training data
            val_data (DataFrame): The validation data
            train_target (Series): The training target
            val_target (Series): The validation target
            window_size (int, optional): The lookback window to use. Defaults to 12.
            batch_size (int, optional): The batch size for the model. Defaults to 32.

        Returns:
            DataLoader: Dataloader consisting of the combined data
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

        return DataLoader(
            dataset=combined_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True,
        )

    def train_on_best(
        self,
        train_data: DataFrame,
        val_data: DataFrame,
        train_target: Series,
        val_target: Series,
        best_params: dict[str, int | float],
        config: TrainingConfig | None = None,
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
            tuple(WeatherLSTM, list[float]): The final LSTM model fit with the associated losses
        """

        if config is None:
            config = TrainingConfig(lr=best_params["lr"])

        combined_loader = self._create_combined_dataset(
            train_data, val_data, train_target, val_target
        )

        model = WeatherLSTM(
            past_input_size=self.past_input_size,
            future_input_size=self.future_input_size,
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            use_residual=True,
        )

        early_stopper = EarlyStopping(patience=5)

        # overwrite the class instance of the loader and then revert
        original_loader = self.train_loader
        self.train_loader = combined_loader

        try:
            result = self.train_model(
                model=model,
                early_stopper=early_stopper,
                config=config,
                use_validation=False,
            )
        finally:
            self.train_loader = original_loader

        return result.model, result.train_losses

    def _validate_prediction_inputs(
        self,
        test_df: DataFrame,
        time_index: pd.Index,
        window_size: int,
        forecast_len: int,
    ) -> None:
        """
        Helper function to ensure that prediction inputs are the proper length

        Args:
            test_df (DataFrame): The dataframe to validate inputs for
            time_index (pd.Index): The time index to validate length of input data
            window_size (int): The lookback window that was used during model fit to predict the next timestamp(s)
            forecast_len (int): Specifies how many timesteps to forecast for
        """
        if len(test_df) < window_size + forecast_len:
            raise ValueError(
                f"Test data length ({len(test_df)}) must be >= "
                f"window_size + forecast_len ({window_size + forecast_len})"
            )

        if len(time_index) != len(test_df):
            raise ValueError(
                f"Time index length ({len(time_index)}) must match "
                f"test_df length ({len(test_df)})"
            )

        missing_cols = [
            col for col in self.KNOWN_FUTURE_COLS if col not in test_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required future columns: {missing_cols}"
            )

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
            forecast_len(int, optional): The nu,ber of timesteps to forecast into the future. Defaults to 4

        Returns:
            Series: Predicted air quality values with associated timestamps as index
        """

        self._validate_prediction_inputs(
            test_df, time_index, window_size, forecast_len
        )

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
        test_future = test_df[self.KNOWN_FUTURE_COLS].values.astype(np.float32)

        with torch.no_grad():

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
        return pd.DataFrame(
            data=np.expm1(preds), index=pred_index, columns=columns
        )


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
        params = {
            "hidden_size": trial.suggest_int("hidden_size", 16, 64),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        }

        model = WeatherLSTM(
            past_input_size=fit_helper.past_input_size,
            future_input_size=fit_helper.future_input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            use_residual=True,
        )
        early_stopper = EarlyStopping(patience=3)
        config = TrainingConfig(lr=params["lr"], num_epochs=20)

        try:
            result = fit_helper.train_model(
                model=model,
                config=config,
                early_stopper=early_stopper,
                trial=trial,
            )
        except optuna.TrialPruned:
            raise

        ml_logger.log_trial(
            trial_number=trial.number,
            params=params,
            train_loss=result.train_losses,
            val_loss=result.val_losses,
        )

        return result.best_val_loss

    return objective
