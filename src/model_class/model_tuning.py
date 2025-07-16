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
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        input_size: int,
        device: str = "cpu",
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.input_size = input_size
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
                X_batch, y_batch = batch["features"].to(self.device), batch[
                    "target"
                ].to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
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
                        X_val, y_val = batch["features"].to(
                            self.device
                        ), batch["target"].to(self.device)
                        val_outputs = model(X_val)
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
        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                X_test, y_test = batch["features"], batch["target"]
                outputs = model(X_test).squeeze()
                preds.append(outputs.numpy())
                targets.append(y_test.numpy())

        y_pred_all = np.concatenate(preds)
        y_true_all = np.concatenate(targets)

        # Inverse transform
        y_pred_real = np.expm1(y_pred_all)
        y_true_real = np.expm1(y_true_all)

        rmse = root_mean_squared_error(y_true_real, y_pred_real)
        mae = mean_absolute_error(y_true_real, y_pred_real)

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
    ) -> WeatherLSTM:
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
            input_size=self.input_size,
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
        test_df: pd.DataFrame,
        time_index: pd.Index,
        window_size: int,
    ) -> pd.Series:
        model.eval()
        preds = []

        with torch.no_grad():
            # Convert entire test_df to numpy for windowing
            data_np = test_df.values.astype(np.float32)

            for i in range(len(data_np) - window_size):
                X = data_np[i : i + window_size]
                X_tensor = torch.tensor(X).unsqueeze(0)  # Add batch dim
                output = model(X_tensor).squeeze().cpu().numpy()
                preds.append(output)

        # Construct timestamps starting from window_size offset
        pred_index = time_index[window_size : window_size + len(preds)]

        # Convert predictions back from log scale
        preds_exp = np.expm1(preds)

        return pd.Series(data=preds_exp, index=pred_index)


def create_objective(
    fit_helper: AirQualityFitHelper, ml_logger: MLFlowLogger
) -> Callable[[Trial], float]:
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
            input_size=fit_helper.input_size,
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
