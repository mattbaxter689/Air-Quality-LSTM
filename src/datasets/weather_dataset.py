import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class LoaderConfig:
    """Config class for the dataloader parameters"""

    window_size: int = 12
    batch_size: int = 32
    shuffle: bool = (False,)
    drop_last: bool = True


@dataclass
class WeatherLoader:
    """Class to return dataloaders in single object"""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class WeatherDataset(Dataset):
    """
    Custom weather dataset for air quality data

    Args:
        Dataset (torch.utils.Dataset): Pytorch Dataset class
    """

    def __init__(
        self,
        weather: pd.DataFrame,
        target: pd.Series,
        window_size: int = 12,
        forecast_len: int = 4,
    ) -> None:
        """
        Initializes instance of custom weather dataset to be used by pytorch

        Args:
            weather (pd.DataFrame): Dataframe containing transformed weather data to use. Sorted by time
            target (pd.Series): Target containing air quality data. Sorted by time
            window_size (int, optional): The lookback window to use for the model. Defaults to 12 time points back.
            forecast_len (int, optional): The number of timesteps to forecast, defaults to 4 time points
        """
        super().__init__()

        known_future_cols = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]

        past_value_cols = [
            col for col in weather.columns if col not in known_future_cols
        ]

        self.weather = weather
        self.window_size = window_size
        self.target = target.values.astype(np.float32)
        self.known_future = weather[known_future_cols].values.astype(
            np.float32
        )
        self.past_vals = weather[past_value_cols].values.astype(np.float32)
        self.forecast_len = forecast_len

    def __len__(self) -> int:
        return len(self.weather) - self.window_size - self.forecast_len + 1

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        x_past = self.past_vals[index : index + self.window_size]
        x_future = self.known_future[
            index
            + self.window_size : index
            + self.window_size
            + self.forecast_len
        ]
        y = self.target[
            index
            + self.window_size : index
            + self.window_size
            + self.forecast_len
        ]

        return {
            "x_past": torch.tensor(x_past),
            "x_future": torch.tensor(x_future),
            "target": torch.tensor(y).float(),
        }


def create_dataloaders(
    train_tf: pd.DataFrame,
    val_tf: pd.DataFrame,
    test_tf: pd.DataFrame,
    train_target: pd.Series,
    val_target: pd.Series,
    test_target: pd.Series,
    loader_config: LoaderConfig = None,
) -> WeatherLoader:
    """
    Create the PyTorch datasets and dataloaders from the respective dataframes

    Args:
        train_tf (pd.DataFrame): Transformed training data
        val_tf (pd.DataFrame): Transformed validation data
        test_tf (pd.DataFrame): Transformed test data
        train_target (pd.Series): Training target variable
        val_target (pd.Series): Validation target variable
        test_target (pd.Series): Test target variable
        loader_config (LoaderConfig, optional): Optional loader config defining parameters for loaders. Defaults to None.

    Returns:
        WeatherLoader: Dataclass that returns all data loaders in single response
    """
    if not loader_config:
        loader_config = LoaderConfig()

    train_dataset = WeatherDataset(
        weather=train_tf,
        target=train_target,
        window_size=loader_config.window_size,
    )
    val_dataset = WeatherDataset(
        weather=val_tf,
        target=val_target,
        window_size=loader_config.window_size,
    )
    test_dataset = WeatherDataset(
        weather=test_tf,
        target=test_target,
        window_size=loader_config.window_size,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        drop_last=loader_config.drop_last,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        drop_last=loader_config.drop_last,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        drop_last=loader_config.drop_last,
    )

    return WeatherLoader(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
