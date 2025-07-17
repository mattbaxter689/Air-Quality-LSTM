import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


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
