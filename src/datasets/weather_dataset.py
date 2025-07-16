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
        self, weather: pd.DataFrame, target: pd.Series, window_size: int = 12
    ):
        """
        Initializes instance of custom weather dataset to be used by pytorch

        Args:
            weather (pd.DataFrame): Dataframe containing transformed weather data to use. Sorted by time
            target (pd.Series): Target containing air quality data. Sorted by time
            window_size (int, optional): The lookback window to use for the model. Defaults to 12 time points back.
        """
        super().__init__()
        self.window_size = window_size
        self.target = target.values.astype(np.float32)
        self.weather = weather.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.weather) - self.window_size

    def __getitem__(self, index) -> dict[str, np.array]:
        # We want to predict the next value in the window
        X = self.weather[index : index + self.window_size]
        y = self.target[index + self.window_size]

        return {"features": torch.tensor(X), "target": torch.tensor(y).float()}
