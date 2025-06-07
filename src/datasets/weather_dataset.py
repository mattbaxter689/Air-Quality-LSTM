import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(
        self, weather: pd.DataFrame, target: pd.Series, window_size: int = 12
    ):
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
