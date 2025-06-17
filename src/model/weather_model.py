import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np
from src.model.early_stopper import EarlyStopping


class WeatherLSTM(nn.Module):
    """
    Pytorch LSTM model to predict air quality for my region

    Args:
        nn (nn.Module): The required inheritance to create custom pytorch models
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        """
        Initialize custom pytorch LSTM model

        Args:
            input_size (int): The size of input data to use in the model
            hidden_size (int, optional): Hidden neurons to be used in model. Defaults to 32.
            num_layers (int, optional): Number of hidden layers for LSTM model. Defaults to 1.
            dropout (float, optional): Dropout for LSTM model. Defaults to 0.2.
        """
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # return 1 output as final
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out.squeeze(1)
