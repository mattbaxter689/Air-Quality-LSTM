from torch import nn


class WeatherLSTM(nn.Module):
    """
    Pytorch LSTM model to predict air quality for my region

    Args:
        nn (nn.Module): The required inheritance to create custom pytorch models
    """

    def __init__(
        self,
        past_input_size: int,
        future_input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        forecast_len: int = 4,
    ):
        """
        Initialize custom pytorch LSTM model

        Args:
            past_input_size (int): The number of known past covariates to supply.
            future_input_size (int): The number of known future time covariates to supply.
            hidden_size (int, optional): Hidden neurons to be used in model. Defaults to 32.
            num_layers (int, optional): Number of hidden layers for LSTM model. Defaults to 1.
            dropout (float, optional): Dropout for LSTM model. Defaults to 0.2.
            forecast_len (int, optional): The number of timesteps to forecast into the future. Defaults to 4
        """
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.forecast_len = forecast_len

        # return 1 output as final
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_past, x_future):
        _, (h, c) = self.encoder(x_past)

        dec_out, _ = self.decoder(x_future, (h, c))
        dec_out = self.dropout(dec_out)

        out = self.fc(dec_out).squeeze(-1)

        return out
