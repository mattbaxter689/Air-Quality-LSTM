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
    ):
        """
        Initialize custom pytorch LSTM model

        Args:
            past_input_size (int): The number of known past covariates to supply.
            future_input_size (int): The number of known future time covariates to supply.
            hidden_size (int, optional): Hidden neurons to be used in model. Defaults to 32.
            num_layers (int, optional): Number of hidden layers for LSTM model. Defaults to 1.
            dropout (float, optional): Dropout for LSTM model. Defaults to 0.2.
        """
        super(WeatherLSTM, self).__init__()

        self.lstm_encoder = WeatherEncoder(
            past_input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lstm_decoder = WeatherDecoder(
            future_input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_past, x_future):
        h, c = self.lstm_encoder(x_past)

        decode_output, _ = self.lstm_decoder(x_future, (h, c))
        decode_output = self.dropout(decode_output)

        output = self.fc(decode_output).squeeze(-1)

        return output


class WeatherEncoder(nn.Module):
    """
    Encoder Class LSTM for my air quality model. Separating from Decoder for a clean
    structure

    Args:
        nn (nn.Module): The required inheritance to create custom pytorch models
    """

    def __init__(
        self,
        past_input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        """
        Initialize custom Encoder

        Args:
            past_input_size (int): The number of known past covariates to supply.
            hidden_size (int, optional): Hidden neurons to be used in model. Defaults to 32.
            num_layers (int, optional): Number of hidden layers for LSTM model. Defaults to 1.
            dropout (float, optional): Dropout for LSTM model. Defaults to 0.2.
        """
        super(WeatherEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x_past):
        _, (h, c) = self.encoder(x_past)

        return h, c


class WeatherDecoder(nn.Module):
    """
    Decoder Class LSTM for my air quality model. Separating from Encoder for a clean
    structure

    Args:
        nn (nn.Module): The required inheritance to create custom pytorch models
    """

    def __init__(
        self,
        future_input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        """
        Initialize custom Decoder

        Args:
            future_input_size (int): The number of known future time covariates to supply.
            hidden_size (int, optional): Hidden neurons to be used in model. Defaults to 32.
            num_layers (int, optional): Number of hidden layers for LSTM model. Defaults to 1.
            dropout (float, optional): Dropout for LSTM model. Defaults to 0.2.
        """
        super(WeatherDecoder, self).__init__()

        self.decoder = nn.LSTM(
            input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_future, hidden):
        decode_out = self.decoder(x_future, hidden)

        return decode_out
