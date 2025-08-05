from torch import nn
import torch
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """
    Config class for LSTM parameters
    """

    past_input_size: int
    future_input_size: int
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    output_size: int = 1


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
        bidirectional: bool = False,
        use_residual: bool = False,
        output_size: int = 1,
    ) -> None:
        """
        Initialize custom pytorch LSTM model

        Args:
            past_input_size (int): The number of known past covariates to supply.
            future_input_size (int): The number of known future time covariates to supply.
            hidden_size (int, optional): Hidden neurons to be used in model. Defaults to 32.
            num_layers (int, optional): Number of hidden layers for LSTM model. Defaults to 1.
            dropout (float, optional): Dropout for LSTM model. Defaults to 0.2.
            bidirectional (bool, optional): Should the LSTM use bidirectional. Defaults to False
            use_residual (bool, optional): Should residuals be used in calculation. Defaults to False
            output_size (int, optional): Output of the target. Defaults to 1
        """
        super(WeatherLSTM, self).__init__()

        # instantiate config to pass to Encoder and Decoder
        self.config = LSTMConfig(
            past_input_size=past_input_size,
            future_input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            output_size=output_size,
        )

        self.use_residual = use_residual

        self.lstm_encoder = WeatherEncoder(config=self.config)
        self.lstm_decoder = WeatherDecoder(config=self.config)

        # Optional to use residual. Helps to learn trends at time
        if future_input_size > 0 and use_residual:
            self.residual_projection = nn.Linear(future_input_size, 1)
        else:
            self.residual_projection = None

    def forward(
        self, x_past: torch.Tensor, x_future: torch.Tensor
    ) -> torch.Tensor:
        h, c = self.lstm_encoder(x_past)

        output = self.lstm_decoder(x_future, (h, c))

        # If using residual, add residual to output
        if self.residual_projection is not None:
            residual = self.residual_projection(x_future).squeeze(-1)
            output = output + residual

        return output


class WeatherEncoder(nn.Module):
    """
    Encoder Class LSTM for my air quality model. Separating from Decoder for a clean
    structure

    Args:
        nn (nn.Module): The required inheritance to create custom pytorch models
    """

    def __init__(self, config: LSTMConfig) -> None:
        """
        Initialize custom Encoder

        Args:
            config (LSTMConfig): Dataclass containing LSTM parameters
        """
        super(WeatherEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional

        self.encoder = nn.LSTM(
            input_size=config.past_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # If we want to use bidirectional, set up fc for projections
        if config.bidirectional:
            self.hidden_projection = nn.Linear(
                config.hidden_size * 2, config.hidden_size
            )
            self.cell_projection = nn.Linear(
                config.hidden_size * 2, config.hidden_size
            )

        else:
            self.hidden_projection = None
            self.cell_projection = None

    def forward(
        self, x_past: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, c) = self.encoder(x_past)

        # If using bidirectional, project hidden
        if self.bidirectional:
            batch_size = h.size(1)
            h = h.view(self.num_layers, 2, batch_size, -1)
            h = (
                h.transpose(1, 2)
                .contiguous()
                .view(self.num_layers, batch_size, -1)
            )
            h = self.hidden_projection(h)

            c = c.view(self.num_layers, 2, batch_size, -1)
            c = (
                c.transpose(1, 2)
                .contiguous()
                .view(self.num_layers, batch_size, -1)
            )
            c = self.cell_projection(c)

        return h, c


class WeatherDecoder(nn.Module):
    """
    Decoder Class LSTM for my air quality model. Separating from Encoder for a clean
    structure

    Args:
        nn (nn.Module): The required inheritance to create custom pytorch models
    """

    def __init__(self, config: LSTMConfig) -> None:
        """
        Initialize custom Decoder

        Args:
            config (LSTMConfig): Dataclass containing LSTM parameters
        """
        super(WeatherDecoder, self).__init__()

        self.decoder = nn.LSTM(
            input_size=config.future_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(config.dropout)

        if config.hidden_size >= 64:
            self.fc = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, config.output_size),
            )
        else:
            self.fc = nn.Linear(config.hidden_size, 1)

    def forward(
        self, x_future: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        decode_out, _ = self.decoder(x_future, hidden)
        decode_out = self.dropout(decode_out)
        output = self.fc(decode_out).squeeze(-1)

        return output
