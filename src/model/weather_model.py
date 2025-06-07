import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np


class WeatherLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # return 1 output as final
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(1)


def train_model(
    model: WeatherLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
) -> tuple[WeatherLSTM, list[float], list[float]]:
    model = model.to("cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_epoch_losses = []
    val_epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch, y_batch = batch["features"].to("cpu"), batch["target"].to(
                "cpu"
            )

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_epoch_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                X_val, y_val = batch["features"].to("cpu"), batch["target"].to(
                    "cpu"
                )
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        val_epoch_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}"
        )

    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
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

    print(f"\nTest RMSE (original scale): {rmse:.4f}")
    print(f"Test MAE  (original scale): {mae:.4f}")

    return model, train_epoch_losses, val_epoch_losses
