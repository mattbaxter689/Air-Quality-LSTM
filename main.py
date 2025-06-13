import pandas as pd
from dotenv import load_dotenv
import numpy as np
from sklearn import set_config
from src.utils.db_utils import DatabaseConnection
from src.utils.utils import time_series_split
from src.transformers.time_transformer import AirQualityProcessor
from src.datasets.weather_dataset import WeatherDataset
from src.model.weather_model import WeatherLSTM, train_model
from src.model.early_stopper import EarlyStopping
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


load_dotenv()
set_config(transform_output="pandas")


def main():

    with DatabaseConnection() as conn:
        df = pd.read_sql(
            """
            SELECT *
            FROM air_quality
            """,
            con=conn,
        )

    # take the log of the aqi, since it is heavily skewed
    # also drop the CO2 and CH$ columns since they are 75% missing for now
    df["log_aqi"] = np.log1p(df["us_aqi"])
    df = (
        df.drop(columns=["us_aqi", "insert_time", "carbon_dioxide", "methane"])
        .sort_values(by="_time", ascending=True)
        .set_index("_time")
    )
    df["_time"] = df.index

    # -------- TRAIN-TEST SPLIT ------------
    train, val, test = time_series_split(df=df)

    # -------- TRANSFORMER SETUP -----------
    target_col = "log_aqi"
    time_col = "_time"
    train_target = train[target_col]
    val_target = val[target_col]
    test_target = test[target_col]

    train = train.drop(columns=[target_col])
    val = val.drop(columns=[target_col])
    test = test.drop(columns=[target_col])

    # extract columns from data that are numeric
    num_cols = [col for col in train.columns if col != time_col]

    processor = AirQualityProcessor(num_cols=num_cols, time_col=time_col)

    # Apply the pipeline. Fit and transform train, trainsform others
    train_transformed = processor.fit_transform(train)
    val_transformed = processor.transform(val)
    test_transformed = processor.transform(test)

    # -------- PYTORCH DATASETS ------------
    train_dataset = WeatherDataset(
        train_transformed, train_target, window_size=12
    )
    val_dataset = WeatherDataset(val_transformed, val_target, window_size=12)
    test_dataset = WeatherDataset(
        test_transformed, test_target, window_size=12
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=False
    )
    # -------- MODEL TRAINING ------------
    model = WeatherLSTM(input_size=15)
    early_stopper = EarlyStopping(patience=2, min_delta=1e-4)
    _, train_losses, val_losses = train_model(
        model=model,
        early_stopper=early_stopper,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
