import pandas as pd
import optuna
from dotenv import load_dotenv
import numpy as np
from sklearn import set_config
from src.utils.db_utils import DatabaseConnection
from src.utils.utils import time_series_split
from src.transformers.time_transformer import AirQualityProcessor
from src.datasets.weather_dataset import WeatherDataset
from src.model.weather_model import WeatherLSTM
from src.model.early_stopper import EarlyStopping
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.model.model_tuning import AirQualityTuner


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
    trainer = AirQualityTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_size=15,
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(trainer.objective, n_trials=10)

    print("Best trial:")
    print(f"Value: {study.best_trial.value}")
    print(f"Params: {study.best_params}")

    # Optionally, retrain best model on all data or test best params
    best_params = study.best_params
    best_model = WeatherLSTM(
        input_size=15,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    )

    early_stopper = EarlyStopping(patience=5)
    best_model, train_losses, val_losses = trainer.train_model(
        best_model,
        lr=best_params["lr"],
        num_epochs=20,
        early_stopper=early_stopper,
    )

    trainer.test_model(best_model)
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
    plt.savefig("figures/loss_curves.png")

    final_model = trainer.train_on_best(
        train_transformed,
        val_transformed,
        train_target,
        val_target,
        best_params,
        num_epochs=20,
    )

    predictions_series = trainer.predict_with_timestamps(
        final_model, test_transformed, test_transformed.index, window_size=12
    )
    window_size = 12

    # Align true values with prediction time frame
    y_true_aligned = np.expm1(test_target.values[window_size:])
    time_index_aligned = test_target.index[window_size:]

    plt.figure(figsize=(15, 5))
    plt.plot(time_index_aligned, y_true_aligned, label="True AQI")
    plt.plot(
        predictions_series.index,
        predictions_series.values,
        linestyle="--",
        label="Predicted AQI",
    )
    plt.xlabel("Time")
    plt.ylabel("Air Quality Index (AQI)")
    plt.title("Predictions vs True Values on Test Set")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/actual_predictions_overlay.png")


if __name__ == "__main__":
    main()
