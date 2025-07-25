import pandas as pd
import optuna
from dotenv import load_dotenv
import numpy as np
from sklearn import set_config
from src.utils.db_utils import DatabaseConnection
from src.utils.utils import time_series_split, warm_cold_start
from src.transformers.time_transformer import AirQualityProcessor
from src.datasets.weather_dataset import WeatherDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.model_class.model_tuning import AirQualityFitHelper, create_objective
from src.utils.mlflow_manager import MLFlowLogger
from src.utils.init_logger import create_logger
import os

load_dotenv()
set_config(transform_output="pandas")
logger = create_logger(name="torch_weather")


def main():
    START_TYPE = os.getenv("START_TYPE")
    df = warm_cold_start(start_type=START_TYPE)

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
        weather=train_transformed, target=train_target, window_size=12
    )
    val_dataset = WeatherDataset(
        weather=val_transformed, target=val_target, window_size=12
    )
    test_dataset = WeatherDataset(
        weather=test_transformed, target=test_target, window_size=12
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, drop_last=False
    )
    # -------- MODEL TRAINING ------------
    trainer = AirQualityFitHelper(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    with MLFlowLogger(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URL")
    ) as ml_logger:
        objective_func = create_objective(
            fit_helper=trainer, ml_logger=ml_logger
        )
        study = optuna.create_study(direction="minimize")
        if START_TYPE == "warm_start":
            params = ml_logger.load_recent_params()
            study.enqueue_trial(params=params)

        study.optimize(objective_func, n_trials=10)

        logger.info("Best trial:")
        logger.info(f"Value: {study.best_trial.value}")
        logger.info(f"Params: {study.best_params}")

        # Optionally, retrain best model on all data or test best params
        best_params = study.best_params

        final_model, train_loss = trainer.train_on_best(
            train_data=train_transformed,
            val_data=val_transformed,
            train_target=train_target,
            val_target=val_target,
            best_params=best_params,
            num_epochs=20,
        )
        rmse, mae = trainer.test_model(model=final_model)
        ml_logger.log_final_model(
            model=final_model,
            params=best_params,
            train_loss=train_loss,
            test_rmse=rmse,
            test_mae=mae,
            processor=processor,
        )

    preds_df = trainer.predict_with_timestamps(
        final_model, test_transformed, test_transformed.index, window_size=12
    )
    forecast_len = preds_df.shape[1]

    plt.figure(figsize=(15, 6))
    plt.plot(
        test_target.index,
        np.expm1(test_target.values),
        label="True AQI",
        color="black",
    )

    # Plot each forecast horizon with different linestyle/color
    colors = plt.cm.viridis(np.linspace(0, 1, forecast_len))
    for i, col in enumerate(preds_df.columns):
        plt.plot(
            preds_df.index,
            preds_df[col],
            label=f"Forecast {col}",
            linestyle="--",
            color=colors[i],
        )

    plt.xlabel("Time")
    plt.ylabel("Air Quality Index (AQI)")
    plt.title("Multi-step Forecasts vs True Values on Test Set")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
