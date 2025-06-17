import mlflow
import mlflow.pytorch
import mlflow.sklearn
import matplotlib.pyplot as plt
import os
import uuid
from typing import Self
from src.model.weather_model import WeatherLSTM
from src.transformers.time_transformer import AirQualityProcessor
import joblib
import logging

logger = logging.getLogger("torch_weather")


class MLFlowLogger:
    def __init__(
        self,
        experiment_name: str = "air_quality_experiment",
        tracking_uri: str | None = "http://localhost:5000",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.final_run_name = f"{self.experiment_name}_final_models"
        self.active_run = None

    def __enter__(self) -> Self:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.active_run = mlflow.start_run()
        logger.info(f"Started MLflow run: {self.active_run.info.run_id}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_trial(
        self, trial_number: int, params: dict, train_loss: list, val_loss: list
    ) -> None:
        with mlflow.start_run(nested=True, run_name=f"trial_{trial_number}"):
            logger.info("Logging results from nested run")
            mlflow.log_params(params)
            for i, (t, v) in enumerate(zip(train_loss, val_loss)):
                mlflow.log_metric("train_loss", t, step=i)
                mlflow.log_metric("val_loss", v, step=i)
            self._log_loss_plot(
                train_loss, f"figures/trial_{trial_number}_loss.png", val_loss
            )

    def log_final_model(
        self,
        model: WeatherLSTM,
        params: dict[str, int | float],
        train_loss: list[float],
        test_rmse: float,
        test_mae: float,
        processor: AirQualityProcessor | None,
    ) -> None:
        logger.info("Logging final model fit and parameters")
        mlflow.set_experiment(self.final_run_name)
        with mlflow.start_run(
            run_name="final_model_" + str(uuid.uuid4())[:8], nested=True
        ):
            mlflow.log_params(params)
            for i, t in enumerate(train_loss):
                mlflow.log_metric("train_loss", t, step=i)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae", test_mae)
            self._log_loss_plot(train_loss, "figures/final_model_loss.png")
            mlflow.pytorch.log_model(model, name="lstm_model")

            if processor is not None:
                self.log_pipeline(processor=processor)

    def log_pipeline(
        self,
        processor: AirQualityProcessor,
        artifact_path: str = "preprocessing_pipeline",
    ):
        if not hasattr(processor, "pipeline"):
            raise AttributeError(
                "Processor instance has no attribute 'pipeline'"
            )
        joblib.dump(processor.pipeline, f"models/{artifact_path}.joblib")
        mlflow.log_artifact(f"models/{artifact_path}.joblib")
        os.remove(f"models/{artifact_path}.joblib")

    def _log_loss_plot(
        self,
        train_loss: list[float],
        filename: str,
        val_loss: list[float] | None = None,
    ) -> None:
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        if val_loss is not None:
            ax.plot(val_loss, label="Validation Loss")
        ax.set_title("Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close(fig)
        os.remove(filename)
