from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Self
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


class AddCyclicalTime(BaseEstimator, TransformerMixin):
    def __init__(self, time_col: str) -> None:
        super().__init__()
        self.time_col = time_col

    def fit(self, X, y=None) -> Self:
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        X["hour"] = X[self.time_col].dt.hour
        X["day_of_week"] = X[self.time_col].dt.dayofweek
        X["month"] = X[self.time_col].dt.month

        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

        X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

        return X[
            [
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                "month_sin",
                "month_cos",
            ]
        ]

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]


class AirQualityProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols: list[str], time_col: str) -> None:
        super().__init__()
        self.num_cols = num_cols
        self.time_col = time_col
        self._pipeline = None
        self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        # Create numeric and time pipelines
        # Cyclical encoding time features does not require
        # scaling
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]
        )
        time_pipe = Pipeline(
            [("time", AddCyclicalTime(time_col=self.time_col))]
        )

        self.preprocesser = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.num_cols),
                ("time", time_pipe, [self.time_col]),
            ],
            verbose_feature_names_out=False,
        )

        self._pipeline = Pipeline([("preprocess", self.preprocesser)])

        return self._pipeline

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def fit(self, X, y=None) -> Self:
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        X_transform = self.pipeline.transform(X)

        # Check if the pipeline is fitted before calling get_feature_names_out
        preprocess = self.pipeline.named_steps["preprocess"]
        if hasattr(preprocess, "get_feature_names_out"):
            feature_names = preprocess.get_feature_names_out()
        else:
            feature_names = [
                f"feature_{i}" for i in range(X_transform.shape[1])
            ]

        return pd.DataFrame(X_transform, columns=feature_names, index=X.index)

    def fit_transform(self, X, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
