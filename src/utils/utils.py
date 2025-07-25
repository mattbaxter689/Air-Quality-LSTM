import pandas as pd
import numpy as np
from src.utils.db_utils import DatabaseConnection
import os


def time_series_split(
    df: pd.DataFrame, train_size=0.8, val_size=0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a time series dataframe into train, val, and test sets in time order.

    Args:
        df (pd.DataFrame): DataFrame indexed or sorted by time.
        train_size (float): Fraction of data to use for training.
        val_size (float): Fraction of data to use for validation (from remaining).

    Returns:
        train_df, val_df, test_df
    """
    total_len = len(df)
    train_end = int(total_len * train_size)
    val_end = train_end + int(total_len * val_size)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    publish_validation_data(val=val_df)

    return train_df, val_df, test_df


def publish_validation_data(val: pd.DataFrame) -> None:
    """
    Publish validation data from model training to db for
    checking data drift

    Args:
        val (pd.DataFrame): The dataframe to upload

    Returns:
        None: None
    """

    with DatabaseConnection() as conn:
        val.to_sql(
            "validation_data", con=conn, if_exists="replace", index=False
        )

    return None


def warm_cold_start(start_type: str) -> pd.DataFrame:

    if not start_type:
        raise ValueError(
            "Please ensure you specify the START_TYPE variable for retraining type"
        )
    print(start_type)
    if start_type not in ["warm_start", "cold_start"]:
        raise ValueError(
            "Please ensure you specify either 'warm_start' or 'cold_start' for retraining type"
        )

    if start_type == "cold_start":
        with DatabaseConnection() as conn:
            df = pd.read_sql(
                """
                SELECT *
                FROM air_quality
                """,
                con=conn,
            )

        # take the log of the aqi, since it is heavily skewed
        # also drop the CO2 and CH4 columns since they are 75% missing for now
        df["log_aqi"] = np.log1p(df["us_aqi"])
        df = (
            df.drop(
                columns=["us_aqi", "insert_time", "carbon_dioxide", "methane"]
            )
            .sort_values(by="_time", ascending=True)
            .set_index("_time")
        )
        df["_time"] = df.index

    if start_type == "warm_start":
        with DatabaseConnection() as conn:
            df = pd.read_sql(
                """
                SELECT *
                FROM air_quality
                where _time >= now() - interval '365 days'
                """,
                con=conn,
            )

        # take the log of the aqi, since it is heavily skewed
        # also drop the CO2 and CH4 columns since they are 75% missing for now
        df["log_aqi"] = np.log1p(df["us_aqi"])
        df = (
            df.drop(
                columns=["us_aqi", "insert_time", "carbon_dioxide", "methane"]
            )
            .sort_values(by="_time", ascending=True)
            .set_index("_time")
        )
        df["_time"] = df.index

    return df
