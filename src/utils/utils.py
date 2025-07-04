import pandas as pd
from src.utils.db_utils import DatabaseConnection


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
