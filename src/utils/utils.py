import pandas as pd


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

    return train_df, val_df, test_df
