import pandas as pd
from dotenv import load_dotenv
import numpy as np
from sklearn import set_config
from src.utils.db_utils import DatabaseConnection
from src.utils.utils import time_series_split
from src.transformers.time_transformer import AirQualityProcessor
from src.datasets.weather_dataset import WeatherDataset
from torch.utils.data import DataLoader

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
    train_transformed = processor.fit_transform(train)

    # Transform val and test using the *fitted* pipeline
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
        train_dataset, batch_size=32, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=False
    )

    for batch in train_loader:
        print(batch["features"])
        print(batch["target"])


if __name__ == "__main__":
    main()
