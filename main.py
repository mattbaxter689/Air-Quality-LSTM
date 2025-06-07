import pandas as pd
from dotenv import load_dotenv
import numpy as np
from sklearn import set_config
from src.utils.db_utils import DatabaseConnection
from src.transformers.time_transformer import AirQualityProcessor

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
    df = df.drop(
        columns=["us_aqi", "insert_time", "carbon_dioxide", "methane"]
    ).sort_values(by="_time", ascending=True)

    num_cols = [col for col in df.columns if col not in ["log_aqi", "_time"]]
    time_col = "_time"
    target = df["log_aqi"]
    features = df.drop(columns="log_aqi")
    air_processor = AirQualityProcessor(num_cols=num_cols, time_col=time_col)

    air_tf = air_processor.fit_transform(features)
    print(air_tf)


if __name__ == "__main__":
    main()
