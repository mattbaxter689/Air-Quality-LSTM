from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class AirQualityHourly(BaseModel, extra="forbid"):
    time: datetime = Field(
        description="The hourly time for the associated data points"
    )
    pm10: float = Field(
        description="PM10 associated with current hour of data"
    )
    pm2_5: float = Field(
        description="PM2.5 associated with current hour of data"
    )
    carbon_monoxide: float = Field(
        description="CO level associated with current hour of data"
    )
    nitrogen_dioxide: float = Field(
        description="NO2 level associated with current hour of data"
    )
    sulphur_dioxide: float = Field(
        description="SO2 level associated with current hour of data"
    )
    ozone: float = Field(
        description="O3 level associated with current level of data"
    )
    uv_index: float = Field(
        description="UV index level associated with current hour"
    )
    dust: float = Field(
        description="Dust level associated with current hour of data"
    )
    aerosol_optical_depth: float = Field(
        description="Aerosol optical value associated with current hour of data"
    )


class AirQuality(BaseModel):
    sequence: List[AirQualityHourly] = Field(
        description="List of 12 consecutive time points used to forecast next hour's air quality"
    )
