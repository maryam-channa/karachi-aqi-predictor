"""
Production AQI feature pipeline for Karachi.

Purpose
-------
1. Fetch the current weather and pollutant observation from OpenWeather.
2. Append the raw observation to the existing Hopsworks raw feature group
   (aqi_features, v3).
3. Maintain a local raw cache at data/historical_aqi.csv.
4. Build the exact 98-feature schema used by the verified
   GradientBoostingRegressor.
5. Write current, unlabeled 98-feature rows to a separate Hopsworks
   serving feature group: aqi_serving_features, version 1.
6. Do not fabricate a next-hour target; labels are created retrospectively
   by the historical backfill/training pipeline.
7. Never fabricate missing hourly history.

Important:
- The current trained model has exactly 98 features.
- The model itself does NOT currently consume weather variables.
  Weather is collected as part of the raw pipeline for future feature
  expansion, but the model feature schema remains exactly 98 columns.
- A model-feature row is written only when the required 24-hour lag history
  is genuinely available and hourly-contiguous.

Environment variables required
-------------------------------
HOPSWORKS_HOST
HOPSWORKS_PROJECT
HOPSWORKS_API_KEY
OPENWEATHER_API_KEY

Optional
--------
RAW_FEATURE_GROUP_NAME=aqi_features
RAW_FEATURE_GROUP_VERSION=3
MODEL_FEATURE_GROUP_NAME=aqi_model_features
MODEL_FEATURE_GROUP_VERSION=1
CITY_NAME=Karachi
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import hopsworks
import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CACHE_PATH = DATA_DIR / "historical_aqi.csv"

HOPSWORKS_HOST = os.getenv(
    "HOPSWORKS_HOST",
    "eu-west.cloud.hopsworks.ai",
)

HOPSWORKS_PROJECT = os.getenv(
    "HOPSWORKS_PROJECT",
    "noismore",
)

HOPSWORKS_API_KEY = os.getenv(
    "HOPSWORKS_API_KEY",
)

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY",
)

LAT = 24.8607
LON = 67.0011
CITY_NAME = os.getenv("CITY_NAME", "Karachi")

RAW_FEATURE_GROUP_NAME = os.getenv(
    "RAW_FEATURE_GROUP_NAME",
    "aqi_features",
)

RAW_FEATURE_GROUP_VERSION = int(
    os.getenv(
        "RAW_FEATURE_GROUP_VERSION",
        "3",
    )
)

REQUEST_TIMEOUT = 20
HOPSWORKS_READ_TIMEOUT = 300


# ============================================================
# EXACT 98-FEATURE MODEL SCHEMA
# ============================================================

FEATURE_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_morning",
    "is_afternoon",
    "is_evening",
    "is_night",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",

    "pm25_lag_1h",
    "pm25_lag_2h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",

    "pm10_lag_1h",
    "pm10_lag_2h",
    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_24h",

    "no2_lag_1h",
    "no2_lag_2h",
    "no2_lag_3h",
    "no2_lag_6h",
    "no2_lag_12h",
    "no2_lag_24h",

    "o3_lag_1h",
    "o3_lag_2h",
    "o3_lag_3h",
    "o3_lag_6h",
    "o3_lag_12h",
    "o3_lag_24h",

    "aqi_lag_1h",
    "aqi_lag_2h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",

    "pm25_rolling_mean_3h",
    "pm25_rolling_std_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_std_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_std_12h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_std_24h",

    "pm10_rolling_mean_3h",
    "pm10_rolling_std_3h",
    "pm10_rolling_mean_6h",
    "pm10_rolling_std_6h",
    "pm10_rolling_mean_12h",
    "pm10_rolling_std_12h",
    "pm10_rolling_mean_24h",
    "pm10_rolling_std_24h",

    "no2_rolling_mean_3h",
    "no2_rolling_std_3h",
    "no2_rolling_mean_6h",
    "no2_rolling_std_6h",
    "no2_rolling_mean_12h",
    "no2_rolling_std_12h",
    "no2_rolling_mean_24h",
    "no2_rolling_std_24h",

    "o3_rolling_mean_3h",
    "o3_rolling_std_3h",
    "o3_rolling_mean_6h",
    "o3_rolling_std_6h",
    "o3_rolling_mean_12h",
    "o3_rolling_std_12h",
    "o3_rolling_mean_24h",
    "o3_rolling_std_24h",

    "aqi_rolling_mean_3h",
    "aqi_rolling_std_3h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_std_6h",
    "aqi_rolling_mean_12h",
    "aqi_rolling_std_12h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_std_24h",

    "pm25_pm10_ratio",
    "pm10_pm25_ratio",
    "pm25_no2_ratio",
    "o3_no2_ratio",

    "pm25_change_1h",
    "pm10_change_1h",
    "no2_change_1h",
    "o3_change_1h",
]

RAW_COLUMNS = [
    "timestamp",
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "o3",
]

if len(FEATURE_COLUMNS) != 98:
    raise RuntimeError(
        f"Internal schema error: expected 98 features, "
        f"found {len(FEATURE_COLUMNS)}."
    )


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(f"[AQI PIPELINE] {message}", flush=True)


def require_environment() -> None:

    missing = []

    if not HOPSWORKS_API_KEY:
        missing.append("HOPSWORKS_API_KEY")

    if not OPENWEATHER_API_KEY:
        missing.append("OPENWEATHER_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing environment variable(s): "
            + ", ".join(missing)
        )


def connect_hopsworks():
    log("Connecting to Hopsworks...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
    )

    fs = project.get_feature_store()

    log("Hopsworks Feature Store connection established.")

    return project, fs


# ============================================================
# OPENWEATHER
# ============================================================

def fetch_openweather_observation() -> tuple[dict, dict]:
    """
    Fetch current weather and current pollutant/AQI observations.

    OpenWeather's air-pollution response provides:
      main.aqi: 1..5
      components.pm2_5
      components.pm10
      components.no2
      components.o3
    """

    weather_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}"
        f"&lon={LON}"
        "&units=metric"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    pollution_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}"
        f"&lon={LON}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    log("Fetching current weather from OpenWeather...")

    weather_response = requests.get(
        weather_url,
        timeout=REQUEST_TIMEOUT,
    )

    weather_response.raise_for_status()

    weather = weather_response.json()

    log("Fetching current pollutant/AQI data from OpenWeather...")

    pollution_response = requests.get(
        pollution_url,
        timeout=REQUEST_TIMEOUT,
    )

    pollution_response.raise_for_status()

    pollution = pollution_response.json()

    if not pollution.get("list"):
        raise RuntimeError(
            "OpenWeather returned no current air-pollution record."
        )

    return weather, pollution


def normalize_current_observation(
    weather: dict,
    pollution: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Convert OpenWeather responses into:
      - one raw AQI/pollution row
      - weather metadata for logging

    Timestamp is taken from the OpenWeather air-pollution record.
    """

    entry = pollution["list"][0]

    components = entry.get("components", {})
    main = entry.get("main", {})

    timestamp = pd.to_datetime(
        entry.get("dt"),
        unit="s",
        utc=True,
    )

    aqi = main.get("aqi")

    required_values = {
        "aqi": aqi,
        "pm25": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
    }

    missing = [
        name
        for name, value in required_values.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            "OpenWeather current pollution response is missing: "
            + ", ".join(missing)
        )

    raw_row = pd.DataFrame(
        [{
            "timestamp": timestamp,
            "aqi": float(aqi),
            "pm25": float(components["pm2_5"]),
            "pm10": float(components["pm10"]),
            "no2": float(components["no2"]),
            "o3": float(components["o3"]),
        }]
    )

    weather_info = {
        "temperature_c": weather.get("main", {}).get("temp"),
        "humidity_percent": weather.get("main", {}).get("humidity"),
        "pressure_hpa": weather.get("main", {}).get("pressure"),
        "wind_speed_mps": weather.get("wind", {}).get("speed"),
        "cloud_percent": weather.get("clouds", {}).get("all"),
    }

    return raw_row, weather_info


# ============================================================
# RAW HISTORY
# ============================================================

def clean_raw_history(df: pd.DataFrame) -> pd.DataFrame:

    missing = [
        c
        for c in RAW_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Raw history is missing columns: "
            + ", ".join(missing)
        )

    df = df[RAW_COLUMNS].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in RAW_COLUMNS[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(subset=RAW_COLUMNS)
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def load_local_raw_cache() -> pd.DataFrame:

    if not RAW_CACHE_PATH.exists():
        return pd.DataFrame(
            columns=RAW_COLUMNS
        )

    df = pd.read_csv(RAW_CACHE_PATH)

    return clean_raw_history(df)


def save_local_raw_cache(df: pd.DataFrame) -> None:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean = clean_raw_history(df)

    clean.to_csv(
        RAW_CACHE_PATH,
        index=False,
    )

    log(
        f"Local raw cache saved: {len(clean)} rows."
    )


def read_raw_history_from_hopsworks(fs) -> pd.DataFrame:

    log(
        f"Reading recent raw history from Hopsworks "
        f"({RAW_FEATURE_GROUP_NAME}, v{RAW_FEATURE_GROUP_VERSION})..."
    )

    fg = fs.get_feature_group(
        name=RAW_FEATURE_GROUP_NAME,
        version=RAW_FEATURE_GROUP_VERSION,
    )

    end_time = pd.Timestamp.now(tz="UTC")

    # 72 hours is enough for the 24-hour lags plus rolling windows.
    start_time = (
        end_time
        - pd.Timedelta(hours=72)
    )

    df = fg.read(
        start_time=start_time,
        end_time=end_time,
        read_options={
            "arrow_flight_config": {
                "timeout": HOPSWORKS_READ_TIMEOUT
            }
        },
    )

    if df is None or df.empty:
        return pd.DataFrame(
            columns=RAW_COLUMNS
        )

    return clean_raw_history(df)


def merge_histories(
    hopsworks_history: pd.DataFrame,
    local_history: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    if not hopsworks_history.empty:
        frames.append(
            hopsworks_history
        )

    if not local_history.empty:
        frames.append(
            local_history
        )

    if not frames:
        return pd.DataFrame(
            columns=RAW_COLUMNS
        )

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    return clean_raw_history(
        merged
    )


# ============================================================
# WRITE RAW OBSERVATION TO HOPSWORKS
# ============================================================

def get_or_create_serving_feature_group(fs):
    """Get/create a feature group for current 98-feature inference rows.

    The existing ``aqi_features`` v3 group is a labeled training dataset
    and requires ``id`` + ``target_aqi``. A brand-new observation does not
    know its next-hour target yet, so it must not be written into that
    labeled group with a fabricated target.
    """

    try:
        return fs.get_feature_group(
            name="aqi_serving_features",
            version=1,
        )
    except Exception:
        return fs.create_feature_group(
            name="aqi_serving_features",
            version=1,
            description=(
                "Karachi AQI serving features: the exact 98 feature schema "
                "used by the production Gradient Boosting model, without a "
                "future target."
            ),
            primary_key=["id"],
            event_time="timestamp",
            online_enabled=False,
            time_travel_format="HUDI",
        )


def write_serving_feature_row(
    fs,
    feature_row: pd.DataFrame,
    current_row: pd.Series,
    timestamp: pd.Timestamp,
) -> None:
    """Write one current, unlabeled 98-feature inference row."""

    timestamp = pd.to_datetime(
        timestamp,
        utc=True,
    )

    # Deterministic integer identifier: one ID per UTC hour.
    feature_id = int(
        timestamp.timestamp() // 3600
    )

    output = feature_row.copy()

    output.insert(
        0,
        "id",
        feature_id,
    )

    output.insert(
        1,
        "timestamp",
        timestamp,
    )

    output.insert(
        2,
        "current_aqi",
        float(current_row["aqi"]),
    )

    serving_columns = [
        "id",
        "timestamp",
        "current_aqi",
    ] + FEATURE_COLUMNS

    output = output[serving_columns]

    # Avoid inserting the same deterministic hourly observation twice.
    try:
        fg = get_or_create_serving_feature_group(fs)
    except Exception:
        raise

    try:
        fg.read(
            start_time=timestamp,
            end_time=timestamp + pd.Timedelta(minutes=1),
            read_options={
                "arrow_flight_config": {
                    "timeout": HOPSWORKS_READ_TIMEOUT
                }
            },
        )
        already_exists = True
    except Exception:
        already_exists = False

    if already_exists:
        log(
            f"Serving feature row already exists for {timestamp}; "
            "skipping duplicate insert."
        )
        return

    log(
        f"Writing 98-feature serving row for {timestamp}..."
    )

    fg.insert(
        output,
        wait=True,
    )

    log(
        "98-feature serving row written successfully."
    )

# ============================================================
# CONTIGUOUS HOURLY HISTORY
# ============================================================

def check_contiguous_history(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> tuple[bool, str]:

    if history.empty:
        return False, "No historical observations are available."

    history = clean_raw_history(
        history
    )

    before = history[
        history["timestamp"] < timestamp
    ].copy()

    if len(before) < 24:
        return (
            False,
            "Fewer than 24 prior observations are available."
        )

    # We need 24 actual hourly steps immediately before the current
    # timestamp. Never fabricate missing rows.
    required = before.tail(24)

    expected = pd.date_range(
        end=timestamp - pd.Timedelta(hours=1),
        periods=24,
        freq="1h",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(
        required["timestamp"]
    )

    if not actual.equals(expected):
        return (
            False,
            "The preceding 24 hours are not contiguous. "
            "The pipeline will not fabricate missing observations."
        )

    return True, "24-hour history is contiguous."


# ============================================================
# 98-FEATURE ENGINEERING
# ============================================================

def build_98_feature_row(
    history: pd.DataFrame,
    current_row: pd.Series,
) -> pd.DataFrame:

    history = clean_raw_history(
        history
    )

    timestamp = pd.to_datetime(
        current_row["timestamp"],
        utc=True,
    )

    working = pd.concat(
        [
            history,
            pd.DataFrame(
                [{
                    "timestamp": timestamp,
                    "aqi": float(current_row["aqi"]),
                    "pm25": float(current_row["pm25"]),
                    "pm10": float(current_row["pm10"]),
                    "no2": float(current_row["no2"]),
                    "o3": float(current_row["o3"]),
                }]
            ),
        ],
        ignore_index=True,
    )

    working = clean_raw_history(
        working
    )

    current_matches = working.index[
        working["timestamp"] == timestamp
    ]

    if len(current_matches) != 1:
        raise RuntimeError(
            "Could not identify exactly one current observation."
        )

    current_index = int(
        current_matches[0]
    )

    ts = working.loc[
        current_index,
        "timestamp",
    ]

    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    day_of_month = int(ts.day)
    month = int(ts.month)
    week_of_year = int(
        ts.isocalendar().week
    )

    row = {
        "pm25": working.loc[current_index, "pm25"],
        "pm10": working.loc[current_index, "pm10"],
        "no2": working.loc[current_index, "no2"],
        "o3": working.loc[current_index, "o3"],

        "hour": hour,
        "day_of_week": dow,
        "day_of_month": day_of_month,
        "month": month,
        "week_of_year": week_of_year,

        "is_weekend": int(dow in [5, 6]),
        "is_morning": int(6 <= hour <= 11),
        "is_afternoon": int(12 <= hour <= 17),
        "is_evening": int(18 <= hour <= 23),
        "is_night": int(0 <= hour <= 5),

        "hour_sin": np.sin(
            2 * np.pi * hour / 24
        ),
        "hour_cos": np.cos(
            2 * np.pi * hour / 24
        ),

        "dow_sin": np.sin(
            2 * np.pi * dow / 7
        ),
        "dow_cos": np.cos(
            2 * np.pi * dow / 7
        ),

        "month_sin": np.sin(
            2 * np.pi * month / 12
        ),
        "month_cos": np.cos(
            2 * np.pi * month / 12
        ),
    }

    lag_hours = [
        1,
        2,
        3,
        6,
        12,
        24,
    ]

    pollutants = [
        "pm25",
        "pm10",
        "no2",
        "o3",
    ]

    # ----------------------------
    # LAGS
    # ----------------------------

    for pollutant in pollutants:

        for lag in lag_hours:

            idx = current_index - lag

            if idx < 0:
                raise RuntimeError(
                    f"Insufficient history for "
                    f"{pollutant}_lag_{lag}h."
                )

            row[
                f"{pollutant}_lag_{lag}h"
            ] = working.loc[
                idx,
                pollutant,
            ]

    for lag in lag_hours:

        idx = current_index - lag

        if idx < 0:
            raise RuntimeError(
                f"Insufficient history for aqi_lag_{lag}h."
            )

        row[
            f"aqi_lag_{lag}h"
        ] = working.loc[
            idx,
            "aqi",
        ]

    # ----------------------------
    # ROLLING FEATURES
    # ----------------------------

    for pollutant in pollutants:

        previous_values = (
            working[pollutant]
            .shift(1)
        )

        for window in [
            3,
            6,
            12,
            24,
        ]:

            start = (
                current_index
                - window
            )

            values = previous_values.iloc[
                max(0, start):
                current_index
            ]

            if len(values) != window:
                raise RuntimeError(
                    f"Insufficient history for "
                    f"{pollutant} rolling {window}h."
                )

            row[
                f"{pollutant}_rolling_mean_{window}h"
            ] = values.mean()

            row[
                f"{pollutant}_rolling_std_{window}h"
            ] = values.std()

    previous_aqi = (
        working["aqi"]
        .shift(1)
    )

    for window in [
        3,
        6,
        12,
        24,
    ]:

        values = previous_aqi.iloc[
            current_index - window:
            current_index
        ]

        if len(values) != window:
            raise RuntimeError(
                f"Insufficient history for "
                f"aqi rolling {window}h."
            )

        row[
            f"aqi_rolling_mean_{window}h"
        ] = values.mean()

        row[
            f"aqi_rolling_std_{window}h"
        ] = values.std()

    # ----------------------------
    # RATIOS
    # ----------------------------

    prev = current_index - 1
    prev2 = current_index - 2

    if prev < 0 or prev2 < 0:
        raise RuntimeError(
            "At least two previous observations are required "
            "for one-hour change features."
        )

    prev_pm25 = working.loc[
        prev,
        "pm25",
    ]

    prev_pm10 = working.loc[
        prev,
        "pm10",
    ]

    prev_no2 = working.loc[
        prev,
        "no2",
    ]

    prev_o3 = working.loc[
        prev,
        "o3",
    ]

    row["pm25_pm10_ratio"] = (
        prev_pm25 / prev_pm10
        if prev_pm10 != 0
        else np.nan
    )

    row["pm10_pm25_ratio"] = (
        prev_pm10 / prev_pm25
        if prev_pm25 != 0
        else np.nan
    )

    row["pm25_no2_ratio"] = (
        prev_pm25 / prev_no2
        if prev_no2 != 0
        else np.nan
    )

    row["o3_no2_ratio"] = (
        prev_o3 / prev_no2
        if prev_no2 != 0
        else np.nan
    )

    row["pm25_change_1h"] = (
        working.loc[prev, "pm25"]
        - working.loc[prev2, "pm25"]
    )

    row["pm10_change_1h"] = (
        working.loc[prev, "pm10"]
        - working.loc[prev2, "pm10"]
    )

    row["no2_change_1h"] = (
        working.loc[prev, "no2"]
        - working.loc[prev2, "no2"]
    )

    row["o3_change_1h"] = (
        working.loc[prev, "o3"]
        - working.loc[prev2, "o3"]
    )

    result = pd.DataFrame(
        [row]
    )

    missing = [
        c
        for c in FEATURE_COLUMNS
        if c not in result.columns
    ]

    if missing:
        raise RuntimeError(
            "98-feature construction failed. Missing: "
            + ", ".join(missing)
        )

    result = result[
        FEATURE_COLUMNS
    ]

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if result.isna().any().any():
        missing_values = result.columns[
            result.isna().any()
        ].tolist()

        raise RuntimeError(
            "98-feature row contains NaN values in: "
            + ", ".join(missing_values)
        )

    return result


# ============================================================
# MODEL FEATURE GROUP
# ============================================================


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> int:

    log("Starting hourly AQI feature pipeline.")

    require_environment()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # 1. Fetch external observation
    # ----------------------------------------

    weather, pollution = (
        fetch_openweather_observation()
    )

    current_row, weather_info = (
        normalize_current_observation(
            weather,
            pollution,
        )
    )

    current_timestamp = pd.to_datetime(
        current_row.iloc[0]["timestamp"],
        utc=True,
    )

    log(
        "Current observation: "
        f"{current_timestamp} | "
        f"AQI={current_row.iloc[0]['aqi']} | "
        f"PM2.5={current_row.iloc[0]['pm25']:.2f} | "
        f"PM10={current_row.iloc[0]['pm10']:.2f} | "
        f"NO2={current_row.iloc[0]['no2']:.2f} | "
        f"O3={current_row.iloc[0]['o3']:.2f}"
    )

    log(
        "Weather: "
        f"temp={weather_info['temperature_c']} C, "
        f"humidity={weather_info['humidity_percent']}%, "
        f"pressure={weather_info['pressure_hpa']} hPa, "
        f"wind={weather_info['wind_speed_mps']} m/s"
    )

    # ----------------------------------------
    # 2. Connect to Hopsworks
    # ----------------------------------------

    _, fs = connect_hopsworks()

    # ----------------------------------------
    # 3. Read existing raw history
    # ----------------------------------------

    try:

        hopsworks_history = (
            read_raw_history_from_hopsworks(fs)
        )

    except Exception as exc:

        log(
            "WARNING: Could not read Hopsworks raw history: "
            f"{exc}"
        )

        hopsworks_history = pd.DataFrame(
            columns=RAW_COLUMNS
        )

    local_history = (
        load_local_raw_cache()
    )

    history_before_current = merge_histories(
        hopsworks_history,
        local_history,
    )

    # ----------------------------------------
    # 4. Update local raw-observation cache.
    # ----------------------------------------
    #
    # The existing aqi_features v3 group is a labeled training dataset.
    # We do NOT insert a current observation into it because its future
    # target_aqi is not known yet.

    combined_history = merge_histories(
        history_before_current,
        current_row,
    )

    save_local_raw_cache(
        combined_history
    )

    # ----------------------------------------
    # 5. Check whether 98 features are possible
    # ----------------------------------------

    contiguous, reason = (
        check_contiguous_history(
            combined_history,
            current_timestamp,
        )
    )

    if not contiguous:

        log(
            "RAW COLLECTION SUCCESS."
        )

        log(
            "98-feature serving row not written yet: "
            + reason
        )

        log(
            "This is intentional. Missing historical observations "
            "will not be fabricated."
        )

        return 0

    # ----------------------------------------
    # 6. Build exact 98 features for the current hour
    # ----------------------------------------

    feature_row = build_98_feature_row(
        combined_history,
        current_row.iloc[0],
    )

    log(
        f"Constructed exactly {feature_row.shape[1]} model features."
    )

    # ----------------------------------------
    # 7. Store unlabeled current features
    # ----------------------------------------
    #
    # target_aqi is deliberately absent here because the next-hour AQI
    # is not known at the current timestamp. The training/backfill
    # pipeline will construct labeled rows retrospectively.

    write_serving_feature_row(
        fs,
        feature_row,
        current_row.iloc[0],
        current_timestamp,
    )

    log(
        "PIPELINE SUCCESS: raw cache updated + "
        "98-feature serving row stored."
    )

    return 0


if __name__ == "__main__":

    try:
        raise SystemExit(
            run_pipeline()
        )

    except KeyboardInterrupt:
        log("Pipeline interrupted.")
        raise SystemExit(130)

    except Exception as exc:
        log(
            f"PIPELINE FAILED: {exc}"
        )
        raise SystemExit(1)
