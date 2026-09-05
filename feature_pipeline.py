"""
Production AQI feature pipeline for Karachi.

Final production alignment
--------------------------
1. Fetch live weather + pollutant observations from OpenWeather.
2. Maintain a local live observation cache.
3. Recalculate the same modeling-oriented 0-500 AQI definition used by
   calculate_aqi_0_500.py whenever enough continuous history exists.
4. Combine the verified 2-year AQI/weather dataset with live observations.
5. Build the exact 163-feature schema used by the final Recursive
   Random Forest model.
6. Save the latest 163-feature inference row locally.
7. Persist the raw live observation to Hopsworks on a best-effort basis.
   A Hopsworks upload/materialization failure must NOT fail the GitHub job.
8. Never fabricate missing hourly observations. Until a genuine 24-hour
   contiguous live window exists, raw collection succeeds but feature
   generation is deferred.

Important:
- OpenWeather main.aqi is a 1-5 index and is stored only as
  openweather_aqi/raw metadata.
- The final model uses the custom 0-500 AQI definition from
  calculate_aqi_0_500.py.
- The final model uses exactly 163 features.
- Weather is part of the final model feature schema.
"""

from __future__ import annotations

import os
from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

VERIFIED_DATASET_PATH = DATA_DIR / "karachi_aqi_weather_2years.csv"
RAW_HISTORY_PATH = DATA_DIR / "historical_aqi_2years_raw.csv"
LIVE_CACHE_PATH = DATA_DIR / "live_aqi_weather.csv"
LATEST_FEATURE_ROW_PATH = DATA_DIR / "latest_recursive_163_features.csv"

MODEL_METADATA_PATH = (
    BASE_DIR / "models" / "recursive_72h" / "metadata.pkl"
)

HOPSWORKS_HOST = os.getenv(
    "HOPSWORKS_HOST",
    "eu-west.cloud.hopsworks.ai",
)
HOPSWORKS_PROJECT = os.getenv(
    "HOPSWORKS_PROJECT",
    "noismore",
)
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

RAW_FEATURE_GROUP_NAME = os.getenv(
    "RAW_FEATURE_GROUP_NAME",
    "aqi_raw_observations",
)
RAW_FEATURE_GROUP_VERSION = int(
    os.getenv("RAW_FEATURE_GROUP_VERSION", "1")
)

REQUEST_TIMEOUT = 30
HOPSWORKS_READ_TIMEOUT = 300


# ============================================================
# FINAL 163-FEATURE SCHEMA
# ============================================================

RECURSIVE_LAGS = [1, 2, 3, 6, 12, 24]
RECURSIVE_WINDOWS = [3, 6, 12, 24]

RECURSIVE_VARIABLES = [
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "o3",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
]

try:
    import joblib

    _metadata = joblib.load(MODEL_METADATA_PATH)
    FEATURE_COLUMNS = list(_metadata["feature_columns"])
except Exception as exc:
    raise RuntimeError(
        "Could not load final recursive model metadata from "
        f"{MODEL_METADATA_PATH}: {exc}"
    ) from exc

if len(FEATURE_COLUMNS) != 163:
    raise RuntimeError(
        f"Final model metadata must contain exactly 163 features; "
        f"found {len(FEATURE_COLUMNS)}."
    )


# ============================================================
# AQI BREAKPOINTS — same definition as calculate_aqi_0_500.py
# ============================================================

PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

O3_BREAKPOINTS = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 50.4, 301, 500),
]

NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 2049, 301, 500),
]

SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
]


def calculate_subindex(concentration, breakpoints):
    if concentration is None or pd.isna(concentration):
        return np.nan

    concentration = float(concentration)
    if concentration < 0:
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            if c_high == c_low:
                return float(i_high)

            value = (
                (i_high - i_low)
                / (c_high - c_low)
            ) * (concentration - c_low) + i_low

            return float(value)

    if concentration > breakpoints[-1][1]:
        return 500.0

    return np.nan


def ug_m3_to_ppm(value, molecular_weight):
    return float(value) * 24.45 / (molecular_weight * 1000.0)


def ug_m3_to_ppb(value, molecular_weight):
    return float(value) * 24.45 / molecular_weight


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(f"[AQI PIPELINE] {message}", flush=True)


def require_environment() -> None:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "Missing environment variable: OPENWEATHER_API_KEY"
        )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("Dataframe must contain timestamp.")

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        utc=True,
        errors="coerce",
    ).dt.floor("h")

    for column in out.columns:
        if column != "timestamp":
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    out = (
        out.dropna(subset=["timestamp"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return out


# ============================================================
# OPENWEATHER
# ============================================================

def fetch_openweather_observation() -> tuple[dict, dict]:
    weather_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&units=metric"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    pollution_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    log("Fetching current weather from OpenWeather...")
    weather_response = requests.get(
        weather_url,
        timeout=REQUEST_TIMEOUT,
    )
    weather_response.raise_for_status()

    log("Fetching current pollutant data from OpenWeather...")
    pollution_response = requests.get(
        pollution_url,
        timeout=REQUEST_TIMEOUT,
    )
    pollution_response.raise_for_status()

    weather = weather_response.json()
    pollution = pollution_response.json()

    if not pollution.get("list"):
        raise RuntimeError(
            "OpenWeather returned no current air-pollution record."
        )

    return weather, pollution


def normalize_current_observation(
    weather: dict,
    pollution: dict,
) -> pd.DataFrame:

    entry = pollution["list"][0]
    components = entry.get("components", {})
    main = entry.get("main", {})

    timestamp = pd.to_datetime(
        entry.get("dt"),
        unit="s",
        utc=True,
    ).floor("h")

    required = {
        "openweather_aqi": main.get("aqi"),
        "pm25": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "nh3": components.get("nh3"),
    }

    missing = [
        name for name, value in required.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            "OpenWeather current pollution response is missing: "
            + ", ".join(missing)
        )

    return pd.DataFrame([{
        "timestamp": timestamp,
        "openweather_aqi": float(required["openweather_aqi"]),
        "pm25": float(required["pm25"]),
        "pm10": float(required["pm10"]),
        "no2": float(required["no2"]),
        "o3": float(required["o3"]),
        "so2": float(required["so2"]),
        "co": float(required["co"]),
        "nh3": float(required["nh3"]),
        "temperature_2m": float(
            weather.get("main", {}).get("temp")
        ),
        "relative_humidity_2m": float(
            weather.get("main", {}).get("humidity")
        ),
        "surface_pressure": float(
            weather.get("main", {}).get("pressure")
        ),
        "wind_speed_10m": float(
            weather.get("wind", {}).get("speed")
        ) * 3.6,
        "cloud_cover": float(
            weather.get("clouds", {}).get("all")
        ),
    }])


# ============================================================
# LOCAL LIVE CACHE
# ============================================================

LIVE_COLUMNS = [
    "timestamp",
    "openweather_aqi",
    "pm25",
    "pm10",
    "no2",
    "o3",
    "so2",
    "co",
    "nh3",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
]


def load_live_cache() -> pd.DataFrame:
    if not LIVE_CACHE_PATH.exists():
        return pd.DataFrame(columns=LIVE_COLUMNS)

    df = pd.read_csv(LIVE_CACHE_PATH)

    missing = [
        c for c in LIVE_COLUMNS
        if c not in df.columns
    ]

    if missing:
        log(
            "WARNING: Existing live cache has an older schema; "
            f"rebuilding it. Missing: {', '.join(missing)}"
        )
        return pd.DataFrame(columns=LIVE_COLUMNS)

    return clean_dataframe(df[LIVE_COLUMNS])


def save_live_cache(df: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    clean = clean_dataframe(df)

    clean.to_csv(
        LIVE_CACHE_PATH,
        index=False,
    )

    log(f"Local live cache saved: {len(clean)} rows.")
    return clean


# ============================================================
# EXACT 0-500 AQI CALCULATION FOR COMBINED RAW HISTORY
# ============================================================

def calculate_0_500_aqi_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the same AQI construction used by
    calculate_aqi_0_500.py.

    Rows without a complete required averaging window remain NaN.
    Missing hours are never fabricated.
    """

    required = [
        "timestamp",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
    ]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            "Raw AQI history is missing: "
            + ", ".join(missing)
        )

    df = clean_dataframe(raw[required])

    df["pm25_24h"] = (
        df["pm25"]
        .rolling(window=24, min_periods=24)
        .mean()
    )

    df["pm10_24h"] = (
        df["pm10"]
        .rolling(window=24, min_periods=24)
        .mean()
    )

    df["o3_8h"] = (
        df["o3"]
        .rolling(window=8, min_periods=8)
        .mean()
    )

    df["co_8h"] = (
        df["co"]
        .rolling(window=8, min_periods=8)
        .mean()
    )

    df["aqi_pm25"] = df["pm25_24h"].apply(
        lambda x: calculate_subindex(
            x,
            PM25_BREAKPOINTS,
        )
    )

    df["aqi_pm10"] = df["pm10_24h"].apply(
        lambda x: calculate_subindex(
            x,
            PM10_BREAKPOINTS,
        )
    )

    df["aqi_o3"] = (
        df["o3_8h"]
        .apply(
            lambda x: (
                calculate_subindex(
                    ug_m3_to_ppm(x, 48.00),
                    O3_BREAKPOINTS,
                )
                if pd.notna(x)
                else np.nan
            )
        )
    )

    df["aqi_co"] = (
        df["co_8h"]
        .apply(
            lambda x: (
                calculate_subindex(
                    ug_m3_to_ppm(x, 28.01),
                    CO_BREAKPOINTS,
                )
                if pd.notna(x)
                else np.nan
            )
        )
    )

    df["aqi_no2"] = (
        df["no2"]
        .apply(
            lambda x: calculate_subindex(
                ug_m3_to_ppb(x, 46.0055),
                NO2_BREAKPOINTS,
            )
        )
    )

    df["aqi_so2"] = (
        df["so2"]
        .apply(
            lambda x: calculate_subindex(
                ug_m3_to_ppb(x, 64.066),
                SO2_BREAKPOINTS,
            )
        )
    )

    aqi_columns = [
        "aqi_pm25",
        "aqi_pm10",
        "aqi_o3",
        "aqi_co",
        "aqi_no2",
        "aqi_so2",
    ]

    df["aqi"] = (
        df[aqi_columns]
        .max(axis=1, skipna=True)
        .clip(lower=0, upper=500)
        .round()
    )

    return df[
        ["timestamp", "aqi"]
    ].copy()


# ============================================================
# HISTORY + LIVE COMBINATION
# ============================================================

def build_raw_history_for_aqi(
    live_cache: pd.DataFrame,
) -> pd.DataFrame:

    if not RAW_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Verified raw historical dataset not found: "
            f"{RAW_HISTORY_PATH}"
        )

    historical = pd.read_csv(RAW_HISTORY_PATH)

    required = [
        "timestamp",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
        "nh3",
    ]

    missing = [
        c for c in required
        if c not in historical.columns
    ]

    if missing:
        raise ValueError(
            "Verified raw historical dataset is missing: "
            + ", ".join(missing)
        )

    historical = historical[required].copy()
    historical = clean_dataframe(historical)

    live_raw = live_cache[
        [
            "timestamp",
            "pm25",
            "pm10",
            "no2",
            "o3",
            "so2",
            "co",
            "nh3",
        ]
    ].copy()

    merged = pd.concat(
        [historical, live_raw],
        ignore_index=True,
    )

    return clean_dataframe(merged)


def build_model_history(
    live_cache: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the feature-engineering history from:
      - verified 2-year AQI + weather data
      - live observations collected by this pipeline

    The verified historical dataset already contains the final 0-500 AQI.
    Live AQI is recalculated using the same formula.
    """

    historical_path = VERIFIED_DATASET_PATH

    if not historical_path.exists():
        raise FileNotFoundError(
            f"Verified combined dataset not found: "
            f"{historical_path}"
        )

    historical = pd.read_csv(historical_path)

    required_historical = [
        "timestamp",
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "cloud_cover",
    ]

    missing = [
        c for c in required_historical
        if c not in historical.columns
    ]

    if missing:
        raise ValueError(
            "Verified combined dataset is missing: "
            + ", ".join(missing)
        )

    historical = historical[required_historical].copy()
    historical = clean_dataframe(historical)

    # Recalculate exact 0-500 AQI for live records.
    combined_raw = build_raw_history_for_aqi(live_cache)
    live_aqi = calculate_0_500_aqi_from_raw(combined_raw)

    live_joined = live_cache[
        [
            "timestamp",
            "pm25",
            "pm10",
            "no2",
            "o3",
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
        ]
    ].merge(
        live_aqi,
        on="timestamp",
        how="left",
    )

    live_joined = live_joined[
        [
            "timestamp",
            "aqi",
            "pm25",
            "pm10",
            "no2",
            "o3",
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
        ]
    ]

    model_history = pd.concat(
        [historical, live_joined],
        ignore_index=True,
    )

    model_history = clean_dataframe(model_history)

    return model_history


# ============================================================
# CONTIGUOUS HISTORY CHECK
# ============================================================

def check_contiguous_history(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> tuple[bool, str]:

    valid = history[
        history["timestamp"] < timestamp
    ].copy()

    valid = valid.dropna(
        subset=[
            "aqi",
            "pm25",
            "pm10",
            "no2",
            "o3",
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
        ]
    )

    if len(valid) < 24:
        return (
            False,
            "Fewer than 24 complete prior observations are available.",
        )

    required = valid.tail(24)["timestamp"].sort_values()

    expected = pd.date_range(
        end=timestamp - pd.Timedelta(hours=1),
        periods=24,
        freq="1h",
        tz="UTC",
    )

    if (
        len(required) != 24
        or not (
            pd.DatetimeIndex(required).to_numpy()
            == pd.DatetimeIndex(expected).to_numpy()
        ).all()
    ):
        return (
            False,
            "The preceding 24 hours are not complete and contiguous. "
            "The pipeline will not fabricate missing observations.",
        )

    return True, "24-hour history is contiguous."


# ============================================================
# EXACT 163-FEATURE ENGINEERING
# ============================================================

def build_recursive_163_features(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:

    history = history.copy()
    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        utc=True,
    )

    history = (
        history.drop_duplicates(
            "timestamp",
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    timestamp = pd.Timestamp(timestamp).tz_convert("UTC")

    if len(history) < 25:
        raise ValueError(
            "At least 25 hours of history are required."
        )

    current = history.iloc[-1]

    # The trained schema starts with 10 calendar features.
    hour = timestamp.hour
    dow = timestamp.dayofweek
    month = timestamp.month
    doy = timestamp.dayofyear

    row = {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "day_of_year": doy,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }

    for variable in RECURSIVE_VARIABLES:
        row[variable] = float(current[variable])

        for lag in RECURSIVE_LAGS:
            row[f"{variable}_lag_{lag}h"] = float(
                history[variable].iloc[-lag]
            )

    for variable in RECURSIVE_VARIABLES:
        for window in RECURSIVE_WINDOWS:
            values = (
                history[variable]
                .tail(window)
                .to_numpy(dtype=float)
            )

            row[f"{variable}_mean_{window}h"] = float(
                values.mean()
            )
            row[f"{variable}_std_{window}h"] = float(
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )

    for window in [6, 12, 24]:
        values = (
            history["aqi"]
            .tail(window)
            .to_numpy(dtype=float)
        )
        x = np.arange(len(values), dtype=float)

        row[f"aqi_slope_{window}h"] = float(
            np.polyfit(x, values, 1)[0]
        )

    result = pd.DataFrame([row])

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in result.columns
    ]

    if missing:
        raise RuntimeError(
            "163-feature construction failed. Missing: "
            + ", ".join(missing)
        )

    result = result[FEATURE_COLUMNS]

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if result.isna().any().any():
        bad = result.columns[
            result.isna().any()
        ].tolist()

        raise RuntimeError(
            "163-feature row contains NaN values in: "
            + ", ".join(bad)
        )

    return result


# ============================================================
# LOCAL FEATURE OUTPUT
# ============================================================

def save_latest_feature_row(
    feature_row: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> None:

    output = feature_row.copy()
    output.insert(
        0,
        "timestamp",
        pd.Timestamp(timestamp).tz_convert("UTC"),
    )

    output.to_csv(
        LATEST_FEATURE_ROW_PATH,
        index=False,
    )

    log(
        f"Latest 163-feature row saved locally: "
        f"{LATEST_FEATURE_ROW_PATH}"
    )


# ============================================================
# HOPSWORKS — BEST EFFORT ONLY
# ============================================================

def connect_hopsworks():
    if not HOPSWORKS_API_KEY:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not configured."
        )

    log("Connecting to Hopsworks...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
    )

    fs = project.get_feature_store()

    log("Hopsworks Feature Store connection established.")

    return fs


def get_or_create_raw_feature_group(fs):
    try:
        fg = fs.get_feature_group(
            name=RAW_FEATURE_GROUP_NAME,
            version=RAW_FEATURE_GROUP_VERSION,
        )
        if fg is not None:
            return fg
    except Exception:
        pass

    log(
        f"Creating raw Feature Group "
        f"{RAW_FEATURE_GROUP_NAME} v{RAW_FEATURE_GROUP_VERSION}..."
    )

    return fs.create_feature_group(
        name=RAW_FEATURE_GROUP_NAME,
        version=RAW_FEATURE_GROUP_VERSION,
        description=(
            "Persistent Karachi hourly raw OpenWeather observations. "
            "The aqi field in this legacy raw group is the OpenWeather "
            "1-5 index; final 0-500 AQI is derived locally."
        ),
        primary_key=["id"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="DELTA",
    )


def write_raw_observation_to_hopsworks(
    fs,
    current_row: pd.DataFrame,
) -> None:
    """
    Keep the existing raw Feature Group contract so this pipeline
    does not attempt to migrate its schema while Hopsworks quota is
    constrained.
    """

    row = current_row.copy()

    row["id"] = (
        row["timestamp"]
        .map(lambda x: int(pd.Timestamp(x).timestamp() // 3600))
        .astype("int64")
    )

    output = pd.DataFrame({
        "id": row["id"].astype("int64"),
        "timestamp": pd.to_datetime(
            row["timestamp"],
            utc=True,
        ),
        "aqi": row["openweather_aqi"].astype("int64"),
        "pm25": row["pm25"].astype(float),
        "pm10": row["pm10"].astype(float),
        "no2": row["no2"].astype(float),
        "o3": row["o3"].astype(float),
    })

    fg = get_or_create_raw_feature_group(fs)

    log(
        f"Writing raw observation to "
        f"{RAW_FEATURE_GROUP_NAME} v{RAW_FEATURE_GROUP_VERSION}..."
    )

    fg.insert(
        output,
        write_options={
            "wait_for_job": False,
        },
    )

    log(
        "Hopsworks raw observation submitted successfully."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> int:

    log(
        "Starting hourly AQI feature pipeline "
        "(final 163-feature version)."
    )

    require_environment()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Live OpenWeather observation
    # --------------------------------------------------------

    weather, pollution = fetch_openweather_observation()

    current = normalize_current_observation(
        weather,
        pollution,
    )

    current_timestamp = pd.Timestamp(
        current.iloc[0]["timestamp"]
    ).tz_convert("UTC")

    log(
        "Current observation: "
        f"{current_timestamp} | "
        f"OpenWeather AQI={current.iloc[0]['openweather_aqi']:.0f} | "
        f"PM2.5={current.iloc[0]['pm25']:.2f} | "
        f"PM10={current.iloc[0]['pm10']:.2f} | "
        f"NO2={current.iloc[0]['no2']:.2f} | "
        f"O3={current.iloc[0]['o3']:.2f}"
    )

    log(
        "Weather: "
        f"temp={current.iloc[0]['temperature_2m']:.2f} C, "
        f"humidity={current.iloc[0]['relative_humidity_2m']:.0f}%, "
        f"pressure={current.iloc[0]['surface_pressure']:.0f} hPa, "
        f"wind={current.iloc[0]['wind_speed_10m'] / 3.6:.2f} m/s"
    )

    # --------------------------------------------------------
    # 2. Update local live cache FIRST
    # --------------------------------------------------------

    live_cache = load_live_cache()

    live_cache = pd.concat(
        [live_cache, current[LIVE_COLUMNS]],
        ignore_index=True,
    )

    live_cache = save_live_cache(live_cache)

    # --------------------------------------------------------
    # 3. Build final 0-500/model history
    # --------------------------------------------------------

    try:
        model_history = build_model_history(live_cache)
    except Exception as exc:
        log(
            "WARNING: Could not build model history yet: "
            f"{exc}"
        )
        model_history = None

    if model_history is not None:
        # Live 0-500 AQI, when it is genuinely calculable.
        latest_model_row = model_history[
            model_history["timestamp"] == current_timestamp
        ]

        if (
            not latest_model_row.empty
            and pd.notna(latest_model_row.iloc[0]["aqi"])
        ):
            current_aqi_500 = float(
                latest_model_row.iloc[0]["aqi"]
            )
            log(
                f"Current model-scale AQI (0-500): "
                f"{current_aqi_500:.0f}"
            )
        else:
            current_aqi_500 = None
            log(
                "Current 0-500 AQI is not yet calculable from a "
                "complete continuous averaging window."
            )

        # ----------------------------------------------------
        # 4. Build current 163-feature row only when valid
        # ----------------------------------------------------

        if current_aqi_500 is not None:
            contiguous, reason = check_contiguous_history(
                model_history,
                current_timestamp,
            )

            if contiguous:
                # Replace the latest row with the current complete live state.
                current_feature_state = model_history[
                    model_history["timestamp"] == current_timestamp
                ]

                feature_row = build_recursive_163_features(
                    current_feature_state.append(
                        # pandas append removed in modern pandas; this branch
                        # is never used because the state is already present.
                        {},
                        ignore_index=True,
                    )
                    if False else model_history[
                        model_history["timestamp"] <= current_timestamp
                    ],
                    current_timestamp,
                )

                save_latest_feature_row(
                    feature_row,
                    current_timestamp,
                )

                log(
                    f"Constructed exactly {feature_row.shape[1]} "
                    "final model features."
                )
                log(
                    "163-feature local inference row is ready."
                )
            else:
                log(
                    "163-feature serving row deferred: "
                    + reason
                )
        else:
            log(
                "163-feature serving row deferred until a complete "
                "0-500 AQI averaging window exists."
            )

    # --------------------------------------------------------
    # 5. Best-effort Hopsworks persistence
    # --------------------------------------------------------

    if HOPSWORKS_API_KEY:
        try:
            fs = connect_hopsworks()

            try:
                write_raw_observation_to_hopsworks(
                    fs,
                    current,
                )
            except Exception as exc:
                log(
                    "WARNING: Hopsworks raw write/materialization "
                    f"failed; local cache remains authoritative: {exc}"
                )

        except Exception as exc:
            log(
                "WARNING: Hopsworks unavailable; "
                f"continuing with local cache only: {exc}"
            )
    else:
        log(
            "WARNING: HOPSWORKS_API_KEY is not configured; "
            "using local cache only."
        )

    log(
        "PIPELINE SUCCESS: local live cache updated; "
        "Hopsworks persistence was best-effort."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_pipeline())
    except KeyboardInterrupt:
        log("Pipeline interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        log(f"PIPELINE FAILED: {exc}")
        raise SystemExit(1)
