import os
import shutil
import joblib
import requests
import pandas as pd
import numpy as np
import hopsworks
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
import time
import shap


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌎",
    layout="wide"
)

HOPSWORKS_HOST = os.getenv(
    "HOPSWORKS_HOST",
    "eu-west.cloud.hopsworks.ai"
)

HOPSWORKS_PROJECT = os.getenv(
    "HOPSWORKS_PROJECT",
    "noismore"
)

HOPSWORKS_API_KEY = os.getenv(
    "HOPSWORKS_API_KEY"
)

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

LAT = 24.8607
LON = 67.0011

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 4

MODEL_NAME = "karachi_aqi_random_forest"
MODEL_VERSION = 7

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

FEATURE_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "feature_metadata.pkl"
)

try:
    _feature_metadata = joblib.load(
        FEATURE_METADATA_PATH
    )

    if isinstance(_feature_metadata, dict):
        FEATURE_COLUMNS = list(
            _feature_metadata["feature_columns"]
        )
    else:
        FEATURE_COLUMNS = list(
            _feature_metadata
        )

except Exception:
    FEATURE_COLUMNS = []



# ============================================================
# AQI CATEGORIES — 0–500 SCALE
# ============================================================

AQI_CATEGORY_RANGES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


# ============================================================
# PREMIUM DASHBOARD CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(14,165,233,.12), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(34,197,94,.08), transparent 25%),
            #07111f;
        color: #E5E7EB;
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.88);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1,h2,h3,h4 { color:#F8FAFC !important; letter-spacing:-.02em; }
    p,label,.stCaption { color:#CBD5E1; }

    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 22px;
        background: linear-gradient(135deg,rgba(15,23,42,.96),rgba(15,39,64,.78));
        box-shadow: 0 18px 50px rgba(0,0,0,.22);
        margin-bottom: 1.2rem;
    }

    .hero-title { font-size:2.25rem; font-weight:800; color:#F8FAFC; margin:0; }
    .hero-subtitle { color:#94A3B8; font-size:1rem; margin-top:.35rem; }

    .status-pill {
        display:inline-block;
        padding:.42rem .8rem;
        border-radius:999px;
        background:rgba(34,197,94,.12);
        border:1px solid rgba(34,197,94,.28);
        color:#86EFAC;
        font-size:.82rem;
        font-weight:700;
        margin-top:.8rem;
    }

    .glass-card {
        background:rgba(15,23,42,.76);
        border:1px solid rgba(148,163,184,.13);
        border-radius:18px;
        padding:1.2rem;
        box-shadow:0 12px 35px rgba(0,0,0,.16);
        margin-bottom:1rem;
    }

    .section-title { font-size:1.15rem; font-weight:750; color:#F8FAFC; margin-bottom:.75rem; }

    .aqi-card {
        min-height:245px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        border-radius:20px;
        padding:1.5rem;
        background:linear-gradient(145deg,rgba(30,41,59,.95),rgba(15,23,42,.82));
        border:1px solid rgba(148,163,184,.15);
    }

    .aqi-label {
        color:#94A3B8;
        font-size:.9rem;
        text-transform:uppercase;
        letter-spacing:.12em;
        font-weight:700;
    }

    .aqi-number { font-size:5rem; line-height:1; font-weight:850; margin:.45rem 0; }
    .aqi-category {
        font-size:1.05rem;
        font-weight:750;
        padding:.38rem .9rem;
        border-radius:999px;
        background:rgba(255,255,255,.07);
    }

    .metric-tile {
        background:rgba(2,6,23,.45);
        border:1px solid rgba(148,163,184,.10);
        border-radius:14px;
        padding:.9rem;
        text-align:center;
        min-height:92px;
    }

    .metric-label {
        color:#94A3B8;
        font-size:.76rem;
        text-transform:uppercase;
        letter-spacing:.07em;
    }

    .metric-value { color:#F8FAFC; font-size:1.45rem; font-weight:800; margin-top:.25rem; }

    .forecast-card {
        border-radius:16px;
        padding:1rem;
        text-align:center;
        background:rgba(2,6,23,.48);
        border:1px solid rgba(148,163,184,.12);
        min-height:150px;
    }

    .forecast-date {
        color:#94A3B8;
        font-size:.78rem;
        text-transform:uppercase;
        letter-spacing:.08em;
    }

    .forecast-value { color:#F8FAFC; font-size:2rem; font-weight:850; margin:.35rem 0; }
    .forecast-category { font-weight:700; font-size:.9rem; }

    .pollutant-card {
        background:rgba(2,6,23,.48);
        border:1px solid rgba(148,163,184,.10);
        border-radius:14px;
        padding:.9rem;
        min-height:100px;
    }

    .pollutant-name { color:#94A3B8; font-size:.78rem; font-weight:700; text-transform:uppercase; }
    .pollutant-value { color:#F8FAFC; font-size:1.45rem; font-weight:800; margin-top:.25rem; }
    .pollutant-unit { color:#64748B; font-size:.72rem; }

    .model-strip {
        border-radius:16px;
        padding:1rem;
        background:linear-gradient(90deg,rgba(14,165,233,.10),rgba(34,197,94,.07));
        border:1px solid rgba(148,163,184,.12);
    }

    .model-name { color:#F8FAFC; font-size:1.05rem; font-weight:800; }
    .model-detail { color:#94A3B8; font-size:.78rem; margin-top:.2rem; }

    [data-testid="stMetric"] { background:transparent; }
    [data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
    .stPlotlyChart { border-radius:14px; overflow:hidden; }
    div[data-testid="stAlert"] { border-radius:12px; }
    footer { visibility:hidden; }

    .stButton > button {
        border-radius:10px;
        font-weight:700;
        border:1px solid rgba(148,163,184,.18);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOCAL MODEL
# ============================================================

@st.cache_resource
def load_local_random_forest_model():

    try:

        st.info(
            "Loading local trained model: Random Forest"
        )

        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "random_forest_model.pkl"
        )

        if not os.path.exists(model_path):

            st.error(
                f"Local model not found: {model_path}"
            )

            return None

        loaded_model = joblib.load(
            model_path
        )

        # ----------------------------------------------------
        # Verify the model schema
        # ----------------------------------------------------

        model_feature_count = getattr(
            loaded_model,
            "n_features_in_",
            None
        )

        if model_feature_count is None:
            st.error(
                "Loaded model does not expose n_features_in_."
            )
            return None

        if int(model_feature_count) != 98:
            st.error(
                f"Expected 98 model features, but loaded {model_feature_count}."
            )
            return None

        model_features = list(FEATURE_COLUMNS)

        st.write(
            f"Loaded model features: {len(model_features)}"
        )

        if len(model_features) != 98:

            st.error(
                f"Expected 98 model features, "
                f"but loaded {len(model_features)}."
            )

            return None

        st.success(
            "Random Forest model loaded successfully ✅ 98 features."
        )

        return loaded_model

    except Exception as e:

        st.error(
            f"Error loading local trained model: {e}"
        )

        return None

@st.cache_resource
def load_recursive_72h_model():
    """Load the verified 163-feature recursive Random Forest artifacts."""

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir,
            "models",
            "recursive_72h",
            "recursive_random_forest_compressed.pkl",
        )
        imputer_path = os.path.join(
            base_dir,
            "models",
            "recursive_72h",
            "imputer.pkl",
        )
        metadata_path = os.path.join(
            base_dir,
            "models",
            "recursive_72h",
            "metadata.pkl",
        )

        if not all(os.path.exists(p) for p in [model_path, imputer_path, metadata_path]):
            missing = [p for p in [model_path, imputer_path, metadata_path] if not os.path.exists(p)]
            st.error("Missing recursive model artifact(s): " + ", ".join(missing))
            return None, None, None

        model = joblib.load(model_path)
        imputer = joblib.load(imputer_path)
        metadata = joblib.load(metadata_path)
        feature_columns = list(metadata.get("feature_columns", []))

        model_feature_count = getattr(model, "n_features_in_", None)
        if int(model_feature_count or -1) != 163 or len(feature_columns) != 163:
            st.error(
                f"Recursive model schema mismatch: model={model_feature_count}, metadata={len(feature_columns)}; expected 163."
            )
            return None, None, None

        return model, imputer, feature_columns

    except Exception as exc:
        st.error(f"Error loading recursive 72-hour model: {exc}")
        return None, None, None


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


def build_recursive_163_features(history, timestamp):
    """Reproduce the exact 163-feature schema used by recursive_72h.py."""

    row = {}
    timestamp = pd.Timestamp(timestamp).tz_convert("UTC")

    hour = timestamp.hour
    dow = timestamp.dayofweek
    month = timestamp.month
    doy = timestamp.dayofyear

    row["hour"] = hour
    row["day_of_week"] = dow
    row["month"] = month
    row["day_of_year"] = doy
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if len(history) < 25:
        raise ValueError("At least 25 hours of history are required.")

    current = history.iloc[-1]

    for variable in RECURSIVE_VARIABLES:
        row[variable] = float(current[variable])
        for lag in RECURSIVE_LAGS:
            row[f"{variable}_lag_{lag}h"] = float(history[variable].iloc[-lag])

    for variable in RECURSIVE_VARIABLES:
        for window in RECURSIVE_WINDOWS:
            values = history[variable].tail(window).to_numpy(dtype=float)
            row[f"{variable}_mean_{window}h"] = float(values.mean())
            row[f"{variable}_std_{window}h"] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0
            )

    for window in [6, 12, 24]:
        values = history["aqi"].tail(window).to_numpy(dtype=float)
        x = np.arange(len(values), dtype=float)
        row[f"aqi_slope_{window}h"] = float(np.polyfit(x, values, 1)[0])

    return pd.DataFrame([row])


# ============================================================
# OPENWEATHER REQUEST HELPER
# ============================================================

def _openweather_get(url, timeout_seconds=60, attempts=3):
    """
    Perform an OpenWeather request with a generous read timeout and
    a small retry/backoff policy. This is important because the
    OpenWeather air-pollution forecast endpoint can occasionally
    take longer than a normal 15-second request.
    """

    last_error = None

    for attempt in range(1, attempts + 1):

        try:

            response = requests.get(
                url,
                timeout=(10, timeout_seconds)
            )

            if response.status_code == 200:
                return response

            # Retry transient server/rate-limit errors.
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"HTTP {response.status_code}"
                )

                if attempt < attempts:
                    time.sleep(2 * attempt)
                    continue

            return response

        except requests.RequestException as e:

            last_error = e

            if attempt < attempts:
                time.sleep(2 * attempt)
                continue

    if last_error is not None:
        raise last_error

    return None


# ============================================================
# CURRENT WEATHER
# ============================================================

@st.cache_data(ttl=3600)
def fetch_current_weather():

    if not OPENWEATHER_API_KEY:

        st.error(
            "OPENWEATHER_API_KEY environment variable is not set."
        )

        return None

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}"
        f"&lon={LON}"
        "&units=metric"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    try:

        response = _openweather_get(
            url,
            timeout_seconds=45,
            attempts=2
        )

        if response is not None and response.status_code == 200:
            return response.json()

        status = (
            response.status_code
            if response is not None
            else "no response"
        )

        st.error(
            f"Weather API error: {status}"
        )

        return None

    except requests.RequestException as e:

        st.error(
            f"Weather request failed after retries: {e}"
        )

        return None


# ============================================================
# AQI FORECAST
# ============================================================

def _build_fallback_aqi_forecast(current_air_data, hours=72):
    """
    Build a last-known-pollutant fallback when the OpenWeather
    forecast endpoint is temporarily unreachable.

    The ML model still performs the AQI prediction recursively;
    only the future pollutant inputs are held at the latest
    available OpenWeather values.
    """

    if not current_air_data:
        return None

    records = current_air_data.get("list", [])

    if not records:
        return None

    latest = records[0]
    components = latest.get("components", {})

    required_components = (
        "pm2_5",
        "pm10",
        "no2",
        "o3"
    )

    if any(
        components.get(name) is None
        for name in required_components
    ):
        return None

    base_dt = int(latest.get("dt", datetime.now(timezone.utc).timestamp()))

    fallback_list = []

    for hour in range(1, hours + 1):

        fallback_list.append(
            {
                "dt": base_dt + hour * 3600,
                "components": {
                    "pm2_5": float(components["pm2_5"]),
                    "pm10": float(components["pm10"]),
                    "no2": float(components["no2"]),
                    "o3": float(components["o3"])
                }
            }
        )

    return {
        "list": fallback_list,
        "_fallback": True
    }


@st.cache_data(ttl=3600)
def fetch_aqi_forecast():

    if not OPENWEATHER_API_KEY:

        st.error(
            "OPENWEATHER_API_KEY environment variable is not set."
        )

        return None

    forecast_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
        f"?lat={LAT}"
        f"&lon={LON}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    try:

        response = _openweather_get(
            forecast_url,
            timeout_seconds=60,
            attempts=3
        )

        if response is not None and response.status_code == 200:

            data = response.json()

            if data.get("list"):
                st.success(
                    "OpenWeather AQI forecast loaded successfully."
                )
                return data

        status = (
            response.status_code
            if response is not None
            else "no response"
        )

        st.warning(
            "OpenWeather AQI forecast is temporarily unavailable "
            f"(status: {status}). Trying the current air-quality "
            "endpoint as a fallback..."
        )

    except requests.RequestException as e:

        st.warning(
            "OpenWeather AQI forecast timed out after retries. "
            "Trying the current air-quality endpoint as a fallback..."
        )

    # --------------------------------------------------------
    # FALLBACK: current air pollution snapshot
    # --------------------------------------------------------

    current_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}"
        f"&lon={LON}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    try:

        response = _openweather_get(
            current_url,
            timeout_seconds=45,
            attempts=2
        )

        if response is not None and response.status_code == 200:

            current_data = response.json()

            fallback = _build_fallback_aqi_forecast(
                current_data,
                hours=72
            )

            if fallback is not None:

                st.warning(
                    "Using the latest OpenWeather pollutant values "
                    "for future hours because the forecast endpoint "
                    "did not respond. ML AQI prediction remains active."
                )

                return fallback

        status = (
            response.status_code
            if response is not None
            else "no response"
        )

        st.error(
            "OpenWeather current air-quality fallback also failed: "
            f"{status}"
        )

        return None

    except requests.RequestException as e:

        st.error(
            "OpenWeather current air-quality fallback failed: "
            f"{e}"
        )

        return None


# ============================================================
# SELECT ONE RECORD PER DAY
# ============================================================

def select_three_daily_records(data):

    if not data:
        return []

    forecast_list = data.get(
        "list",
        []
    )

    if not forecast_list:
        return []

    daily_records = {}

    for entry in forecast_list:

        timestamp = entry.get("dt")

        if not timestamp:
            continue

        date = datetime.fromtimestamp(
            timestamp
        )

        date_key = date.date()

        if date_key not in daily_records:

            daily_records[date_key] = entry

        if len(daily_records) >= 3:
            break

    selected_records = sorted(
        daily_records.values(),
        key=lambda x: x.get("dt", 0)
    )

    return selected_records[:3]


# ============================================================
# PROCESS FORECAST DATA
# ============================================================


# ============================================================
# HISTORICAL FEATURE DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_historical_feature_data():
    """Load the verified 2-year 0–500 AQI + weather dataset."""

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "karachi_aqi_weather_2years.csv",
    )

    required = [
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

    try:
        if not os.path.exists(data_path):
            st.error(f"2-year combined dataset not found: {data_path}")
            return None

        df = pd.read_csv(data_path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error("Combined dataset is missing columns: " + ", ".join(missing))
            return None

        df = df[required].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for column in required[1:]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = (
            df.dropna(subset=required)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(df) < 25:
            st.error(f"Only {len(df)} valid historical rows are available; at least 25 are required.")
            return None

        return df

    except Exception as exc:
        st.error(f"Could not load 2-year AQI/weather dataset: {exc}")
        return None


# ============================================================
# MODEL FEATURE INFERENCE ROW
# ============================================================

def build_98_feature_row(
    history,
    timestamp,
    pm25,
    pm10,
    no2,
    o3,
    aqi,
    feature_columns
):

    working = history[
        [
            "timestamp",
            "aqi",
            "pm25",
            "pm10",
            "no2",
            "o3"
        ]
    ].copy()

    timestamp = pd.to_datetime(
        timestamp,
        utc=True
    )

    new_row = pd.DataFrame(
        [{
            "timestamp": timestamp,
            "aqi": float(aqi),
            "pm25": float(pm25),
            "pm10": float(pm10),
            "no2": float(no2),
            "o3": float(o3)
        }]
    )

    working = pd.concat(
        [working, new_row],
        ignore_index=True
    )

    working = (
        working
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    current_index = working.index[
        working["timestamp"] == timestamp
    ][-1]

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    ts = working.loc[
        current_index,
        "timestamp"
    ]

    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    day_of_month = int(ts.day)
    month = int(ts.month)
    week_of_year = int(ts.isocalendar().week)

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
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12)
    }

    lag_hours = [1, 2, 3, 6, 12, 24]
    pollutants = ["pm25", "pm10", "no2", "o3"]

    # --------------------------------------------------------
    # Past pollutant and AQI lags
    # --------------------------------------------------------

    for pollutant in pollutants:
        for lag in lag_hours:
            idx = current_index - lag
            row[f"{pollutant}_lag_{lag}h"] = (
                working.loc[idx, pollutant]
                if idx >= 0 else np.nan
            )

    for lag in lag_hours:
        idx = current_index - lag
        row[f"aqi_lag_{lag}h"] = (
            working.loc[idx, "aqi"]
            if idx >= 0 else np.nan
        )

    # --------------------------------------------------------
    # Rolling features.
    #
    # This matches featurespt.py:
    # previous_values = data[pollutant].shift(1)
    # previous_aqi = data["aqi"].shift(1)
    #
    # Therefore the rolling windows contain values BEFORE
    # the current prediction timestamp.
    # --------------------------------------------------------

    for pollutant in pollutants:
        previous_values = working[pollutant].shift(1)

        for window in [3, 6, 12, 24]:
            values = previous_values.iloc[
                max(0, current_index - window + 1):
                current_index + 1
            ]

            row[
                f"{pollutant}_rolling_mean_{window}h"
            ] = values.mean()

            row[
                f"{pollutant}_rolling_std_{window}h"
            ] = values.std()

    previous_aqi = working["aqi"].shift(1)

    for window in [3, 6, 12, 24]:
        values = previous_aqi.iloc[
            max(0, current_index - window + 1):
            current_index + 1
        ]

        row[
            f"aqi_rolling_mean_{window}h"
        ] = values.mean()

        row[
            f"aqi_rolling_std_{window}h"
        ] = values.std()

    # --------------------------------------------------------
    # Ratios and one-hour changes
    # --------------------------------------------------------

    prev = current_index - 1
    prev2 = current_index - 2

    if prev >= 0:
        prev_pm25 = working.loc[prev, "pm25"]
        prev_pm10 = working.loc[prev, "pm10"]
        prev_no2 = working.loc[prev, "no2"]
        prev_o3 = working.loc[prev, "o3"]
    else:
        prev_pm25 = prev_pm10 = prev_no2 = prev_o3 = np.nan

    row["pm25_pm10_ratio"] = (
        prev_pm25 / prev_pm10
        if pd.notna(prev_pm10) and prev_pm10 != 0
        else np.nan
    )

    row["pm10_pm25_ratio"] = (
        prev_pm10 / prev_pm25
        if pd.notna(prev_pm25) and prev_pm25 != 0
        else np.nan
    )

    row["pm25_no2_ratio"] = (
        prev_pm25 / prev_no2
        if pd.notna(prev_no2) and prev_no2 != 0
        else np.nan
    )

    row["o3_no2_ratio"] = (
        prev_o3 / prev_no2
        if pd.notna(prev_no2) and prev_no2 != 0
        else np.nan
    )

    if prev >= 0 and prev2 >= 0:
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
    else:
        row["pm25_change_1h"] = np.nan
        row["pm10_change_1h"] = np.nan
        row["no2_change_1h"] = np.nan
        row["o3_change_1h"] = np.nan

    result = pd.DataFrame([row])

    # Exact trained schema and order.
    missing = [
        c for c in feature_columns
        if c not in result.columns
    ]

    if missing:
        raise ValueError(
            "Could not construct all 98 model features. "
            f"Missing: {missing}"
        )

    result = result[feature_columns]

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result


# ============================================================
# BUILD 72-HOUR RECURSIVE FORECAST USING THE 163-FEATURE MODEL
# ============================================================

def process_recursive_72h_forecast(
    forecast_data,
    historical_data,
    current_weather,
    model,
    imputer,
    feature_columns,
):
    """Generate 72 recursive hourly predictions and aggregate them into 3 days."""

    if historical_data is None or model is None or imputer is None:
        return None

    history = historical_data.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    history = (
        history.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Start from the most recent local 0–500 AQI observation.
    latest = history.iloc[-1].copy()
    latest_timestamp = pd.Timestamp(latest["timestamp"]).tz_convert("UTC")

    # Use current OpenWeather pollutants when available, while keeping
    # the 0–500 AQI from the historical dataset as the recursive starting value.
    if forecast_data and forecast_data.get("list"):
        first_api = sorted(forecast_data["list"], key=lambda x: x.get("dt", 0))[0]
        components = first_api.get("components", {})
    else:
        components = {}

    latest_aqi_500 = float(latest["aqi"])

    current_values = {
        "timestamp": latest_timestamp,
        "aqi": latest_aqi_500,
        "pm25": float(components.get("pm2_5", latest["pm25"])),
        "pm10": float(components.get("pm10", latest["pm10"])),
        "no2": float(components.get("no2", latest["no2"])),
        "o3": float(components.get("o3", latest["o3"])),
        "temperature_2m": float(
            current_weather.get("main", {}).get("temp", latest["temperature_2m"])
            if current_weather else latest["temperature_2m"]
        ),
        "relative_humidity_2m": float(
            current_weather.get("main", {}).get("humidity", latest["relative_humidity_2m"])
            if current_weather else latest["relative_humidity_2m"]
        ),
        "surface_pressure": float(
            current_weather.get("main", {}).get("pressure", latest["surface_pressure"])
            if current_weather else latest["surface_pressure"]
        ),
        "wind_speed_10m": float(
            current_weather.get("wind", {}).get("speed", latest["wind_speed_10m"])
            if current_weather else latest["wind_speed_10m"]
        ) * 3.6,
        "cloud_cover": float(
            current_weather.get("clouds", {}).get("all", latest["cloud_cover"])
            if current_weather else latest["cloud_cover"]
        ),
    }

    # Replace the same timestamp with the freshest current inputs.
    history = history[history["timestamp"] != latest_timestamp].copy()
    history = pd.concat([history, pd.DataFrame([current_values])], ignore_index=True)
    history = history.sort_values("timestamp").reset_index(drop=True)

    # Normalize OpenWeather forecast timestamps and keep the first 72 future hours.
    api_forecasts = {}
    for entry in (forecast_data or {}).get("list", []):
        if not entry.get("dt"):
            continue
        ts = pd.to_datetime(entry["dt"], unit="s", utc=True).floor("h")
        if ts > latest_timestamp:
            api_forecasts[ts] = entry

    predictions = []
    feature_rows = []
    hourly_map = {}

    previous_aqi = float(history["aqi"].iloc[-1])

    # This mirrors the validated recursive experiment: future weather is
    # held at the latest available weather input unless a separate weather
    # forecast source is supplied.
    held_weather = {
        "temperature_2m": float(history["temperature_2m"].iloc[-1]),
        "relative_humidity_2m": float(history["relative_humidity_2m"].iloc[-1]),
        "surface_pressure": float(history["surface_pressure"].iloc[-1]),
        "wind_speed_10m": float(history["wind_speed_10m"].iloc[-1]),
        "cloud_cover": float(history["cloud_cover"].iloc[-1]),
    }

    for step in range(72):
        target_ts = pd.Timestamp(history["timestamp"].iloc[-1]).tz_convert("UTC") + pd.Timedelta(hours=1)
        api_item = api_forecasts.get(target_ts)
        components = api_item.get("components", {}) if api_item else {}

        future_row = {
            "timestamp": target_ts,
            "aqi": previous_aqi,
            "pm25": float(components.get("pm2_5", history["pm25"].iloc[-1])),
            "pm10": float(components.get("pm10", history["pm10"].iloc[-1])),
            "no2": float(components.get("no2", history["no2"].iloc[-1])),
            "o3": float(components.get("o3", history["o3"].iloc[-1])),
            **held_weather,
        }

        # First build features from the history available immediately before
        # the next target step, exactly as in the validated recursive model.
        X = build_recursive_163_features(
            history,
            target_ts,
        )
        X = X[feature_columns].replace([np.inf, -np.inf], np.nan)
        X_i = imputer.transform(X)

        prediction = float(
            np.clip(
                model.predict(X_i)[0],
                0,
                500,
            )
        )

        hourly_map[target_ts] = prediction
        predictions.append(prediction)
        feature_rows.append(X)

        future_row["aqi"] = prediction
        history = pd.concat(
            [history, pd.DataFrame([future_row])],
            ignore_index=True,
        )
        previous_aqi = prediction

    hourly_predictions = np.asarray(predictions, dtype=float)

    day_predictions = []
    day_dates = []
    day_min = []
    day_max = []

    first_ts = min(hourly_map.keys())

    for day_index in range(3):
        start = first_ts + pd.Timedelta(hours=24 * day_index)
        end = start + pd.Timedelta(hours=24)
        values = [
            v for ts, v in hourly_map.items()
            if start <= ts < end
        ]
        if len(values) != 24:
            return None
        day_predictions.append(float(np.mean(values)))
        day_dates.append(start.strftime("%Y-%m-%d"))
        day_min.append(float(np.min(values)))
        day_max.append(float(np.max(values)))

    feature_df = pd.concat(feature_rows, ignore_index=True)

    # First recursive feature row is used for SHAP explanation.
    shap_features = feature_df.iloc[[0]].copy()

    return {
        "hourly_predictions": hourly_map,
        "daily_predictions": np.asarray(day_predictions, dtype=float),
        "forecast_dates": day_dates,
        "daily_min": day_min,
        "daily_max": day_max,
        "shap_features": shap_features,
        "first_forecast_timestamp": first_ts,
    }


# ============================================================
# MODEL VALIDATION / AQI HELPERS
# ============================================================

def get_aqi_category(aqi):
    try:
        value = float(aqi)
    except (TypeError, ValueError):
        return "Unknown"

    if value < 0:
        return "Unknown"

    for low, high, name, _ in AQI_CATEGORY_RANGES:
        if low <= value <= high:
            return name

    return "Hazardous"


def get_aqi_color(aqi):
    try:
        value = float(aqi)
    except (TypeError, ValueError):
        return "#94A3B8"

    for low, high, _, color in AQI_CATEGORY_RANGES:
        if low <= value <= high:
            return color

    return "#7e0023"


def create_aqi_chart(predictions, dates):
    values = [float(v) for v in predictions]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=values,
            marker_color=[get_aqi_color(v) for v in values],
            text=[f"{v:.2f}" for v in values],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Predicted AQI: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        height=400,
        margin=dict(l=50, r=20, t=20, b=50),
        yaxis=dict(range=[0, 500], dtick=50, title="AQI (0–500)"),
        xaxis=dict(type="category", title=None),
        showlegend=False,
    )

    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ============================================================
# POLLUTANT CHART
# ============================================================

def create_pollutant_chart(
    input_data
):

    if input_data is None:
        return None

    pollutants = (
        input_data[
            [
                "pm25",
                "pm10",
                "no2",
                "o3"
            ]
        ]
        .iloc[0]
        .to_dict()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(

            x=list(
                pollutants.keys()
            ),

            y=list(
                pollutants.values()
            ),

            text=[
                f"{value:.2f}"
                for value in pollutants.values()
            ],

            textposition="auto"
        )
    )

    fig.update_layout(

        

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#e5e7eb"
        ),

        height=350
    )

    return fig


# ============================================================
# MAIN APP
# ============================================================

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_metric_tile(label, value):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pollutant_card(name, value):
    number = _safe_float(value)
    text = "?" if number is None else f"{number:.2f}"
    st.markdown(
        f"""
        <div class="pollutant-card">
            <div class="pollutant-name">{name}</div>
            <div class="pollutant-value">{text}</div>
            <div class="pollutant-unit">μg/m³</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_shap_explanation(model, feature_row, feature_columns, max_features=12):
    """
    Create a SHAP explanation for one Random Forest prediction.
    Returns a DataFrame containing the most influential features.
    """

    try:

        if model is None or feature_row is None:
            return None

        X_explain = pd.DataFrame(
            feature_row,
            columns=feature_columns
        )

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            X_explain
        )

        values = np.asarray(shap_values)

        if values.ndim == 2:
            values = values[0]

        values = values.astype(float).flatten()

        if len(values) != len(feature_columns):
            return None

        explanation = pd.DataFrame({
            "feature": feature_columns,
            "shap_value": values,
            "impact": np.abs(values)
        })

        explanation = (
            explanation
            .sort_values(
                "impact",
                ascending=False
            )
            .head(max_features)
            .sort_values(
                "shap_value"
            )
        )

        return explanation

    except Exception as e:

        st.warning(
            f"SHAP explanation unavailable: {e}"
        )

        return None


def main():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🌎 Karachi Air Quality</div>
            <div class="hero-subtitle">
                Machine-learning powered 72-hour AQI forecasting for Karachi, Pakistan
            </div>
            <div class="hero-status">🤖 72-HOUR ML SYSTEM OPERATIONAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Preparing the latest Karachi air-quality forecast..."):
        model, imputer, model_features = load_recursive_72h_model()
        current_weather = fetch_current_weather()
        forecast_data = fetch_aqi_forecast()
        historical_data = fetch_historical_feature_data()

    if model is None or imputer is None or model_features is None:
        st.error("The 72-hour recursive model could not be loaded.")
        st.stop()

    if forecast_data is None:
        st.error("Current AQI forecast data is unavailable.")
        st.stop()

    if historical_data is None:
        st.error("The 2-year AQI + weather history is unavailable.")
        st.stop()

    forecast_result = process_recursive_72h_forecast(
        forecast_data,
        historical_data,
        current_weather,
        model,
        imputer,
        model_features,
    )

    if forecast_result is None:
        st.error("The 72-hour AQI forecast could not be generated.")
        st.stop()

    if forecast_data.get("_fallback", False):
        st.warning(
            "OpenWeather's forecast endpoint was unavailable. "
            "The latest pollutant values are being held constant for future hours."
        )

    predictions = forecast_result["daily_predictions"]
    forecast_dates = forecast_result["forecast_dates"]
    daily_min = forecast_result["daily_min"]
    daily_max = forecast_result["daily_max"]
    shap_features = forecast_result["shap_features"]

    # Current observed values come from the latest verified 0–500 dataset row.
    latest = historical_data.iloc[-1]
    current_aqi = _safe_float(latest.get("aqi"))
    current_pm25 = _safe_float(latest.get("pm25"))
    current_pm10 = _safe_float(latest.get("pm10"))
    current_no2 = _safe_float(latest.get("no2"))
    current_o3 = _safe_float(latest.get("o3"))
    last_timestamp = pd.Timestamp(latest["timestamp"]).tz_convert("UTC")

    current_category = get_aqi_category(current_aqi)
    current_color = get_aqi_color(current_aqi)

    # Validation metrics are the verified held-out recursive evaluation.
    metadata_path = os.path.join(
        MODEL_DIR,
        "recursive_72h",
        "metadata.pkl",
    )

    validation = {}
    try:
        validation = joblib.load(metadata_path).get("day_results", {})
    except Exception:
        validation = {}

    # Hazardous alerts on the 0–500 scale.
    if current_aqi is not None and current_aqi >= 301:
        st.error(
            "🚨 Hazardous AQI Alert: Air quality is hazardous. "
            "Avoid prolonged outdoor exposure and consider protective measures."
        )
    elif current_aqi is not None and current_aqi >= 201:
        st.warning(
            "⚠️ Very Unhealthy AQI Alert: Air quality is very unhealthy. "
            "Sensitive groups should reduce outdoor exposure."
        )
    elif current_aqi is not None and current_aqi >= 151:
        st.warning(
            "⚠️ Unhealthy AQI Alert: Consider reducing prolonged outdoor exposure."
        )

    # --------------------------------------------------------
    # Current AQI + weather
    # --------------------------------------------------------
    left, right = st.columns([0.9, 1.35], gap="large")

    with left:
        aqi_value = f"{current_aqi:.0f}" if current_aqi is not None else "?"
        st.markdown(
            f"""
            <div class="aqi-card">
                <div class="aqi-label">Current AQI</div>
                <div class="aqi-number" style="color:{current_color};">{aqi_value}</div>
                <div class="aqi-category" style="color:{current_color};">{current_category}</div>
                <div style="color:#64748B;font-size:.76rem;margin-top:.8rem;">
                    Latest observed: {last_timestamp.strftime("%d %b %Y, %H:%M UTC")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="glass-card">🌎 Karachi — Current Conditions</div>',
            unsafe_allow_html=True,
        )
        weather_cols = st.columns(4)

        if current_weather:
            temp = current_weather.get("main", {}).get("temp")
            humidity = current_weather.get("main", {}).get("humidity")
            feels_like = current_weather.get("main", {}).get("feels_like")
            description = (
                current_weather.get("weather", [{}])[0]
                .get("description", "Unavailable")
                .capitalize()
            )

            with weather_cols[0]:
                render_metric_tile(
                    "Temperature",
                    f"{float(temp):.1f}°C" if _safe_float(temp) is not None else "?",
                )
            with weather_cols[1]:
                render_metric_tile(
                    "Feels Like",
                    f"{float(feels_like):.1f}°C" if _safe_float(feels_like) is not None else "?",
                )
            with weather_cols[2]:
                render_metric_tile(
                    "Humidity",
                    f"{humidity}%" if humidity is not None else "?",
                )
            with weather_cols[3]:
                render_metric_tile("Conditions", description)
        else:
            st.info("Weather information is temporarily unavailable.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 3-day forecast
    # --------------------------------------------------------
    st.markdown(
        '<div class="glass-card"><div class="section-title">📅 3-Day AQI Forecast</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Each day represents the mean predicted AQI across its 24-hour recursive forecast block."
    )

    forecast_cols = st.columns(3, gap="medium")
    for i, col in enumerate(forecast_cols):
        value = float(predictions[i])
        category = get_aqi_category(value)
        color = get_aqi_color(value)

        with col:
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-date">DAY {i + 1} · {forecast_dates[i]}</div>
                    <div class="forecast-value" style="color:{color};">{value:.2f}</div>
                    <div class="forecast-category" style="color:{color};">{category}</div>
                    <div style="color:#64748B;font-size:.72rem;margin-top:.45rem;">
                        Min {daily_min[i]:.1f} · Max {daily_max[i]:.1f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart = create_aqi_chart(predictions, forecast_dates)
    st.plotly_chart(
        chart,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Validation performance
    # --------------------------------------------------------
    st.markdown(
        '<div class="glass-card"><div class="section-title">📈 Model Validation Performance</div>',
        unsafe_allow_html=True,
    )

    validation_cols = st.columns(3, gap="medium")
    for i, key in enumerate(["Day 1", "Day 2", "Day 3"]):
        metrics = validation.get(key, {})
        r2 = float(metrics.get("r2", 0.0))
        with validation_cols[i]:
            status = "PASS · R² > 0.70" if r2 > 0.70 else "Below target"
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-date">{key.upper()} VALIDATION R²</div>
                    <div class="forecast-value">{r2:.4f}</div>
                    <div class="forecast-category">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Validation R² values come from the held-out historical recursive 72-hour evaluation; they are model-validation metrics, not live-future R² values."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------
    st.markdown(
        '<div class="glass-card"><div class="section-title">🔍 Why This Prediction?</div>',
        unsafe_allow_html=True,
    )

    shap_explanation = create_shap_explanation(
        model,
        shap_features,
        model_features,
        max_features=12,
    )

    if shap_explanation is not None and not shap_explanation.empty:
        st.markdown(
            '<div style="color:#94A3B8;font-size:.85rem;margin-bottom:.8rem;">SHAP shows how the most influential features contributed to the first forecasted hourly AQI value.</div>',
            unsafe_allow_html=True,
        )

        shap_chart = go.Figure()
        shap_chart.add_trace(
            go.Bar(
                x=shap_explanation["shap_value"],
                y=shap_explanation["feature"],
                orientation="h",
                hovertemplate=(
                    "<b>%{y}</b><br>SHAP impact: %{x:.4f}<extra></extra>"
                ),
            )
        )
        shap_chart.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="SHAP Value",
            yaxis_title=None,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
        )
        shap_chart.update_xaxes(
            zeroline=True,
            zerolinecolor="rgba(148,163,184,.25)",
            gridcolor="rgba(148,163,184,.10)",
        )
        shap_chart.update_yaxes(
            gridcolor="rgba(148,163,184,.06)"
        )
        st.plotly_chart(
            shap_chart,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.caption(
            "Positive SHAP values push the prediction higher; negative values push it lower."
        )
    else:
        st.info("SHAP explanation is temporarily unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Pollutants
    # --------------------------------------------------------
    st.markdown(
        '<div class="glass-card"><div class="section-title">🧪 Current Pollutant Levels</div>',
        unsafe_allow_html=True,
    )

    pollutant_cols = st.columns(4, gap="medium")
    with pollutant_cols[0]:
        render_pollutant_card("PM2.5", current_pm25)
    with pollutant_cols[1]:
        render_pollutant_card("PM10", current_pm10)
    with pollutant_cols[2]:
        render_pollutant_card("NO2", current_no2)
    with pollutant_cols[3]:
        render_pollutant_card("O3", current_o3)

    pollutant_chart = create_pollutant_chart(
        pd.DataFrame(
            [{
                "pm25": current_pm25,
                "pm10": current_pm10,
                "no2": current_no2,
                "o3": current_o3,
            }]
        )
    )
    if pollutant_chart is not None:
        pollutant_chart.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False,
        )
        pollutant_chart.update_yaxes(
            gridcolor="rgba(148,163,184,.10)",
            zeroline=False,
        )
        pollutant_chart.update_xaxes(showgrid=False)
        st.plotly_chart(
            pollutant_chart,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Forecast table
    # --------------------------------------------------------
    st.markdown(
        '<div class="glass-card"><div class="section-title">📊 Forecast Details</div>',
        unsafe_allow_html=True,
    )

    forecast_table = pd.DataFrame(
        {
            "Horizon": [
                "Day 1 · 0–24h",
                "Day 2 · 24–48h",
                "Day 3 · 48–72h",
            ],
            "Date": forecast_dates,
            "Predicted Mean AQI": [round(float(v), 2) for v in predictions],
            "Minimum AQI": [round(float(v), 2) for v in daily_min],
            "Maximum AQI": [round(float(v), 2) for v in daily_max],
            "Category": [get_aqi_category(v) for v in predictions],
        }
    )

    st.dataframe(
        forecast_table,
        width="stretch",
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # ML system
    # --------------------------------------------------------
    st.markdown(
        '<div class="glass-card"><div class="section-title">🤖 Machine Learning System</div>'
        '<div class="model-strip">',
        unsafe_allow_html=True,
    )

    model_cols = st.columns(5, gap="medium")
    model_items = [
        ("MODEL", "Random Forest", "Recursive 72-hour forecasting"),
        ("FEATURES", "163", "Verified training schema"),
        ("DATA", "2+ Years", "Hourly AQI + weather"),
        ("FORECAST", "72 Hours", "3 × 24-hour blocks"),
        (
            "R² VALIDATION",
            f"{validation.get('Day 1', {}).get('r2', 0):.3f} / {validation.get('Day 2', {}).get('r2', 0):.3f} / {validation.get('Day 3', {}).get('r2', 0):.3f}",
            "Day 1 / Day 2 / Day 3",
        ),
    ]

    for col, (label, value, detail) in zip(model_cols, model_items):
        with col:
            st.markdown(
                f"""
                <div>
                    <div class="metric-label">{label}</div>
                    <div class="model-name">{value}</div>
                    <div class="model-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 .5rem;color:#64748B;font-size:.78rem;">
            Karachi AQI Forecast • OpenWeather • Open-Meteo historical weather • 2+ year dataset • 163-Feature Recursive Random Forest
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
