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
# AQI CATEGORIES
# ============================================================

AQI_CATEGORIES = {
    1: {
        "name": "Good",
        "color": "#00e400"
    },
    2: {
        "name": "Fair",
        "color": "#ffff00"
    },
    3: {
        "name": "Moderate",
        "color": "#ff7e00"
    },
    4: {
        "name": "Poor",
        "color": "#ff0000"
    },
    5: {
        "name": "Very Poor",
        "color": "#99004c"
    }
}


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

    # ========================================================
    # LOCAL HISTORICAL CACHE
    # ========================================================
    #
    # The 7-day Hopsworks query was successfully tested from
    # PowerShell and produced 144 rows. The Streamlit app uses this verified local cache and
    # does not query the Hopsworks Arrow Flight service at runtime.
    #
    # The verified result is stored in:
    #     data/historical_aqi.csv
    #
    # Normal Streamlit startup therefore uses this local cache
    # and does not make an Arrow Flight request.
    # ========================================================

    required = [
        "timestamp",
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "o3"
    ]

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "historical_aqi.csv"
    )

    st.info(
        "Loading historical AQI data from local cache..."
    )

    try:

        if not os.path.exists(data_path):

            st.error(
                "Historical AQI cache not found: "
                + data_path
            )

            st.info(
                "Run the verified Hopsworks 7-day download "
                "command to create data/historical_aqi.csv."
            )

            return None

        df = pd.read_csv(
            data_path
        )

        # ----------------------------------------------------
        # REQUIRED COLUMN CHECK
        # ----------------------------------------------------

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            st.error(
                "Historical AQI cache is missing columns: "
                + ", ".join(missing)
            )

            return None

        # ----------------------------------------------------
        # KEEP ONLY RAW FEATURES USED BY THE FEATURE BUILDER
        # ----------------------------------------------------

        df = df[required].copy()

        # ----------------------------------------------------
        # NORMALIZE TIMESTAMP
        # ----------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce"
        )

        # ----------------------------------------------------
        # NORMALIZE NUMERIC COLUMNS
        # ----------------------------------------------------

        for column in required[1:]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # CLEAN AND SORT
        # ----------------------------------------------------

        df = (
            df
            .dropna(subset=required)
            .drop_duplicates(
                subset=["timestamp"],
                keep="last"
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # MINIMUM HISTORY CHECK
        # ----------------------------------------------------

        if len(df) < 25:

            st.error(
                f"Only {len(df)} valid historical rows are "
                "available. At least 25 are required to build "
                "the model feature row."
            )

            return None

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        st.success(
            f"Historical AQI cache loaded successfully ✅ "
            f"{len(df)} rows."
        )

        st.caption(
            f"Historical range: {df['timestamp'].min()} → "
            f"{df['timestamp'].max()}"
        )

        return df

    except Exception as e:

        st.error(
            f"Could not read local historical AQI cache: {e}"
        )

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
# BUILD 3-DAY FORECAST USING THE MODEL FEATURE SCHEMA
# ============================================================

def process_forecast_data(
    data,
    historical_data,
    model,
    feature_columns
):

    if not data or historical_data is None:
        return None, None, None

    forecast_list = data.get("list", [])

    if not forecast_list:
        return None, None, None

    # OpenWeather provides hourly pollution forecasts.
    future_records = []

    for entry in forecast_list:
        if not entry.get("dt"):
            continue

        future_records.append(entry)

    future_records = sorted(
        future_records,
        key=lambda x: x["dt"]
    )

    if not future_records:
        return None, None, None

    # --------------------------------------------------------
    # Start from the latest observed Hopsworks row.
    # --------------------------------------------------------

    history = historical_data.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        utc=True
    )

    history = (
        history
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    last_observed = history.iloc[-1].copy()

    # Only use future API rows after the latest observation.
    future_records = [
        entry
        for entry in future_records
        if pd.to_datetime(
            datetime.fromtimestamp(
                entry["dt"],
                tz=timezone.utc
            ),
            utc=True
        ) > last_observed["timestamp"]
    ]

    if not future_records:
        return None, None, None

    predictions_by_timestamp = {}
    feature_rows = []
    pollutant_rows = []

    previous_aqi = float(last_observed["aqi"])

    # --------------------------------------------------------
    # Recursive hourly forecasting.
    #
    # First row: latest observed hour -> predicts next hour.
    # Each later row uses the previous predicted AQI as the
    # current AQI, exactly as required by the AQI lag features.
    # --------------------------------------------------------

    current_timestamp = last_observed["timestamp"]

    # The first prediction should be for the first future hour.
    first_entry = future_records[0]
    first_ts = pd.to_datetime(
        datetime.fromtimestamp(
            first_entry["dt"],
            tz=timezone.utc
        ),
        utc=True
    )

    if first_ts > current_timestamp + pd.Timedelta(hours=1):
        # We cannot safely bridge a missing hourly observation.
        st.warning(
            "OpenWeather forecast does not begin immediately after "
            "the latest Hopsworks observation. Using the first "
            "available forecast hour."
        )

    # Predict the first available future hour from the latest
    # observed row.
    first_components = first_entry.get("components", {})

    first_features = build_98_feature_row(
        history,
        current_timestamp,
        last_observed["pm25"],
        last_observed["pm10"],
        last_observed["no2"],
        last_observed["o3"],
        last_observed["aqi"],
        feature_columns
    )

    first_prediction = float(
        np.clip(
            model.predict(first_features.to_numpy())[0],
            1,
            5
        )
    )

    predictions_by_timestamp[first_ts] = first_prediction

    # Append first future observation with predicted AQI.
    first_future_row = {
        "timestamp": first_ts,
        "aqi": first_prediction,
        "pm25": first_components.get("pm2_5", np.nan),
        "pm10": first_components.get("pm10", np.nan),
        "no2": first_components.get("no2", np.nan),
        "o3": first_components.get("o3", np.nan)
    }

    feature_rows.append(
        first_features.assign(
            forecast_timestamp=first_ts,
            predicted_aqi=first_prediction
        )
    )

    pollutant_rows.append(first_future_row)

    recursive_history = pd.concat(
        [
            history,
            pd.DataFrame([first_future_row])
        ],
        ignore_index=True
    )

    recursive_history = (
        recursive_history
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    previous_aqi = first_prediction

    # --------------------------------------------------------
    # Remaining hourly predictions.
    # --------------------------------------------------------

    for entry in future_records[1:]:

        timestamp = pd.to_datetime(
            datetime.fromtimestamp(
                entry["dt"],
                tz=timezone.utc
            ),
            utc=True
        )

        components = entry.get(
            "components",
            {}
        )

        pm25 = components.get("pm2_5", np.nan)
        pm10 = components.get("pm10", np.nan)
        no2 = components.get("no2", np.nan)
        o3 = components.get("o3", np.nan)

        # Use the forecast AQI from the previous hour as the
        # current AQI for the next recursive step.
        current_aqi = previous_aqi

        current_features = build_98_feature_row(
            recursive_history,
            timestamp,
            pm25,
            pm10,
            no2,
            o3,
            current_aqi,
            feature_columns
        )

        prediction = float(
            np.clip(
                model.predict(current_features.to_numpy())[0],
                1,
                5
            )
        )

        predictions_by_timestamp[timestamp] = prediction

        future_row = {
            "timestamp": timestamp,
            "aqi": prediction,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "o3": o3
        }

        feature_rows.append(
            current_features.assign(
                forecast_timestamp=timestamp,
                predicted_aqi=prediction
            )
        )

        pollutant_rows.append(future_row)

        recursive_history = pd.concat(
            [
                recursive_history,
                pd.DataFrame([future_row])
            ],
            ignore_index=True
        )

        recursive_history = (
            recursive_history
            .drop_duplicates(
                subset=["timestamp"],
                keep="last"
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        previous_aqi = prediction

    # --------------------------------------------------------
    # Select exactly one prediction per day, matching the
    # existing dashboard's 3-day layout.
    # --------------------------------------------------------

    daily_predictions = {}
    daily_pollutants = {}

    for timestamp, prediction in predictions_by_timestamp.items():

        date_key = timestamp.date()

        if date_key not in daily_predictions:
            daily_predictions[date_key] = prediction

        if date_key not in daily_pollutants:
            daily_pollutants[date_key] = {
                "timestamp": timestamp,
                "aqi": prediction,
                "pm25": np.nan,
                "pm10": np.nan,
                "no2": np.nan,
                "o3": np.nan
            }

    if len(daily_predictions) < 3:
        return None, None, None

    selected_dates = sorted(
        daily_predictions.keys()
    )[:3]

    predictions = np.array(
        [
            daily_predictions[d]
            for d in selected_dates
        ],
        dtype=float
    )

    forecast_dates = [
        d.strftime("%Y-%m-%d")
        for d in selected_dates
    ]

    # Pollutant chart uses the first forecast observation of the
    # corresponding day.
    pollutant_df = pd.DataFrame(
        [
            next(
                row for row in pollutant_rows
                if row["timestamp"].date() == d
            )
            for d in selected_dates
        ]
    )

    # Preserve the model-feature rows for diagnostics.
    feature_df = pd.concat(
        feature_rows,
        ignore_index=True
    )

    pollutant_df.attrs["shap_features"] = (
    feature_df[feature_columns].iloc[[0]].copy()
    )

    feature_df.attrs["hourly_predictions"] = predictions_by_timestamp
    feature_df.attrs["selected_dates"] = forecast_dates
    feature_df.attrs["daily_predictions"] = predictions

    return (
        pollutant_df,
        predictions,
        forecast_dates
    )


# ============================================================
# PREDICT AQI

def predict_aqi(
    model,
    input_data
):

    try:

        if input_data is None:
            return None

        model_feature_count = getattr(
            model,
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
                f"Expected model to use 98 features, but it uses {model_feature_count}."
            )
            return None

        required_features = list(FEATURE_COLUMNS)

        missing_features = [
            feature
            for feature in required_features
            if feature not in input_data.columns
        ]

        if missing_features:
            st.error(
                "Missing model features: "
                + str(missing_features)
            )
            return None

        model_input = input_data[
            required_features
        ].copy()

        predictions = model.predict(
            model_input.to_numpy()
        )

        return np.clip(
            predictions,
            1,
            5
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        return None

# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(
    aqi
):

    value = int(
        round(
            float(aqi)
        )
    )

    return AQI_CATEGORIES.get(
        value,
        {
            "name": "Unknown"
        }
    )["name"]


# ============================================================
# AQI COLOR
# ============================================================

def get_aqi_color(
    aqi
):

    value = int(
        round(
            float(aqi)
        )
    )

    return AQI_CATEGORIES.get(
        value,
        {
            "color": "#808080"
        }
    )["color"]


# ============================================================
# AQI CHART
# ============================================================

def create_aqi_chart(
    predictions,
    dates
):

    bar_colors = [

        get_aqi_color(
            prediction
        )

        for prediction in predictions
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(

            x=dates,

            y=predictions,

            marker_color=bar_colors,

            text=[
                f"{prediction:.2f}"
                for prediction in predictions
            ],

            textposition="auto"
        )
    )

    fig.update_layout(

        

        xaxis=dict(type="category"),

        yaxis=dict(

            range=[
                0,
                5.5
            ],

            dtick=1
        ),

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#e5e7eb"
        ),

        height=400
    )

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
                Machine-learning powered AQI forecasting for Karachi, Pakistan
            </div>
            <div class="hero-status">🤖 ML SYSTEM OPERATIONAL</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Preparing the latest Karachi air-quality forecast..."):
        model = load_local_random_forest_model()
        current_weather = fetch_current_weather()
        forecast_data = fetch_aqi_forecast()
        historical_data = fetch_historical_feature_data()

    if model is None:
        st.error("The trained model could not be loaded.")
        st.stop()

    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count != 98:
        st.error(f"Model schema error: expected 98 features, found {model_feature_count}.")
        st.stop()

    model_features = list(FEATURE_COLUMNS)
    if len(model_features) != 98:
        st.error(f"Feature metadata error: expected 98 features, found {len(model_features)}.")
        st.stop()

    if forecast_data is None:
        st.error("Current AQI forecast data is unavailable.")
        st.stop()

    if historical_data is None:
        st.error("Historical AQI data is unavailable.")
        st.stop()

    processed_data, predictions, forecast_dates = process_forecast_data(
        forecast_data, historical_data, model, model_features
    )

    if forecast_data.get("_fallback", False):
        st.warning(
            "OpenWeather's forecast endpoint is temporarily unavailable. "
            "The model is using the latest available pollutant values as fallback inputs."
        )

    if processed_data is None or predictions is None or forecast_dates is None:
        st.error("The 3-day AQI forecast could not be generated.")
        st.stop()

    latest = historical_data.iloc[-1]
    current_aqi = _safe_float(latest.get("aqi"))
    current_pm25 = _safe_float(latest.get("pm25"))
    current_pm10 = _safe_float(latest.get("pm10"))
    current_no2 = _safe_float(latest.get("no2"))
    current_o3 = _safe_float(latest.get("o3"))

    current_category = get_aqi_category(current_aqi) if current_aqi is not None else "Unknown"
    current_color = get_aqi_color(current_aqi) if current_aqi is not None else "#94A3B8"
    last_timestamp = latest["timestamp"]

    # Hazardous AQI alert
    if current_aqi is not None and current_aqi >= 5:
        st.error(
            "🚨 Hazardous AQI Alert: Air quality is Very Poor. "
            "Consider reducing prolonged outdoor exposure."
        )

    # Current AQI + weather
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
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            '<div class="glass-card">🌎 Karachi — Current Conditions</div>',
            unsafe_allow_html=True
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
                    f"{float(temp):.1f}°C" if _safe_float(temp) is not None else "?"
                )
            with weather_cols[1]:
                render_metric_tile(
                    "Feels Like",
                    f"{float(feels_like):.1f}°C" if _safe_float(feels_like) is not None else "?"
                )
            with weather_cols[2]:
                render_metric_tile("Humidity", f"{humidity}%" if humidity is not None else "?")
            with weather_cols[3]:
                render_metric_tile("Conditions", description)
        else:
            st.info("Weather information is temporarily unavailable.")

        st.markdown("</div>", unsafe_allow_html=True)

    # 3-day forecast
    st.markdown(
        '<div class="glass-card"><div class="section-title">📅 3-Day AQI Forecast</div>',
        unsafe_allow_html=True
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
                    <div class="forecast-date">{forecast_dates[i]}</div>
                    <div class="forecast-value" style="color:{color};">{value:.2f}</div>
                    <div class="forecast-category" style="color:{color};">{category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    chart = create_aqi_chart(predictions, forecast_dates)
    chart.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        hovermode="x unified"
    )
    chart.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    chart.update_xaxes(showgrid=False)

    st.plotly_chart(
        chart,
        width="stretch",
        config={"displayModeBar": False, "responsive": True}
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # SHAP Explainability
    st.markdown(
        '<div class="glass-card"><div class="section-title">🔍 Why This Prediction?</div>',
        unsafe_allow_html=True
    )

    shap_features = processed_data.attrs.get(
        "shap_features"
    )

    shap_explanation = create_shap_explanation(
        model,
        shap_features,
        model_features,
        max_features=12
    )

    if shap_explanation is not None and not shap_explanation.empty:

        st.markdown(
            """
            <div style="color:#94A3B8;font-size:.85rem;margin-bottom:.8rem;">
                SHAP shows how the most influential features contributed
                to the first forecasted AQI value.
            </div>
            """,
            unsafe_allow_html=True
        )

        shap_chart = go.Figure()

        shap_chart.add_trace(
            go.Bar(
                x=shap_explanation["shap_value"],
                y=shap_explanation["feature"],
                orientation="h",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP impact: %{x:.4f}"
                    "<extra></extra>"
                )
            )
        )

        shap_chart.update_layout(
            height=430,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            xaxis_title="SHAP Value",
            yaxis_title=None,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#CBD5E1"
            )
        )

        shap_chart.update_xaxes(
            zeroline=True,
            zerolinecolor="rgba(148,163,184,.25)",
            gridcolor="rgba(148,163,184,.10)"
        )

        shap_chart.update_yaxes(
            gridcolor="rgba(148,163,184,.06)"
        )

        st.plotly_chart(
            shap_chart,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

        st.caption(
            "Positive SHAP values push the prediction higher; "
            "negative values push it lower."
        )

    else:

        st.info(
            "SHAP explanation is temporarily unavailable."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Pollutants
    st.markdown(
        '<div class="glass-card"><div class="section-title">🧪 Current Pollutant Levels</div>',
        unsafe_allow_html=True
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

    pollutant_chart = create_pollutant_chart(pd.DataFrame([{"pm25": current_pm25, "pm10": current_pm10, "no2": current_no2, "o3": current_o3}]))
    if pollutant_chart is not None:
        pollutant_chart.update_layout(
            
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False
        )
        pollutant_chart.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
        pollutant_chart.update_xaxes(showgrid=False)

        st.plotly_chart(
            pollutant_chart,
            width="stretch",
            config={"displayModeBar": False, "responsive": True}
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Forecast table
    st.markdown(
        '<div class="glass-card"><div class="section-title">📊 Forecast Details</div>',
        unsafe_allow_html=True
    )

    forecast_table = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted AQI": [round(float(v), 2) for v in predictions],
        "Category": [get_aqi_category(v) for v in predictions]
    })

    st.dataframe(forecast_table, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ML system
    st.markdown(
        '<div class="glass-card"><div class="section-title">🤖 Machine Learning System</div>'
        '<div class="model-strip">',
        unsafe_allow_html=True
    )

    model_cols = st.columns(5, gap="medium")
    model_items = [
        ("MODEL", "Random Forest", "RandomForestRegressor"),
        ("FEATURES", "98", "Verified feature schema"),
        ("TRAINING GROUP", "v4", "aqi_features"),
        ("SERVING GROUP", "v1", "aqi_serving_features"),
        ("MODEL REGISTRY", "v7", "karachi_aqi_random_forest"),
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
                unsafe_allow_html=True
            )

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 .5rem;color:#64748B;font-size:.78rem;">
            Karachi AQI Forecast • OpenWeather • Hopsworks • 98-Feature Random Forest
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()

