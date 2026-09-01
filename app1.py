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
    :root {
        --bg: #07111f;
        --panel: #0b1729;
        --panel-2: #0f1e33;
        --border: rgba(148,163,184,.14);
        --muted: #91a4bd;
        --text: #f8fafc;
        --accent: #38bdf8;
        --good: #22c55e;
        --moderate: #facc15;
        --usg: #fb923c;
        --unhealthy: #ef4444;
        --very: #a855f7;
        --hazard: #991b1b;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(56,189,248,.12), transparent 24%),
            radial-gradient(circle at 92% 7%, rgba(34,197,94,.08), transparent 20%),
            linear-gradient(180deg, #06101c 0%, #081321 55%, #07111f 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(6,16,28,.72);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    h1,h2,h3,h4 { color: var(--text) !important; }
    p,label,.stCaption { color: #c7d3e3; }

    .hero-v2 {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1rem;
        background:
            linear-gradient(135deg, rgba(15,36,61,.98), rgba(10,22,39,.96)),
            radial-gradient(circle at 90% 10%, rgba(56,189,248,.10), transparent 38%);
        box-shadow: 0 22px 60px rgba(0,0,0,.24);
    }

    .hero-v2:after {
        content: "";
        position: absolute;
        right: -70px;
        top: -70px;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: rgba(56,189,248,.06);
        filter: blur(4px);
    }

    .eyebrow {
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-size: .68rem;
        font-weight: 800;
        margin-bottom: .3rem;
    }

    .hero-title-v2 {
        color: #f8fafc;
        font-size: 2.45rem;
        font-weight: 900;
        letter-spacing: -.04em;
        margin: 0;
    }

    .hero-subtitle-v2 {
        color: #9fb0c5;
        margin-top: .35rem;
        font-size: .92rem;
    }

    .hero-badges {
        display:flex;
        flex-wrap:wrap;
        gap:.5rem;
        margin-top:.9rem;
    }

    .badge {
        display:inline-flex;
        align-items:center;
        gap:.35rem;
        padding:.42rem .7rem;
        border-radius:999px;
        border:1px solid rgba(148,163,184,.15);
        background:rgba(255,255,255,.035);
        color:#dbeafe;
        font-size:.74rem;
        font-weight:750;
    }

    .live-dot {
        width:8px;
        height:8px;
        border-radius:50%;
        background:#22c55e;
        box-shadow:0 0 0 4px rgba(34,197,94,.12);
    }

    .section-bar {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:1rem;
        border:1px solid var(--border);
        border-radius:18px;
        padding:.95rem 1.15rem;
        margin:1rem 0 .7rem;
        background:linear-gradient(135deg, rgba(15,30,51,.96), rgba(8,18,32,.96));
    }

    .section-heading {
        font-size:1rem;
        font-weight:850;
        color:#f8fafc;
    }

    .section-kicker {
        color:#7f93ad;
        font-size:.7rem;
        text-transform:uppercase;
        letter-spacing:.09em;
        font-weight:750;
    }

    .subsection-title {
        color:#cbd5e1;
        font-size:.78rem;
        font-weight:850;
        letter-spacing:.04em;
        text-transform:uppercase;
        margin:1rem 0 .45rem;
    }

    .panel {
        border:1px solid var(--border);
        border-radius:20px;
        padding:1rem;
        background:linear-gradient(145deg, rgba(15,30,51,.74), rgba(5,14,26,.83));
        box-shadow:0 12px 38px rgba(0,0,0,.16);
    }

    .panel-tight {
        padding:.8rem;
    }

    .aqi-hero-card {
        min-height:350px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        border-radius:22px;
        padding:1rem;
        background:
            radial-gradient(circle at 50% 38%, rgba(255,255,255,.045), transparent 37%),
            linear-gradient(145deg, rgba(24,40,63,.95), rgba(7,18,32,.98));
        border:1px solid rgba(148,163,184,.16);
        box-shadow: inset 0 1px rgba(255,255,255,.03), 0 18px 50px rgba(0,0,0,.2);
    }

    .aqi-card-label {
        color:#94a3b8;
        font-size:.72rem;
        text-transform:uppercase;
        letter-spacing:.16em;
        font-weight:800;
    }

    .aqi-big {
        font-size:4.7rem;
        line-height:.95;
        font-weight:950;
        letter-spacing:-.06em;
        margin-top:.1rem;
    }

    .aqi-pill {
        display:inline-flex;
        align-items:center;
        gap:.4rem;
        padding:.42rem .82rem;
        border-radius:999px;
        font-size:.82rem;
        font-weight:850;
        background:rgba(255,255,255,.06);
        border:1px solid rgba(255,255,255,.08);
        margin-top:.45rem;
    }

    .micro {
        color:#7287a2;
        font-size:.68rem;
        margin-top:.55rem;
    }

    .stat-card {
        border-radius:16px;
        border:1px solid rgba(148,163,184,.12);
        background:rgba(4,12,22,.58);
        padding:.85rem;
        min-height:96px;
    }

    .stat-label {
        color:#8195b0;
        font-size:.66rem;
        text-transform:uppercase;
        letter-spacing:.08em;
        font-weight:800;
    }

    .stat-value {
        color:#f8fafc;
        font-size:1.35rem;
        font-weight:900;
        margin-top:.22rem;
    }

    .stat-detail {
        color:#64748b;
        font-size:.65rem;
        margin-top:.1rem;
    }

    .forecast-card-v2 {
        border-radius:18px;
        border:1px solid rgba(148,163,184,.12);
        background:linear-gradient(145deg, rgba(3,11,22,.88), rgba(9,20,35,.88));
        padding:1rem;
        min-height:172px;
        position:relative;
        overflow:hidden;
    }

    .forecast-card-v2:before {
        content:"";
        position:absolute;
        inset:0 auto 0 0;
        width:4px;
        background:var(--forecast-color);
    }

    .forecast-horizon {
        color:#8094ad;
        font-size:.65rem;
        text-transform:uppercase;
        letter-spacing:.1em;
        font-weight:800;
    }

    .forecast-date-v2 {
        color:#e2e8f0;
        font-size:.72rem;
        margin-top:.18rem;
    }

    .forecast-value-v2 {
        font-size:2.15rem;
        font-weight:950;
        letter-spacing:-.05em;
        margin:.4rem 0 .05rem;
    }

    .forecast-risk {
        font-size:.72rem;
        font-weight:850;
    }

    .forecast-range {
        color:#6f829c;
        font-size:.65rem;
        margin-top:.55rem;
    }

    .trend-badge {
        display:inline-flex;
        padding:.32rem .6rem;
        border-radius:999px;
        background:rgba(56,189,248,.08);
        border:1px solid rgba(56,189,248,.18);
        color:#7dd3fc;
        font-size:.68rem;
        font-weight:750;
        margin-top:.55rem;
    }

    .metric-number {
        color:#f8fafc;
        font-size:1.9rem;
        font-weight:950;
        letter-spacing:-.03em;
    }

    .metric-small {
        color:#7f93ad;
        font-size:.67rem;
        margin-top:.2rem;
    }

    .model-row {
        display:grid;
        grid-template-columns: 1.1fr 1fr 1fr 1fr 1fr;
        gap:.65rem;
    }

    .model-box {
        border:1px solid rgba(148,163,184,.11);
        border-radius:14px;
        padding:.85rem;
        background:rgba(3,10,20,.5);
    }

    .model-title {
        color:#7f93ad;
        font-size:.62rem;
        text-transform:uppercase;
        letter-spacing:.08em;
        font-weight:800;
    }

    .model-value {
        color:#f8fafc;
        font-size:1rem;
        font-weight:900;
        margin-top:.25rem;
    }

    .model-sub {
        color:#64748b;
        font-size:.62rem;
        margin-top:.15rem;
    }

    .architecture {
        display:flex;
        align-items:center;
        gap:.35rem;
        flex-wrap:wrap;
        margin-top:.4rem;
    }

    .arch-step {
        flex:1 1 140px;
        min-width:120px;
        padding:.75rem;
        border:1px solid rgba(148,163,184,.11);
        border-radius:14px;
        background:rgba(4,12,22,.56);
        text-align:center;
    }

    .arch-step b {
        display:block;
        color:#e2e8f0;
        font-size:.72rem;
    }

    .arch-step span {
        color:#667a95;
        font-size:.58rem;
    }

    .arch-arrow {
        color:#38bdf8;
        font-size:.9rem;
        font-weight:900;
    }

    [data-testid="stDataFrame"] {
        border-radius:14px;
        overflow:hidden;
    }

    .stPlotlyChart {
        border-radius:16px;
        overflow:hidden;
    }

    div[data-testid="stAlert"] {
        border-radius:14px;
    }

    .stButton > button {
        border-radius:10px;
        font-weight:800;
        border:1px solid rgba(148,163,184,.18);
        background:rgba(255,255,255,.025);
    }

    @media (max-width: 900px) {
        .model-row { grid-template-columns:1fr 1fr; }
        .hero-title-v2 { font-size:2rem; }
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
# ENHANCED DASHBOARD HELPERS
# ============================================================

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aqi_category_key(aqi):
    value = _safe_float(aqi)
    if value is None:
        return "Unknown"
    for low, high, name, _ in AQI_CATEGORY_RANGES:
        if low <= value <= high:
            return name
    return "Hazardous"


def render_stat_card(label, value, detail=""):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title, kicker=None):
    kicker_html = (
        f'<div class="section-kicker">{kicker}</div>'
        if kicker else ""
    )
    st.markdown(
        f"""
        <div class="section-bar">
            <div class="section-heading">{title}</div>
            {kicker_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pollutant_card(name, value, unit="µg/m³"):
    number = _safe_float(value)
    text = "—" if number is None else f"{number:.2f}"
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{name}</div>
            <div class="stat-value">{text}</div>
            <div class="stat-detail">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_aqi_gauge(aqi_value):
    value = max(0.0, min(500.0, float(aqi_value)))

    steps = [
        {"range": [0, 50], "color": "rgba(34,197,94,.22)"},
        {"range": [50, 100], "color": "rgba(250,204,21,.20)"},
        {"range": [100, 150], "color": "rgba(251,146,60,.20)"},
        {"range": [150, 200], "color": "rgba(239,68,68,.20)"},
        {"range": [200, 300], "color": "rgba(168,85,247,.18)"},
        {"range": [300, 500], "color": "rgba(153,27,27,.22)"},
    ]

    color = get_aqi_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "font": {"size": 54, "color": color},
                "suffix": "",
            },
            gauge={
                "axis": {
                    "range": [0, 500],
                    "tickvals": [0, 50, 100, 150, 200, 300, 500],
                    "tickfont": {"size": 9, "color": "#8195ad"},
                    "tickcolor": "#8195ad",
                },
                "bar": {
                    "color": color,
                    "thickness": 0.28,
                },
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": steps,
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.85,
                    "value": value,
                },
            },
            domain={"x": [0.05, 0.95], "y": [0.02, 0.96]},
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def create_hourly_forecast_chart(historical_data, hourly_predictions):
    actual = historical_data[["timestamp", "aqi"]].tail(96).copy()
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True)

    forecast_df = pd.DataFrame(
        {
            "timestamp": list(hourly_predictions.keys()),
            "aqi": list(hourly_predictions.values()),
        }
    )
    forecast_df["timestamp"] = pd.to_datetime(
        forecast_df["timestamp"], utc=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["aqi"],
            mode="lines",
            name="Historical AQI",
            line=dict(color="#38bdf8", width=3),
            hovertemplate="%{x|%d %b %H:%M}<br>AQI: %{y:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["aqi"],
            mode="lines+markers",
            name="72h Forecast",
            line=dict(color="#facc15", width=3, dash="dash"),
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %H:%M}<br>Forecast AQI: %{y:.1f}<extra></extra>",
        )
    )

    # AQI risk bands.
    bands = [
        (0, 50, "rgba(34,197,94,.07)"),
        (50, 100, "rgba(250,204,21,.06)"),
        (100, 150, "rgba(251,146,60,.05)"),
        (150, 200, "rgba(239,68,68,.045)"),
        (200, 300, "rgba(168,85,247,.04)"),
        (300, 500, "rgba(153,27,27,.04)"),
    ]

    for low, high, fill in bands:
        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=fill,
            line_width=0,
            layer="below",
        )

    first_forecast = forecast_df["timestamp"].min()

    fig.add_vline(
        x=first_forecast,
        line_width=2,
        line_dash="dot",
        line_color="rgba(248,250,252,.45)",
        annotation_text="NOW / FORECAST",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#cbd5e1"),
    )

    for hours, label in [
        (24, "DAY 1 → DAY 2"),
        (48, "DAY 2 → DAY 3"),
    ]:
        if len(forecast_df) >= hours:
            x = forecast_df["timestamp"].iloc[hours - 1]
            fig.add_vline(
                x=x,
                line_width=1,
                line_dash="dash",
                line_color="rgba(148,163,184,.22)",
            )
            fig.add_annotation(
                x=x,
                y=500,
                text=label,
                showarrow=False,
                yanchor="top",
                yshift=-4,
                font=dict(size=8, color="#7f93ad"),
            )

    for level in [50, 100, 150, 200, 300]:
        fig.add_hline(
            y=level,
            line_width=1,
            line_dash="dot",
            line_color="rgba(148,163,184,.16)",
        )

    fig.update_layout(
        height=470,
        margin=dict(l=50, r=20, t=20, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        yaxis=dict(
            range=[0, 500],
            title="AQI (0–500)",
            gridcolor="rgba(148,163,184,.08)",
            zeroline=False,
        ),
        xaxis=dict(
            title=None,
            gridcolor="rgba(148,163,184,.04)",
        ),
    )
    return fig


def create_model_performance_chart(validation):
    labels = ["Day 1", "Day 2", "Day 3"]
    values = [
        float(validation.get(day, {}).get("r2", 0.0))
        for day in labels
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#38bdf8", "#22c55e", "#a78bfa"],
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: R² = %{x:.4f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=0.70,
        line_width=2,
        line_dash="dash",
        line_color="#facc15",
        annotation_text="Required 0.70",
        annotation_position="top",
        annotation_font=dict(size=10, color="#facc15"),
    )

    fig.update_layout(
        height=230,
        margin=dict(l=20, r=80, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(
            range=[0, 1.05],
            title="R²",
            gridcolor="rgba(148,163,184,.07)",
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(148,163,184,.05)",
        ),
        showlegend=False,
    )
    return fig


def create_history_trend_chart(historical_data):
    data = historical_data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    daily = (
        data.set_index("timestamp")["aqi"]
        .resample("1D")
        .mean()
        .dropna()
        .tail(30)
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["timestamp"],
            y=daily["aqi"],
            mode="lines+markers",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,.08)",
            hovertemplate="%{x|%d %b}<br>Daily mean AQI: %{y:.1f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="rgba(34,197,94,.55)",
    )
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="rgba(250,204,21,.55)",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=45, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        showlegend=False,
        yaxis=dict(
            range=[0, max(120, float(daily["aqi"].max()) * 1.15)],
            title="Daily mean AQI",
            gridcolor="rgba(148,163,184,.08)",
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,.04)",
        ),
    )
    return fig


def create_category_distribution_chart(historical_data):
    series = historical_data["aqi"].astype(float)
    counts = [
        int(((series >= low) & (series <= high)).sum())
        for low, high, _, _ in AQI_CATEGORY_RANGES
    ]
    labels = [name for _, _, name, _ in AQI_CATEGORY_RANGES]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=[
                color for _, _, _, color in AQI_CATEGORY_RANGES
            ],
            text=counts,
            textposition="auto",
            hovertemplate="%{x}: %{y:,} observations<extra></extra>",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=25, r=20, t=10, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        showlegend=False,
        xaxis=dict(
            tickangle=-18,
            gridcolor="rgba(148,163,184,.03)",
        ),
        yaxis=dict(
            title="Hourly observations",
            gridcolor="rgba(148,163,184,.08)",
        ),
    )
    return fig


def create_correlation_heatmap(historical_data):
    cols = [
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

    corr = historical_data[cols].corr()

    labels = [
        "AQI",
        "PM2.5",
        "PM10",
        "NO2",
        "O3",
        "Temp",
        "Humidity",
        "Pressure",
        "Wind",
        "Cloud",
    ]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#7f1d1d"],
                [0.25, "#991b1b"],
                [0.5, "#0f172a"],
                [0.75, "#0e7490"],
                [1.0, "#22c55e"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} ↔ %{y}<br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                title="r",
                thickness=12,
                len=0.7,
            ),
        )
    )

    fig.update_layout(
        height=470,
        margin=dict(l=30, r=30, t=20, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def create_pollutant_profile_chart(historical_data):
    latest = historical_data.iloc[-1]
    names = ["PM2.5", "PM10", "NO2", "O3"]
    values = [
        float(latest["pm25"]),
        float(latest["pm10"]),
        float(latest["no2"]),
        float(latest["o3"]),
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=["#fb7185", "#f59e0b", "#60a5fa", "#34d399"],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f} µg/m³<extra></extra>",
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=45, r=70, t=10, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="rgba(148,163,184,.08)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.04)"),
        showlegend=False,
    )
    return fig


# ============================================================
# MAIN APP
# ============================================================


def create_shap_explanation(model, feature_row, feature_columns, max_features=12):
    """Create a SHAP explanation for one recursive Random Forest forecast step."""
    try:
        if model is None or feature_row is None:
            return None

        X_explain = pd.DataFrame(feature_row, columns=feature_columns)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

        values = np.asarray(shap_values)
        if values.ndim == 2:
            values = values[0]
        values = values.astype(float).flatten()

        if len(values) != len(feature_columns):
            return None

        explanation = pd.DataFrame({
            "feature": feature_columns,
            "shap_value": values,
            "impact": np.abs(values),
        })

        return (
            explanation
            .sort_values("impact", ascending=False)
            .head(max_features)
            .sort_values("shap_value")
            .reset_index(drop=True)
        )
    except Exception as exc:
        st.warning(f"SHAP explanation unavailable: {exc}")
        return None

def main():
    st.markdown(
        """
        <div class="hero-v2">
            <div class="eyebrow">AQI Intelligence Center • Karachi, Pakistan</div>
            <div class="hero-title-v2">Karachi Air Quality</div>
            <div class="hero-subtitle-v2">
                Machine-learning powered 72-hour air-quality forecasting with live operational inputs and historical analytics.
            </div>
            <div class="hero-badges">
                <div class="badge"><span class="live-dot"></span> LIVE SYSTEM</div>
                <div class="badge">0–500 AQI</div>
                <div class="badge">163 FEATURES</div>
                <div class="badge">72-HOUR FORECAST</div>
                <div class="badge">SHAP EXPLAINABILITY</div>
            </div>
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
    hourly_predictions = forecast_result["hourly_predictions"]
    shap_features = forecast_result["shap_features"]

    latest = historical_data.iloc[-1]
    current_aqi = float(latest["aqi"])
    current_pm25 = float(latest["pm25"])
    current_pm10 = float(latest["pm10"])
    current_no2 = float(latest["no2"])
    current_o3 = float(latest["o3"])
    last_timestamp = pd.Timestamp(latest["timestamp"]).tz_convert("UTC")

    current_category = get_aqi_category(current_aqi)
    current_color = get_aqi_color(current_aqi)

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

    # --------------------------------------------------------
    # STATUS + CURRENT SNAPSHOT
    # --------------------------------------------------------
    if current_aqi >= 301:
        st.error(
            "🚨 Hazardous AQI Alert — avoid prolonged outdoor exposure and take protective measures."
        )
    elif current_aqi >= 201:
        st.warning(
            "⚠️ Very Unhealthy AQI — sensitive groups should reduce outdoor exposure."
        )
    elif current_aqi >= 151:
        st.warning(
            "⚠️ Unhealthy AQI — consider reducing prolonged outdoor exposure."
        )

    render_section(
        "Live AQI Command Center",
        "Current conditions • latest verified local observation",
    )

    left, middle, right = st.columns([0.9, 1.1, 1.1], gap="medium")

    with left:
        st.markdown(
            f"""
            <div class="aqi-hero-card">
                <div class="aqi-card-label">Current AQI</div>
                <div class="aqi-big" style="color:{current_color};">{current_aqi:.0f}</div>
                <div class="aqi-pill" style="color:{current_color};">
                    <span style="width:7px;height:7px;border-radius:50%;background:{current_color};display:inline-block;"></span>
                    {current_category}
                </div>
                <div class="micro">
                    Last observed: {last_timestamp.strftime("%d %b %Y • %H:%M UTC")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        render_stat_card(
            "Temperature",
            f"{float(current_weather.get('main', {}).get('temp')):.1f}°C"
            if current_weather else "—",
            "Current air temperature",
        )
        render_stat_card(
            "Feels Like",
            f"{float(current_weather.get('main', {}).get('feels_like')):.1f}°C"
            if current_weather else "—",
            "Perceived temperature",
        )

    with right:
        render_stat_card(
            "Humidity",
            f"{current_weather.get('main', {}).get('humidity')}%"
            if current_weather else "—",
            "Relative humidity",
        )
        condition = (
            current_weather.get("weather", [{}])[0]
            .get("description", "Unavailable")
            .capitalize()
            if current_weather
            else "Unavailable"
        )
        render_stat_card(
            "Conditions",
            condition,
            "OpenWeather current weather",
        )

    # --------------------------------------------------------
    # FORECAST HERO
    # --------------------------------------------------------
    render_section(
        "72-Hour Forecast",
        "Recursive hourly predictions aggregated into 3 daily blocks",
    )

    forecast_cols = st.columns(3, gap="medium")

    first_day_value = float(predictions[0])
    last_day_value = float(predictions[-1])
    overall_change = last_day_value - current_aqi

    if overall_change > 3:
        trend_text = "↗ Expected worsening"
        trend_color = "#fb7185"
    elif overall_change < -3:
        trend_text = "↘ Expected improvement"
        trend_color = "#4ade80"
    else:
        trend_text = "→ Expected stability"
        trend_color = "#7dd3fc"

    for i, col in enumerate(forecast_cols):
        value = float(predictions[i])
        category = get_aqi_category(value)
        color = get_aqi_color(value)

        with col:
            st.markdown(
                f"""
                <div class="forecast-card-v2" style="--forecast-color:{color};">
                    <div class="forecast-horizon">DAY {i + 1}</div>
                    <div class="forecast-date-v2">{forecast_dates[i]}</div>
                    <div class="forecast-value-v2" style="color:{color};">{value:.2f}</div>
                    <div class="forecast-risk" style="color:{color};">{category}</div>
                    <div class="forecast-range">
                        Range {daily_min[i]:.1f} – {daily_max[i]:.1f}
                    </div>
                    <div class="trend-badge" style="color:{trend_color};border-color:{trend_color}33;">
                        {trend_text if i == 2 else ("↗ Rising" if predictions[i] > current_aqi else "→ Near current")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart = create_hourly_forecast_chart(
        historical_data,
        hourly_predictions,
    )
    st.plotly_chart(
        chart,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    stat_cols = st.columns(4, gap="medium")

    hourly_values = np.asarray(
        list(hourly_predictions.values()),
        dtype=float,
    )

    with stat_cols[0]:
        render_stat_card(
            "72h mean",
            f"{hourly_values.mean():.1f}",
            "Mean forecast AQI",
        )

    with stat_cols[1]:
        render_stat_card(
            "72h peak",
            f"{hourly_values.max():.1f}",
            "Highest hourly forecast",
        )

    with stat_cols[2]:
        render_stat_card(
            "72h minimum",
            f"{hourly_values.min():.1f}",
            "Lowest hourly forecast",
        )

    with stat_cols[3]:
        render_stat_card(
            "Net change",
            f"{overall_change:+.1f}",
            "Day 3 mean − current AQI",
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------
    render_section(
        "Model Validation",
        "Held-out recursive 72-hour evaluation",
    )

    perf_left, perf_right = st.columns([1.55, 1], gap="medium")

    with perf_left:
        perf_chart = create_model_performance_chart(validation)
        st.plotly_chart(
            perf_chart,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    with perf_right:
        all_pass = all(
            float(validation.get(day, {}).get("r2", 0.0)) > 0.70
            for day in ["Day 1", "Day 2", "Day 3"]
        )

        st.markdown(
            f"""
            <div class="panel" style="height:100%;">
                <div class="stat-label">REQUIREMENT STATUS</div>
                <div style="font-size:2rem;font-weight:950;color:{'#4ade80' if all_pass else '#fb7185'};margin-top:.3rem;">
                    {'ALL PASSED' if all_pass else 'REVIEW REQUIRED'}
                </div>
                <div style="color:#94a3b8;font-size:.74rem;margin-top:.35rem;">
                    Required threshold: R² &gt; 0.70
                </div>
                <div style="height:1px;background:rgba(148,163,184,.10);margin:1rem 0;"></div>
                <div class="stat-label">VALIDATION TYPE</div>
                <div style="color:#e2e8f0;font-weight:800;margin-top:.25rem;">
                    Chronological held-out recursive evaluation
                </div>
                <div style="color:#64748b;font-size:.67rem;margin-top:.25rem;">
                    Metrics reflect historical model validation, not live-future R².
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # VALIDATION COMPARISON TABLE
    # --------------------------------------------------------
    validation_table = pd.DataFrame(
        [
            {
                "Forecast Horizon": "Day 1 (0–24h)",
                "RMSE": float(validation["Day 1"]["rmse"]),
                "MAE": float(validation["Day 1"]["mae"]),
                "R²": float(validation["Day 1"]["r2"]),
            },
            {
                "Forecast Horizon": "Day 2 (24–48h)",
                "RMSE": float(validation["Day 2"]["rmse"]),
                "MAE": float(validation["Day 2"]["mae"]),
                "R²": float(validation["Day 2"]["r2"]),
            },
            {
                "Forecast Horizon": "Day 3 (48–72h)",
                "RMSE": float(validation["Day 3"]["rmse"]),
                "MAE": float(validation["Day 3"]["mae"]),
                "R²": float(validation["Day 3"]["r2"]),
            },
        ]
    )

    st.markdown(
        '<div class="subsection-title">Validation Comparison</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        validation_table.style.format(
            {
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "R²": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Forecast Horizon": st.column_config.TextColumn(
                "Forecast Horizon",
            ),
            "RMSE": st.column_config.NumberColumn(
                "RMSE",
                format="%.4f",
            ),
            "MAE": st.column_config.NumberColumn(
                "MAE",
                format="%.4f",
            ),
            "R²": st.column_config.NumberColumn(
                "R²",
                format="%.4f",
            ),
        },
    )

    st.caption(
        "Day 1 = 0–24h • Day 2 = 24–48h • Day 3 = 48–72h. "
        "These R² values are validation metrics, not live-future R² values."
    )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------
    render_section(
        "Why This Forecast?",
        "SHAP feature attribution for the first recursive forecast step",
    )

    shap_explanation = create_shap_explanation(
        model,
        shap_features,
        model_features,
        max_features=12,
    )

    if shap_explanation is not None and not shap_explanation.empty:
        shap_explanation["direction"] = np.where(
            shap_explanation["shap_value"] >= 0,
            "Pushes AQI higher",
            "Pushes AQI lower",
        )

        shap_chart = go.Figure()

        shap_chart.add_trace(
            go.Bar(
                x=shap_explanation["shap_value"],
                y=shap_explanation["feature"],
                orientation="h",
                marker_color=[
                    "#fb7185" if v >= 0 else "#38bdf8"
                    for v in shap_explanation["shap_value"]
                ],
                customdata=shap_explanation["direction"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP: %{x:.4f}<br>"
                    "%{customdata}"
                    "<extra></extra>"
                ),
            )
        )

        shap_chart.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            showlegend=False,
            xaxis=dict(
                title="Impact on predicted AQI",
                gridcolor="rgba(148,163,184,.08)",
                zeroline=True,
                zerolinecolor="rgba(248,250,252,.22)",
            ),
            yaxis=dict(
                gridcolor="rgba(148,163,184,.05)",
            ),
        )

        shap_cols = st.columns([1.5, 1], gap="medium")

        with shap_cols[0]:
            st.plotly_chart(
                shap_chart,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )

        with shap_cols[1]:
            top = (
                shap_explanation.assign(
                    abs_impact=lambda x: x["shap_value"].abs()
                )
                .sort_values("abs_impact", ascending=False)
                .head(5)
            )

            st.markdown(
                '<div class="panel"><div class="stat-label">TOP DRIVERS</div>',
                unsafe_allow_html=True,
            )

            for _, row in top.iterrows():
                value = float(row["shap_value"])
                color = "#fb7185" if value >= 0 else "#38bdf8"
                arrow = "↑" if value >= 0 else "↓"

                st.markdown(
                    f"""
                    <div style="padding:.55rem 0;border-bottom:1px solid rgba(148,163,184,.08);">
                        <div style="display:flex;justify-content:space-between;gap:1rem;">
                            <span style="color:#e2e8f0;font-size:.74rem;">{row['feature']}</span>
                            <span style="color:{color};font-weight:900;font-size:.74rem;">{arrow} {abs(value):.3f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Positive SHAP values push the forecast upward; negative SHAP values pull it downward."
        )
    else:
        st.info("SHAP explanation is temporarily unavailable.")

    # --------------------------------------------------------
    # POLLUTION PROFILE
    # --------------------------------------------------------
    render_section(
        "Pollution Profile",
        "Latest verified pollutant concentrations",
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

    st.plotly_chart(
        create_pollutant_profile_chart(historical_data),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    # --------------------------------------------------------
    # HISTORICAL ANALYTICS
    # --------------------------------------------------------
    render_section(
        "Historical AQI Intelligence",
        "30-day trend • 2-year category distribution • pollutant/weather relationships",
    )

    history_cols = st.columns(2, gap="medium")

    with history_cols[0]:
        st.markdown(
            '<div class="panel"><div class="section-heading">Recent 30-Day AQI Trend</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            create_history_trend_chart(historical_data),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with history_cols[1]:
        st.markdown(
            '<div class="panel"><div class="section-heading">2-Year AQI Category Distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            create_category_distribution_chart(historical_data),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="panel" style="margin-top:.8rem;"><div class="section-heading">AQI Correlation Map</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Correlation is descriptive association in the final 2-year combined dataset; it is not causal evidence."
    )
    st.plotly_chart(
        create_correlation_heatmap(historical_data),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------
    render_section(
        "Forecast Details",
        "24-hour block summary",
    )

    forecast_table = pd.DataFrame(
        {
            "Horizon": [
                "Day 1 · 0–24h",
                "Day 2 · 24–48h",
                "Day 3 · 48–72h",
            ],
            "Date": forecast_dates,
            "Predicted Mean AQI": [
                round(float(v), 2)
                for v in predictions
            ],
            "Minimum AQI": [
                round(float(v), 2)
                for v in daily_min
            ],
            "Maximum AQI": [
                round(float(v), 2)
                for v in daily_max
            ],
            "Category": [
                get_aqi_category(v)
                for v in predictions
            ],
        }
    )

    st.dataframe(
        forecast_table,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # SYSTEM ARCHITECTURE
    # --------------------------------------------------------
    render_section(
        "How the System Works",
        "End-to-end architecture",
    )

    with st.expander(
        "Open architecture & methodology",
        expanded=True,
    ):
        st.markdown(
            """
            <div class="architecture">
                <div class="arch-step">
                    <b>OpenWeather</b>
                    <span>Live air pollution + weather inputs</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <b>Historical Data</b>
                    <span>2+ years of AQI + weather</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <b>Feature Engineering</b>
                    <span>163 temporal, lag, rolling & trend features</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <b>Recursive Random Forest</b>
                    <span>One-hour prediction repeated for 72 hours</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <b>SHAP</b>
                    <span>Feature-level forecast explanation</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <b>Streamlit</b>
                    <span>Interactive AQI intelligence dashboard</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# --------------------------------------------------------
    # MODEL SYSTEM STRIP
    # --------------------------------------------------------
    render_section(
        "Machine Learning System",
        "Deployment-ready model summary",
    )

    model_items = [
        ("MODEL", "Recursive Random Forest", "72-hour forecasting"),
        ("FEATURES", "163", "Verified training schema"),
        ("DATA", "2+ Years", "Hourly AQI + weather"),
        ("FORECAST", "72 Hours", "3 × 24-hour blocks"),
        (
            "VALIDATION R²",
            f"{validation.get('Day 1', {}).get('r2', 0):.3f} / "
            f"{validation.get('Day 2', {}).get('r2', 0):.3f} / "
            f"{validation.get('Day 3', {}).get('r2', 0):.3f}",
            "Day 1 / Day 2 / Day 3",
        ),
    ]

    st.markdown(
        '<div class="model-row">',
        unsafe_allow_html=True,
    )

    for label, value, detail in model_items:
        st.markdown(
            f"""
            <div class="model-box">
                <div class="model-title">{label}</div>
                <div class="model-value">{value}</div>
                <div class="model-sub">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center;padding:1.4rem 0 .7rem;color:#5f738d;font-size:.68rem;">
            Karachi AQI Intelligence Center • OpenWeather • Open-Meteo historical weather
            • 2+ year dataset • 163-feature Recursive Random Forest • SHAP
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
