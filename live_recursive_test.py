import os
import joblib
import numpy as np
import pandas as pd
import requests

from pathlib import Path


LAT = 24.8607
LON = 67.0011

HISTORY_FILE = Path(
    "data/karachi_aqi_weather_2years.csv"
)

MODEL_FILE = Path(
    "models/recursive_72h/recursive_random_forest_compressed.pkl"
)

IMPUTER_FILE = Path(
    "models/recursive_72h/imputer.pkl"
)

META_FILE = Path(
    "models/recursive_72h/metadata.pkl"
)


# ============================================================
# FEATURE BUILDER — EXACT TRAINING SCHEMA
# ============================================================

LAGS = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]

VARIABLES = [
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


def make_features(history, timestamp):

    row = {}

    hour = timestamp.hour
    dow = timestamp.dayofweek
    month = timestamp.month
    doy = timestamp.dayofyear

    row["hour"] = hour
    row["day_of_week"] = dow
    row["month"] = month
    row["day_of_year"] = doy

    row["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    row["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    row["dow_sin"] = np.sin(
        2 * np.pi * dow / 7
    )

    row["dow_cos"] = np.cos(
        2 * np.pi * dow / 7
    )

    row["doy_sin"] = np.sin(
        2 * np.pi * doy / 365.25
    )

    row["doy_cos"] = np.cos(
        2 * np.pi * doy / 365.25
    )

    current = history.iloc[-1]

    for variable in VARIABLES:

        row[variable] = float(
            current[variable]
        )

        for lag in LAGS:

            row[
                f"{variable}_lag_{lag}h"
            ] = float(
                history[
                    variable
                ].iloc[-lag]
            )

    for variable in VARIABLES:

        for window in WINDOWS:

            values = (
                history[
                    variable
                ]
                .tail(window)
                .to_numpy()
            )

            row[
                f"{variable}_mean_{window}h"
            ] = float(
                values.mean()
            )

            row[
                f"{variable}_std_{window}h"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

    for window in [6, 12, 24]:

        values = (
            history["aqi"]
            .tail(window)
            .to_numpy()
        )

        x = np.arange(
            len(values),
            dtype=float
        )

        row[
            f"aqi_slope_{window}h"
        ] = float(
            np.polyfit(
                x,
                values,
                1
            )[0]
        )

    return pd.DataFrame([row])


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("=" * 60)
print("LIVE RECURSIVE 72-HOUR MODEL TEST")
print("=" * 60)

model = joblib.load(
    MODEL_FILE
)

imputer = joblib.load(
    IMPUTER_FILE
)

metadata = joblib.load(
    META_FILE
)

feature_columns = metadata[
    "feature_columns"
]

print(
    "Model features:",
    len(feature_columns)
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

history = pd.read_csv(
    HISTORY_FILE
)

history["timestamp"] = pd.to_datetime(
    history["timestamp"],
    utc=True
)

for column in VARIABLES:
    history[column] = pd.to_numeric(
        history[column],
        errors="coerce"
    )

history = (
    history
    .dropna(
        subset=VARIABLES
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# Use only the latest continuous segment.
diff = history["timestamp"].diff()

history["segment"] = (
    diff != pd.Timedelta(hours=1)
).cumsum()

segments = [
    g.reset_index(drop=True)
    for _, g in history.groupby(
        "segment"
    )
]

history = max(
    segments,
    key=len
)

if len(history) < 25:
    raise RuntimeError(
        "Not enough continuous history."
    )


# ============================================================
# OPENWEATHER CURRENT POLLUTION
# ============================================================

api_key = os.getenv(
    "OPENWEATHER_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "OPENWEATHER_API_KEY is not set."
    )

current_url = (
    "https://api.openweathermap.org/data/2.5/"
    "air_pollution"
)

response = requests.get(
    current_url,
    params={
        "lat": LAT,
        "lon": LON,
        "appid": api_key,
    },
    timeout=60
)

response.raise_for_status()

current_data = response.json()

current_item = (
    current_data["list"][0]
)

current_ts = (
    pd.to_datetime(
        current_item["dt"],
        unit="s",
        utc=True
    )
    .floor("h")
)
components = current_item[
    "components"
]


# ============================================================
# CURRENT WEATHER
# ============================================================

weather_url = (
    "https://api.openweathermap.org/data/2.5/"
    "weather"
)

weather_response = requests.get(
    weather_url,
    params={
        "lat": LAT,
        "lon": LON,
        "appid": api_key,
        "units": "metric",
    },
    timeout=60
)

weather_response.raise_for_status()

weather_data = (
    weather_response.json()
)

weather_row = {
    "temperature_2m":
        float(
            weather_data["main"]["temp"]
        ),

    "relative_humidity_2m":
        float(
            weather_data["main"]["humidity"]
        ),

    "surface_pressure":
        float(
            weather_data["main"]["pressure"]
        ),

    "wind_speed_10m":
        float(
            weather_data["wind"]["speed"]
        ) * 3.6,

    "cloud_cover":
        float(
            weather_data.get(
                "clouds",
                {}
            ).get(
                "all",
                0
            )
        ),
}


# ============================================================
# ALIGN CURRENT OBSERVATION
# ============================================================

current_aqi_500 = float(
    history["aqi"].iloc[-1]
)

current_row = {
    "timestamp": current_ts,

    "aqi": current_aqi_500,
    "pm25": float(
        components["pm2_5"]
    ),

    "pm10": float(
        components["pm10"]
    ),

    "no2": float(
        components["no2"]
    ),

    "o3": float(
        components["o3"]
    ),

    **weather_row,
}


# Remove any old rows at the same timestamp.
history = history[
    history["timestamp"] != current_ts
].copy()

history = pd.concat(
    [
        history[
            [
                "timestamp",
                *VARIABLES,
            ]
        ],
        pd.DataFrame(
            [current_row]
        ),
    ],
    ignore_index=True
)

history = (
    history
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# VERIFY FEATURE SCHEMA
# ============================================================

test_features = make_features(
    history,
    current_ts + pd.Timedelta(hours=1)
)

missing = [
    c
    for c in feature_columns
    if c not in test_features.columns
]

extra = [
    c
    for c in test_features.columns
    if c not in feature_columns
]

print(
    "Missing model features:",
    missing
)

print(
    "Extra features:",
    extra
)

if missing or extra:
    raise RuntimeError(
        "Live feature schema does not match "
        "trained model."
    )


# ============================================================
# RECURSIVE FORECAST
# ============================================================

predictions = []

# We don't have future weather forecasts in this test,
# so current weather is held constant.
# OpenWeather future pollution forecast values are used
# when available.

pollution_forecast_url = (
    "https://api.openweathermap.org/data/2.5/"
    "air_pollution/forecast"
)

forecast_response = requests.get(
    pollution_forecast_url,
    params={
        "lat": LAT,
        "lon": LON,
        "appid": api_key,
    },
    timeout=60
)

forecast_response.raise_for_status()

forecast_data = (
    forecast_response.json()
)

future = forecast_data.get(
    "list",
    []
)

future = sorted(
    future,
    key=lambda x: x["dt"]
)

previous_aqi = float(
    history["aqi"].iloc[-1]
)

for step in range(72):

    target_ts = (
        history["timestamp"].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    matching = None

    for item in future:

        item_ts = pd.to_datetime(
            item["dt"],
            unit="s",
            utc=True
        )

        if item_ts == target_ts:
            matching = item
            break

    if matching is not None:

        c = matching.get(
            "components",
            {}
        )

        pm25 = float(
            c.get(
                "pm2_5",
                history["pm25"].iloc[-1]
            )
        )

        pm10 = float(
            c.get(
                "pm10",
                history["pm10"].iloc[-1]
            )
        )

        no2 = float(
            c.get(
                "no2",
                history["no2"].iloc[-1]
            )
        )

        o3 = float(
            c.get(
                "o3",
                history["o3"].iloc[-1]
            )
        )

    else:

        pm25 = float(
            history["pm25"].iloc[-1]
        )

        pm10 = float(
            history["pm10"].iloc[-1]
        )

        no2 = float(
            history["no2"].iloc[-1]
        )

        o3 = float(
            history["o3"].iloc[-1]
        )

    # Future pollution inputs.
    new_input = {
        "timestamp": target_ts,
        "aqi": previous_aqi,
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "o3": o3,
        **weather_row,
    }

    X = make_features(
        history,
        target_ts
    )

    X = X[
        feature_columns
    ]

    X_i = imputer.transform(
        X
    )

    prediction = float(
        model.predict(
            X_i
        )[0]
    )

    prediction = float(
        np.clip(
            prediction,
            0,
            500
        )
    )

    predictions.append(
        {
            "timestamp": target_ts,
            "predicted_aqi": prediction,
        }
    )

    # Recursive AQI update.
    new_input["aqi"] = prediction

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [new_input]
            ),
        ],
        ignore_index=True
    )

    previous_aqi = prediction


# ============================================================
# RESULTS
# ============================================================

forecast = pd.DataFrame(
    predictions
)

forecast["day"] = (
    np.arange(len(forecast))
    // 24
) + 1

print("\n" + "=" * 60)
print("72-HOUR LIVE FORECAST")
print("=" * 60)

for day in [1, 2, 3]:

    values = (
        forecast[
            forecast["day"] == day
        ]["predicted_aqi"]
    )

    print(
        f"Day {day}: "
        f"mean={values.mean():.2f}, "
        f"min={values.min():.2f}, "
        f"max={values.max():.2f}"
    )

print(
    "\nFirst 10 predictions:"
)

print(
    forecast.head(10)
    .to_string(index=False)
)

print(
    "\nLive recursive forecast completed."
)