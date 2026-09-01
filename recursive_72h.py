import os
import joblib
import numpy as np
import pandas as pd
import hopsworks

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


INPUT = Path("data/karachi_aqi_weather_2years.csv")
OUTPUT_DIR = Path("models/recursive_72h")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAGS = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]


def make_features(history, timestamp):
    row = {}

    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

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

    variables = [
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

    current = history.iloc[-1]

    for variable in variables:
        row[variable] = float(current[variable])

        for lag in LAGS:
            if len(history) < lag + 1:
                raise ValueError("Insufficient history")
            row[f"{variable}_lag_{lag}h"] = float(
                history[variable].iloc[-lag]
            )

    for variable in variables:
        for window in WINDOWS:
            values = history[variable].tail(window).to_numpy()

            row[f"{variable}_mean_{window}h"] = float(values.mean())
            row[f"{variable}_std_{window}h"] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0
            )

    # AQI trend
    for window in [6, 12, 24]:
        values = history["aqi"].tail(window).to_numpy()
        x = np.arange(len(values), dtype=float)
        row[f"aqi_slope_{window}h"] = float(
            np.polyfit(x, values, 1)[0]
        )

    return pd.DataFrame([row])


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

numeric = [
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

for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = (
    df
    .dropna(subset=numeric + ["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# Keep only continuous hourly data.
diff = df["timestamp"].diff()
df["segment"] = (
    diff != pd.Timedelta(hours=1)
).cumsum()


# ============================================================
# BUILD ONE-STEP TRAINING DATA
# ============================================================

samples = []

for _, group in df.groupby("segment", sort=False):

    group = group.reset_index(drop=True)

    if len(group) < 30:
        continue

    for i in range(24, len(group) - 1):

        history = group.iloc[: i + 1]
        timestamp = group["timestamp"].iloc[i]

        X = make_features(
            history,
            timestamp,
        )

        X["target"] = float(
            group["aqi"].iloc[i + 1]
        )

        X["timestamp"] = timestamp

        samples.append(X)


if not samples:
    raise RuntimeError(
        "No training samples could be constructed."
    )

train_data = pd.concat(
    samples,
    ignore_index=True,
)

feature_columns = [
    c for c in train_data.columns
    if c not in {"timestamp", "target"}
]


# ============================================================
# CHRONOLOGICAL TRAIN / TEST
# ============================================================

split = int(len(train_data) * 0.80)

train = train_data.iloc[:split].copy()
test = train_data.iloc[split:].copy()

X_train = train[feature_columns]
y_train = train["target"]

X_test = test[feature_columns]
y_test = test["target"]


# ============================================================
# MODEL
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_i = imputer.fit_transform(X_train)
X_test_i = imputer.transform(X_test)

model = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_i,
    y_train,
)

one_step_pred = model.predict(
    X_test_i
)

one_step_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        one_step_pred,
    )
)

one_step_mae = mean_absolute_error(
    y_test,
    one_step_pred,
)

one_step_r2 = r2_score(
    y_test,
    one_step_pred,
)

print("=" * 60)
print("ONE-STEP MODEL")
print("=" * 60)

print(
    f"RMSE: {one_step_rmse:.6f}"
)

print(
    f"MAE:  {one_step_mae:.6f}"
)

print(
    f"R2:   {one_step_r2:.6f}"
)


# ============================================================
# RECURSIVE 72-HOUR EVALUATION
# ============================================================

segments = [
    g.reset_index(drop=True)
    for _, g in df.groupby(
        "segment",
        sort=False
    )
    if len(g) >= 200
]

if not segments:
    raise RuntimeError(
        "No sufficiently long continuous segment found."
    )

segment = segments[-1]

# Reserve final 72 hours as forecast target.
origin_index = len(segment) - 73

history = segment.iloc[
    : origin_index + 1
].copy()

actual_future = segment.iloc[
    origin_index + 1 :
    origin_index + 73
].copy()

if len(actual_future) != 72:
    raise RuntimeError(
        "Could not obtain a full 72-hour evaluation window."
    )

predictions = []

for step in range(72):

    timestamp = (
        history["timestamp"].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    X_next = make_features(
        history,
        timestamp,
    )

    X_next = X_next[
        feature_columns
    ]

    X_next_i = imputer.transform(
        X_next
    )

    prediction = float(
        model.predict(
            X_next_i
        )[0]
    )

    predictions.append(
        prediction
    )

    # Recursive update:
    # keep known future weather/pollutants,
    # replace only AQI with the model prediction.
    new_row = (
        actual_future.iloc[step]
        .copy()
    )

    new_row["aqi"] = prediction

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [new_row]
            ),
        ],
        ignore_index=True,
    )


predictions = np.asarray(
    predictions
)

actual = (
    actual_future["aqi"]
    .to_numpy()
)


# ============================================================
# DAY-WISE METRICS
# ============================================================

print("\n" + "=" * 60)
print("RECURSIVE 72-HOUR RESULTS")
print("=" * 60)

results = {}

for name, start, end in [
    ("Day 1", 0, 24),
    ("Day 2", 24, 48),
    ("Day 3", 48, 72),
]:

    y_true = actual[start:end]
    y_pred = predictions[start:end]

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    results[name] = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "status": (
            "PASS"
            if r2 > 0.70
            else "FAIL"
        ),
    }

    print(
        f"{name}: "
        f"RMSE={rmse:.6f} | "
        f"MAE={mae:.6f} | "
        f"R2={r2:.6f} | "
        f"{'PASS' if r2 > 0.70 else 'FAIL'}"
    )


# ============================================================
# SAVE FINAL MODEL ARTIFACTS
# ============================================================

MODEL_ARTIFACT = (
    OUTPUT_DIR
    / "recursive_random_forest_compressed.pkl"
)

IMPUTER_ARTIFACT = (
    OUTPUT_DIR
    / "imputer.pkl"
)

METADATA_ARTIFACT = (
    OUTPUT_DIR
    / "metadata.pkl"
)

# Save compressed model so it remains below GitHub's
# 100 MB individual-file limit.
joblib.dump(
    model,
    MODEL_ARTIFACT,
    compress=3,
)

joblib.dump(
    imputer,
    IMPUTER_ARTIFACT,
)

metadata = {
    "feature_columns": feature_columns,
    "feature_count": len(feature_columns),

    "model_type": (
        "Recursive Random Forest"
    ),

    "aqi_scale": "0-500",

    "training_data": str(INPUT),

    "training_rows": len(train),
    "testing_rows": len(test),
    "train_ratio": 0.80,

    "random_state": 42,

    "one_step_metrics": {
        "rmse": float(one_step_rmse),
        "mae": float(one_step_mae),
        "r2": float(one_step_r2),
    },

    "one_step_r2": float(
        one_step_r2
    ),

    "day_results": results,

    "forecast_definition": {
        "day1": "Hours 1-24",
        "day2": "Hours 25-48",
        "day3": "Hours 49-72",
    },

    "validation_method": (
        "Chronological held-out recursive 72-hour evaluation"
    ),

    "model_artifact": str(
        MODEL_ARTIFACT
    ),
}

joblib.dump(
    metadata,
    METADATA_ARTIFACT,
)

print(
    "\n" + "=" * 60
)

print(
    "FINAL MODEL ARTIFACTS SAVED"
)

print(
    "=" * 60
)

print(
    f"Model:    {MODEL_ARTIFACT}"
)

print(
    f"Imputer:  {IMPUTER_ARTIFACT}"
)

print(
    f"Metadata: {METADATA_ARTIFACT}"
)


# ============================================================
# HOPSWORKS MODEL REGISTRY
# ============================================================

def register_recursive_model():
    """
    Register the final recursive 72-hour Random Forest
    in the Hopsworks Model Registry.

    A new model version is created automatically by Hopsworks.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "REGISTERING FINAL RECURSIVE MODEL"
    )

    print(
        "=" * 60
    )

    host = os.getenv(
        "HOPSWORKS_HOST",
        "eu-west.cloud.hopsworks.ai",
    )

    project_name = os.getenv(
        "HOPSWORKS_PROJECT",
        "noismore",
    )

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        print(
            "HOPSWORKS_API_KEY is not set."
        )
        print(
            "Local training completed, but "
            "Model Registry registration was skipped."
        )
        return None

    try:
        print(
            "Connecting to Hopsworks..."
        )

        project = hopsworks.login(
            host=host,
            project=project_name,
            api_key_value=api_key,
        )

        print(
            "Connected to Hopsworks."
        )

        registry = (
            project.get_model_registry()
        )

        registered_model = (
            registry.python.create_model(
                name=(
                    "karachi_aqi_recursive_random_forest_72h"
                ),

                description=(
                    "Karachi 0-500 AQI 72-hour "
                    "recursive Random Forest using "
                    "163 engineered pollution, "
                    "weather and temporal features."
                ),

                metrics={
                    "one_step_rmse": float(
                        one_step_rmse
                    ),

                    "one_step_mae": float(
                        one_step_mae
                    ),

                    "one_step_r2": float(
                        one_step_r2
                    ),

                    "day1_rmse": float(
                        results["Day 1"]["rmse"]
                    ),

                    "day1_mae": float(
                        results["Day 1"]["mae"]
                    ),

                    "day1_r2": float(
                        results["Day 1"]["r2"]
                    ),

                    "day2_rmse": float(
                        results["Day 2"]["rmse"]
                    ),

                    "day2_mae": float(
                        results["Day 2"]["mae"]
                    ),

                    "day2_r2": float(
                        results["Day 2"]["r2"]
                    ),

                    "day3_rmse": float(
                        results["Day 3"]["rmse"]
                    ),

                    "day3_mae": float(
                        results["Day 3"]["mae"]
                    ),

                    "day3_r2": float(
                        results["Day 3"]["r2"]
                    ),

                    "feature_count": float(
                        len(feature_columns)
                    ),
                },
            )
        )

        registered_model.save(
            str(MODEL_ARTIFACT),
            keep_original_files=True,
        )

        print(
            "\nHopsworks Model Registry: PASSED"
        )

        print(
            "Registered model:"
        )

        print(
            "karachi_aqi_recursive_random_forest_72h"
        )

        return registered_model

    except Exception as exc:
        print(
            "\nHopsworks Model Registry: FAILED"
        )

        print(
            f"Registry error: {exc}"
        )

        return None


registered_model = (
    register_recursive_model()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "RECURSIVE 72-HOUR PIPELINE COMPLETE"
)

print(
    "=" * 60
)

for day_name in [
    "Day 1",
    "Day 2",
    "Day 3",
]:

    print(
        f"{day_name} R2: "
        f"{results[day_name]['r2']:.6f} "
        f"-> {results[day_name]['status']}"
    )

print(
    "\nOne-step R2: "
    f"{one_step_r2:.6f}"
)

print(
    "\nModel artifacts:"
)

print(
    MODEL_ARTIFACT
)

print(
    IMPUTER_ARTIFACT
)

print(
    METADATA_ARTIFACT
)
