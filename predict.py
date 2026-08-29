import os
import joblib
import hopsworks
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 4

MODEL_PATH = r"models/random_forest_model.pkl"
IMPUTER_PATH = r"models/imputer.pkl"
METADATA_PATH = r"models/feature_metadata.pkl"

LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]


# ============================================================
# AQI CATEGORY
# ============================================================

def aqi_category(aqi):

    rounded = int(round(aqi))

    categories = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }

    return categories.get(rounded, "Unknown")


# ============================================================
# CONFIGURATION CHECK
# ============================================================

def check_configuration():

    if not HOPSWORKS_HOST:
        raise ValueError("HOPSWORKS_HOST is not set.")

    if not HOPSWORKS_PROJECT:
        raise ValueError("HOPSWORKS_PROJECT is not set.")

    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")


# ============================================================
# LOAD MODEL
# ============================================================

def load_artifacts():

    print("\nLoading model artifacts...")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    metadata = joblib.load(METADATA_PATH)

    feature_columns = metadata["feature_columns"]

    print("Model: Random Forest")
    print(f"Expected features: {len(feature_columns)}")

    if len(feature_columns) != 98:
        raise ValueError("98 feature requirement failed.")

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None
    )

    if model_feature_count is None:
        raise ValueError(
            "Model does not expose n_features_in_."
        )

    if int(model_feature_count) != len(feature_columns):
        raise ValueError(
            f"Model expects {model_feature_count} features, "
            f"but metadata contains {len(feature_columns)}."
        )

    if len(feature_columns) != 98:
        raise ValueError(
            f"Expected exactly 98 features, "
            f"got {len(feature_columns)}."
        )

    print(
        "98-feature model schema verification: PASSED"
    )

    print("98 feature requirement: PASSED")
    print("Feature schema: VERIFIED")

    return model, imputer, feature_columns


# ============================================================
# CONNECT TO HOPSWORKS
# ============================================================

def connect_hopsworks():

    print("\nConnecting to Hopsworks...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        engine="python"
    )

    print(f"Connected to project: {project.name}")

    return project


# ============================================================
# READ HISTORICAL DATA
# ============================================================

def get_historical_data(project):

    fs = project.get_feature_store()

    print(
        f"\nReading Feature Group: "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    df = fg.read()

    if df.empty:
        raise ValueError("Feature Group contains no data.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    print(f"Rows available: {len(df)}")

    print("\nLatest observed data:")

    print(
        df[
            [
                "timestamp",
                "aqi",
                "pm25",
                "pm10",
                "no2",
                "o3"
            ]
        ].tail(5).to_string(index=False)
    )

    return df


# ============================================================
# BUILD NEXT-HOUR FEATURE ROW
# ============================================================

def build_next_hour_features(df, feature_columns):

    if len(df) < 25:
        raise ValueError(
            "At least 25 historical hourly rows are required."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    latest_timestamp = df["timestamp"].iloc[-1]

    forecast_timestamp = (
        latest_timestamp + pd.Timedelta(hours=1)
    )

    row = {}

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We are predicting AQI at t+1.
    # Therefore the feature timestamp is t+1,
    # but pollutant/AQI history comes only from t and earlier.
    # --------------------------------------------------------

    hour = forecast_timestamp.hour
    day_of_week = forecast_timestamp.dayofweek
    day_of_month = forecast_timestamp.day
    month = forecast_timestamp.month
    week_of_year = forecast_timestamp.isocalendar().week

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    row["hour"] = hour
    row["day_of_week"] = day_of_week
    row["day_of_month"] = day_of_month
    row["month"] = month
    row["week_of_year"] = int(week_of_year)

    row["is_weekend"] = int(day_of_week in [5, 6])
    row["is_morning"] = int(6 <= hour <= 11)
    row["is_afternoon"] = int(12 <= hour <= 17)
    row["is_evening"] = int(18 <= hour <= 23)
    row["is_night"] = int(0 <= hour <= 5)

    row["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    row["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    row["dow_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    row["dow_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    row["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    row["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    # --------------------------------------------------------
    # CURRENT OBSERVED POLLUTANTS
    #
    # At forecast time t+1, the latest known measurements
    # are from t.
    # --------------------------------------------------------

    for pollutant in ["pm25", "pm10", "no2", "o3"]:

        row[pollutant] = df[pollutant].iloc[-1]

    # --------------------------------------------------------
    # POLLUTANT LAGS
    #
    # For forecast t+1:
    #
    # lag 1h = value at t
    # lag 2h = value at t-1
    # etc.
    # --------------------------------------------------------

    for pollutant in ["pm25", "pm10", "no2", "o3"]:

        for lag in LAG_HOURS:

            row[
                f"{pollutant}_lag_{lag}h"
            ] = df[pollutant].iloc[-lag]

    # --------------------------------------------------------
    # AQI LAGS
    # --------------------------------------------------------

    for lag in LAG_HOURS:

        row[
            f"aqi_lag_{lag}h"
        ] = df["aqi"].iloc[-lag]

    # --------------------------------------------------------
    # ROLLING POLLUTANT FEATURES
    #
    # Use values available before forecast time.
    # --------------------------------------------------------

    for pollutant in ["pm25", "pm10", "no2", "o3"]:

        historical_values = (
            df[pollutant]
            .tail(24)
            .to_numpy()
        )

        for window in ROLLING_WINDOWS:

            values = historical_values[-window:]

            row[
                f"{pollutant}_rolling_mean_{window}h"
            ] = np.mean(values)

            row[
                f"{pollutant}_rolling_std_{window}h"
            ] = (
                np.std(values, ddof=1)
                if len(values) > 1
                else 0.0
            )

    # --------------------------------------------------------
    # AQI ROLLING FEATURES
    # --------------------------------------------------------

    historical_aqi = (
        df["aqi"]
        .tail(24)
        .to_numpy()
    )

    for window in ROLLING_WINDOWS:

        values = historical_aqi[-window:]

        row[
            f"aqi_rolling_mean_{window}h"
        ] = np.mean(values)

        row[
            f"aqi_rolling_std_{window}h"
        ] = (
            np.std(values, ddof=1)
            if len(values) > 1
            else 0.0
        )

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------

    pm25 = df["pm25"].iloc[-1]
    pm10 = df["pm10"].iloc[-1]
    no2 = df["no2"].iloc[-1]
    o3 = df["o3"].iloc[-1]

    row["pm25_pm10_ratio"] = (
        pm25 / pm10 if pm10 != 0 else np.nan
    )

    row["pm10_pm25_ratio"] = (
        pm10 / pm25 if pm25 != 0 else np.nan
    )

    row["pm25_no2_ratio"] = (
        pm25 / no2 if no2 != 0 else np.nan
    )

    row["o3_no2_ratio"] = (
        o3 / no2 if no2 != 0 else np.nan
    )

    # --------------------------------------------------------
    # ONE-HOUR CHANGES
    # --------------------------------------------------------

    for pollutant in ["pm25", "pm10", "no2", "o3"]:

        current = df[pollutant].iloc[-1]
        previous = df[pollutant].iloc[-2]

        row[
            f"{pollutant}_change_1h"
        ] = current - previous

    # --------------------------------------------------------
    # CREATE DATAFRAME IN EXACT MODEL ORDER
    # --------------------------------------------------------

    X = pd.DataFrame(
        [row],
        columns=feature_columns
    )

    missing = [
        c for c in feature_columns
        if c not in row
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} features: {missing}"
        )

    if len(X.columns) != 98:
        raise ValueError(
            f"Expected 98 features, got {len(X.columns)}"
        )

    if list(X.columns) != list(feature_columns):
        raise ValueError(
            "Feature order does not match trained model."
        )

    return X, forecast_timestamp


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" KARACHI AQI NEXT-HOUR LIVE PREDICTION")
    print("=" * 60)

    check_configuration()

    model, imputer, feature_columns = load_artifacts()

    project = connect_hopsworks()

    df = get_historical_data(project)

    print("\nBuilding next-hour 98-feature inference row...")

    X, forecast_timestamp = build_next_hour_features(
        df,
        feature_columns
    )

    print(
        f"Feature row created successfully: "
        f"{len(X.columns)} features"
    )

    # --------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------

    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=feature_columns
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nRunning Random Forest prediction...")

    prediction = float(
        model.predict(X_imputed.to_numpy())[0]
    )

    prediction = max(
        1.0,
        min(5.0, prediction)
    )

    category = aqi_category(prediction)

    latest_timestamp = df["timestamp"].iloc[-1]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Last observed time : {latest_timestamp}"
    )

    print(
        f"Forecast time      : {forecast_timestamp}"
    )

    print(
        f"Current AQI        : {df['aqi'].iloc[-1]}"
    )

    print(
        f"Predicted AQI      : {prediction:.4f}"
    )

    print(
        f"Rounded AQI        : {round(prediction)}"
    )

    print(
        f"Category            : {category}"
    )

    print(
        f"Features used       : {len(feature_columns)}"
    )

    print("=" * 60)

    print("\nLive prediction completed successfully.")


if __name__ == "__main__":
    main()
