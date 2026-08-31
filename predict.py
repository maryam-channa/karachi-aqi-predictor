import os

import joblib
import hopsworks
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Production inference now reads the already-engineered 98-feature
# serving row from Hopsworks Online Feature Store.
FEATURE_GROUP_NAME = "aqi_serving_features"
FEATURE_GROUP_VERSION = 1

MODEL_PATH = r"models/random_forest_model.pkl"
IMPUTER_PATH = r"models/imputer.pkl"
METADATA_PATH = r"models/feature_metadata.pkl"


# ============================================================
# AQI CATEGORY
# ============================================================

def aqi_category(aqi: float) -> str:
    rounded = int(round(aqi))

    categories = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }

    return categories.get(rounded, "Unknown")


# ============================================================
# CONFIGURATION CHECK
# ============================================================

def check_configuration() -> None:
    missing = []

    if not HOPSWORKS_HOST:
        missing.append("HOPSWORKS_HOST")

    if not HOPSWORKS_PROJECT:
        missing.append("HOPSWORKS_PROJECT")

    if not HOPSWORKS_API_KEY:
        missing.append("HOPSWORKS_API_KEY")

    if missing:
        raise ValueError(
            "Missing environment variable(s): "
            + ", ".join(missing)
        )


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
        raise ValueError(
            f"Expected exactly 98 metadata features, "
            f"got {len(feature_columns)}."
        )

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
        engine="python",
    )

    print(f"Connected to project: {project.name}")

    return project


# ============================================================
# READ LATEST SERVING ROW
# ============================================================

def get_latest_serving_row(project, feature_columns):
    fs = project.get_feature_store()

    print(
        f"\nReading serving Feature Group: "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    if fg is None:
        raise ValueError(
            f"Feature Group {FEATURE_GROUP_NAME} v"
            f"{FEATURE_GROUP_VERSION} was not found."
        )

    if not fg.online_enabled:
        raise ValueError(
            "Serving Feature Group is not online-enabled."
        )

    # The serving group is intentionally read from the online store.
    df = fg.read(online=True)

    if df is None or df.empty:
        raise ValueError(
            "Serving Feature Group contains no online data."
        )

    required_metadata = [
        "id",
        "timestamp",
        "current_aqi",
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in df.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Serving row is missing metadata columns: "
            + ", ".join(missing_metadata)
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Serving row is missing model features: "
            + ", ".join(missing_features)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Serving Feature Group contains invalid timestamps."
        )

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    df["current_aqi"] = pd.to_numeric(
        df["current_aqi"],
        errors="coerce",
    )

    if df["id"].isna().any():
        raise ValueError(
            "Serving Feature Group contains invalid IDs."
        )

    if df["current_aqi"].isna().any():
        raise ValueError(
            "Serving Feature Group contains invalid current AQI values."
        )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    latest = df.iloc[-1].copy()

    print(
        f"Online serving rows available: {len(df)}"
    )

    print("\nLatest serving observation:")

    print(
        df[
            [
                "timestamp",
                "current_aqi",
            ]
        ]
        .tail(5)
        .to_string(index=False)
    )

    return latest


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    serving_row: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:

    print(
        "\nBuilding model input from serving Feature Group..."
    )

    values = [
        serving_row[column]
        for column in feature_columns
    ]

    X = pd.DataFrame(
        [values],
        columns=feature_columns,
    )

    if X.shape != (1, 98):
        raise ValueError(
            f"Expected a (1, 98) model matrix, "
            f"got {X.shape}."
        )

    if list(X.columns) != list(feature_columns):
        raise ValueError(
            "Serving feature order does not match trained model."
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().any().any():
        missing = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "Serving model input contains NaN values in: "
            + ", ".join(missing)
        )

    print(
        f"Feature row loaded successfully: "
        f"{len(X.columns)} features"
    )

    return X


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" KARACHI AQI NEXT-HOUR LIVE PREDICTION")
    print("=" * 60)

    check_configuration()

    model, imputer, feature_columns = (
        load_artifacts()
    )

    project = connect_hopsworks()

    serving_row = get_latest_serving_row(
        project,
        feature_columns,
    )

    serving_timestamp = pd.to_datetime(
        serving_row["timestamp"],
        utc=True,
    )

    forecast_timestamp = (
        serving_timestamp
        + pd.Timedelta(hours=1)
    )

    current_aqi = float(
        serving_row["current_aqi"]
    )

    X = build_model_input(
        serving_row,
        feature_columns,
    )

    # --------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------

    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=feature_columns,
    )

    if X_imputed.shape != (1, 98):
        raise ValueError(
            f"Expected a (1, 98) imputed matrix, "
            f"got {X_imputed.shape}."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nRunning Random Forest prediction...")

    prediction = float(
        model.predict(
            X_imputed.to_numpy()
        )[0]
    )

    prediction = max(
        1.0,
        min(5.0, prediction),
    )

    rounded_prediction = int(
        round(prediction)
    )

    category = aqi_category(
        prediction
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Last observed time : {serving_timestamp}"
    )

    print(
        f"Forecast time      : {forecast_timestamp}"
    )

    print(
        f"Current AQI        : {current_aqi:.0f}"
    )

    print(
        f"Predicted AQI      : {prediction:.4f}"
    )

    print(
        f"Rounded AQI        : {rounded_prediction}"
    )

    print(
        f"Category            : {category}"
    )

    print(
        f"Features used       : {len(feature_columns)}"
    )

    print("=" * 60)

    print(
        "\nLive prediction completed successfully."
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nPrediction interrupted.")
        raise SystemExit(130)

    except Exception as exc:
        print(
            f"\nPrediction FAILED: {exc}"
        )
        raise SystemExit(1)
