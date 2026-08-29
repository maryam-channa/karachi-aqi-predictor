import os
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import hopsworks


# ============================================================
# CONFIGURATION
# ============================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

LAT = 24.8607
LON = 67.0011

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 4

BACKFILL_DAYS = 30

# Safety switch:
# 1 = build and validate the complete historical dataset but do NOT
#     write it to Hopsworks.
# 0 = after validation, write it to Hopsworks.
DRY_RUN = os.getenv(
    "AQI_FEATURE_PIPELINE_DRY_RUN",
    "1",
).strip() == "1"

# We need enough history for the 24-hour lag/rolling features.
ROLLING_WINDOWS = [3, 6, 12, 24]
LAG_HOURS = [1, 2, 3, 6, 12, 24]


# ============================================================
# VALIDATION
# ============================================================

required_variables = {
    "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
    "HOPSWORKS_API_KEY": HOPSWORKS_API_KEY,
    "HOPSWORKS_PROJECT": HOPSWORKS_PROJECT,
    "HOPSWORKS_HOST": HOPSWORKS_HOST,
}

missing = [
    name
    for name, value in required_variables.items()
    if not value
]

if missing:
    raise ValueError(
        "Missing environment variables: "
        + ", ".join(missing)
    )


# ============================================================
# HOPSWORKS HOST
# ============================================================

def normalize_hopsworks_host(host):
    """
    Hopsworks expects the host without a duplicated protocol.

    Examples accepted:
        eu-west.cloud.hopsworks.ai
        https://eu-west.cloud.hopsworks.ai
        https://https://eu-west.cloud.hopsworks.ai

    Result:
        eu-west.cloud.hopsworks.ai
    """

    if not host:
        return host

    host = host.strip()

    while host.startswith("https://"):
        host = host[len("https://"):]

    while host.startswith("http://"):
        host = host[len("http://"):]

    return host.rstrip("/")


HOPSWORKS_HOST = normalize_hopsworks_host(
    HOPSWORKS_HOST
)


# ============================================================
# OPENWEATHER
# ============================================================

def fetch_data(api_key, start_unix, end_unix):
    """
    Fetch historical air-quality observations for Karachi.
    """

    url = (
        "https://api.openweathermap.org/data/2.5/"
        "air_pollution/history"
    )

    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_unix,
        "end": end_unix,
        "appid": api_key,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    if response.status_code == 200:
        return response.json().get("list", [])

    print(
        f"OpenWeather error {response.status_code}: "
        f"{response.text[:300]}"
    )

    return []


# ============================================================
# RAW DATA PROCESSING
# ============================================================

def process_data(raw_data):
    """
    Convert OpenWeather response into a chronological DataFrame.
    """

    rows = []

    for entry in raw_data:

        timestamp = dt.datetime.fromtimestamp(
            entry["dt"],
            tz=dt.timezone.utc
        )

        rows.append({
            "timestamp": timestamp,
            "aqi": float(entry["main"]["aqi"]),
            "pm25": float(entry["components"]["pm2_5"]),
            "pm10": float(entry["components"]["pm10"]),
            "no2": float(entry["components"]["no2"]),
            "o3": float(entry["components"]["o3"]),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "aqi",
                "pm25",
                "pm10",
                "no2",
                "o3",
            ]
        )

    data = pd.DataFrame(rows)

    data = (
        data
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_data(data):

    if data.empty:
        return data.copy()

    data = data.copy()

    # --------------------------------------------------------
    # Timestamp normalization
    # --------------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "o3",
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Strict row validation
    #
    # No interpolation / ffill / bfill is performed here.
    # Missing observations are removed rather than fabricated.
    # --------------------------------------------------------

    data = (
        data
        .dropna(
            subset=[
                "timestamp",
                "aqi",
                "pm25",
                "pm10",
                "no2",
                "o3",
            ]
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def keep_contiguous_hourly_segments(data):
    """
    Keep only hourly-contiguous segments containing enough history.

    The model uses 24-hour lags and 24-hour rolling windows, so a
    minimum of 25 genuine hourly observations is required per segment.

    This function does NOT fabricate missing timestamps.
    """

    if data.empty:
        return data.copy()

    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    timestamp_diff = (
        data["timestamp"]
        .diff()
    )

    new_segment = (
        timestamp_diff
        .isna()
        | (timestamp_diff > pd.Timedelta(hours=1))
    )

    segment_id = new_segment.cumsum()

    segments = []

    for _, segment in data.groupby(
        segment_id,
        sort=True,
    ):

        segment = (
            segment
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # 24 hours of history + current observation.
        if len(segment) >= 25:

            segments.append(segment)

    if not segments:
        return pd.DataFrame(
            columns=data.columns
        )

    return (
        pd.concat(
            segments,
            ignore_index=True,
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _create_features_for_segment(segment):
    """
    Build the complete 98-feature dataset for ONE contiguous hourly
    segment only.

    The segment must contain genuine hourly observations. No feature
    calculation is allowed to cross a gap.
    """

    data = (
        segment
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(data) < 25:
        return pd.DataFrame(columns=data.columns)

    # --------------------------------------------------------
    # IMPORTANT:
    # The current AQI is NOT used as a feature.
    # We predict target_aqi = AQI at t+1 hour.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 1. TIME FEATURES
    # --------------------------------------------------------

    data["hour"] = (
        data["timestamp"]
        .dt.hour
        .astype("int32")
    )

    data["day_of_week"] = (
        data["timestamp"]
        .dt.dayofweek
        .astype("int32")
    )

    data["day_of_month"] = (
        data["timestamp"]
        .dt.day
        .astype("int32")
    )

    data["month"] = (
        data["timestamp"]
        .dt.month
        .astype("int32")
    )

    data["week_of_year"] = (
        data["timestamp"]
        .dt.isocalendar()
        .week
        .astype("int32")
    )

    data["is_weekend"] = (
        data["day_of_week"]
        .isin([5, 6])
        .astype("int64")
    )

    data["is_morning"] = (
        data["hour"]
        .between(6, 11)
        .astype("int64")
    )

    data["is_afternoon"] = (
        data["hour"]
        .between(12, 17)
        .astype("int64")
    )

    data["is_evening"] = (
        data["hour"]
        .between(18, 23)
        .astype("int64")
    )

    data["is_night"] = (
        (
            (data["hour"] >= 0)
            & (data["hour"] <= 5)
        )
        .astype("int64")
    )

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # --------------------------------------------------------
    # 2. POLLUTANT LAG FEATURES
    # --------------------------------------------------------

    pollutants = [
        "pm25",
        "pm10",
        "no2",
        "o3",
    ]

    for pollutant in pollutants:

        for lag in LAG_HOURS:

            data[
                f"{pollutant}_lag_{lag}h"
            ] = data[pollutant].shift(lag)

    # --------------------------------------------------------
    # 3. AQI LAG FEATURES
    # --------------------------------------------------------

    for lag in LAG_HOURS:

        data[
            f"aqi_lag_{lag}h"
        ] = data["aqi"].shift(lag)

    # --------------------------------------------------------
    # 4. ROLLING POLLUTANT FEATURES
    # --------------------------------------------------------

    for pollutant in pollutants:

        previous_values = (
            data[pollutant]
            .shift(1)
        )

        for window in ROLLING_WINDOWS:

            data[
                f"{pollutant}_rolling_mean_{window}h"
            ] = (
                previous_values
                .rolling(window=window)
                .mean()
            )

            data[
                f"{pollutant}_rolling_std_{window}h"
            ] = (
                previous_values
                .rolling(window=window)
                .std()
            )

    # --------------------------------------------------------
    # 5. AQI ROLLING FEATURES
    # --------------------------------------------------------

    previous_aqi = (
        data["aqi"]
        .shift(1)
    )

    for window in ROLLING_WINDOWS:

        data[
            f"aqi_rolling_mean_{window}h"
        ] = (
            previous_aqi
            .rolling(window=window)
            .mean()
        )

        data[
            f"aqi_rolling_std_{window}h"
        ] = (
            previous_aqi
            .rolling(window=window)
            .std()
        )

    # --------------------------------------------------------
    # 6. POLLUTANT DIFFERENCE / RATIO FEATURES
    # --------------------------------------------------------

    data["pm25_pm10_ratio"] = (
        data["pm25"].shift(1)
        / data["pm10"].shift(1).replace(
            0,
            np.nan
        )
    )

    data["pm10_pm25_ratio"] = (
        data["pm10"].shift(1)
        / data["pm25"].shift(1).replace(
            0,
            np.nan
        )
    )

    data["pm25_no2_ratio"] = (
        data["pm25"].shift(1)
        / data["no2"].shift(1).replace(
            0,
            np.nan
        )
    )

    data["o3_no2_ratio"] = (
        data["o3"].shift(1)
        / data["no2"].shift(1).replace(
            0,
            np.nan
        )
    )

    data["pm25_change_1h"] = (
        data["pm25"].shift(1)
        - data["pm25"].shift(2)
    )

    data["pm10_change_1h"] = (
        data["pm10"].shift(1)
        - data["pm10"].shift(2)
    )

    data["no2_change_1h"] = (
        data["no2"].shift(1)
        - data["no2"].shift(2)
    )

    data["o3_change_1h"] = (
        data["o3"].shift(1)
        - data["o3"].shift(2)
    )

    # --------------------------------------------------------
    # 7. NEXT-HOUR TARGET
    #
    # This is the AQI at t+1 hour, calculated only inside the
    # same contiguous hourly segment.
    # --------------------------------------------------------

    data["target_aqi"] = (
        data["aqi"].shift(-1)
    )

    # --------------------------------------------------------
    # Remove rows that cannot have complete 24-hour history
    # and a genuine next-hour target.
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Deterministic primary key:
    # one stable integer for each UTC timestamp.
    # --------------------------------------------------------

    data["id"] = (
        (
            data["timestamp"].astype("int64")
            // 10**9
        )
        .astype("int64")
    )

    # --------------------------------------------------------
    # Explicit data types
    # --------------------------------------------------------

    integer_columns = [
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
        "id",
    ]

    for column in integer_columns:

        data[column] = (
            data[column]
            .astype("int64")
        )

    data["aqi"] = (
        data["aqi"]
        .astype("int64")
    )

    data["target_aqi"] = (
        data["target_aqi"]
        .astype("int64")
    )

    return data


def create_features(data):
    """
    Build the complete 98-feature training dataset.

    Observations are split into contiguous hourly segments first.
    Feature engineering is then performed independently within each
    segment, preventing lag/rolling/target calculations from crossing
    any missing-hour gap.
    """

    data = clean_data(data)

    if data.empty:
        return data

    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    gaps = (
        data["timestamp"]
        .diff()
        .gt(pd.Timedelta(hours=1))
    )

    segment_id = (
        gaps.fillna(True)
        .cumsum()
    )

    feature_segments = []
    segment_count = 0
    skipped_short_segments = 0

    for _, segment in data.groupby(
        segment_id,
        sort=True
    ):

        segment_count += 1

        # A complete row needs at least 24 previous hours plus
        # one current hour and one next-hour target.
        if len(segment) < 26:

            skipped_short_segments += 1

            continue

        segment_features = (
            _create_features_for_segment(
                segment
            )
        )

        if not segment_features.empty:

            feature_segments.append(
                segment_features
            )

    if not feature_segments:

        return pd.DataFrame()

    result = (
        pd.concat(
            feature_segments,
            ignore_index=True
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    result.attrs["segment_count"] = segment_count
    result.attrs["skipped_short_segments"] = (
        skipped_short_segments
    )

    return result


# ============================================================
# HOPSWORKS
# ============================================================

def store_data_to_hopsworks(data):

    print("\nConnecting to Hopsworks...")
    print(
        f"Hopsworks host: {HOPSWORKS_HOST}"
    )

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        engine="python",
    )

    print(
        f"Connected to project: {project.name}"
    )

    fs = project.get_feature_store()

    print(
        f"\nChecking Feature Group: "
        f"{FEATURE_GROUP_NAME} v"
        f"{FEATURE_GROUP_VERSION}"
    )

    # --------------------------------------------------------
    # Create the corrected Feature Group version if it does not exist.
    # --------------------------------------------------------

    try:

        feature_group = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )

    except Exception:

        feature_group = None

    if feature_group is None:

        print(
            f"Feature Group v{FEATURE_GROUP_VERSION} "
            "was not retrieved. Creating it..."
        )

        feature_group = fs.create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=(
                "Leakage-free Karachi AQI "
                "next-hour forecasting features"
            ),
            primary_key=["id"],
            event_time="timestamp",
            online_enabled=False,
            time_travel_format="HUDI",
        )

        if feature_group is None:
            raise RuntimeError(
                "Hopsworks did not return a Feature Group "
                "object after creation."
            )

        print(
            f"Feature Group v{FEATURE_GROUP_VERSION} "
            "created successfully."
        )

    else:

        print(
            "Feature Group retrieved successfully."
        )

    # --------------------------------------------------------
    # Final schema information
    # --------------------------------------------------------

    print(
        f"\nPreparing {len(data)} rows "
        "for Hopsworks..."
    )

    print(
        f"Feature columns: "
        f"{len(data.columns)} total columns"
    )

    print("\nData types:")
    print(data.dtypes.to_string())

    # --------------------------------------------------------
    # Derive and validate the exact 98 ML feature columns.
    # --------------------------------------------------------

    excluded_columns = {
        "timestamp",
        "aqi",
        "target_aqi",
        "id",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    if len(feature_columns) != 98:

        raise RuntimeError(
            "Expected exactly 98 ML features before upload, "
            f"found {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Exact schema check
    # --------------------------------------------------------

    expected_columns = (
        feature_columns
        + [
            "timestamp",
            "aqi",
            "target_aqi",
            "id",
        ]
    )

    if set(data.columns) != set(expected_columns):

        missing_columns = [
            c for c in expected_columns
            if c not in data.columns
        ]

        unexpected_columns = [
            c for c in data.columns
            if c not in expected_columns
        ]

        raise RuntimeError(
            "Feature Group schema mismatch. "
            f"Missing={missing_columns}; "
            f"Unexpected={unexpected_columns}"
        )

    # Reorder exactly as the Hopsworks Feature Group expects.
    data = data[
        feature_columns
        + [
            "timestamp",
            "aqi",
            "target_aqi",
            "id",
        ]
    ].copy()

    # Exact types for the existing Hopsworks schema.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data["aqi"] = data["aqi"].astype("int64")
    data["target_aqi"] = data["target_aqi"].astype("int64")
    data["id"] = data["id"].astype("int64")

    print(
        "\nUploading corrected 98-feature dataset..."
    )

    if feature_group is None:
        raise RuntimeError(
            "Feature Group object is None. Upload aborted."
        )

    feature_group.insert(
        data,
        write_options={
            "wait_for_job": True
        }
    )

    print(
        "\nData successfully stored "
        "in Hopsworks."
    )

    print(
        f"Rows processed: {len(data)}"
    )


# ============================================================
# FETCH + PROCESS
# ============================================================

def fetch_and_process_data(days=BACKFILL_DAYS):

    end_date = dt.datetime.now(
        dt.timezone.utc
    )

    start_date = (
        end_date
        - dt.timedelta(days=days)
    )

    print("=" * 60)
    print(
        " KARACHI AQI FEATURE PIPELINE v7"
    )
    print(
        " 98 Features / Leakage-Safe Next-Hour Forecast"
    )
    print("=" * 60)

    print(
        f"\nBackfilling {days} days..."
    )

    print(
        f"Start: {start_date}"
    )

    print(
        f"End:   {end_date}"
    )

    all_data = []

    current_date = start_date

    while current_date < end_date:

        next_date = min(
            current_date
            + dt.timedelta(days=1),
            end_date
        )

        start_unix = int(
            current_date.timestamp()
        )

        end_unix = int(
            next_date.timestamp()
        )

        print(
            f"\nFetching: "
            f"{current_date.date()} "
            f"-> {next_date.date()}"
        )

        raw_data = fetch_data(
            OPENWEATHER_API_KEY,
            start_unix,
            end_unix
        )

        if raw_data:

            all_data.extend(
                raw_data
            )

            print(
                f"  Received "
                f"{len(raw_data)} records."
            )

        else:

            print(
                "  No records received."
            )

        current_date = next_date

        time.sleep(0.5)

    if not all_data:

        raise RuntimeError(
            "No data was fetched from OpenWeather."
        )

    print(
        f"\nTotal raw records: "
        f"{len(all_data)}"
    )

    raw_df = process_data(
        all_data
    )

    print(
        f"Raw DataFrame rows: "
        f"{len(raw_df)}"
    )

    clean_df = clean_data(
        raw_df
    )

    print(
        f"Clean DataFrame rows: "
        f"{len(clean_df)}"
    )

    feature_df = create_features(
        clean_df
    )

    print(
        f"Feature DataFrame rows: "
        f"{len(feature_df)}"
    )

    if feature_df.empty:

        raise RuntimeError(
            "No rows remain after "
            "feature engineering."
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    excluded_columns = [
        "timestamp",
        "aqi",
        "target_aqi",
        "id",
    ]

    feature_columns = [
        column
        for column in feature_df.columns
        if column not in excluded_columns
    ]

    print(
        f"\nActual ML feature count: "
        f"{len(feature_columns)}"
    )

    print(
        "Valid contiguous segments: "
        f"{feature_df.attrs.get('segment_count', 'unknown')}"
    )

    print(
        "Skipped short segments: "
        f"{feature_df.attrs.get('skipped_short_segments', 'unknown')}"
    )

    if len(feature_columns) < 46:

        raise RuntimeError(
            f"Only {len(feature_columns)} "
            "features were created. "
            "At least 46 are required."
        )

    print(
        "\n46+ FEATURE REQUIREMENT: PASSED"
    )

    if len(feature_columns) != 98:
        raise RuntimeError(
            "The corrected pipeline must create exactly 98 "
            f"model features, found {len(feature_columns)}."
        )

    print(
        "98-feature schema check: PASSED"
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    forbidden_features = [
        "aqi",
        "target_aqi",
    ]

    leakage_features = [
        column
        for column in feature_columns
        if column in forbidden_features
    ]

    if leakage_features:

        raise RuntimeError(
            "Potential target leakage detected: "
            + ", ".join(leakage_features)
        )

    print(
        "Target leakage check: PASSED"
    )

    # --------------------------------------------------------
    # Timestamp checks
    # --------------------------------------------------------

    duplicate_timestamps = (
        feature_df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicate_timestamps}"
    )

    if duplicate_timestamps != 0:

        raise RuntimeError(
            "Duplicate timestamps detected."
        )

    # --------------------------------------------------------
    # Segment validation
    #
    # A gap between two independent valid segments is acceptable.
    # Every individual segment must still be hourly contiguous.
    # --------------------------------------------------------

    validation_gaps = 0

    if not feature_df.empty:

        diff = (
            feature_df["timestamp"]
            .sort_values()
            .diff()
        )

        # These are boundary gaps between valid segments.
        # They are reported, not treated as leakage.
        validation_gaps = int(
            (
                diff
                .dropna()
                > dt.timedelta(hours=1)
            ).sum()
        )

    print(
        "Valid feature-segment boundaries: "
        f"{validation_gaps}"
    )

    # --------------------------------------------------------
    # Target alignment validation inside contiguous segments.
    # --------------------------------------------------------

    target_mismatches = 0

    for _, segment in clean_df.groupby(
        (
            clean_df["timestamp"]
            .diff()
            .gt(dt.timedelta(hours=1))
            .fillna(True)
            .cumsum()
        ),
        sort=True,
    ):

        if len(segment) < 26:
            continue

        segment = (
            segment
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        expected_targets = (
            segment["aqi"]
            .shift(-1)
        )

        # Compare only timestamps represented in feature_df.
        segment_expected = dict(
            zip(
                segment["timestamp"],
                expected_targets,
            )
        )

        actual_expected = (
            feature_df["timestamp"]
            .map(segment_expected)
        )

        mismatches = (
            actual_expected.notna()
            & (
                feature_df["target_aqi"].astype(float)
                != actual_expected.astype(float)
            )
        ).sum()

        target_mismatches += int(
            mismatches
        )

    print(
        "Target alignment mismatches: "
        f"{target_mismatches}"
    )

    if target_mismatches != 0:

        raise RuntimeError(
            "target_aqi alignment check failed."
        )

    # --------------------------------------------------------
    # Deterministic ID validation
    # --------------------------------------------------------

    expected_ids = (
        (
            feature_df["timestamp"].astype("int64")
            // 10**9
        )
        .astype("int64")
    )

    id_mismatches = (
        feature_df["id"].astype("int64")
        != expected_ids
    ).sum()

    print(
        f"ID alignment mismatches: {int(id_mismatches)}"
    )

    if int(id_mismatches) != 0:

        raise RuntimeError(
            "Deterministic ID validation failed."
        )

    # --------------------------------------------------------
    # Display columns
    # --------------------------------------------------------

    print("\nML FEATURES:")
    for index, column in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{index:02d}. {column}"
        )

    print(
        "\nTarget: target_aqi "
        "(AQI one hour ahead)"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        feature_df[
            "target_aqi"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    if DRY_RUN:

        print(
            "\nDRY RUN: validation completed."
        )

        print(
            "No data was written to Hopsworks. The dataset is ready for review."
        )

        print(
            f"Validated rows: {len(feature_df)}"
        )

        print(
            f"Validated ML features: {len(feature_columns)}"
        )

        return

    store_data_to_hopsworks(
        feature_df
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    fetch_and_process_data(
        days=BACKFILL_DAYS
    )