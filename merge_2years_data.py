import pandas as pd
from pathlib import Path


POLLUTION_FILE = Path(
    "data/historical_aqi_2years_aqi.csv"
)

WEATHER_FILE = Path(
    "data/historical_weather_2years.csv"
)

OUTPUT_FILE = Path(
    "data/karachi_aqi_weather_2years.csv"
)


# ============================================================
# LOAD
# ============================================================

pollution = pd.read_csv(
    POLLUTION_FILE
)

weather = pd.read_csv(
    WEATHER_FILE
)


# ============================================================
# NORMALIZE TIMESTAMPS
# ============================================================

pollution["timestamp"] = pd.to_datetime(
    pollution["timestamp"],
    utc=True,
    errors="coerce"
)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    utc=True,
    errors="coerce"
)


pollution = pollution.dropna(
    subset=["timestamp"]
)

weather = weather.dropna(
    subset=["timestamp"]
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

pollution = (
    pollution
    .drop_duplicates("timestamp")
    .sort_values("timestamp")
    .reset_index(drop=True)
)

weather = (
    weather
    .drop_duplicates("timestamp")
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# MERGE ON EXACT UTC HOUR
# ============================================================

merged = pollution.merge(
    weather,
    on="timestamp",
    how="inner",
    validate="one_to_one"
)


# ============================================================
# SORT
# ============================================================

merged = (
    merged
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# VALIDATION
# ============================================================

print("=" * 60)
print("KARACHI 2-YEAR POLLUTION + WEATHER MERGE")
print("=" * 60)

print(
    f"Pollution rows: {len(pollution)}"
)

print(
    f"Weather rows:   {len(weather)}"
)

print(
    f"Merged rows:    {len(merged)}"
)

print(
    f"Start: {merged['timestamp'].min()}"
)

print(
    f"End:   {merged['timestamp'].max()}"
)

print(
    "\nDuplicates:",
    merged["timestamp"].duplicated().sum()
)


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

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

missing = [
    column
    for column in required
    if column not in merged.columns
]

if missing:
    raise RuntimeError(
        "Missing columns: "
        + ", ".join(missing)
    )


print(
    "\nMissing required values:"
)

print(
    merged[required]
    .isna()
    .sum()
    .to_string()
)


# ============================================================
# HOURLY GAP CHECK
# ============================================================

timestamps = (
    pd.DatetimeIndex(
        merged["timestamp"]
    )
    .sort_values()
)

differences = (
    timestamps[1:]
    - timestamps[:-1]
)

gaps = differences[
    differences > pd.Timedelta(hours=1)
]

missing_hours = int(
    sum(
        (gap / pd.Timedelta(hours=1)) - 1
        for gap in gaps
    )
)

print(
    f"\nGaps > 1 hour: {len(gaps)}"
)

print(
    f"Missing hours between merged records: "
    f"{missing_hours}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE.resolve()
)


print(
    "\nMerge completed successfully."
)