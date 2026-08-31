import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

LAT = 24.8607
LON = 67.0011

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# At least 2 years of historical hourly data.
START_DATE = datetime(
    2024, 9, 1, 0, 0, 0, tzinfo=timezone.utc
)

END_DATE = datetime(
    2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc
)

# Conservative chunk size based on the successful API tests.
CHUNK_DAYS = 7

OUTPUT = Path(
    "data/historical_aqi_2years_raw.csv"
)

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 5


# ============================================================
# VALIDATION
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "OPENWEATHER_API_KEY environment variable is not set."
    )

if END_DATE <= START_DATE:
    raise ValueError(
        "END_DATE must be after START_DATE."
    )


# ============================================================
# FETCH ONE CHUNK
# ============================================================

def fetch_chunk(start_dt, end_dt):
    start_unix = int(
        start_dt.timestamp()
    )

    end_unix = int(
        end_dt.timestamp()
    )

    url = (
        "https://api.openweathermap.org/data/2.5/"
        "air_pollution/history"
    )

    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_unix,
        "end": end_unix,
        "appid": API_KEY,
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:

                payload = response.json()

                return payload.get(
                    "list",
                    []
                )

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:

                last_error = (
                    f"HTTP {response.status_code}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(
                        RETRY_SLEEP_SECONDS * attempt
                    )

                continue

            raise RuntimeError(
                f"OpenWeather HTTP "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            )

        except requests.RequestException as exc:

            last_error = str(exc)

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_SLEEP_SECONDS * attempt
                )

    raise RuntimeError(
        f"Request failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# PARSE RECORDS
# ============================================================

def parse_records(records):

    rows = []

    for item in records:

        timestamp = item.get("dt")

        if timestamp is None:
            continue

        components = item.get(
            "components",
            {}
        )

        main = item.get(
            "main",
            {}
        )

        rows.append(
            {
                "timestamp": pd.to_datetime(
                    timestamp,
                    unit="s",
                    utc=True,
                ).floor("h"),

                # Keep OpenWeather's original 1-5 AQI
                # for traceability.
                "openweather_aqi": (
                    float(main["aqi"])
                    if main.get("aqi") is not None
                    else None
                ),

                "pm25": (
                    float(
                        components["pm2_5"]
                    )
                    if components.get("pm2_5") is not None
                    else None
                ),

                "pm10": (
                    float(
                        components["pm10"]
                    )
                    if components.get("pm10") is not None
                    else None
                ),

                "no2": (
                    float(
                        components["no2"]
                    )
                    if components.get("no2") is not None
                    else None
                ),

                "o3": (
                    float(
                        components["o3"]
                    )
                    if components.get("o3") is not None
                    else None
                ),

                "so2": (
                    float(
                        components["so2"]
                    )
                    if components.get("so2") is not None
                    else None
                ),

                "co": (
                    float(
                        components["co"]
                    )
                    if components.get("co") is not None
                    else None
                ),

                "nh3": (
                    float(
                        components["nh3"]
                    )
                    if components.get("nh3") is not None
                    else None
                ),
            }
        )

    return rows


# ============================================================
# MAIN BACKFILL
# ============================================================

def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("KARACHI 2+ YEAR OPENWEATHER POLLUTION BACKFILL")
    print("=" * 60)

    print(
        f"Start: {START_DATE}"
    )

    print(
        f"End:   {END_DATE}"
    )

    print(
        f"Chunk size: {CHUNK_DAYS} days"
    )

    all_rows = []

    current = START_DATE

    total_chunks = 0

    while current < END_DATE:

        chunk_end = min(
            current + timedelta(
                days=CHUNK_DAYS
            ),
            END_DATE,
        )

        total_chunks += 1

        print(
            f"\nChunk {total_chunks}: "
            f"{current} -> {chunk_end}"
        )

        records = fetch_chunk(
            current,
            chunk_end,
        )

        print(
            f"API records returned: "
            f"{len(records)}"
        )

        all_rows.extend(
            parse_records(records)
        )

        current = chunk_end

        # Small delay between requests.
        if current < END_DATE:
            time.sleep(1)

    if not all_rows:
        raise RuntimeError(
            "No historical records were returned."
        )

    df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REQUIRED DATA CHECK
    # --------------------------------------------------------

    required = [
        "timestamp",
        "openweather_aqi",
        "pm25",
        "pm10",
        "no2",
        "o3",
    ]

    df = df.dropna(
        subset=required
    )

    # --------------------------------------------------------
    # SAVE RAW DATASET
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # VALIDATION SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)

    print(
        f"Rows:       {len(df)}"
    )

    print(
        f"Duplicates: "
        f"{df['timestamp'].duplicated().sum()}"
    )

    print(
        f"Start:      {df['timestamp'].min()}"
    )

    print(
        f"End:        {df['timestamp'].max()}"
    )

    print(
        f"Saved to:   {OUTPUT.resolve()}"
    )

    print("\nMissing values:")

    print(
        df[required]
        .isna()
        .sum()
        .to_string()
    )

    # --------------------------------------------------------
    # HOURLY GAP CHECK
    # --------------------------------------------------------

    timestamps = (
        pd.DatetimeIndex(
            df["timestamp"]
        )
        .sort_values()
    )

    gaps = (
        timestamps[1:]
        - timestamps[:-1]
    )

    large_gaps = (
        gaps > pd.Timedelta(hours=1)
    )

    print(
        "\nGaps greater than 1 hour:",
        int(large_gaps.sum())
    )

    print("\nLatest records:")

    print(
        df.tail(5)
        .to_string(index=False)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()