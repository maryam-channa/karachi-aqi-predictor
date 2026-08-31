import requests
import pandas as pd
import time
from datetime import date, timedelta
from pathlib import Path


LAT = 24.8607
LON = 67.0011

START_DATE = date(2024, 9, 1)
END_DATE = date(2026, 8, 31)

OUTPUT = Path("data/historical_weather_2years.csv")

CHUNK_DAYS = 31


def fetch_weather(start_date, end_date):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m,"
            "cloud_cover"
        ),
        "timezone": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 60)
    print("KARACHI 2+ YEAR HISTORICAL WEATHER BACKFILL")
    print("=" * 60)

    all_frames = []

    current = START_DATE
    chunk_number = 0

    while current < END_DATE:

        chunk_number += 1

        chunk_end = min(
            current + timedelta(days=CHUNK_DAYS - 1),
            END_DATE
        )

        print(
            f"\nChunk {chunk_number}: "
            f"{current} -> {chunk_end}"
        )

        data = fetch_weather(
            current,
            chunk_end
        )

        hourly = data.get(
            "hourly",
            {}
        )

        if not hourly or "time" not in hourly:
            raise RuntimeError(
                f"No hourly weather data returned "
                f"for {current} -> {chunk_end}"
            )

        frame = pd.DataFrame(hourly)

        frame["timestamp"] = pd.to_datetime(
            frame["time"],
            utc=True
        )

        frame = frame.drop(
            columns=["time"]
        )

        all_frames.append(frame)

        print(
            f"Records returned: {len(frame)}"
        )

        current = chunk_end + timedelta(days=1)

        if current < END_DATE:
            time.sleep(1)

    df = pd.concat(
        all_frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "cloud_cover",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing weather columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # MISSING VALUE CHECK
    # --------------------------------------------------------

    missing_counts = (
        df[required]
        .isna()
        .sum()
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

    differences = (
        timestamps[1:]
        - timestamps[:-1]
    )

    gaps = differences[
        differences > pd.Timedelta(hours=1)
    ]

    total_missing_hours = int(
        sum(
            (
                gap / pd.Timedelta(hours=1)
            ) - 1
            for gap in gaps
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("WEATHER BACKFILL COMPLETE")
    print("=" * 60)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Duplicates: "
        f"{df['timestamp'].duplicated().sum()}"
    )

    print(
        f"Start: "
        f"{df['timestamp'].min()}"
    )

    print(
        f"End: "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Hourly gaps: {len(gaps)}"
    )

    print(
        f"Missing hours: "
        f"{total_missing_hours}"
    )

    print(
        "\nMissing values:"
    )

    print(
        missing_counts.to_string()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT.resolve()
    )

    print("\nLatest records:")

    print(
        df.tail(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
