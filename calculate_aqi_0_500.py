import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = Path(
    "data/historical_aqi_2years_raw.csv"
)

OUTPUT = Path(
    "data/historical_aqi_2years_aqi.csv"
)


# ============================================================
# AQI BREAKPOINT TABLES
# EPA-style AQI breakpoints
# ============================================================

PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

O3_BREAKPOINTS = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 50.4, 301, 500),
]

NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 2049, 301, 500),
]

SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
]


# ============================================================
# GENERIC AQI INTERPOLATION
# ============================================================

def calculate_subindex(
    concentration,
    breakpoints,
):
    if concentration is None:
        return np.nan

    if pd.isna(concentration):
        return np.nan

    concentration = float(
        concentration
    )

    if concentration < 0:
        return np.nan

    for (
        c_low,
        c_high,
        i_low,
        i_high,
    ) in breakpoints:

        if (
            concentration >= c_low
            and concentration <= c_high
        ):

            if c_high == c_low:
                return float(i_high)

            value = (
                (
                    i_high - i_low
                )
                / (
                    c_high - c_low
                )
            ) * (
                concentration - c_low
            ) + i_low

            return float(
                value
            )

    # Above the last supported breakpoint:
    # cap the reported AQI at 500.
    if concentration > breakpoints[-1][1]:
        return 500.0

    return np.nan


# ============================================================
# UNIT CONVERSIONS
# ============================================================

def ug_m3_to_ppm(
    value,
    molecular_weight,
):
    """
    Convert micrograms/m3 to ppm using
    the conventional 25 C molar-volume approximation.
    """

    return (
        float(value)
        * 24.45
        / (
            molecular_weight
            * 1000.0
        )
    )


def ug_m3_to_ppb(
    value,
    molecular_weight,
):
    """
    Convert micrograms/m3 to ppb.
    """

    return (
        float(value)
        * 24.45
        / molecular_weight
    )


# ============================================================
# AQI CATEGORY
# ============================================================

def aqi_category(aqi):

    if pd.isna(aqi):
        return "Unknown"

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Input dataset not found: "
            f"{INPUT.resolve()}"
        )

    print("=" * 60)
    print("OPENWEATHER → 0-500 AQI DATASET")
    print("=" * 60)

    df = pd.read_csv(
        INPUT
    )

    required = [
        "timestamp",
        "openweather_aqi",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    numeric_columns = [
        "openweather_aqi",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=[
                "timestamp",
                "pm25",
                "pm10",
                "no2",
                "o3",
            ]
        )
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # ROLLING AVERAGES
    # --------------------------------------------------------
    #
    # Particle pollution:
    # approximately 24-hour average.
    #
    # Ozone and CO:
    # approximately 8-hour average.
    #
    # NO2 / SO2:
    # hourly value.
    #
    # This is a modeling-oriented hourly AQI construction,
    # not a claim that each row is an official regulatory
    # daily AQI observation.
    # --------------------------------------------------------

    df["pm25_24h"] = (
        df["pm25"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    df["pm10_24h"] = (
        df["pm10"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    df["o3_8h"] = (
        df["o3"]
        .rolling(
            window=8,
            min_periods=8,
        )
        .mean()
    )

    df["co_8h"] = (
        df["co"]
        .rolling(
            window=8,
            min_periods=8,
        )
        .mean()
    )

    # --------------------------------------------------------
    # POLLUTANT SUB-INDICES
    # --------------------------------------------------------

    df["aqi_pm25"] = (
        df["pm25_24h"]
        .apply(
            lambda x:
            calculate_subindex(
                x,
                PM25_BREAKPOINTS,
            )
        )
    )

    df["aqi_pm10"] = (
        df["pm10_24h"]
        .apply(
            lambda x:
            calculate_subindex(
                x,
                PM10_BREAKPOINTS,
            )
        )
    )

    # O3: μg/m3 → ppm
    df["o3_8h_ppm"] = (
        df["o3_8h"]
        .apply(
            lambda x:
            (
                ug_m3_to_ppm(
                    x,
                    48.00,
                )
                if pd.notna(x)
                else np.nan
            )
        )
    )

    df["aqi_o3"] = (
        df["o3_8h_ppm"]
        .apply(
            lambda x:
            calculate_subindex(
                x,
                O3_BREAKPOINTS,
            )
        )
    )

    # CO: μg/m3 → ppm
    df["co_8h_ppm"] = (
        df["co_8h"]
        .apply(
            lambda x:
            (
                ug_m3_to_ppm(
                    x,
                    28.01,
                )
                if pd.notna(x)
                else np.nan
            )
        )
    )

    df["aqi_co"] = (
        df["co_8h_ppm"]
        .apply(
            lambda x:
            calculate_subindex(
                x,
                CO_BREAKPOINTS,
            )
        )
    )

    # NO2: μg/m3 → ppb
    df["no2_ppb"] = (
        df["no2"]
        .apply(
            lambda x:
            (
                ug_m3_to_ppb(
                    x,
                    46.0055,
                )
                if pd.notna(x)
                else np.nan
            )
        )
    )

    df["aqi_no2"] = (
        df["no2_ppb"]
        .apply(
            lambda x:
            calculate_subindex(
                x,
                NO2_BREAKPOINTS,
            )
        )
    )

    # SO2: μg/m3 → ppb
    df["so2_ppb"] = (
        df["so2"]
        .apply(
            lambda x:
            (
                ug_m3_to_ppb(
                    x,
                    64.066,
                )
                if pd.notna(x)
                else np.nan
            )
        )
    )

    df["aqi_so2"] = (
        df["so2_ppb"]
        .apply(
            lambda x:
            calculate_subindex(
                x,
                SO2_BREAKPOINTS,
            )
        )
    )

    # --------------------------------------------------------
    # FINAL AQI
    # --------------------------------------------------------

    aqi_columns = [
        "aqi_pm25",
        "aqi_pm10",
        "aqi_o3",
        "aqi_co",
        "aqi_no2",
        "aqi_so2",
    ]

    # Maximum pollutant-specific sub-index
    # defines the composite AQI.
    df["aqi"] = (
        df[aqi_columns]
        .max(
            axis=1,
            skipna=True,
        )
    )

    df["aqi"] = (
        df["aqi"]
        .clip(
            lower=0,
            upper=500,
        )
        .round()
    )

    df["aqi_category"] = (
        df["aqi"]
        .apply(aqi_category)
    )

    # --------------------------------------------------------
    # FINAL DATASET
    # --------------------------------------------------------

    final_columns = [
        "timestamp",
        "aqi",
        "aqi_category",
        "openweather_aqi",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
        "nh3",
    ]

    output_df = df[
        final_columns
    ].copy()

    # The first ~24 rows cannot have a full PM 24h window.
    # Remove rows where a valid final AQI could not be formed.
    output_df = (
        output_df
        .dropna(
            subset=["aqi"]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n===== AQI DATASET =====")

    print(
        f"Rows: {len(output_df)}"
    )

    print(
        f"Start: "
        f"{output_df['timestamp'].min()}"
    )

    print(
        f"End: "
        f"{output_df['timestamp'].max()}"
    )

    print(
        "Duplicate timestamps:",
        output_df["timestamp"]
        .duplicated()
        .sum(),
    )

    print(
        "Missing AQI:",
        output_df["aqi"]
        .isna()
        .sum(),
    )

    print(
        "\n===== AQI SUMMARY ====="
    )

    print(
        output_df["aqi"]
        .describe()
        .to_string()
    )

    print(
        "\n===== AQI CATEGORIES ====="
    )

    print(
        output_df["aqi_category"]
        .value_counts()
        .to_string()
    )

    print(
        "\n===== AQI SOURCE COMPARISON ====="
    )

    print(
        output_df[
            [
                "timestamp",
                "aqi",
                "openweather_aqi",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT.resolve()
    )

    print(
        "\nAQI 0-500 calculation completed successfully."
    )


if __name__ == "__main__":
    main()