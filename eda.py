import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA = Path("data/historical_aqi.csv")
OUT = Path("eda_results")
OUT.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA.resolve()}"
    )

df = pd.read_csv(DATA)


# ============================================================
# BASIC CLEANING
# ============================================================

required_columns = [
    "timestamp",
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "o3",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

df = df[required_columns].copy()

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce",
)

for column in required_columns[1:]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = (
    df
    .dropna(subset=required_columns)
    .drop_duplicates(subset=["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())


# ============================================================
# TIME PERIOD
# ============================================================

start_time = df["timestamp"].min()
end_time = df["timestamp"].max()

print("\n===== DATASET PERIOD =====")
print(f"Start: {start_time}")
print(f"End:   {end_time}")
print(f"Rows:  {len(df)}")


# ============================================================
# AQI SUMMARY
# ============================================================

aqi = df["aqi"]

print("\n===== AQI SUMMARY =====")
print(f"Mean AQI:   {aqi.mean():.4f}")
print(f"Median AQI: {aqi.median():.4f}")
print(f"Minimum AQI:{aqi.min():.4f}")
print(f"Maximum AQI:{aqi.max():.4f}")
print(f"Std AQI:    {aqi.std():.4f}")


# ============================================================
# AQI CATEGORY COUNTS
# ============================================================

aqi_categories = (
    df["aqi"]
    .round()
    .astype(int)
    .value_counts()
    .sort_index()
)

print("\n===== AQI CATEGORY DISTRIBUTION =====")
print(aqi_categories)


# ============================================================
# CORRELATION WITH AQI
# ============================================================

numeric = df[
    ["aqi", "pm25", "pm10", "no2", "o3"]
]

correlation = numeric.corr()

print("\n===== CORRELATION WITH AQI =====")

aqi_correlation = (
    correlation["aqi"]
    .sort_values(
        ascending=False
    )
)

print(aqi_correlation)


# ============================================================
# STRONGEST AQI CORRELATION
# ============================================================

pollutant_correlations = (
    aqi_correlation
    .drop(labels=["aqi"])
)

strongest_pollutant = (
    pollutant_correlations
    .abs()
    .idxmax()
)

strongest_value = (
    pollutant_correlations[strongest_pollutant]
)

print("\n===== KEY CORRELATION FINDING =====")
print(
    f"Strongest AQI correlation: "
    f"{strongest_pollutant} "
    f"({strongest_value:.4f})"
)


# ============================================================
# NUMERIC SUMMARY
# ============================================================

print("\n===== NUMERIC MEANS =====")
print(
    numeric.mean()
    .sort_values(
        ascending=False
    )
)


# ============================================================
# ADD TIME FEATURES FOR EDA
# ============================================================

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek


# ============================================================
# HOURLY AQI TREND
# ============================================================

hourly_aqi = (
    df.groupby("hour")["aqi"]
    .mean()
)

print("\n===== HOURLY AQI AVERAGE =====")
print(hourly_aqi)


highest_aqi_hour = int(
    hourly_aqi.idxmax()
)

lowest_aqi_hour = int(
    hourly_aqi.idxmin()
)

print(
    f"Highest average AQI hour: "
    f"{highest_aqi_hour}:00"
)

print(
    f"Lowest average AQI hour: "
    f"{lowest_aqi_hour}:00"
)


# ============================================================
# DAY-OF-WEEK AQI TREND
# ============================================================

daily_aqi = (
    df.groupby("day_of_week")["aqi"]
    .mean()
)

print("\n===== DAY-OF-WEEK AQI AVERAGE =====")
print(daily_aqi)


# ============================================================
# PLOT 1: AQI OVER TIME
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["aqi"],
    linewidth=1.5,
)

plt.title("AQI Trend Over Time")
plt.xlabel("Timestamp")
plt.ylabel("AQI")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUT / "aqi_trend.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 2: POLLUTANT TRENDS
# ============================================================

plt.figure(figsize=(14, 7))

for column in [
    "pm25",
    "pm10",
    "no2",
    "o3",
]:
    plt.plot(
        df["timestamp"],
        df[column],
        label=column,
        linewidth=1.2,
    )

plt.title("Pollutant Trends Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Concentration")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUT / "pollutant_trends.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 3: NUMERIC DISTRIBUTIONS
# ============================================================

numeric.hist(
    figsize=(14, 10),
)

plt.suptitle(
    "AQI and Pollutant Distributions"
)

plt.tight_layout()

plt.savefig(
    OUT / "numeric_distributions.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 4: CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    correlation,
    aspect="auto",
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index,
)

plt.title(
    "AQI and Pollutant Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    OUT / "correlation_heatmap.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 5: HOURLY AQI PATTERN
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_aqi.index,
    hourly_aqi.values,
    marker="o",
)

plt.title(
    "Average AQI by Hour of Day"
)

plt.xlabel("Hour")
plt.ylabel("Average AQI")
plt.xticks(
    range(0, 24, 2)
)

plt.tight_layout()

plt.savefig(
    OUT / "hourly_aqi_pattern.png",
    dpi=150,
)

plt.close()


# ============================================================
# KEY EDA FINDINGS
# ============================================================

print("\n===== EDA KEY FINDINGS =====")

print(
    f"1. Dataset contains {len(df)} hourly observations."
)

print(
    f"2. AQI ranges from {aqi.min():.2f} "
    f"to {aqi.max():.2f}, with a mean of "
    f"{aqi.mean():.2f}."
)

print(
    f"3. {strongest_pollutant} has the strongest "
    f"absolute correlation with AQI "
    f"({strongest_value:.4f})."
)

print(
    f"4. The highest average AQI occurs around "
    f"{highest_aqi_hour}:00."
)

print(
    f"5. The lowest average AQI occurs around "
    f"{lowest_aqi_hour}:00."
)

print(
    "6. Time-series and distribution plots were "
    "generated to identify temporal and statistical trends."
)


# ============================================================
# FINISHED
# ============================================================

print("\nEDA completed successfully.")

print(
    f"Results saved in: {OUT.resolve()}"
)