import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/historical_aqi.csv")
OUT = Path("eda_results")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe(include="all"))

numeric = df.select_dtypes(include="number")

if not numeric.empty:
    print("\n===== CORRELATION WITH AQI =====")
    aqi_cols = [c for c in numeric.columns if c.lower() == "aqi"]
    
    if aqi_cols:
        print(
            numeric.corr(numeric_only=True)[aqi_cols[0]]
            .sort_values(ascending=False)
        )

    print("\n===== NUMERIC TRENDS =====")
    print(numeric.mean().sort_values(ascending=False))

    numeric.hist(figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(OUT / "numeric_distributions.png", dpi=150)
    plt.close()

print("\nEDA completed successfully.")
print(f"Results saved in: {OUT.resolve()}")
