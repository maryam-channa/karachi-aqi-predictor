# Karachi AQI Predictor

An end-to-end machine learning system for forecasting the Air Quality Index (AQI) of Karachi, Pakistan for the next 72 hours.

## Project Overview

This project combines historical air-pollution observations, historical weather data, machine-learning-based forecasting, SHAP explainability, automated pipelines, and a Streamlit dashboard.

The final forecasting system uses a recursive Random Forest model trained with 163 engineered features and produces hourly predictions for a 72-hour forecast horizon.

The 72-hour forecast is evaluated as three separate 24-hour periods:

- Day 1: 0–24 hours
- Day 2: 24–48 hours
- Day 3: 48–72 hours

## Main Objectives

- Collect historical AQI and pollutant observations.
- Build a two-year historical dataset.
- Convert pollutant observations into a 0–500 AQI target.
- Integrate historical weather information.
- Engineer temporal, lag, rolling, and trend features.
- Train and compare machine-learning models.
- Generate recursive 72-hour AQI forecasts.
- Explain predictions using SHAP.
- Provide a real-time Streamlit dashboard.
- Automate data collection and training through GitHub Actions.
- Store ML assets and pipeline data using the project infrastructure.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Hopsworks
- GitHub Actions
- Streamlit
- Flask
- OpenWeather API
- Open-Meteo Archive API
- SHAP
- Git / GitHub
- Plotly

## Data Sources

### OpenWeather

OpenWeather is used for current atmospheric and air-pollution information and operational forecast inputs.

### Open-Meteo

Open-Meteo Archive API is used for historical hourly weather backfill because the available OpenWeather subscription did not provide historical weather access through One Call 3.0.

Historical weather variables include:

- Temperature
- Relative humidity
- Surface pressure
- Wind speed
- Cloud cover

## Historical Dataset

The final historical pollution dataset covers more than two years:

**September 2024 – August 2026**

The final combined dataset contains approximately 17,000 hourly pollution observations matched with historical weather observations.

The pollution dataset contains:

- AQI
- PM2.5
- PM10
- NO2
- O3
- SO2
- CO
- NH3

The merged pollution + weather dataset contains 16,968 matched rows.

Duplicate timestamps were removed and missing feature values were checked.

The pollution source contains some temporal gaps. Missing observations were not fabricated.

## AQI Scale

The final target is represented on a 0–500 AQI scale.

Categories used in the dashboard include:

- Good
- Moderate
- Unhealthy for Sensitive Groups
- Unhealthy
- Very Unhealthy
- Hazardous

## Feature Engineering

The final recursive model uses 163 engineered features.

Feature groups include:

### Calendar Features

- Hour
- Day of week
- Month
- Day of year
- Cyclical hour features
- Cyclical day-of-week features
- Cyclical day-of-year features

### Pollution Features

- Current pollutant values
- AQI history
- PM2.5
- PM10
- NO2
- O3

### Weather Features

- Temperature
- Humidity
- Pressure
- Wind speed
- Cloud cover

### Lag Features

Historical lag values include:

- 1 hour
- 2 hours
- 3 hours
- 6 hours
- 12 hours
- 24 hours

### Rolling Statistics

Rolling means and standard deviations are calculated over:

- 3 hours
- 6 hours
- 12 hours
- 24 hours

### Trend Features

AQI trend slopes are calculated over:

- 6 hours
- 12 hours
- 24 hours

## Model Development

Several forecasting formulations and machine-learning models were evaluated during development.

Models explored included:

- Ridge Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- TensorFlow MLP
- Recursive Random Forest

The final system uses a recursive Random Forest because it achieved the required three-day validation performance.

## Final Forecasting Method

The final model predicts AQI one hour ahead.

The prediction is then fed back into the feature-generation process as the predicted AQI for the next recursive step.

This process is repeated for 72 hours.

The 72 hourly predictions are evaluated in three blocks:

- Day 1 = hours 1–24
- Day 2 = hours 25–48
- Day 3 = hours 49–72

## Final Model Performance

Held-out recursive validation produced the following results:

| Forecast Horizon | RMSE | MAE | R² |
|---|---:|---:|---:|
| Day 1 (0–24h) | 0.5054 | 0.4320 | 0.8074 |
| Day 2 (24–48h) | 0.4243 | 0.3620 | 0.9677 |
| Day 3 (48–72h) | 0.5999 | 0.4966 | 0.7725 |

All three horizons achieved the required:

**R² > 0.70**

One-step validation performance:

- RMSE: 1.5852
- MAE: 0.5802
- R²: 0.9933

## Explainability

SHAP is integrated into the Streamlit dashboard.

The dashboard displays the most influential model features and their contribution to the forecast.

Positive SHAP values indicate features that push the model prediction higher, while negative values indicate features that push the prediction lower.

## Streamlit Dashboard

The dashboard provides:

- Current AQI
- Current weather
- Current pollutant levels
- 72-hour AQI forecast
- Day 1 / Day 2 / Day 3 forecast summaries
- AQI forecast chart
- Forecast details table
- SHAP explanation
- Model validation performance
- Hazardous AQI alerts
- Machine-learning system information

The final dashboard uses a 0–500 AQI scale and the 163-feature recursive Random Forest.

## Automation

GitHub Actions is used for project automation.

The workflow contains:

- Hourly feature pipeline schedule
- Daily training schedule
- Manual workflow dispatch
- Python environment setup
- Dependency installation
- Feature pipeline execution
- Model training pipeline execution

The existing workflow includes the original project training pipeline. The final recursive 72-hour model is maintained separately in `recursive_72h.py`.

## Hopsworks

Hopsworks is used as part of the project's feature-store and machine-learning pipeline infrastructure.

The project successfully connected to Hopsworks and used feature-store resources during development.

## Important Methodological Note

The reported Day 1, Day 2, and Day 3 R² values are held-out historical recursive validation metrics.

They should not be interpreted as live real-time R² values because future real-world observations are not available at the time of prediction.

The live prediction pipeline uses currently available operational API information and recursively generated AQI predictions.

## Project Structure

```text
karachi-aqi-predictor/
│
├── app1.py
├── api.py
├── recursive_72h.py
├── live_recursive_test.py
├── backfill_2years.py
├── backfill_weather_2years.py
├── calculate_aqi_0_500.py
├── merge_2years_data.py
│
├── data/
│   ├── historical_aqi_2years_raw.csv
│   ├── historical_aqi_2years_aqi.csv
│   ├── historical_weather_2years.csv
│   └── karachi_aqi_weather_2years.csv
│
├── models/
│   └── recursive_72h/
│       ├── imputer.pkl
│       ├── metadata.pkl
│       └── recursive_random_forest_compressed.pkl
│
├── eda.py
├── eda_results/
│
├── training/
│   └── train_models_v1.py
│
├── deployment/
│   └── predictor.py
│
├── requirements.txt
├── requirements-tensorflow.txt
└── .github/
    └── workflows/
        └── main.yml