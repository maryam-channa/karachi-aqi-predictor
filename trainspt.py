import os
import joblib
import hopsworks
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_VIEW_NAME = "aqi_prediction_view"
FEATURE_VIEW_VERSION = 2

MODEL_DIR = "models"


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():

    missing = []

    if not HOPSWORKS_HOST:
        missing.append("HOPSWORKS_HOST")

    if not HOPSWORKS_PROJECT:
        missing.append("HOPSWORKS_PROJECT")

    if not HOPSWORKS_API_KEY:
        missing.append("HOPSWORKS_API_KEY")

    if missing:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing)
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return rmse, mae, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" KARACHI AQI MODEL TRAINING PIPELINE")
    print(" Feature View + 98 Features")
    print("=" * 60)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    check_environment()

    print(f"\nHopsworks host: {HOPSWORKS_HOST}")
    print(f"Project: {HOPSWORKS_PROJECT}")

    # --------------------------------------------------------
    # Login
    # --------------------------------------------------------

    print("\nConnecting to Hopsworks...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,

    )

    print(f"Connected to project: {project.name}")

    # --------------------------------------------------------
    # Feature Store
    # --------------------------------------------------------

    fs = project.get_feature_store()

    # --------------------------------------------------------
    # Feature View
    # --------------------------------------------------------

    print(
        f"\nLoading Feature View: "
        f"{FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}"
    )

    feature_view = fs.get_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION
    )

    # --------------------------------------------------------
    # Read training data
    # --------------------------------------------------------

    print("\nReading training data from Feature View...")

    X, y = feature_view.training_data(
        start_time=None,
        end_time=None
    )

    print(f"Feature rows loaded: {len(X)}")
    print(f"Feature columns loaded: {len(X.columns)}")

    if y is None:
        raise ValueError(
            "Feature View did not return label data."
        )

    print(f"Label rows loaded: {len(y)}")
    print(f"Label columns loaded: {len(y.columns)}")

    print("\nFeature columns:")
    print(list(X.columns))

    print("\nLabel columns:")
    print(list(y.columns))

    # --------------------------------------------------------
    # Verify label
    # --------------------------------------------------------

    if "target_aqi" not in y.columns:

        # Some Hopsworks versions may return the label
        # under a different representation. Print enough
        # information to diagnose instead of guessing.

        raise ValueError(
            "target_aqi was not found in the label dataframe. "
            f"Available labels: {list(y.columns)}"
        )

    # --------------------------------------------------------
    # Convert to normal pandas DataFrames
    # --------------------------------------------------------

    X = pd.DataFrame(X).copy()
    y = pd.DataFrame(y).copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = pd.to_numeric(
        y["target_aqi"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove accidental non-feature columns
    # --------------------------------------------------------

    forbidden_columns = [
        "timestamp",
        "id",
        "aqi",
        "target_aqi",
    ]

    feature_columns = [
        column
        for column in X.columns
        if column not in forbidden_columns
    ]

    # --------------------------------------------------------
    # Verify 46-feature requirement
    # --------------------------------------------------------

    print(
        f"\nInput feature count: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) < 46:

        raise ValueError(
            "46-feature requirement FAILED. "
            f"Only {len(feature_columns)} features found."
        )

    print(
        "46+ feature requirement: PASSED"
    )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    X = X[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    valid_target = target.notna()

    X = X.loc[
        valid_target
    ].reset_index(drop=True)

    target = target.loc[
        valid_target
    ].reset_index(drop=True)

    print(
        f"Final usable rows: {len(X)}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = target.iloc[
        :split_index
    ].copy()

    y_test = target.iloc[
        split_index:
    ].copy()

    print("\n" + "=" * 60)
    print(" CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 60)

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    print(
        "\nApplying median imputation..."
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" TRAINING RANDOM FOREST")
    print("=" * 60)

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train_imputed,
        y_train
    )

    rf_train_pred = rf_model.predict(
        X_train_imputed
    )

    rf_test_pred = rf_model.predict(
        X_test_imputed
    )

    rf_train_rmse, rf_train_mae, rf_train_r2 = (
        calculate_metrics(
            y_train,
            rf_train_pred
        )
    )

    rf_test_rmse, rf_test_mae, rf_test_r2 = (
        calculate_metrics(
            y_test,
            rf_test_pred
        )
    )

    print("\nRandom Forest Results")

    print(
        f"Train RMSE: {rf_train_rmse:.6f}"
    )

    print(
        f"Test RMSE:  {rf_test_rmse:.6f}"
    )

    print(
        f"Test MAE:   {rf_test_mae:.6f}"
    )

    print(
        f"Test RÂ²:    {rf_test_r2:.6f}"
    )

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" TRAINING GRADIENT BOOSTING")
    print("=" * 60)

    gbr_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    gbr_model.fit(
        X_train_imputed,
        y_train
    )

    gbr_train_pred = gbr_model.predict(
        X_train_imputed
    )

    gbr_test_pred = gbr_model.predict(
        X_test_imputed
    )

    gbr_train_rmse, gbr_train_mae, gbr_train_r2 = (
        calculate_metrics(
            y_train,
            gbr_train_pred
        )
    )

    gbr_test_rmse, gbr_test_mae, gbr_test_r2 = (
        calculate_metrics(
            y_test,
            gbr_test_pred
        )
    )

    print("\nGradient Boosting Results")

    print(
        f"Train RMSE: {gbr_train_rmse:.6f}"
    )

    print(
        f"Test RMSE:  {gbr_test_rmse:.6f}"
    )

    print(
        f"Test MAE:   {gbr_test_mae:.6f}"
    )

    print(
        f"Test RÂ²:    {gbr_test_r2:.6f}"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "rmse": rf_test_rmse,
                "mae": rf_test_mae,
                "r2": rf_test_r2,
            },
            {
                "model": "Gradient Boosting",
                "rmse": gbr_test_rmse,
                "mae": gbr_test_mae,
                "r2": gbr_test_r2,
            },
        ]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    best_row = comparison.loc[
        comparison["rmse"].idxmin()
    ]

    best_model_name = best_row[
        "model"
    ]

    print(
        f"\nBEST MODEL: "
        f"{best_model_name}"
    )

    print(
        f"RMSE: {best_row['rmse']:.6f}"
    )

    print(
        f"MAE: {best_row['mae']:.6f}"
    )

    print(
        f"RÂ²: {best_row['r2']:.6f}"
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    rf_path = os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )

    gbr_path = os.path.join(
        MODEL_DIR,
        "gradient_boosting_model.pkl"
    )

    imputer_path = os.path.join(
        MODEL_DIR,
        "imputer.pkl"
    )

    metadata_path = os.path.join(
        MODEL_DIR,
        "feature_metadata.pkl"
    )

    joblib.dump(
        rf_model,
        rf_path
    )

    joblib.dump(
        gbr_model,
        gbr_path
    )

    joblib.dump(
        imputer,
        imputer_path
    )

    metadata = {
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "target_column": "target_aqi",
        "feature_view": FEATURE_VIEW_NAME,
        "feature_view_version": FEATURE_VIEW_VERSION,
        "best_model": best_model_name,

        "random_forest": {
            "rmse": float(rf_test_rmse),
            "mae": float(rf_test_mae),
            "r2": float(rf_test_r2),
        },

        "gradient_boosting": {
            "rmse": float(gbr_test_rmse),
            "mae": float(gbr_test_mae),
            "r2": float(gbr_test_r2),
        },
    }

    joblib.dump(
        metadata,
        metadata_path
    )

    print(
        "\nLocal models saved."
    )

    # --------------------------------------------------------
    # Model Registry
    # --------------------------------------------------------

    print(
        "\nConnecting to Model Registry..."
    )

    mr = project.get_model_registry()

    # --------------------------------------------------------
    # Random Forest Registry
    # --------------------------------------------------------

    print(
        "\nRegistering Random Forest..."
    )

    rf_registered = mr.python.create_model(
        name="karachi_aqi_random_forest",
        description=(
            "Random Forest Karachi AQI prediction "
            "using 98 engineered features from "
            "aqi_prediction_view v1."
        ),
        metrics={
            "rmse": float(rf_test_rmse),
            "mae": float(rf_test_mae),
            "r2": float(rf_test_r2),
            "feature_count": float(len(feature_columns)),
        }
    )

    rf_registered.save(
        rf_path,
        keep_original_files=True
    )

    print(
        "Random Forest registered successfully."
    )

    # --------------------------------------------------------
    # Gradient Boosting Registry
    # --------------------------------------------------------

    print(
        "\nRegistering Gradient Boosting..."
    )

    gbr_registered = mr.python.create_model(
        name="karachi_aqi_gradient_boosting",
        description=(
            "Gradient Boosting Karachi AQI prediction "
            "using 98 engineered features from "
            "aqi_prediction_view v1."
        ),
        metrics={
            "rmse": float(gbr_test_rmse),
            "mae": float(gbr_test_mae),
            "r2": float(gbr_test_r2),
            "feature_count": float(len(feature_columns)),
        }
    )

    gbr_registered.save(
        gbr_path,
        keep_original_files=True
    )

    print(
        "Gradient Boosting registered successfully."
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        " MODEL TRAINING COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)

    print(
        f"Feature View: "
        f"{FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}"
    )

    print(
        f"Input features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test)}"
    )

    print(
        f"Best Model: "
        f"{best_model_name}"
    )

    print(
        f"Best RMSE: "
        f"{best_row['rmse']:.6f}"
    )

    print(
        f"Best MAE: "
        f"{best_row['mae']:.6f}"
    )

    print(
        f"Best RÂ²: "
        f"{best_row['r2']:.6f}"
    )

    print(
        "\nModels registered in Hopsworks."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

