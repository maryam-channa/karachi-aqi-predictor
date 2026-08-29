"""
AQI MODEL TRAINING PIPELINE v2

Production-oriented Karachi AQI forecasting training pipeline.

Data source:
    Hopsworks Feature View:
        aqi_prediction_view, version 3

Training dataset:
    Training Dataset version 1

Feature source:
    aqi_features, version 4

Model input:
    98 engineered features

Target:
    target_aqi = next-hour AQI

Models:
    1. Persistence Baseline
    2. Ridge Regression
    3. Random Forest
    4. Gradient Boosting
    5. TensorFlow MLP (if TensorFlow is installed)

Validation:
    Strict chronological 80/20 split.
    No random shuffling across train/test boundary.

Artifacts:
    models/ridge_model.pkl
    models/random_forest_model.pkl
    models/gradient_boosting_model.pkl
    models/tensorflow_model.keras
    models/tensorflow_preprocessing.pkl
    models/imputer.pkl
    models/feature_metadata.pkl
    models/training_metadata_v2.pkl
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

HOPSWORKS_HOST = os.getenv(
    "HOPSWORKS_HOST"
)

HOPSWORKS_PROJECT = os.getenv(
    "HOPSWORKS_PROJECT"
)

HOPSWORKS_API_KEY = os.getenv(
    "HOPSWORKS_API_KEY"
)


# ------------------------------------------------------------
# HOPSWORKS FEATURE VIEW
# ------------------------------------------------------------

FEATURE_VIEW_NAME = (
    "aqi_prediction_view"
)

FEATURE_VIEW_VERSION = 3


# ------------------------------------------------------------
# HOPSWORKS TRAINING DATASET
# ------------------------------------------------------------

TRAINING_DATASET_VERSION = 1


# ------------------------------------------------------------
# MODEL DIRECTORY
# ------------------------------------------------------------

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------

TRAIN_RATIO = 0.80

RANDOM_STATE = 42

TARGET_COLUMN = (
    "target_aqi"
)


# ------------------------------------------------------------
# NON-MODEL COLUMNS
# ------------------------------------------------------------

NON_MODEL_COLUMNS = {
    "timestamp",
    "aqi",
    "target_aqi",
    "id",
}


# ============================================================
# METRICS
# ============================================================

def rmse(
    y_true,
    y_pred
):
    """
    Calculate Root Mean Squared Error.
    """

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def evaluate(
    y_true,
    y_pred
):
    """
    Calculate regression metrics.
    """

    return {
        "rmse": rmse(
            y_true,
            y_pred
        ),

        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "r2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
    }


def print_metrics(
    name,
    metrics
):
    """
    Print model metrics in a readable format.
    """

    print(
        f"{name:<25}"
        f" RMSE={metrics['rmse']:.6f}"
        f" MAE={metrics['mae']:.6f}"
        f" R2={metrics['r2']:.6f}"
    )


# ============================================================
# HOPSWORKS CONNECTION
# ============================================================

def connect_hopsworks():
    """
    Connect to Hopsworks.
    """

    missing = []

    if not HOPSWORKS_HOST:
        missing.append(
            "HOPSWORKS_HOST"
        )

    if not HOPSWORKS_PROJECT:
        missing.append(
            "HOPSWORKS_PROJECT"
        )

    if not HOPSWORKS_API_KEY:
        missing.append(
            "HOPSWORKS_API_KEY"
        )

    if missing:

        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    print(
        "\nConnecting to Hopsworks..."
    )

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
    )

    print(
        "Hopsworks connection established."
    )

    return project


# ============================================================
# LOAD 98-FEATURE SCHEMA
# ============================================================

def load_feature_schema():
    """
    Load and validate the authoritative
    98-feature schema.
    """

    metadata_path = (
        MODEL_DIR
        / "feature_metadata.pkl"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            "Required feature metadata file was not found:\n"
            f"{metadata_path}"
        )

    with metadata_path.open(
        "rb"
    ) as handle:

        metadata = pickle.load(
            handle
        )

    if not isinstance(
        metadata,
        dict
    ):

        raise RuntimeError(
            "feature_metadata.pkl does not contain a dictionary."
        )

    feature_columns = list(
        metadata.get(
            "feature_columns",
            []
        )
    )

    print(
        f"Feature metadata contains "
        f"{len(feature_columns)} features."
    )

    if len(feature_columns) != 98:

        raise RuntimeError(
            "The project requires exactly 98 model features, "
            f"but feature_metadata.pkl contains "
            f"{len(feature_columns)}."
        )

    forbidden = [
        column
        for column in feature_columns
        if column in NON_MODEL_COLUMNS
    ]

    if forbidden:

        raise RuntimeError(
            "Forbidden columns detected in model feature schema: "
            + ", ".join(forbidden)
        )

    print(
        "Verified model schema: 98 features."
    )

    return feature_columns


# ============================================================
# LOAD HOPSWORKS TRAINING DATA
# ============================================================

def load_training_data(
    feature_columns
):
    """
    Load Training Dataset v1 from
    Feature View v3.

    IMPORTANT:
    The installed Hopsworks SDK requires
    training_dataset_version explicitly.
    """

    project = connect_hopsworks()

    fs = project.get_feature_store()

    print(
        "\n============================================================"
    )

    print(
        " HOPSWORKS TRAINING DATA"
    )

    print(
        "============================================================"
    )

    print(
        f"Feature View: "
        f"{FEATURE_VIEW_NAME} "
        f"v{FEATURE_VIEW_VERSION}"
    )

    print(
        f"Training Dataset Version: "
        f"{TRAINING_DATASET_VERSION}"
    )

    fv = fs.get_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
    )

    if fv is None:

        raise RuntimeError(
            "Could not retrieve the Hopsworks Feature View."
        )

    print(
        "\nReading training data from Hopsworks..."
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Your installed Hopsworks SDK requires:
    #
    # get_training_data(
    #     training_dataset_version=...
    # )
    #
    # Training Dataset v1 was successfully created.
    # --------------------------------------------------------

    try:

        X, y = fv.get_training_data(
            training_dataset_version=(
                TRAINING_DATASET_VERSION
            ),

            read_options={
                "arrow_flight_config": {
                    "timeout": 300
                }
            },
        )

    except Exception as e:

        raise RuntimeError(
            "Failed to read Hopsworks Training Dataset "
            f"v{TRAINING_DATASET_VERSION}: {e}"
        ) from e

    if X is None:

        raise RuntimeError(
            "Hopsworks returned X=None."
        )

    if y is None:

        raise RuntimeError(
            "Hopsworks returned y=None."
        )

    X = pd.DataFrame(
        X
    ).copy()

    y = pd.DataFrame(
        y
    ).copy()

    print(
        f"\nRaw X shape: {X.shape}"
    )

    print(
        f"Raw y shape: {y.shape}"
    )

    # --------------------------------------------------------
    # TARGET COLUMN
    # --------------------------------------------------------

    print(
        "\nTarget columns returned:"
    )

    print(
        list(y.columns)
    )

    if TARGET_COLUMN not in y.columns:

        if y.shape[1] == 1:

            y.columns = [
                TARGET_COLUMN
            ]

        else:

            raise RuntimeError(
                "Could not identify target_aqi in "
                "the Feature View training output."
            )

    # --------------------------------------------------------
    # FEATURE CHECK
    # --------------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in X.columns
    ]

    if missing_features:

        raise RuntimeError(
            "The Feature View training dataset is missing "
            "the following required model features:\n"
            + "\n".join(
                missing_features
            )
        )

    # --------------------------------------------------------
    # EXPLICITLY SELECT ONLY 98 FEATURES
    # --------------------------------------------------------

    X = X[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = y[
        TARGET_COLUMN
    ].copy()

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMOVE INVALID TARGET ROWS
    # --------------------------------------------------------

    valid_target = (
        y.notna()
        .to_numpy()
        .ravel()
    )

    X = X.loc[
        valid_target
    ].reset_index(
        drop=True
    )

    y = y.loc[
        valid_target
    ].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if len(X) != len(y):

        raise RuntimeError(
            "X/y row count mismatch after cleaning: "
            f"X={len(X)}, y={len(y)}"
        )

    if X.shape[1] != 98:

        raise RuntimeError(
            "Training matrix does not contain exactly "
            f"98 features. Found {X.shape[1]}."
        )

    if len(X) < 100:

        raise RuntimeError(
            "Too few training rows for a meaningful "
            f"time-series experiment: {len(X)}"
        )

    print(
        f"\nValidated training rows: {len(X)}"
    )

    print(
        f"Validated training features: {X.shape[1]}"
    )

    print(
        "98-feature training schema: PASSED"
    )

    # --------------------------------------------------------
    # NaN SUMMARY
    # --------------------------------------------------------

    total_missing = int(
        X.isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values before imputation: "
        f"{total_missing}"
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    return X, y


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    X,
    y
):
    """
    Strict chronological 80/20 split.

    No shuffle.
    """

    split_index = int(
        len(X)
        * TRAIN_RATIO
    )

    if split_index <= 0:

        raise RuntimeError(
            "Invalid chronological split."
        )

    if split_index >= len(X):

        raise RuntimeError(
            "Chronological split leaves no test rows."
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    print(
        "\n============================================================"
    )

    print(
        " CHRONOLOGICAL TRAIN / TEST SPLIT"
    )

    print(
        "============================================================"
    )

    print(
        f"Total rows:     {len(X)}"
    )

    print(
        f"Training rows:  {len(X_train)}"
    )

    print(
        f"Testing rows:   {len(X_test)}"
    )

    print(
        f"Train ratio:    {TRAIN_RATIO:.0%}"
    )

    print(
        "Shuffle:        DISABLED"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

def persistence_baseline(
    X_test,
    y_test
):
    """
    Persistence baseline.

    Predict next-hour AQI using the most
    recent known AQI value:

        prediction = aqi_lag_1h
    """

    if "aqi_lag_1h" not in X_test.columns:

        raise RuntimeError(
            "aqi_lag_1h is required for "
            "the persistence baseline."
        )

    prediction = pd.to_numeric(
        X_test[
            "aqi_lag_1h"
        ],
        errors="coerce"
    ).to_numpy()

    target = y_test.to_numpy()

    valid = (
        np.isfinite(
            prediction
        )
        &
        np.isfinite(
            target
        )
    )

    if not valid.any():

        raise RuntimeError(
            "Persistence baseline has no valid test predictions."
        )

    return evaluate(
        target[valid],
        prediction[valid],
    )


# ============================================================
# SCIKIT-LEARN MODELS
# ============================================================

def train_sklearn_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train Ridge, Random Forest,
    and Gradient Boosting.
    """

    results = {}

    fitted_models = {}

    # --------------------------------------------------------
    # IMPUTER
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test_imp = (
        imputer.transform(
            X_test
        )
    )

    with open(
        MODEL_DIR
        / "imputer.pkl",
        "wb"
    ) as handle:

        pickle.dump(
            imputer,
            handle
        )

    print(
        "\nMedian imputation fitted "
        "using training data only."
    )

    # ========================================================
    # RIDGE
    # ========================================================

    print(
        "\nTraining Ridge Regression..."
    )

    ridge = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                Ridge(
                    alpha=1.0
                )
            ),
        ]
    )

    ridge.fit(
        X_train_imp,
        y_train
    )

    ridge_pred = ridge.predict(
        X_test_imp
    )

    ridge_metrics = evaluate(
        y_test,
        ridge_pred
    )

    results[
        "Ridge Regression"
    ] = ridge_metrics

    fitted_models[
        "Ridge Regression"
    ] = ridge

    with open(
        MODEL_DIR
        / "ridge_model.pkl",
        "wb"
    ) as handle:

        pickle.dump(
            ridge,
            handle
        )

    print_metrics(
        "Ridge Regression",
        ridge_metrics
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    random_forest = (
        RandomForestRegressor(
            n_estimators=500,

            random_state=(
                RANDOM_STATE
            ),

            n_jobs=-1,

            max_features="sqrt",

            min_samples_leaf=1,
        )
    )

    random_forest.fit(
        X_train_imp,
        y_train
    )

    rf_pred = random_forest.predict(
        X_test_imp
    )

    rf_metrics = evaluate(
        y_test,
        rf_pred
    )

    results[
        "Random Forest"
    ] = rf_metrics

    fitted_models[
        "Random Forest"
    ] = random_forest

    with open(
        MODEL_DIR
        / "random_forest_model.pkl",
        "wb"
    ) as handle:

        pickle.dump(
            random_forest,
            handle
        )

    print_metrics(
        "Random Forest",
        rf_metrics
    )

    # ========================================================
    # GRADIENT BOOSTING
    # ========================================================

    print(
        "\nTraining Gradient Boosting..."
    )

    gradient_boosting = (
        GradientBoostingRegressor(
            n_estimators=300,

            learning_rate=0.03,

            max_depth=3,

            loss="squared_error",

            random_state=(
                RANDOM_STATE
            ),
        )
    )

    gradient_boosting.fit(
        X_train_imp,
        y_train
    )

    gbr_pred = (
        gradient_boosting.predict(
            X_test_imp
        )
    )

    gbr_metrics = evaluate(
        y_test,
        gbr_pred
    )

    results[
        "Gradient Boosting"
    ] = gbr_metrics

    fitted_models[
        "Gradient Boosting"
    ] = gradient_boosting

    with open(
        MODEL_DIR
        / "gradient_boosting_model.pkl",
        "wb"
    ) as handle:

        pickle.dump(
            gradient_boosting,
            handle
        )

    print_metrics(
        "Gradient Boosting",
        gbr_metrics
    )

    return (
        results,
        fitted_models,
        imputer,
    )


# ============================================================
# TENSORFLOW
# ============================================================

def try_train_tensorflow(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train TensorFlow MLP if TensorFlow
    is installed.
    """

    try:

        import tensorflow as tf

    except ImportError:

        print(
            "\nTensorFlow is not installed."
        )

        print(
            "TensorFlow MLP will be skipped."
        )

        return (
            None,
            None,
        )

    print(
        "\n============================================================"
    )

    print(
        " TENSORFLOW MLP"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        RANDOM_STATE
    )

    np.random.seed(
        RANDOM_STATE
    )

    tf.random.set_seed(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test_imp = (
        imputer.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_imp
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_imp
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    X_train_scaled.shape[1],
                )
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.15
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1
            ),
        ]
    )

    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=0.001
            )
        ),

        loss="mse",

        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",

            patience=20,

            restore_best_weights=True,
        )
    )

    print(
        "Training TensorFlow MLP..."
    )

    model.fit(
        X_train_scaled,

        y_train.to_numpy(),

        validation_split=0.15,

        epochs=300,

        batch_size=32,

        shuffle=False,

        callbacks=[
            early_stopping
        ],

        verbose=0,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = (
        model
        .predict(
            X_test_scaled,
            verbose=0
        )
        .reshape(-1)
    )

    metrics = evaluate(
        y_test,
        predictions
    )

    print_metrics(
        "TensorFlow MLP",
        metrics
    )

    # --------------------------------------------------------
    # Save TensorFlow model
    # --------------------------------------------------------

    model.save(
        MODEL_DIR
        / "tensorflow_model.keras"
    )

    # --------------------------------------------------------
    # Save preprocessing
    # --------------------------------------------------------

    with open(
        MODEL_DIR
        / "tensorflow_preprocessing.pkl",
        "wb"
    ) as handle:

        pickle.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
            },
            handle
        )

    return (
        metrics,
        model,
    )


# ============================================================
# VERIFY MODEL SCHEMA
# ============================================================

def verify_model_schema(
    model,
    feature_columns
):
    """
    Verify that the fitted model contains
    exactly the expected 98 features.
    """

    expected_count = len(feature_columns)

    if expected_count != 98:
        raise RuntimeError(
            f"Expected exactly 98 features, "
            f"but feature metadata contains {expected_count}."
        )

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None
    )

    if model_feature_count is None:
        raise RuntimeError(
            "Best model does not expose n_features_in_. "
            "Cannot verify the fitted feature count."
        )

    if int(model_feature_count) != expected_count:
        raise RuntimeError(
            "Best model feature count mismatch: "
            f"model={model_feature_count}, "
            f"expected={expected_count}."
        )

    if hasattr(model, "feature_names_in_"):

        fitted_features = list(
            model.feature_names_in_
        )

        if fitted_features != list(feature_columns):
            raise RuntimeError(
                "Best model feature order does not "
                "match feature_metadata.pkl."
            )

        print(
            "98-feature names and order verification: PASSED"
        )

    else:

        print(
            "Model does not expose feature_names_in_; "
            "verified n_features_in_=98 against "
            "the authoritative feature schema."
        )

    print(
        "98-feature model schema verification: PASSED"
    )


def save_metadata(
    feature_columns,
    results,
    best_model_name,
    best_metrics,
    training_rows,
    testing_rows,
):
    """
    Save training metadata.
    """

    training_metadata = {

        "feature_columns": (
            feature_columns
        ),

        "feature_count": (
            len(feature_columns)
        ),

        "target_column": (
            TARGET_COLUMN
        ),

        "feature_view": (
            FEATURE_VIEW_NAME
        ),

        "feature_view_version": (
            FEATURE_VIEW_VERSION
        ),

        "training_dataset_version": (
            TRAINING_DATASET_VERSION
        ),

        "training_rows": (
            int(training_rows)
        ),

        "testing_rows": (
            int(testing_rows)
        ),

        "train_ratio": (
            TRAIN_RATIO
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "models": (
            results
        ),

        "best_model": (
            best_model_name
        ),

        "best_model_metrics": (
            best_metrics
        ),

        "target_definition": (
            "AQI one hour ahead"
        ),

        "evaluation": (
            "Strict chronological 80/20 split"
        ),

        "shuffle": False,
    }

    metadata_path = (
        MODEL_DIR
        / "training_metadata_v2.pkl"
    )

    with metadata_path.open(
        "wb"
    ) as handle:

        pickle.dump(
            training_metadata,
            handle
        )

    # --------------------------------------------------------
    # Feature metadata
    # --------------------------------------------------------

    feature_metadata = {

        "feature_columns": (
            feature_columns
        ),

        "feature_count": (
            len(feature_columns)
        ),

        "target_column": (
            TARGET_COLUMN
        ),

        "feature_view": (
            FEATURE_VIEW_NAME
        ),

        "feature_view_version": (
            FEATURE_VIEW_VERSION
        ),

        "training_dataset_version": (
            TRAINING_DATASET_VERSION
        ),

        "best_model": (
            best_model_name
        ),

        "models": (
            results
        ),

        "training_rows": (
            int(training_rows)
        ),

        "testing_rows": (
            int(testing_rows)
        ),
    }

    feature_metadata_path = (
        MODEL_DIR
        / "feature_metadata.pkl"
    )

    with feature_metadata_path.open(
        "wb"
    ) as handle:

        pickle.dump(
            feature_metadata,
            handle
        )

    print(
        "\nTraining metadata saved:"
    )

    print(
        training_metadata
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        " KARACHI AQI MODEL TRAINING PIPELINE v2"
    )

    print(
        " 98 Features / Next-Hour Target / "
        "Chronological Evaluation"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    feature_columns = (
        load_feature_schema()
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X, y = (
        load_training_data(
            feature_columns
        )
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(
        X,
        y
    )

    # --------------------------------------------------------
    # Results dictionary
    # --------------------------------------------------------

    results = {}

    # --------------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        " PERSISTENCE BASELINE"
    )

    print(
        "============================================================"
    )

    persistence_metrics = (
        persistence_baseline(
            X_test,
            y_test
        )
    )

    results[
        "Persistence Baseline"
    ] = persistence_metrics

    print_metrics(
        "Persistence Baseline",
        persistence_metrics
    )

    # --------------------------------------------------------
    # sklearn models
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        " SCIKIT-LEARN MODELS"
    )

    print(
        "============================================================"
    )

    (
        sklearn_results,
        fitted_models,
        imputer,
    ) = train_sklearn_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results.update(
        sklearn_results
    )

    # --------------------------------------------------------
    # TensorFlow
    # --------------------------------------------------------

    (
        tensorflow_metrics,
        tensorflow_model,
    ) = try_train_tensorflow(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    if tensorflow_metrics is not None:

        results[
            "TensorFlow MLP"
        ] = tensorflow_metrics

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = (
        pd.DataFrame
        .from_dict(
            results,
            orient="index"
        )
        .reset_index()
        .rename(
            columns={
                "index": "model"
            }
        )
        .sort_values(
            by="rmse",
            ascending=True
        )
    )

    print(
        "\n============================================================"
    )

    print(
        " MODEL COMPARISON"
    )

    print(
        "============================================================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select best fitted sklearn model
    #
    # Persistence is a baseline, not a trainable artifact.
    # TensorFlow is stored separately as Keras.
    # --------------------------------------------------------

    fitted_results = {}

    for name in fitted_models:

        if name in results:

            fitted_results[
                name
            ] = results[
                name
            ]

    if not fitted_results:

        raise RuntimeError(
            "No fitted sklearn models are available."
        )

    best_model_name = min(
        fitted_results,

        key=lambda name:
        fitted_results[
            name
        ][
            "rmse"
        ]
    )

    best_model = (
        fitted_models[
            best_model_name
        ]
    )

    best_metrics = (
        fitted_results[
            best_model_name
        ]
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        " BEST FITTED SKLEARN MODEL"
    )

    print(
        "============================================================"
    )

    print(
        f"Model: {best_model_name}"
    )

    print(
        f"RMSE:  {best_metrics['rmse']:.6f}"
    )

    print(
        f"MAE:   {best_metrics['mae']:.6f}"
    )

    print(
        f"R2:    {best_metrics['r2']:.6f}"
    )

    # --------------------------------------------------------
    # Schema verification
    # --------------------------------------------------------

    verify_model_schema(
        best_model,
        feature_columns
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        feature_columns=(
            feature_columns
        ),

        results=(
            results
        ),

        best_model_name=(
            best_model_name
        ),

        best_metrics=(
            best_metrics
        ),

        training_rows=(
            len(X_train)
        ),

        testing_rows=(
            len(X_test)
        ),
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        " TRAINING COMPLETE"
    )

    print(
        "============================================================"
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
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print(
        f"Feature View: "
        f"{FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}"
    )

    print(
        f"Training Dataset: "
        f"v{TRAINING_DATASET_VERSION}"
    )

    print(
        f"Best sklearn model: "
        f"{best_model_name}"
    )

    print(
        "\nSaved artifacts:"
    )

    print(
        "  models/ridge_model.pkl"
    )

    print(
        "  models/random_forest_model.pkl"
    )

    print(
        "  models/gradient_boosting_model.pkl"
    )

    print(
        "  models/imputer.pkl"
    )

    print(
        "  models/feature_metadata.pkl"
    )

    print(
        "  models/training_metadata_v2.pkl"
    )

    if tensorflow_model is not None:

        print(
            "  models/tensorflow_model.keras"
        )

        print(
            "  models/tensorflow_preprocessing.pkl"
        )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Review the model metrics before registering "
        "the winning model in Hopsworks Model Registry."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()