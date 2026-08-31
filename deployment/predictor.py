import os

import joblib
import numpy as np
import pandas as pd


class Predictor:

    def __init__(self):
        model_dir = (
            os.environ.get("MODEL_FILES_PATH")
            or os.environ.get("ARTIFACT_FILES_PATH")
            or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
            )
        )

        model_path = os.path.join(
            model_dir,
            "random_forest_model.pkl",
        )

        metadata_path = os.path.join(
            model_dir,
            "feature_metadata.pkl",
        )

        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)

        self.feature_columns = self.metadata.get(
            "feature_columns",
            [],
        )

        if len(self.feature_columns) != 98:
            raise RuntimeError(
                "Expected exactly 98 feature columns in metadata, "
                f"got {len(self.feature_columns)}"
            )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if model_feature_count != 98:
            raise RuntimeError(
                f"Expected 98 model features, "
                f"got {model_feature_count}"
            )

        print(
            "Karachi AQI Random Forest loaded successfully."
        )

    def predict(self, inputs):

        if isinstance(inputs, dict):
            if "instances" in inputs:
                inputs = inputs["instances"]
            elif "inputs" in inputs:
                inputs = inputs["inputs"]

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()

        elif isinstance(inputs, dict):
            data = pd.DataFrame([inputs])

        else:
            data = pd.DataFrame(inputs)

        # Remove metadata/target columns that are not model features.
        forbidden = [
            "timestamp",
            "aqi",
            "target_aqi",
            "id",
            "current_aqi",
        ]

        data = data.drop(
            columns=[
                column
                for column in forbidden
                if column in data.columns
            ],
            errors="ignore",
        )

        missing_features = [
            column
            for column in self.feature_columns
            if column not in data.columns
        ]

        # When named features are supplied, enforce the trained schema.
        # For a raw 98-value array, preserve the supplied order.
        if missing_features and data.shape[1] != 98:
            raise ValueError(
                "Prediction input is missing model features: "
                + ", ".join(missing_features)
            )

        if data.shape[1] != 98:
            raise ValueError(
                f"Expected exactly 98 input features, "
                f"received {data.shape[1]}"
            )

        if not missing_features:
            data = data[
                self.feature_columns
            ]

        data = data.apply(
            pd.to_numeric,
            errors="coerce",
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if data.isna().any().any():
            missing_values = data.columns[
                data.isna().any()
            ].tolist()

            raise ValueError(
                "Prediction input contains missing or "
                "non-numeric feature values in: "
                + ", ".join(missing_values)
            )

        # The model was fitted without sklearn feature names.
        # Pass a NumPy array to avoid the feature-name warning.
        prediction = self.model.predict(
            data.to_numpy()
        )

        return prediction.tolist()
