from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Predictor:
    """Local inference wrapper for the final 163-feature recursive AQI model.

    This API-side predictor intentionally does not depend on Hopsworks Model
    Serving. It loads the verified local model artifacts committed with the
    project and enforces the exact trained feature schema.
    """

    EXPECTED_FEATURE_COUNT = 163

    def __init__(self):
        project_root = Path(
            os.environ.get(
                "MODEL_FILES_PATH",
                Path(__file__).resolve().parents[1] / "models" / "recursive_72h",
            )
        )

        if project_root.is_file():
            raise RuntimeError(
                f"MODEL_FILES_PATH points to a file, expected a model directory: {project_root}"
            )

        self.model_path = project_root / "recursive_random_forest_compressed.pkl"
        self.imputer_path = project_root / "imputer.pkl"
        self.metadata_path = project_root / "metadata.pkl"

        missing = [
            str(path)
            for path in (
                self.model_path,
                self.imputer_path,
                self.metadata_path,
            )
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing final recursive model artifact(s): "
                + ", ".join(missing)
            )

        self.model = joblib.load(self.model_path)
        self.imputer = joblib.load(self.imputer_path)
        self.metadata = joblib.load(self.metadata_path)

        self.feature_columns = list(
            self.metadata.get("feature_columns", [])
        )

        if len(self.feature_columns) != self.EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Expected exactly 163 feature columns in metadata, "
                f"got {len(self.feature_columns)}"
            )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if model_feature_count != self.EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Expected exactly 163 model features, "
                f"got {model_feature_count}"
            )

        print(
            "Karachi AQI Recursive Random Forest loaded successfully "
            f"({self.EXPECTED_FEATURE_COUNT} features).",
            flush=True,
        )

    def _normalise_input(self, inputs) -> pd.DataFrame:
        """Convert supported JSON/pandas/list inputs into a feature dataframe."""

        if isinstance(inputs, dict):
            if "instances" in inputs:
                inputs = inputs["instances"]
            elif "inputs" in inputs:
                inputs = inputs["inputs"]
            else:
                # A named single-row feature mapping.
                inputs = [inputs]

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()
        elif isinstance(inputs, dict):
            data = pd.DataFrame([inputs])
        else:
            data = pd.DataFrame(inputs)

        if data.empty:
            raise ValueError("Prediction input is empty.")

        return data

    def predict(self, inputs):
        data = self._normalise_input(inputs)

        # Named-feature input: enforce the exact trained order.
        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in data.columns
        ]

        if not missing_features:
            data = data[self.feature_columns]
        elif data.shape[1] == self.EXPECTED_FEATURE_COUNT:
            # Raw 163-value arrays are accepted in supplied order.
            pass
        else:
            raise ValueError(
                "Prediction input is missing model features: "
                + ", ".join(missing_features)
            )

        if data.shape[1] != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected exactly {self.EXPECTED_FEATURE_COUNT} input features, "
                f"received {data.shape[1]}"
            )

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)

        if data.isna().any().any():
            bad_columns = data.columns[data.isna().any()].tolist()
            raise ValueError(
                "Prediction input contains missing or non-numeric values in: "
                + ", ".join(map(str, bad_columns))
            )

        transformed = self.imputer.transform(
            pd.DataFrame(data, columns=self.feature_columns)
        )

        prediction = self.model.predict(transformed)
        prediction = np.clip(
            np.asarray(prediction, dtype=float),
            0,
            500,
        )

        return prediction.tolist()
