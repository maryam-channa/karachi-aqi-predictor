import os
import joblib
import pandas as pd
import numpy as np


class Predictor:

    def __init__(self):
        model_dir = (
            os.environ.get("MODEL_FILES_PATH")
            or os.environ.get("ARTIFACT_FILES_PATH")
            or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        )

        model_path = os.path.join(
            model_dir,
            "gradient_boosting_model.pkl"
        )

        self.model = joblib.load(model_path)

        if getattr(self.model, "n_features_in_", None) != 98:
            raise RuntimeError(
                f"Expected 98 model features, "
                f"got {getattr(self.model, 'n_features_in_', None)}"
            )

        print(
            "Karachi AQI Gradient Boosting loaded successfully."
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

        # If the request contains the 4 non-ML columns,
        # remove them before prediction.
        forbidden = [
            "timestamp",
            "aqi",
            "target_aqi",
            "id"
        ]

        data = data.drop(
            columns=[
                c for c in forbidden
                if c in data.columns
            ],
            errors="ignore"
        )

        if data.shape[1] != 98:
            raise ValueError(
                f"Expected exactly 98 input features, "
                f"received {data.shape[1]}"
            )

        data = data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if data.isna().any().any():
            raise ValueError(
                "Prediction input contains missing or "
                "non-numeric feature values."
            )

        prediction = self.model.predict(data)

        return prediction.tolist()


