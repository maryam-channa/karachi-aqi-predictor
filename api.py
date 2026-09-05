from flask import Flask, jsonify, request

from deployment.predictor import Predictor

app = Flask(__name__)

predictor = Predictor()


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "Karachi AQI Prediction API",
            "model": "Recursive Random Forest",
            "features": 163,
            "scale": "0-500",
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify(
            {
                "error": "Request body must contain valid JSON."
            }
        ), 400

    try:
        prediction = predictor.predict(payload)

        return jsonify(
            {
                "prediction": prediction,
                "model": "Recursive Random Forest",
                "features": 163,
                "scale": "0-500",
            }
        )

    except ValueError as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 400

    except Exception as exc:
        app.logger.exception("Prediction service failed.")
        return jsonify(
            {
                "error": f"Prediction service failed: {exc}"
            }
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "5000")),
        debug=False,
    )
