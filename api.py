from flask import Flask, request, jsonify
from deployment.predictor import Predictor

app = Flask(__name__)

predictor = Predictor()


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Karachi AQI Prediction API"
    })


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify({
                "error": "Request body must contain valid JSON."
            }), 400

        prediction = predictor.predict(payload)

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
