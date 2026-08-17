import os

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model("ev_battery_failure_model.keras")
scaler = joblib.load("ev_battery_scaler.pkl")

# Features used by the model
FEATURE_NAMES = [
    "battery_health_percent",
    "state_of_health",
    "charge_efficiency",
    "charging_quality_score",
    "voltage_imbalance",
    "odometer_km",
    "cycle_count",
    "cell_voltage_std",
    "aging_score",
    "capacity_loss_percent"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EV Battery Failure Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }

        h1 {
            text-align: center;
        }

        .description {
            text-align: center;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }

        input {
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
        }

        button {
            width: 100%;
            padding: 12px;
            margin-top: 15px;
            font-size: 16px;
            cursor: pointer;
        }

        .result {
            margin-top: 25px;
            padding: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8d7da;
        }
    </style>
</head>

<body>

<h1>EV Battery Failure Prediction</h1>

<p class="description">
Enter the current battery readings below to estimate battery failure risk.
</p>

<form method="POST" action="/predict">

    {% for feature in features %}
    <div class="form-group">
        <label for="{{ feature }}">{{ feature }}</label>
        <input
            type="number"
            step="any"
            name="{{ feature }}"
            id="{{ feature }}"
            required
        >
    </div>
    {% endfor %}

    <button type="submit">Predict Battery Failure Risk</button>

</form>

{% if result %}
<div class="result">
    {{ result }}
    <br>
    Failure Probability: {{ probability }}
</div>
{% endif %}

{% if error %}
<div class="error">
    Error: {{ error }}
</div>
{% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_NAMES,
        result=None,
        probability=None,
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from the web form
        input_data = [
            float(request.form[feature])
            for feature in FEATURE_NAMES
        ]

        input_array = np.array([input_data])

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_array)

        # Generate prediction probability
        probability = float(
            model.predict(input_scaled, verbose=0)[0][0]
        )

        # Use the same threshold used during evaluation
        prediction = int(probability >= 0.5)

        if prediction == 1:
            result = "Battery Failure Risk Detected"
        else:
            result = "No Battery Failure Risk Detected"

        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            result=result,
            probability=f"{probability * 100:.2f}%",
            error=None
        )

    except Exception as e:

        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            result=None,
            probability=None,
            error=str(e)
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json()

        missing_features = [
            feature
            for feature in FEATURE_NAMES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        input_data = np.array([
            [
                float(data[feature])
                for feature in FEATURE_NAMES
            ]
        ])

        input_scaled = scaler.transform(input_data)

        probability = float(
            model.predict(input_scaled, verbose=0)[0][0]
        )

        prediction = int(probability >= 0.5)

        result = (
            "Battery Failure Risk"
            if prediction == 1
            else "No Battery Failure Risk"
        )

        return jsonify({
            "prediction": prediction,
            "result": result,
            "failure_probability": round(probability, 4)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )
