from pathlib import Path
import joblib

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory
)


# Project paths
ROOT = Path(__file__).resolve().parent.parent

BACKEND = ROOT / "backend"
GRAPHS = ROOT / "graphs"

TEMPLATES = ROOT / "frontend" / "templates"
STATIC = ROOT / "frontend" / "static"


# Flask application
app = Flask(
    __name__,
    template_folder=str(TEMPLATES),
    static_folder=str(STATIC)
)


# Load trained models
def load_artifacts():

    model_logreg_path = BACKEND / "model_logreg.pkl"
    model_xgb_path = BACKEND / "model_xgb.pkl"
    vectorizer_path = BACKEND / "vectorizer.pkl"

    required_files = [
        model_logreg_path,
        model_xgb_path,
        vectorizer_path
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Trained model files are missing: "
            + ", ".join(missing_files)
            + ". Run python train_models.py first."
        )

    logreg_model = joblib.load(model_logreg_path)
    xgb_model = joblib.load(model_xgb_path)
    vectorizer = joblib.load(vectorizer_path)

    return logreg_model, xgb_model, vectorizer


logreg_model, xgb_model, vectorizer = load_artifacts()


# Load model accuracies
def read_accuracy():

    accuracy_file = BACKEND / "model_accuracy.txt"

    values = {
        "logistic_regression": "Not available",
        "xgboost": "Not available"
    }

    if accuracy_file.exists():

        lines = accuracy_file.read_text(
            encoding="utf-8"
        ).splitlines()

        try:
            values["logistic_regression"] = (
                f"{float(lines[0].split(':')[-1].strip()):.2%}"
            )

            values["xgboost"] = (
                f"{float(lines[1].split(':')[-1].strip()):.2%}"
            )

        except (IndexError, ValueError):
            pass

    return values


accuracies = read_accuracy()


# Home page
@app.route("/")
def home():

    return render_template("index.html")


# Serve graphs from the root graphs folder
@app.route("/graphs/<path:filename>")
def graph_file(filename):

    return send_from_directory(
        str(GRAPHS),
        filename
    )


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    input_text = request.form.get("text", "")

    # Also allow JSON requests
    if not input_text and request.is_json:

        payload = request.get_json(silent=True) or {}
        input_text = payload.get("text", "")

    if not isinstance(input_text, str):
        input_text = ""

    input_text = input_text.strip()

    if not input_text:

        return jsonify({
            "error": "Please enter some text before predicting."
        }), 400

    # Transform input text using saved TF-IDF vectorizer
    transformed_text = vectorizer.transform([
        input_text
    ])

    # Generate predictions
    logreg_prediction = int(
        logreg_model.predict(transformed_text)[0]
    )

    xgb_prediction = int(
        xgb_model.predict(transformed_text)[0]
    )

    # Get Hate class probabilities
    logreg_probability = None
    xgb_probability = None

    if hasattr(logreg_model, "predict_proba"):

        logreg_probability = float(
            logreg_model.predict_proba(transformed_text)[0][1]
        )

    if hasattr(xgb_model, "predict_proba"):

        xgb_probability = float(
            xgb_model.predict_proba(transformed_text)[0][1]
        )

    # Format results
    def prediction_result(
        prediction,
        probability,
        accuracy
    ):

        return {
            "prediction": (
                "Hate Speech"
                if prediction == 1
                else "Not Hate Speech"
            ),

            "hate_probability": (
                f"{probability:.2%}"
                if probability is not None
                else "Not available"
            ),

            "accuracy": accuracy
        }

    result = {

        "input_text": input_text,

        "logistic_regression": prediction_result(
            logreg_prediction,
            logreg_probability,
            accuracies["logistic_regression"]
        ),

        "xgboost": prediction_result(
            xgb_prediction,
            xgb_probability,
            accuracies["xgboost"]
        )
    }

    return jsonify(result)


# Run Flask server
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
