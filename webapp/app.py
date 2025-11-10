from flask import Flask, render_template, request
import time
from prometheus_client import Counter, Histogram, start_http_server
from panelClassification.pipeline.predict import PredictionPipeline
from pathlib import Path

app = Flask(__name__)

# ---- Project + classes ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]

# ---- Prometheus ----
start_http_server(9101)
PREDICTIONS = Counter("predictions_total", "Total number of predictions")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Latency of predictions")
ERRORS = Counter("prediction_errors_total", "Total prediction errors")
BY_CLASS = Counter("predictions_by_class_total", "Predictions by class label", ["label"])

# ---- Cache model to avoid reloading ----
model_cache = {}

def get_model(family):
    # Return cached if already loaded
    if family in model_cache:
        return model_cache[family]

    # Build expected path for this family
    model_path = PROJECT_ROOT / "artifacts" / "training" / family / "model.keras"

    # If file/dir doesn't exist → signal missing model
    if not model_path.exists():
        return None

    # Otherwise load and cache
    model_cache[family] = PredictionPipeline(
        model_path=model_path,
        model_family=family,
        class_names=CLASS_NAMES
    )
    return model_cache[family]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files.get("image")
        if not img_file:
            return render_template("index.html", error="No image uploaded.")

        img_bytes = img_file.read()
        family = request.form.get("family", "auto")
        top_k = int(request.form.get("top_k", 5))

        pipeline = get_model(family)

        # If model not available for this family, show note in UI
        if pipeline is None:
            ERRORS.inc()
            return render_template(
                "index.html",
                error=f"Model for '{family}' is not trained or missing yet. Please select another model."
            )

        t0 = time.time()
        # Always return only top-1
        result = pipeline.predict_from_bytes(img_bytes, top_k=1)
        INFERENCE_LATENCY.observe(time.time() - t0)
        PREDICTIONS.inc()
        BY_CLASS.labels(result["top1"]["label"]).inc()

        # Keep only top1 info for UI
        result = {"top1": result["top1"]}
        return render_template("index.html", result=result)


    except Exception as e:
        ERRORS.inc()
        return render_template("index.html", error=str(e))


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
