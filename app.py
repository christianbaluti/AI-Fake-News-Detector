from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins on the /predict route

# In-memory log storage
logs = []

# Helper function for logging
def log_message(level, message):
    log_entry = {"level": level, "message": message}
    logs.append(log_entry)
    print(f"[{level}] {message}")

# Load model and tokenizer
try:
    model_path = "fake-news-detector"  # Path to the saved model directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message("INFO", f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    log_message("INFO", "Model and tokenizer loaded successfully.")
except Exception as e:
    log_message("ERROR", f"Error loading model or tokenizer: {e}")
    raise e

@app.route("/", methods=["GET"])
def home():
    """Home route to test API is running."""
    log_message("INFO", "Accessed home route.")
    return jsonify({"message": "Fake News Detector API is running."}), 200

@app.route("/logs", methods=["GET"])
def get_logs():
    """Route to fetch logs."""
    log_message("INFO", "Accessed logs route.")
    return jsonify(logs), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction route to classify input text."""
    try:
        log_message("INFO", "Received a POST request to /predict.")
        # Get input text from the POST request
        data = request.json
        if not data or "text" not in data:
            log_message("WARNING", "No 'text' field in the request JSON.")
            return jsonify({"error": "Request must contain a 'text' field."}), 400

        text = data.get("text", "")
        log_message("INFO", f"Input text: {text}")

        # Tokenize input
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
        log_message("DEBUG", f"Tokenized input: {inputs}")

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        if predicted_class==1:
            predicted_class = "True"
        if predicted_class==0:
            predicted_class = "Very Doubtful"
        
        log_message("INFO", f"Prediction: {predicted_class}")

        # Return result
        result = {"prediction": str(predicted_class)}
        log_message("INFO", f"Response: {result}")
        return jsonify(result), 200
    except Exception as e:
        log_message("ERROR", f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    log_message("INFO", "Starting Flask app.")
    app.run(debug=True, host='0.0.0.0', port=5000)  # Ensure it's publicly accessible
