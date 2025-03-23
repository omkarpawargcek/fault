from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "AI-Based Fault Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({"message": "Prediction endpoint is working!"})

if __name__ == "__main__":
    app.run(debug=True)
