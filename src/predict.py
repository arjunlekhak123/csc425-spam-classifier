"""
predict.py — classify a new message from the command line.

Usage:
    python src/predict.py "Congratulations! You won a free prize, click here"
"""
import sys
import joblib
from preprocess import clean_text

MODEL_PATH = "models/baseline_nb.joblib"


def predict(message: str):
    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(message)
    label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0].max()
    return ("spam" if label == 1 else "ham"), proba


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your message here"')
        sys.exit(1)
    msg = " ".join(sys.argv[1:])
    label, conf = predict(msg)
    print(f"Prediction: {label}  (confidence {conf:.1%})")
