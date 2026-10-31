"""
train.py — train the baseline spam classifier (TF-IDF + Multinomial Naive Bayes)
and report precision / recall / F1 and a confusion matrix.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import prepare

MODEL_PATH = "models/baseline_nb.joblib"


def main():
    df = prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"],
        test_size=0.20, random_state=42, stratify=df["target"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Classification report (0=ham, 1=spam):\n")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
