"""
preprocess.py — load the SMS Spam Collection and clean the text.

Expected raw file: data/raw/SMSSpamCollection  (tab-separated: label <TAB> message)
Download from: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
"""
import re
import string
import pandas as pd

RAW_PATH = "data/raw/SMSSpamCollection"


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the tab-separated dataset into a DataFrame with columns: label, message."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["target"] = (df["label"] == "spam").astype(int)  # spam=1, ham=0
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers, and remove punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare(path: str = RAW_PATH) -> pd.DataFrame:
    df = load_data(path)
    df["clean"] = df["message"].apply(clean_text)
    return df


if __name__ == "__main__":
    df = prepare()
    print(df.head())
    print("\nClass balance:\n", df["label"].value_counts())
