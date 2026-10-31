# SpamShield — SMS / Email Spam Classifier

An AI system that reads a short text message and classifies it as **spam** (unwanted / malicious) or **ham** (legitimate), using natural-language processing and supervised machine learning.

> CSC 425 — Artificial Intelligence · Term Project · Fall 2026 · Northern Kentucky University
> Instructor: Dr. Junxiu Zhou

---

## Team Members

| Member | Roles | Responsibilities |
|---|---|---|
| **Arjun Lekhak** (solo) | Data & ML lead, evaluation, documentation | Dataset acquisition and cleaning; preprocessing and baseline model; experiments and error analysis; repository maintenance; reports, demo, and final presentation. |

> This is an individual project. I have contacted the instructor to confirm that completing the term project solo is acceptable. Roles may be adjusted if a teammate joins later.

---

## Project Description

Spam messages waste time and are a common vehicle for phishing and scams. **SpamShield** tackles this as a binary **text-classification** problem.

- **Input:** a single raw text message (SMS or short email body)
- **AI task:** binary text classification (NLP + supervised ML)
- **Output:** a predicted label — `spam` or `ham` — with a confidence score

The project starts from a simple, strong baseline and improves it step by step.

### Planned approach
1. **Baseline** — TF-IDF features + Multinomial Naïve Bayes (and Logistic Regression for comparison), built with scikit-learn.
2. **Improvement** — feature engineering, Linear SVM, and ensemble models; hyperparameter tuning.
3. **Stretch goal** — fine-tune a lightweight transformer (e.g., DistilBERT) to benchmark deep learning against the classical pipeline.

### Evaluation
Because the data is class-imbalanced, the model is judged mainly on **precision, recall, F1-score**, and the **confusion matrix** — with special attention to keeping false positives (real messages flagged as spam) low — rather than on accuracy alone.

---

## Dataset / Resources

**Primary dataset — SMS Spam Collection (UCI Machine Learning Repository)**
- ~**5,574** labeled English text messages (~4,827 ham / ~747 spam)
- Two fields per record: `label` (ham/spam) and `message` (raw text)
- Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Download the dataset into `data/raw/` (the raw data is **not** committed — see `.gitignore`).

**Preprocessing:** lowercasing, punctuation/stop-word removal, tokenization, optional stemming/lemmatization, then TF-IDF vectorization; stratified train/test split to preserve the class ratio.

### References
1. Almeida, T. A., Gómez Hidalgo, J. M., & Silva, T. P. (2013). *Towards SMS Spam Filtering: Results under a New Dataset.* Int. J. of Information Security Science, 2(1), 1–18.
2. SMS Spam Collection Dataset — UCI ML Repository. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
3. scikit-learn — *Working With Text Data.* https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
4. scikit-learn — *Text feature extraction.* https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
5. NLTK — *Natural Language Toolkit.* https://www.nltk.org/

---

## Project Structure

```
csc425-spam-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # original dataset (not committed)
│   └── processed/    # cleaned / vectorized data
├── notebooks/
│   └── 01_explore.ipynb   # EDA and baseline experiments
├── src/
│   ├── preprocess.py # text cleaning + TF-IDF
│   ├── train.py      # train & save the baseline model
│   └── predict.py    # classify a new message from the command line
├── models/           # saved trained models (not committed)
└── reports/          # proposal, figures, final report
```

---

## Getting Started

```bash
# 1. clone and enter the repo
git clone https://github.com/<your-username>/csc425-spam-classifier.git
cd csc425-spam-classifier

# 2. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. place the dataset in data/raw/ , then:
python src/train.py                       # trains and saves the baseline model
python src/predict.py "WIN a free prize, click now!"   # -> spam / ham + confidence
```

## Timeline

| Period | Target |
|---|---|
| Now – Oct 11 | Define problem, set up data & GitHub (Milestone 1) |
| Oct 11 – 25 | Explore/preprocess data; build baseline |
| Oct 26 – Nov 15 | Develop & improve the AI approach |
| Nov 16 – 29 | Evaluate, analyze, refine |
| Nov 30 – Dec 11 | Finalize report, demo, presentation |

## License

Course project for CSC 425. Dataset © its original authors (see references).
