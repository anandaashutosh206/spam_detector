"""
Spam Mail Detector  -  AI/ML Internship Task (QSkill)
=====================================================
Classifies SMS/email messages as SPAM or HAM using classic NLP + ML.

Pipeline:
  1. Load dataset (SMS Spam Collection)
  2. Preprocess text  (lowercase, remove punctuation/numbers, tokenize, remove stopwords)
  3. Feature extraction (TF-IDF, with unigrams + bigrams)
  4. Train/test split (80/20, stratified)
  5. Train Naive Bayes and Logistic Regression
  6. Evaluate (accuracy, precision, recall, F1, confusion matrix)
  7. Save best model + predict on new messages

Usage:
  python spam_detector.py                     # train, evaluate, save model
  python spam_detector.py --predict "text"    # classify a message with saved model
  python spam_detector.py --interactive       # type messages and get predictions
"""

import argparse
import os
import re
import sys
import urllib.request

import joblib
import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "SMSSpamCollection.tsv")
MODEL_PATH = "spam_model.joblib"
OUTPUT_DIR = "outputs"
DATA_URLS = [
    # Mirror of the UCI SMS Spam Collection (tab separated: label \t message)
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
]
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    for url in DATA_URLS:
        try:
            print(f"[info] Downloading dataset from {url}")
            urllib.request.urlretrieve(url, DATA_PATH)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Failed: {e}")
    sys.exit(
        "Could not download the dataset. Download the SMS Spam Collection from\n"
        "https://archive.ics.uci.edu/dataset/228/sms+spam+collection and save a\n"
        f"tab-separated file (label<TAB>message) as {DATA_PATH}"
    )


def load_data():
    if not os.path.exists(DATA_PATH):
        download_dataset()
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"],
                     encoding="latin-1")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# ----------------------------------------------------------------------------
# 2. Preprocess text
# ----------------------------------------------------------------------------
def _load_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:  # NLTK data missing -> try download, else fall back
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except Exception:
            return set(ENGLISH_STOP_WORDS)


STOPWORDS = _load_stopwords()


def preprocess(text: str) -> str:
    """lowercase -> strip urls/numbers/punctuation -> tokenize -> drop stopwords."""
    text = str(text).lower()                       # lowercasing
    text = re.sub(r"http\S+|www\.\S+", " url ", text)   # keep a 'url' signal
    text = re.sub(r"\d+", " ", text)                # remove numbers
    text = re.sub(r"[^a-z\s]", " ", text)           # remove punctuation/symbols
    tokens = text.split()                           # tokenization
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]  # stopwords
    return " ".join(tokens)


# ----------------------------------------------------------------------------
# 3-6. Build, train, evaluate
# ----------------------------------------------------------------------------
def build_models():
    tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)  # noqa: E731
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, C=10, class_weight="balanced")),
        ]),
    }


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))
    return metrics, confusion_matrix(y_test, y_pred)


def plot_confusion(cm, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_class_distribution(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    ax = sns.countplot(x="label", data=df, hue="label", palette=["#4C9F70", "#D9534F"], legend=False)
    for c in ax.containers:
        ax.bar_label(c)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def top_spam_words(model, n=15):
    """Words most indicative of spam (from the Logistic Regression coefficients)."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = vec.get_feature_names_out()
    top = clf.coef_[0].argsort()[-n:][::-1]
    return [names[i] for i in top]


def train():
    df = load_data()
    print(f"[info] Loaded {len(df)} messages")
    print(df["label"].value_counts().to_string())
    plot_class_distribution(df)

    print("\n[info] Preprocessing text ...")
    df["clean"] = df["message"].apply(preprocess)
    print("Example:")
    print("  original :", df['message'][2][:100])
    print("  cleaned  :", df['clean'][2][:100])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label_num"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label_num"])
    print(f"\n[info] Train size: {len(X_train)} | Test size: {len(X_test)}")

    results, trained = [], {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics, cm = evaluate(name, model, X_test, y_test)
        plot_confusion(cm, name)
        results.append(metrics)
        trained[name] = model

    summary = pd.DataFrame(results).set_index("Model").round(4)
    print("\n===== Model comparison =====")
    print(summary.to_string())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, "results.csv"))

    best_name = summary["F1"].idxmax()
    print(f"\n[info] Best model by F1: {best_name}")
    joblib.dump(trained[best_name], MODEL_PATH)
    print(f"[info] Saved model -> {MODEL_PATH}")

    if "Logistic Regression" in trained:
        print("\nTop spam-indicating words:", ", ".join(top_spam_words(trained["Logistic Regression"])))

    demo = [
        "Congratulations! You've won a FREE iPhone. Click http://claim-now.com to claim your prize!",
        "URGENT! Your account has been suspended. Call 09061234567 now to avoid charges",
        "Hey, are we still meeting for lunch tomorrow at 1?",
        "Can you send me the notes from today's class?",
    ]
    print("\n===== Demo predictions =====")
    for m in demo:
        label, prob = predict(m, trained[best_name])
        print(f"[{label.upper():4}] ({prob:.1%} spam)  {m}")


# ----------------------------------------------------------------------------
# 7. Predict
# ----------------------------------------------------------------------------
def predict(message, model=None):
    if model is None:
        if not os.path.exists(MODEL_PATH):
            sys.exit("No saved model found. Run `python spam_detector.py` first.")
        model = joblib.load(MODEL_PATH)
    clean = preprocess(message)
    prob = model.predict_proba([clean])[0][1]
    return ("spam" if prob >= 0.5 else "ham"), prob


def interactive():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if model is None:
        sys.exit("No saved model found. Run `python spam_detector.py` first.")
    print("Type a message to classify (empty line to quit).")
    while True:
        msg = input("\n> ").strip()
        if not msg:
            break
        label, prob = predict(msg, model)
        print(f"  => {label.upper()}  (spam probability: {prob:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spam Mail Detector")
    ap.add_argument("--predict", type=str, help="classify a single message")
    ap.add_argument("--interactive", action="store_true", help="interactive mode")
    args = ap.parse_args()

    if args.predict:
        lbl, p = predict(args.predict)
        print(f"{lbl.upper()} (spam probability: {p:.1%})")
    elif args.interactive:
        interactive()
    else:
        train()
