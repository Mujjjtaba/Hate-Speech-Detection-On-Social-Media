from pathlib import Path
import html
import re
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BACKEND = ROOT / "backend"
GRAPHS = ROOT / "graphs"

BACKEND.mkdir(exist_ok=True)
GRAPHS.mkdir(exist_ok=True)

ENTRIES_FILE = (
    DATA / "2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n====================================")
    print("HATE SPEECH DETECTION PROJECT")
    print("====================================\n")

    # Load the entries CSV. Its label column already contains
    # the binary classes: hate and nothate.
    print("Loading dataset...")

    if not ENTRIES_FILE.exists():
        raise FileNotFoundError(
            "Entries CSV not found inside the data folder."
        )

    df = pd.read_csv(ENTRIES_FILE, low_memory=False)

    required_columns = ["id", "text", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")

    df = df[["id", "text", "label"]].copy()

    print("Original dataset records:", len(df))

    # Data cleaning
    print("\nCleaning dataset...")

    df = df.dropna(subset=["text", "label"]).copy()

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[df["label"].isin(["hate", "nothate"])].copy()

    df["text"] = df["text"].map(clean_text)

    df = df[df["text"].str.len() > 0].copy()

    df = df.drop_duplicates(
        subset=["text", "label"]
    ).copy()

    # Exclude identical text assigned to conflicting classes.
    conflicting_labels = df.groupby("text")["label"].nunique()
    conflicting_texts = conflicting_labels[
        conflicting_labels > 1
    ].index

    df = df[~df["text"].isin(conflicting_texts)].copy()

    # Binary encoding: Hate = 1, Not Hate = 0.
    df["label"] = df["label"].map({
        "hate": 1,
        "nothate": 0
    })

    if df.empty or df["label"].nunique() != 2:
        raise ValueError(
            "Both Hate and Not Hate classes must be present."
        )

    print("Records after cleaning:", len(df))

    print("\nFinal label distribution:")
    print(
        df["label"].value_counts().rename({
            0: "Not Hate",
            1: "Hate"
        })
    )

    # Stratified 80:20 train-test split
    print("\nSplitting dataset...")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    print("Training records:", len(X_train_text))
    print("Testing records:", len(X_test_text))

    # TF-IDF: fit only on training text to avoid data leakage.
    print("\nApplying TF-IDF vectorization...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("TF-IDF features:", X_train.shape[1])

    # Logistic Regression
    print("\nTraining Logistic Regression...")

    logreg = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    start_time = time.perf_counter()
    logreg.fit(X_train, y_train)
    logreg_training_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    logreg_predictions = logreg.predict(X_test)
    logreg_prediction_time = time.perf_counter() - start_time

    print("Logistic Regression completed.")

    # XGBoost
    print("\nTraining XGBoost...")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    start_time = time.perf_counter()
    xgb.fit(X_train, y_train)
    xgb_training_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    xgb_predictions = xgb.predict(X_test)
    xgb_prediction_time = time.perf_counter() - start_time

    print("XGBoost completed.")

    # Model evaluation
    print("\nEvaluating models...")

    models = {
        "Logistic Regression": logreg_predictions,
        "XGBoost": xgb_predictions
    }

    training_times = {
        "Logistic Regression": logreg_training_time,
        "XGBoost": xgb_training_time
    }

    prediction_times = {
        "Logistic Regression": logreg_prediction_time,
        "XGBoost": xgb_prediction_time
    }

    results = {}

    for model_name, predictions in models.items():

        results[model_name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Training Time": training_times[model_name],
            "Prediction Time": prediction_times[model_name]
        }

    logreg_probability = logreg.predict_proba(X_test)[:, 1]
    xgb_probability = xgb.predict_proba(X_test)[:, 1]

    results["Logistic Regression"]["ROC AUC"] = roc_auc_score(
        y_test, logreg_probability
    )

    results["XGBoost"]["ROC AUC"] = roc_auc_score(
        y_test, xgb_probability
    )

    results_df = pd.DataFrame(results).T

    print("\nMODEL PERFORMANCE COMPARISON")
    print(results_df.round(4))

    # Save trained models and vectorizer
    print("\nSaving trained models...")

    joblib.dump(logreg, BACKEND / "model_logreg.pkl")
    joblib.dump(xgb, BACKEND / "model_xgb.pkl")
    joblib.dump(vectorizer, BACKEND / "vectorizer.pkl")

    # Save accuracy values
    with open(
        BACKEND / "model_accuracy.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f"Logistic Regression Accuracy: "
            f"{accuracy_score(y_test, logreg_predictions):.4f}\n"
        )
        file.write(
            f"XGBoost Accuracy: "
            f"{accuracy_score(y_test, xgb_predictions):.4f}\n"
        )

    # Save all evaluation metrics
    results_df.to_csv(BACKEND / "model_metrics.csv")

    # Save classification reports
    with open(
        BACKEND / "classification_reports.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("LOGISTIC REGRESSION REPORT\n\n")
        file.write(
            classification_report(
                y_test,
                logreg_predictions,
                target_names=["Not Hate", "Hate"],
                zero_division=0
            )
        )

        file.write("\n\nXGBOOST REPORT\n\n")
        file.write(
            classification_report(
                y_test,
                xgb_predictions,
                target_names=["Not Hate", "Hate"],
                zero_division=0
            )
        )

    # Label distribution graph
    print("\nGenerating visualizations...")

    plt.figure(figsize=(7, 5))

    sns.countplot(
        x=df["label"].map({
            0: "Not Hate",
            1: "Hate"
        })
    )

    plt.title("Hate Speech Label Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(GRAPHS / "label_distribution.png")
    plt.close()

    # Model accuracy graph
    plt.figure(figsize=(7, 5))

    sns.barplot(
        x=list(results.keys()),
        y=[
            results["Logistic Regression"]["Accuracy"],
            results["XGBoost"]["Accuracy"]
        ]
    )

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(GRAPHS / "model_accuracy.png")
    plt.close()

    # Training time graph
    plt.figure(figsize=(7, 5))

    sns.barplot(
        x=list(training_times.keys()),
        y=list(training_times.values())
    )

    plt.title("Model Training Time Comparison")
    plt.ylabel("Time in Seconds")
    plt.tight_layout()
    plt.savefig(GRAPHS / "model_training_time.png")
    plt.close()

    # Confusion matrices
    logreg_cm = confusion_matrix(
        y_test, logreg_predictions, labels=[0, 1]
    )

    xgb_cm = confusion_matrix(
        y_test, xgb_predictions, labels=[0, 1]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        logreg_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Hate", "Hate"],
        yticklabels=["Not Hate", "Hate"],
        ax=axes[0]
    )

    axes[0].set_title("Logistic Regression Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        xgb_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Not Hate", "Hate"],
        yticklabels=["Not Hate", "Hate"],
        ax=axes[1]
    )

    axes[1].set_title("XGBoost Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(GRAPHS / "confusion_matrices.png")
    plt.close()

    # Word clouds
    wordcloud_details = [
        (1, "wordcloud_hate.png", "Hate Speech Word Cloud"),
        (0, "wordcloud_not_hate.png", "Not Hate Speech Word Cloud")
    ]

    for label, filename, title in wordcloud_details:

        text_data = " ".join(
            df.loc[df["label"] == label, "text"].astype(str)
        )

        if text_data.strip():

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white"
            ).generate(text_data)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(GRAPHS / filename)
            plt.close()

    # Final summary
    print("\n====================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("====================================")

    print("\nFinal Model Results:")
    print(results_df.round(4))

    print("\nSaved model files in backend folder.")
    print("Saved graphs in graphs folder.")
    print("Classification reports saved.")

    print("\nTo start Flask, execute:")
    print("python backend/app.py")

    print("\n====================================")


if __name__ == "__main__":
    main()
