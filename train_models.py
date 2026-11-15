
# ============================================================
# HATE SPEECH DETECTION
# Logistic Regression + XGBoost + TF-IDF
# ============================================================

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
# 1. PROJECT PATHS
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
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = html.unescape(str(text))

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 3. MAIN FUNCTION
# ============================================================

def main():

    print("\n====================================")
    print("HATE SPEECH DETECTION PROJECT")
    print("====================================\n")


    # ========================================================
    # 4. LOAD DATASET
    # ========================================================

    print("Loading dataset...")

    if not ENTRIES_FILE.exists():

        raise FileNotFoundError(
            "Entries CSV not found inside the data folder."
        )

    df = pd.read_csv(
        ENTRIES_FILE,
        low_memory=False
    )

    # Check required columns
    required_columns = ["id", "text", "label"]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    df = df[["id", "text", "label"]].copy()

    print("Original dataset records:", len(df))


    # ========================================================
    # 5. DATA CLEANING
    # ========================================================

    print("\nCleaning dataset...")

    # Remove missing values
    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    # Normalize labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only the two expected classes
    df = df[
        df["label"].isin(["hate", "nothate"])
    ].copy()

    # Convert text to cleaned format
    df["text"] = df["text"].map(clean_text)

    # Remove empty text
    df = df[
        df["text"].str.len() > 0
    ].copy()

    # Remove duplicate records
    df = df.drop_duplicates(
        subset=["text", "label"]
    ).copy()

    # Remove identical text with conflicting labels
    conflicting_labels = (
        df.groupby("text")["label"].nunique()
    )

    conflicting_texts = conflicting_labels[
        conflicting_labels > 1
    ].index

    df = df[
        ~df["text"].isin(conflicting_texts)
    ].copy()

    # Encode binary labels
    # Hate = 1
    # Not Hate = 0

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


    # ========================================================
    # 6. TRAIN TEST SPLIT
    # ========================================================

    print("\nSplitting dataset...")

    X = df["text"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("Training records:", len(X_train_text))
    print("Testing records:", len(X_test_text))


    # ========================================================
    # 7. TF-IDF VECTORIZATION
    # ========================================================

    print("\nApplying TF-IDF vectorization...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    # Fit only on training data
    X_train = vectorizer.fit_transform(
        X_train_text
    )

    # Transform test data using the same fitted vectorizer
    X_test = vectorizer.transform(
        X_test_text
    )

    print("TF-IDF features:", X_train.shape[1])


    # ========================================================
    # 8. LOGISTIC REGRESSION
    # ========================================================

    print("\nTraining Logistic Regression...")

    logreg = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    start_time = time.perf_counter()

    logreg.fit(
        X_train,
        y_train
    )

    logreg_training_time = (
        time.perf_counter() - start_time
    )

    start_time = time.perf_counter()

    logreg_predictions = logreg.predict(
        X_test
    )

    logreg_prediction_time = (
        time.perf_counter() - start_time
    )

    print("Logistic Regression completed.")


    # ========================================================
    # 9. XGBOOST
    # ========================================================

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

    xgb.fit(
        X_train,
        y_train
    )

    xgb_training_time = (
        time.perf_counter() - start_time
    )

    start_time = time.perf_counter()

    xgb_predictions = xgb.predict(
        X_test
    )

    xgb_prediction_time = (
        time.perf_counter() - start_time
    )

    print("XGBoost completed.")


    # ========================================================
    # 10. MODEL EVALUATION
    # ========================================================

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

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Training Time": training_times[model_name],

            "Prediction Time": prediction_times[model_name]
        }

    # ROC-AUC
    logreg_probability = logreg.predict_proba(X_test)[:, 1]
    xgb_probability = xgb.predict_proba(X_test)[:, 1]

    results["Logistic Regression"]["ROC AUC"] = (
        roc_auc_score(y_test, logreg_probability)
    )

    results["XGBoost"]["ROC AUC"] = (
        roc_auc_score(y_test, xgb_probability)
    )

    results_df = pd.DataFrame(results).T

    print("\nMODEL PERFORMANCE COMPARISON")

    print(results_df.round(4))


    # ========================================================
    # 11. SAVE TRAINED MODELS
    # ========================================================

    print("\nSaving trained models...")

    joblib.dump(
        logreg,
        BACKEND / "model_logreg.pkl"
    )

    joblib.dump(
        xgb,
        BACKEND / "model_xgb.pkl"
    )

    joblib.dump(
        vectorizer,
        BACKEND / "vectorizer.pkl"
    )

    # Save accuracy
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

    # Save detailed metrics
    results_df.to_csv(
        BACKEND / "model_metrics.csv"
    )


    # ========================================================
    # 12. CLASSIFICATION REPORTS
    # ========================================================

    with open(
        BACKEND / "classification_reports.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LOGISTIC REGRESSION REPORT\n\n"
        )

        file.write(
            classification_report(
                y_test,
                logreg_predictions,
                target_names=["Not Hate", "Hate"],
                zero_division=0
            )
        )

        file.write(
            "\n\nXGBOOST REPORT\n\n"
        )

        file.write(
            classification_report(
                y_test,
                xgb_predictions,
                target_names=["Not Hate", "Hate"],
                zero_division=0
            )
        )


    # ========================================================
    # 13. LABEL DISTRIBUTION GRAPH
    # ========================================================

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

    plt.savefig(
        GRAPHS / "label_distribution.png"
    )

    plt.close()


    # ========================================================
    # 14. MODEL ACCURACY GRAPH
    # ========================================================

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

    plt.savefig(
        GRAPHS / "model_accuracy.png"
    )

    plt.close()


    # ========================================================
    # 15. TRAINING TIME GRAPH
    # ========================================================

    plt.figure(figsize=(7, 5))

    sns.barplot(
        x=list(training_times.keys()),
        y=list(training_times.values())
    )

    plt.title("Model Training Time Comparison")

    plt.ylabel("Time in Seconds")

    plt.tight_layout()

    plt.savefig(
        GRAPHS / "model_training_time.png"
    )

    plt.close()


    # ========================================================
    # 16. CONFUSION MATRICES
    # ========================================================

    logreg_cm = confusion_matrix(
        y_test,
        logreg_predictions,
        labels=[0, 1]
    )

    xgb_cm = confusion_matrix(
        y_test,
        xgb_predictions,
        labels=[0, 1]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    sns.heatmap(
        logreg_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Hate", "Hate"],
        yticklabels=["Not Hate", "Hate"],
        ax=axes[0]
    )

    axes[0].set_title(
        "Logistic Regression Confusion Matrix"
    )

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

    axes[1].set_title(
        "XGBoost Confusion Matrix"
    )

    axes[1].set_xlabel("Predicted")

    axes[1].set_ylabel("Actual")


    plt.tight_layout()

    plt.savefig(
        GRAPHS / "confusion_matrices.png"
    )

    plt.close()


    # ========================================================
    # 17. WORD CLOUDS
    # ========================================================

    wordcloud_details = [

        (1, "wordcloud_hate.png", "Hate Speech Word Cloud"),

        (0, "wordcloud_not_hate.png", "Not Hate Speech Word Cloud")

    ]

    for label, filename, title in wordcloud_details:

        text_data = " ".join(
            df.loc[
                df["label"] == label,
                "text"
            ].astype(str)
        )

        if text_data.strip():

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white"
            ).generate(text_data)

            plt.figure(figsize=(12, 6))

            plt.imshow(
                wordcloud,
                interpolation="bilinear"
            )

            plt.axis("off")

            plt.title(title)

            plt.tight_layout()

            plt.savefig(
                GRAPHS / filename
            )

            plt.close()


    # ========================================================
    # 18. FINAL OUTPUT
    # ========================================================

    print("\n====================================")

    print("TRAINING COMPLETED SUCCESSFULLY")

    print("====================================")

    print("\nFinal Model Results:")

    print(results_df.round(4))

    print("\nSaved model files in backend folder.")

    print("\nSaved graphs in graphs folder.")

    print("\nClassification reports saved.")

    print("\nTo start Flask, execute:")

    print("python backend/app.py")

    print("\n====================================")


# ============================================================
# 19. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()