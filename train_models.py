# train_models.py

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Load datasets
entries = pd.read_csv('data/2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv')
targets = pd.read_csv('data/2020-12-31-DynamicallyGeneratedHateDataset-targets-v0.1.csv')

# Merge on 'id'
df = entries.merge(targets, on='id')

# Basic cleaning: Only keep 'text' and 'label'
df = df[['text', 'label']]

# Encode labels: hate = 1, not = 0
df['label'] = df['label'].apply(lambda x: 1 if x == 'hate' else 0)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
start_logreg = time.time()
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_preds = logreg.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_preds)
end_logreg = time.time()
logreg_time = end_logreg - start_logreg

# Train XGBoost
start_xgb = time.time()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
end_xgb = time.time()
xgb_time = end_xgb - start_xgb

# Save models and vectorizer
os.makedirs('backend', exist_ok=True)
joblib.dump(logreg, 'backend/model_logreg.pkl')
joblib.dump(xgb, 'backend/model_xgb.pkl')
joblib.dump(vectorizer, 'backend/vectorizer.pkl')

# Save accuracy values
with open('backend/model_accuracy.txt', 'w') as f:
    f.write(f'Logistic Regression Accuracy: {logreg_acc:.4f}\n')
    f.write(f'XGBoost Accuracy: {xgb_acc:.4f}\n')

# Plot: Hate vs Not Hate frequency
os.makedirs('graphs', exist_ok=True)
sns.countplot(x=df['label'].map({0: 'Not Hate', 1: 'Hate'}))
plt.title('Label Distribution')
plt.savefig('graphs/label_distribution.png')
plt.clf()

# Plot: Model Accuracy Comparison
sns.barplot(x=['Logistic Regression', 'XGBoost'], y=[logreg_acc, xgb_acc])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('graphs/model_accuracy.png')
plt.clf()

# Plot: Confusion Matrices
log_cm = confusion_matrix(y_test, logreg_preds)
xgb_cm = confusion_matrix(y_test, xgb_preds)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('XGBoost Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('graphs/confusion_matrices.png')
plt.clf()

# Plot: Training Time Comparison
sns.barplot(x=['Logistic Regression', 'XGBoost'], y=[logreg_time, xgb_time])
plt.title('Model Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.savefig('graphs/model_training_time.png')
plt.clf()

# WordCloud: Hate Speech
hate_text = " ".join(df[df['label'] == 1]['text'])
hate_wc = WordCloud(width=800, height=400, background_color='white').generate(hate_text)
plt.figure(figsize=(10, 5))
plt.imshow(hate_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Hate Speech')
plt.savefig('graphs/wordcloud_hate.png')
plt.clf()

# WordCloud: Not Hate Speech
not_text = " ".join(df[df['label'] == 0]['text'])
not_wc = WordCloud(width=800, height=400, background_color='white').generate(not_text)
plt.figure(figsize=(10, 5))
plt.imshow(not_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Not Hate Speech')
plt.savefig('graphs/wordcloud_not_hate.png')
plt.clf()

# Optional: Save classification reports
with open('backend/classification_reports.txt', 'w') as f:
    f.write("Logistic Regression Report:\n")
    f.write(classification_report(y_test, logreg_preds))
    f.write("\n\nXGBoost Report:\n")
    f.write(classification_report(y_test, xgb_preds))

print("✅ Models trained and all visualizations saved successfully.")
