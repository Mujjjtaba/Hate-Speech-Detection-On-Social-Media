# 🛡️ Hate Speech Detection on Social Media

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189AB4?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A full-stack machine learning web application that detects hate speech in real time using Logistic Regression and XGBoost, trained on 32,000+ labeled social media text samples.**

[🌐 Live Demo (Local)](#-getting-started) · [📄 Project Report](./finalDataScprj.pdf) · [🔗 LinkedIn Post](https://www.linkedin.com/posts/mujtaba11_machinelearning-nlp-ai-activity-7316127508809019392-aZt3) · [📊 Dataset](https://catalog.data.gov/dataset/border-crossing-entry-data-683ae)

</div>

---

## 📌 Overview

This project implements an **end-to-end hate speech detection pipeline** using supervised machine learning and NLP. It processes raw social media text, classifies it as *Hate Speech* or *Not Hate Speech*, and serves predictions through an interactive Flask web application.

> **Course:** INT375 – Data Science Toolbox: Python Programming  
> **Institution:** Lovely Professional University, Punjab  
> **Student:** Mujtaba Kamal · Reg. No: 12319576  
> **Supervisor:** Anand Kumar (UID: 30561)

---

## ✨ Features

- 🔍 **Real-time prediction** — Input any sentence and get instant results from two models
- 🤖 **Dual model comparison** — Logistic Regression vs XGBoost side-by-side
- 📊 **4 built-in visualizations** — Label distribution, model accuracy, training time, confusion matrices
- 🧠 **TF-IDF vectorization** — Captures word importance across 10,000 features
- 💾 **Saved model pipeline** — Pre-trained `.pkl` models for instant inference
- 📱 **Responsive UI** — Clean, mobile-friendly web interface

---

## 🖥️ Web Application

<table>
<tr>
<td align="center"><b>Prediction Interface</b></td>
<td align="center"><b>Results Panel</b></td>
</tr>
<tr>
<td>Enter any sentence → click Predict → instant output from both models</td>
<td>Shows prediction label + accuracy for Logistic Regression & XGBoost</td>
</tr>
</table>

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Logistic Regression** | 69.47% | 0.69 | 0.69 | 0.69 | ~0.2s |
| **XGBoost** | 69.01% | 0.69 | 0.69 | 0.69 | ~2.6s |

> Both models trained on **32,770 samples** (80/20 train-test split), tested on **6,554 samples**

---

## 📉 Visualizations

| Chart | Description |
|-------|-------------|
| **Label Distribution** | ~17,000 hate vs ~15,800 not-hate samples — near-balanced dataset |
| **Model Accuracy Comparison** | Both models achieve ~69% accuracy on test set |
| **Confusion Matrices** | LogReg: 2064 TN / 2489 TP · XGBoost: 2180 TN / 2343 TP |
| **Training Time** | LogReg: ~0.2s · XGBoost: ~2.6s — clear efficiency trade-off |

---

## 🗂️ Project Structure

```
Hate-Speech-Detection-On-Social-Media/
│
├── 📁 backend/
│   ├── model_logreg.pkl          # Trained Logistic Regression model
│   ├── model_xgb.pkl             # Trained XGBoost model
│   ├── vectorizer.pkl            # Fitted TF-IDF vectorizer
│   ├── model_accuracy.txt        # Saved accuracy scores
│   └── classification_reports.txt # Full precision/recall/F1 reports
│
├── 📁 frontend/
│   ├── templates/
│   │   └── index.html            # Main web interface
│   └── static/
│       ├── style.css             # Responsive styling
│       ├── label_distribution.png
│       ├── model_accuracy.png
│       ├── model_training_time.png
│       └── confusion_matrices.png
│
├── 📁 data/
│   ├── entries-v0.1.csv          # Social media text samples
│   └── targets-v0.1.csv          # Hate / not-hate labels
│
├── 📁 graphs/                    # All generated visualization PNGs
│
├── app.py                        # Flask backend & API routes
├── train_models.py               # Full training pipeline
├── requirements.txt              # Python dependencies
├── finalDataScprj.pdf            # Academic project report
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mujjjtaba/Hate-Speech-Detection-On-Social-Media.git
cd Hate-Speech-Detection-On-Social-Media

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the Models (Optional — pre-trained models included)

```bash
# Place dataset CSVs in /data folder, then run:
python train_models.py
```

### Run the Web App

```bash
python app.py
```

Then open your browser and go to → **`http://127.0.0.1:5000`**

---

## 🧪 How It Works

```
User Input Text
      │
      ▼
TF-IDF Vectorizer (10,000 features)
      │
      ├──▶ Logistic Regression ──▶ Hate / Not Hate + Accuracy
      │
      └──▶ XGBoost Classifier  ──▶ Hate / Not Hate + Accuracy
```

1. **Text is vectorized** using a pre-fitted TF-IDF transformer
2. **Both models run inference** independently on the same input
3. **Results returned as JSON** and rendered live in the browser

---

## 📦 Dependencies

```txt
flask
pandas
scikit-learn
xgboost
matplotlib
seaborn
joblib
wordcloud
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

- **Source:** [Dynamically Generated Hate Dataset — Kaggle](https://www.kaggle.com/)
- **Size:** ~32,770 labeled samples
- **Format:** Two CSV files merged on `id` column
- **Labels:** `hate` → 1, `not` → 0
- **Split:** 80% train / 20% test

---

## 🔬 Key Technical Decisions

| Decision | Reason |
|----------|--------|
| TF-IDF over Bag-of-Words | Captures word importance, not just frequency |
| Logistic Regression as baseline | Fast, interpretable, strong for linear separability |
| XGBoost as main model | Handles non-linear patterns in language better |
| Flask over Django | Lightweight, ideal for ML model serving |
| joblib for serialization | Efficient model/vectorizer persistence |

---

## 🔮 Future Scope

- [ ] BERT / RoBERTa transformer integration for context-aware detection
- [ ] Multilingual support (Hindi, Arabic, Spanish)
- [ ] Twitter/Reddit API integration for live feed analysis
- [ ] SHAP/LIME explainability for model transparency
- [ ] Docker containerization for cloud deployment (AWS/GCP)
- [ ] Mobile app / browser extension version

---

## 📚 References

1. Davidson et al. (2017) — *Automated Hate Speech Detection and the Problem of Offensive Language*
2. de Gibert et al. (2018) — *Hate Speech Dataset from a White Supremacy Forum*
3. Sap et al. (2019) — *The Risk of Racial Bias in Hate Speech Detection*
4. [Scikit-learn Documentation](https://scikit-learn.org/)
5. [XGBoost Documentation](https://xgboost.readthedocs.io/)
6. [Flask Documentation](https://flask.palletsprojects.com/)

---

## 👤 Author

**Mujtaba Kamal**  
B.Tech CSE/IT · Lovely Professional University  
Registration No: 12319576

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/mujtaba11)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Mujjjtaba)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">
  <sub>Built with Python, Flask & ❤️ · Lovely Professional University · 2025</sub>
</div>
