# 🧠 Hate Speech Detection on Social Media 🚫💬

A Machine Learning project that uses **NLP techniques** and **supervised learning models** (Logistic Regression & XGBoost) to detect hate speech in social media text. The system is deployed using **Flask** and provides real-time predictions with a clean web interface.

---

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📁 Dataset](#-dataset)
- [⚙️ Technologies Used](#-technologies-used)
- [🛠️ Installation & Setup](#️-installation--setup)
- [🚀 How to Run](#-how-to-run)
- [📊 Model Performance](#-model-performance)
- [🌐 Web Interface Preview](#-web-interface-preview)
- [📈 Visualizations](#-visualizations)
- [🔮 Future Scope](#-future-scope)
- [📚 References](#-references)

---

## 🎯 Project Overview

The goal of this project is to develop a **scalable, accurate, and interpretable hate speech detection system** using supervised learning and deploy it as a real-time **web application**.

### ✔ Key Features:
- Text preprocessing and vectorization using **TF-IDF**
- Dual model pipeline: **Logistic Regression** and **XGBoost**
- Detailed evaluation using accuracy, F1-score, and confusion matrices
- Word Cloud analysis for hate vs non-hate speech
- Web application built using **Flask** with real-time prediction
- Interactive frontend with accuracy display

---

## 📁 Dataset

- Source: [Kaggle - Dynamically Generated Hate Speech Dataset](https://www.kaggle.com/datasets)
- Two files:
  - `entries-v0.1.csv`: Text data
  - `targets-v0.1.csv`: Labels (`hate`, `not`)
- Merged using `id` column for final dataset

---

## ⚙️ Technologies Used

| Tool / Library     | Purpose                                 |
|--------------------|------------------------------------------|
| Python             | Core programming language                |
| Pandas, NumPy      | Data manipulation and analysis           |
| Scikit-learn       | ML models (Logistic Regression, TF-IDF)  |
| XGBoost            | High-performance model                   |
| Matplotlib, Seaborn| Data visualization                       |
| WordCloud          | Word frequency visualization             |
| Flask              | Web application backend                  |
| HTML/CSS/JS        | Frontend interface                       |
| Joblib             | Model saving/loading                     |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mujjjtaba/Hate-Speech-Detection-On-Social-Media.git
   cd Hate-Speech-Detection-On-Social-Media
