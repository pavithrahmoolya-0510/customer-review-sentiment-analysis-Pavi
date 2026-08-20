# Customer Review Sentiment Analysis

## 1. Project Overview

This project develops a binary sentiment classification system for customer/movie reviews using Natural Language Processing (NLP).

The project compares two approaches:

1. TF-IDF + Logistic Regression
2. DistilBERT Transformer

The final system includes a Streamlit web application that accepts a review as input and predicts whether the sentiment is Positive or Negative.

---

## 2. Problem Statement

Businesses receive large volumes of customer reviews and cannot manually read every review.

The objective of this project is to build an NLP-based sentiment classification system that can automatically classify reviews as positive or negative and help customer experience teams identify negative feedback efficiently.

---

## 3. Project Objective

The main objectives are:

- Load and validate the IMDb review dataset.
- Perform exploratory data analysis.
- Clean and preprocess the text data.
- Build a TF-IDF + Logistic Regression baseline.
- Fine-tune a DistilBERT transformer model.
- Compare the baseline and transformer models.
- Perform error analysis.
- Select a suitable model for deployment.
- Build a Streamlit sentiment prediction application.

---

## 4. Dataset

### Dataset

IMDb Large Movie Review Dataset

### Source

Hugging Face Datasets

### Task

Binary sentiment classification.

### Labels

- `0` = Negative
- `1` = Positive

The dataset contains labelled positive and negative movie reviews.

---

## 5. Project Structure

```text
customer-review-sentiment-analysis/
│
├── 01_data_loading-02-eda_preprocessing-03_baseline_model.ipynb
├── 04_distilbert_model.ipynb
├── 05_evaluation_error_analysis.ipynb
│
├── plots/
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── final_model_comparison.csv
│   └── distilbert_development/
│           └──https://drive.google.com/drive/folders/1DPLIFucKCEx_jZnCNhxEaJlKhUAgl_WG?usp=sharing
│              Distilbert_development folder saved as a zip file due to its large model size.   
│
├── app.py
├── requirements.txt
├── README.md
├── report.docx
└── .gitignore
