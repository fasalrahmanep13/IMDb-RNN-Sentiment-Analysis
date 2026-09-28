# 🎬 IMDb Movie Review Sentiment Analysis using RNN

## 📌 Project Overview

This project develops a **Natural Language Processing (NLP)** system that classifies IMDb movie reviews into two sentiment categories:

- 😊 Positive
- 😞 Negative

A **Recurrent Neural Network (RNN)** is used to learn patterns and relationships in sequences of words. The trained model is deployed as an interactive **Streamlit web application**, allowing users to enter a movie review and receive a sentiment prediction with confidence.

---

## 🎯 Objective

The main objectives of this project are:

- Perform sentiment analysis on movie reviews.
- Understand and preprocess sequential text data.
- Build an RNN-based deep learning model.
- Evaluate the model using classification metrics.
- Deploy the trained model using Streamlit.
- Provide an interactive interface for predicting sentiment from new reviews.

---

## 📊 Dataset

The project uses the **IMDb Movie Reviews dataset provided through TensorFlow/Keras**.

### Dataset Details

| Dataset | Details |
|---|---|
| Training Reviews | 25,000 |
| Testing Reviews | 25,000 |
| Total Reviews | 50,000 |
| Classes | Positive / Negative |
| Vocabulary Size | 10,000 |
| Sequence Length | 200 |

The sentiment labels are:

- `0` → Negative
- `1` → Positive

The reviews are already represented as integer sequences using the IMDb word index.

---

## 🔍 Exploratory Data Analysis

The following analysis was performed:

- Dataset size analysis
- Sentiment distribution
- Review sequence length analysis
- Sequence length distribution
- Decoding IMDb integer sequences
- Word frequency analysis
- Positive and negative word analysis
- Padding and truncation analysis

---

## 🛠️ Data Preprocessing

The IMDb reviews are represented as sequences of integers.

To provide a fixed input size to the RNN:

- Vocabulary size was limited to **10,000 words**
- Maximum sequence length was set to **200**
- Shorter sequences were padded
- Longer sequences were truncated
- Padding was performed at the end of the sequence

Final input shapes:

```text
Training data: (25000, 200)
Testing data:  (25000, 200)
