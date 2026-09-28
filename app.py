import json

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    return load_model("imdb_rnn_sentiment_model.keras")


model = load_sentiment_model()


# --------------------------------------------------
# Load IMDb word index
# --------------------------------------------------

@st.cache_data
def load_word_index():
    with open("word_index.json", "r") as f:
        return json.load(f)


word_index = load_word_index()


# --------------------------------------------------
# Preprocess review
# --------------------------------------------------

def preprocess_review(review):

    words = review.lower().split()

    encoded_review = []

    for word in words:

        index = word_index.get(word)

        if index is not None and index < 10000:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=200,
        padding="post",
        truncating="post"
    )

    return padded_review


# --------------------------------------------------
# Predict sentiment
# --------------------------------------------------

def predict_sentiment(review):

    processed_review = preprocess_review(review)

    probability = model.predict(
        processed_review,
        verbose=0
    )[0][0]

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    return sentiment, confidence


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the trained RNN model "
    "will classify it as Positive or Negative."
)

st.divider()


review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was amazing and I really enjoyed it!",
    height=150
)


if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"😊 {sentiment}")

        else:
            st.error(f"😞 {sentiment}")

        st.write(
            f"Confidence: **{confidence:.2%}**"
        )

        st.progress(float(confidence))


st.divider()


st.caption(
    "Model: Simple RNN | Dataset: IMDb | "
    "Sequence Length: 200 | Vocabulary Size: 10,000"
)