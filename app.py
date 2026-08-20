import streamlit as st
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "models/logistic_regression.pkl"
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎬 Customer Review Sentiment Analyzer")

st.write(
    "Enter a movie review to predict whether "
    "the sentiment is positive or negative."
)


# --------------------------------------------------
# User input
# --------------------------------------------------

review = st.text_area(
    "Enter your review:",
    height=200,
    placeholder="Type your movie review here..."
)

predict_button = st.button(
    "Predict Sentiment"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        review_vector = tfidf.transform(
            [review]
        )

        prediction = model.predict(
            review_vector
        )[0]

        probability = model.predict_proba(
            review_vector
        )[0]

        confidence = probability[prediction]

        sentiment = (
            "Positive"
            if prediction == 1
            else "Negative"
        )

        st.subheader(
            f"Prediction: {sentiment}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        st.progress(
            float(confidence)
        )


# --------------------------------------------------
# Sample predictions
# --------------------------------------------------

st.divider()

st.subheader("Sample Reviews")

sample_reviews = {
    "Positive example":
        "This movie was absolutely fantastic. "
        "The acting and story were excellent.",

    "Negative example":
        "This movie was boring and disappointing. "
        "I would not recommend it.",

    "Mixed example":
        "The acting was good, but the story was "
        "slow and disappointing."
}

for title, text in sample_reviews.items():

    st.markdown(f"**{title}**")
    st.write(text)

    sample_vector = tfidf.transform([text])

    sample_prediction = model.predict(
        sample_vector
    )[0]

    sample_probability = model.predict_proba(
        sample_vector
    )[0]

    sample_confidence = sample_probability[
        sample_prediction
    ]

    sample_sentiment = (
        "Positive"
        if sample_prediction == 1
        else "Negative"
    )

    st.write(
        f"Prediction: **{sample_sentiment}**"
    )

    st.write(
        f"Confidence: **{sample_confidence:.2%}**"
    )


# --------------------------------------------------
# Known failure case
# --------------------------------------------------

st.divider()

st.subheader("⚠️ Known Failure Case")

st.write(
    "**Review:** The acting was excellent, but the movie "
    "was painfully boring."
)

st.write("**Actual sentiment:** Positive")
st.write("**Model prediction:** Negative")
st.write("**Confidence:** 83.95%")

st.info(
    "The model struggled with mixed sentiment. "
    "The review contains positive language about the acting "
    "but also strong negative language about the movie. "
    "This demonstrates a limitation of the sentiment classifier identified during error analysis, "
    "when sentiment depends on context."
)