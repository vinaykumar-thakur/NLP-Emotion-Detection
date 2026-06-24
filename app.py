import streamlit as st
import joblib
import json

# Load model files
svm_model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊"
)

st.title("😊 Emotion Detection using SVM")
st.write("Enter a sentence and predict its emotion.")

text = st.text_area("Enter Text")

if st.button("Predict"):
    if text.strip():

        # Transform text
        text_vector = tfidf.transform([text])

        # Predict
        pred = svm_model.predict(text_vector)[0]

        # JSON keys become strings after loading
        emotion = label_map[str(pred)]

        st.success(f"Predicted Emotion: {emotion}")

    else:
        st.warning("Please enter some text.")