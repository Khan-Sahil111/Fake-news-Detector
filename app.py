import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Define text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit App
st.title("📰 Fake News Detector")

input_text = st.text_area("Paste a news article or content below:")

if st.button("Predict"):
    clean = clean_text(input_text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    label = "✅ Real News" if pred == 1 else "❌ Fake News"
    st.subheader(label)
    st.write(f"Confidence: {max(prob)*100:.2f}%")
