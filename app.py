import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import subprocess
import sys

# ============================
# BASIC SETUP
# ============================
st.set_page_config(page_title="Hotel Review Analyzer")

# Download NLTK data if not exists
nltk.download('punkt')

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_sentiment_model():
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("anthonydavidson/balihotel-sentiment-deberta")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("anthonydavidson/balihotel-sentiment-deberta")
    return sentiment_model, sentiment_tokenizer

@st.cache_resource
def load_summarizer():
    summarizer = pipeline(
        "summarization",
        model="anthonydavidson/balihotel-bart-summarizer",
        tokenizer="anthonydavidson/balihotel-bart-summarizer",
    )
    return summarizer

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

sentiment_model, sentiment_tokenizer = load_sentiment_model()
summarizer = load_summarizer()
nlp = load_spacy()

# ============================
# LOAD REVIEWS
# ============================
@st.cache_data
def load_data():
    return pd.read_csv("grouped_reviews.csv")

df = load_data()

# ============================
# FUNCTIONS
# ============================
def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return "positive" if pred == 1 else "negative"

HOTEL_ASPECTS = {
    "room", "rooms", "bed", "bathroom", "toilet", "shower", "staff", "service", "location",
    "pool", "pools", "wifi", "internet", "cleanliness", "food", "breakfast", "dinner",
    "price", "view", "beach", "air conditioning", "ac", "facility", "facilities", "restaurant",
    "budget", "night", "stay", "experience", "floor", "size", "space", "comfort",
    "amenity", "beverage", "drink", "quality", "access", "gym", "spa", "park", "carpark",
    "reception", "management", "value", "cost", "balcony", "hotel", "city", "towels", "towel",
    "atmosphere"
}

def extract_aspects(text):
  doc = nlp(text)

  aspects = []
  for token in doc:
    if token.pos_ in ['NOUN', 'PROPN'] and token.text in HOTEL_ASPECTS:
      aspects.append(token.lemma_.lower())  # use lemma to group "rooms" -> "room"

  return list(set(aspects))   # remove duplicates

def get_aspect_sentences(text, aspects):
  sentences = sent_tokenize(text)

  aspect_map = {}
  for aspect in aspects:
    for sentence in sentences:
      if aspect in sentence.lower():
        aspect_map.setdefault(aspect, []).append(sentence)

  return aspect_map

def aspect_sentiment_pipeline(summary_text):
    aspects = extract_aspects(summary_text)
    aspect_sentences = get_aspect_sentences(summary_text, aspects)

    results = {}

    for asp, sentences in aspect_sentences.items():
        # classify each sentence about this aspect
        sentiments = [predict_sentiment(s) for s in sentences]

        # majority vote (if multiple sentences mention the same aspect)
        final_sentiment = max(set(sentiments), key=sentiments.count)

        results[asp] = {
            "sentiment" : final_sentiment,
            "evidence" : sentences
        }

    return results


def safe_bart_summarize(text, length):
    try:
        # Hard limit input size
        text = text[:1000]
        
        # Length Mapping
        if length == "Short":
            min_len, max_len = 20, 60
        elif length == "Medium":
            min_len, max_len = 70, 100
        elif length == "Long":
            min_len, max_len = 110, 150

        summary = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return summary[0]["summary_text"]

    except Exception as e:
        return "Summary failed"


# ============================
# STREAMLIT UI
# ============================
st.title("Bali Hotel Review Summarizer & Aspect Sentiment Analyzer")

# 1. Select hotel
hotel_names = df["Hotel"].tolist()
hotel = st.selectbox("Select a hotel:", hotel_names)

# 2. Show long reviews
combined = df[df["Hotel"] == hotel]["combined_reviews"].values[0]
text_area = st.text_area("Hotel reviews:", combined, height=300)

# 3. Button
summary_length = st.radio(
    "Summary length:",
    ["Short", "Medium", "Long"],
    horizontal=True
)

if st.button("Summarize Review"):
    with st.spinner("Summarizing..."):
        summary = safe_bart_summarize(text_area, summary_length)

    st.subheader("Summary")
    st.write(summary)

    # Aspect Sentiment
    sentiments = aspect_sentiment_pipeline(summary)
    st.subheader("Aspect Sentiment Analysis")
    # st.json(sentiments)
    for asp, info in sentiments.items():
        st.markdown(f"### {asp.capitalize()} — {info['sentiment']}")
        for s in info["evidence"]:
            st.write("•", s)  
