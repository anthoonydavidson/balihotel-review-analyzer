# Hotel Review Summarizer & Aspect-Based Sentiment Analysis

This project is a **NLP-based web application** that analyzes hotel reviews using **deep learning models**.  
It allows users to:
- View long hotel reviews
- Automatically summarize reviews
- Perform **aspect-based sentiment analysis**
- Interact with the system through a **Streamlit web interface**

The application is built as part of an **academic NLP final project**.

---

## Features

- **Hotel Review Selection**
  - Browse hotels and view aggregated customer reviews

- **Neural Review Summarization**
  - Uses a **BART** model for abstractive summarization
  - Adjustable summary length

- **Aspect-Based Sentiment Analysis**
  - Automatically extracts hotel-related aspects (e.g., room, staff, location)
  - Predicts sentiment per aspect using a **fine-tuned DeBERTa model**

---

## Models Used

### 1. Sentiment Classification
- **Model**: DeBERTa (fine-tuned)
- **Task**: Binary sentiment classification (Positive / Negative)
- **Hosted on**: Hugging Face Hub

### 2. Summarization
- **Model**: BART
- **Task**: Abstractive summarization
- **Hosted on**: Hugging Face Hub

> ⚠️ Models are **not stored in GitHub** due to size limitations.  
> They are automatically downloaded from Hugging Face Hub when the app runs.

---

## Project Structure
- app.py # Streamlit application
- NLP_Final_Project.ipynb # Training & experimentation notebook
- combine (1).csv # Dataset used (scraped from Tripadvisor)
- grouped_reviews.csv # Processed hotel reviews
- requirements.txt # Python dependencies
- README.md

---

## Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/anthoonydavidson/balihotel-review-analyzer.git
cd balihotel-review-analyzer
```

### 2️⃣ Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate    # for Windows
source venv/bin/activate  # for macOS / Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### ▶️ Run the Streamlit App
```bash
streamlit run app.py
```
