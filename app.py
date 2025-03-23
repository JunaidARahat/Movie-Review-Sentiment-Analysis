import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("IMDB Sentiment Analysis App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:\\Movie Review Project 2025\\Movie-Review-Sentiment-Analysis\\Data\\IMDB Dataset.csv")
    return df

df = load_data()

# Function to clean text
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    text = text.lower().strip()  # Convert to lowercase and strip spaces
    return text

df['review'] = df['review'].apply(clean_text)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

# Initialize stopwords, tokenizer, and lemmatizer
stop_words = set(stopwords.words('english'))
w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

# Function for stopword removal and lemmatization
def preprocess_text(text):
    words = [lemmatizer.lemmatize(word) for word in w_tokenizer.tokenize(text) if word not in stop_words]
    return ' '.join(words)

df['review'] = df['review'].apply(preprocess_text)

# Encode labels
encoder = LabelEncoder()
df['encoded_sentiment'] = encoder.fit_transform(df['sentiment'])

# Reduce dataset size for efficiency
df_sample = df.sample(n=5000, random_state=42)

# Train-test split
train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(
    df_sample['review'], df_sample['encoded_sentiment'], test_size=0.2, random_state=42
)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True)
X_train = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, solver='saga', n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "SGD Classifier": SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3)
}

# Train models
for model_name, model in models.items():
    model.fit(X_train, train_sentiments)

# Streamlit Sidebar
st.sidebar.title("Sentiment Analysis")
selected_model = st.sidebar.selectbox("Choose a model:", list(models.keys()))
user_review = st.sidebar.text_area("Enter a review for sentiment analysis:")

# Predict Sentiment
if user_review:
    cleaned_review = preprocess_text(clean_text(user_review))
    transformed_review = vectorizer.transform([cleaned_review])
    prediction = models[selected_model].predict(transformed_review)[0]
    sentiment_label = "Positive" if prediction == 1 else "Negative"
    st.sidebar.write(f"**Predicted Sentiment:** {sentiment_label}")
