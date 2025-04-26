# email_spam_checker.py

import re
import string
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved models
vectorizer = joblib.load('TF-IDF vectorizer.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')

# Preprocessing function
def preprocess_email(text):
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Main Function
def predict_email():
    email = input("Enter the email text: ")
    processed_email = preprocess_email(email)
    vectorized_email = vectorizer.transform([processed_email])
    prediction = nb_model.predict(vectorized_email)
    
    if prediction[0] == 1:
        print("\n🔴 The email is predicted as SPAM.")
    else:
        print("\n🟢 The email is predicted as NOT SPAM.")

if __name__ == "__main__":
    predict_email()
