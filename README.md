📧 Email Spam Detection using Machine Learning
<div align="center"> <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Python"> <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"> <img src="https://img.shields.io/badge/NLP-Project-orange.svg" alt="NLP"> <img src="https://img.shields.io/badge/Machine_Learning-Scikit--learn-brightgreen.svg" alt="Scikit-learn"> </div>
📚 Project Overview
This project builds an intelligent system that automatically classifies emails as Spam or Not Spam (Ham).
It uses Natural Language Processing (NLP) techniques combined with Machine Learning models to predict spam emails with high accuracy.

🔹 Dataset Source: Kaggle - Spam Detection Dataset
🔹 Goal: Preprocess email text, balance the dataset, train different models, compare their performances, and deploy the best one for real-world use.

🛠️ Tech Stack
Programming Language: Python
Libraries:
pandas
nltk
scikit-learn
seaborn
matplotlib
joblib

Algorithms:
Naive Bayes Classifier
Logistic Regression
Support Vector Machine (SVM)
Text Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)

📂 Project Structure
bash
Copy
Edit
📁 Email-Spam-Detection
 ├── 📄 emails.csv                   # Kaggle Dataset
 ├── 📄 email_spam_checker.py        # CLI script to predict email spam
 ├── 📄 tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
 ├── 📄 naive_bayes_model.pkl         # Saved Naive Bayes model
 ├── 📄 svm_model.pkl                 # Saved SVM model
 ├── 📄 logistic_regression_model.pkl # Saved Logistic Regression model
 ├── 📄 README.md                     # Project documentation

📊 Problem Understanding
Spam emails are a major issue in communication.
Detecting spam is crucial to:
Save time.
Protect against fraud and scams.
Improve email systems' efficiency.
Machine Learning allows automatic detection based on patterns in email texts without manual rules.

🔎 Project Workflow
1. Dataset Loading
Dataset loaded with columns:
email — the email body,
spam — label (1 = spam, 0 = not spam).

2. Data Visualization
Plotted the distribution of spam and ham emails to understand class imbalance.

3. Balancing the Dataset
Equalized the number of spam and ham emails using sampling to avoid bias during training.

4. Data Preprocessing
Removed:
Email addresses
URLs and links
Numbers
Punctuations
Performed:
Tokenization
Stopwords removal
Lemmatization
Lowercasing

5. Text Vectorization
Used TF-IDF to convert cleaned text into numerical vectors for machine learning algorithms.

6. Model Building
Trained three different models:
Naive Bayes
Logistic Regression
Support Vector Machine (SVM)

7. Model Evaluation
Used Classification Reports and Confusion Matrices to evaluate models.
Compared models’ Accuracy Scores visually.

8. Model Saving
Saved the best models and vectorizer using joblib for future reuse.

9. User Interface
Developed a simple Python CLI where users can input any email and predict whether it's spam or not.

🚀 How to Run Locally
1. Clone the Repository
git clone (https://github.com/AssadRaza10/spam-mail-detector-ml.git)
cd Email-Spam-Detection

2. Install Requirements
pip install pandas scikit-learn nltk seaborn matplotlib joblib

3. Setup NLTK (only once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

4. Run the Spam Checker
python email_spam_checker.py
You will be asked to paste your email content — the model will immediately predict if it’s SPAM or NOT SPAM!

🎯 Sample Predictions
Enter the email text: Congratulations! You have won a free iPhone. Click here to claim.
🔴 Prediction: SPAM

Enter the email text: Hi John, please review the attached project plan and let me know your thoughts.
🟢 Prediction: NOT SPAM

📈 Project Results
Naive Bayes performed the best for this text classification task.
Balanced datasets significantly improved model generalization.
Visual comparison of models helped in making an informed deployment choice.

📈 Visualization Examples

Model	Accuracy (%)
Support Vector Machine	99%
Logistic Regression	98%
Naive Bayes	99%
(Actual accuracy may slightly vary depending on sampling random states.)

🔥 Future Work
Build a full web application (using Flask/Streamlit).

Deploy models as APIs.

Improve model accuracy with deep learning (e.g., LSTM, BERT).

Introduce multilingual support (detect spam in other languages too).

Collect a more diverse and larger email dataset.

🧠 Key Learnings
Text data preprocessing is crucial for NLP tasks.

Model selection should be based on balanced evaluation, not just raw accuracy.

Saving and reusing models accelerates deployment.

Clear data visualization helps communicate insights effectively.

📜 License
This project is licensed under the MIT License — feel free to use, modify, and distribute!

🤝 Acknowledgements
Kaggle Datasets
NLTK Documentation
Scikit-learn Documentation

📬 Contact
Assad Raza
📧 assad.raza.101@gmail.com
💼 Aspiring Data Scientist | Python Developer | NLP Enthusiast

🌟 Thank you for visiting this project!
