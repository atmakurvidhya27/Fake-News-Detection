# Fake-News-Detection 
# Fake News Detection

This project uses Machine Learning to classify news articles as Fake or Real.

## Features
- Detects fake news
- Classifies news as Fake or Real
- Uses Python and Machine Learning

## Technologies Used
- Python
- Pandas
- Scikit-learn
- 
Introduction

Fake News Detection is a machine learning-based project that identifies whether a news article or headline is real or fake. With the rapid growth of social media and online platforms, false information spreads quickly and can mislead people. This project uses Python and machine learning techniques to analyze news content and classify it as real or fake, helping users access reliable information.

Objectives

To detect and classify fake news accurately.
To reduce the spread of misinformation.
To apply machine learning techniques to text analysis.
To improve awareness about the authenticity of online news.
To provide users with a simple tool for verifying news articles.

Procedure

Collect a dataset containing real and fake news articles.
Preprocess the text data by removing unnecessary words and symbols.
Convert text into numerical features using TF-IDF Vectorization.
Split the dataset into training and testing data.
Train a machine learning model (e.g., Logistic Regression).
Test the model and evaluate its accuracy.
Accept user input and predict whether the news is real or fake.
Display the prediction result to the user 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
news = [
    "Scientists discover new species of bird",
    "Aliens have landed in the city center",
    "Government announces new education policy",
    "Magic potion cures all diseases instantly"
]

labels = ["Real", "Fake", "Real", "Fake"]

# Convert text into numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# User input
user_news = input("Enter a news headline: ")

# Predict
user_vector = vectorizer.transform([user_news])
prediction = model.predict(user_vector)

print("Prediction:", prediction[0])
Enter a news headline: Aliens are controlling the weather
Prediction: Fake

system successfully analyzes the entered news headline and predicts whether it is Real or Fake using a machine learning model trained on sample news data. This helps users identify potentially misleading information.
##author
Atmakur vidhya 
