# sms-spam-detection-using-nlp-1
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')


data = {
    "text": [
        "Win a free iPhone now! Click here to claim.",
        "Congratulations! You've won a lottery worth $1M! Call now.",
        "Hello, how are you doing today?",
        "Important meeting scheduled at 5 PM.",
        "Get cheap loans instantly. No credit check needed!",
        "Let's catch up for lunch tomorrow!",
        "Exclusive offer! Buy 1 Get 1 Free. Limited time only!",
        "Can you send me the report by EOD?",
        "URGENT: Your account has been compromised. Reset password now!",
        "Hey, are we still on for the weekend trip?"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  
}

df = pd.DataFrame(data)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


def predict_spam(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"


new_message = "You have won a free vacation! Click here to claim."
print(f"Message: {new_message} -> {predict_spam(new_message)}")
