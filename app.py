import pandas as pd
import re
import joblib
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load dataset
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

# Data Preprocessing
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

df['message'] = df['message'].apply(preprocess_text)

# Split the data
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('static/confusion_matrix.png')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message = preprocess_text(message)

    # Load the model and TF-IDF vectorizer
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')

    # Transform the message using TF-IDF
    message_tfidf = tfidf.transform([message])

    # Make prediction
    prediction = model.predict(message_tfidf)
    prediction_text = 'Spam' if prediction[0] == 1 else 'Ham'
    
    return render_template('index.html', prediction_text=f'The message is: {prediction_text}', accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
