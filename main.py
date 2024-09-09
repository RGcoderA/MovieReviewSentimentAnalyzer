import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Load the dataset
data = pd.read_csv('movie_reviews.csv')

# Preprocess the reviews
data['review'] = data['review'].apply(preprocess_text)

# Split the data into training and test sets
X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict on the test set and print accuracy
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Predict sentiment for user input
def predict_sentiment(review):
    review_processed = preprocess_text(review)
    review_vectorized = vectorizer.transform([review_processed])
    prediction = model.predict(review_vectorized)[0]
    return prediction

if __name__ == "__main__":
    review = input("Enter a movie review: ")
    result = predict_sentiment(review)
    print(f"The sentiment of the review is: {result}")
