import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import emoji
import re


def give_emoji_free_text(text):
    return emoji.replace_emoji(text, replace="")

# Pre-processing: lowercase, remove HTML tags, URLs, hashtags, mentions, emojis...
def clean_text(df):
    df['cleaned_text'] = df['text'].astype(str).str.lower()
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'#\w+', '', x))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'@\w+', '', x))
    df['cleaned_text'] = df['cleaned_text'].apply(give_emoji_free_text)
    return df

# Read datasets from the csv file
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
evaluate_df = pd.read_csv('./evaluation.csv')

# Pre-processing(data cleaning)
train_df = clean_text(train_df)
test_df = clean_text(test_df)

# Features extraction
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

# Fit and transform training data
X_train = tfidf.fit_transform(train_df['cleaned_text'])
Y_train = train_df['score']

# Transform testing data
X_test = tfidf.transform(test_df['cleaned_text'])
Y_test = test_df['score']

# Build a MultinomialNB Classifier
model = MultinomialNB()

# Model training
model.fit(X_train, Y_train)

# Predict
predicted = model.predict(X_test)

# Evaluation on the test set
accuracy = accuracy_score(Y_test, predicted)
report = classification_report(Y_test, predicted)
conf_matrix = confusion_matrix(Y_test, predicted)

# Print evaluation results
print("Test Set Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Evaluation on the evaluation dataset
evaluate_df = clean_text(evaluate_df)
X_evaluate = tfidf.transform(evaluate_df['cleaned_text'])
predicted_eval = model.predict(X_evaluate)

# Print predictions
print("\nPredictions on the evaluation dataset:\n", predicted_eval)