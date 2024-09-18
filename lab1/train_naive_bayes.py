import pandas as pd
from multinomial_naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import emoji
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm, cmap='Blues')  # Display the matrix with a blue color map
    plt.title(title)
    fig.colorbar(cax)

    # Labeling axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])

    # Adding text annotations (the confusion matrix values)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

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

# Fit and transform training data
Y_train = train_df['score']

Y_test = test_df['score']

# Build a MultinomialNB Classifier
model = MultinomialNB()

# Model training
model.fit(train_df['cleaned_text'], Y_train)

# Predict
predicted = model.predict(test_df['cleaned_text'])

# Evaluation on the test set
accuracy = accuracy_score(Y_test, predicted)
report = classification_report(Y_test, predicted)
conf_matrix = confusion_matrix(Y_test, predicted)

# Print evaluation results
print("Test Set Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

plot_confusion_matrix(Y_test, predicted, title="Test Confusion Matrix")

# Evaluation on the evaluation dataset
evaluate_df = clean_text(evaluate_df)
X_evaluate = evaluate_df['cleaned_text']
Y_evaluate = evaluate_df['score']
predicted_eval = model.predict(X_evaluate)

print("Evaluation Set Accuracy: {:.2f}%".format(accuracy_score(Y_evaluate, predicted_eval) * 100))
print("\nClassification Report:\n", classification_report(Y_evaluate, predicted_eval))
print("\nConfusion Matrix:\n", confusion_matrix(Y_evaluate, predicted_eval))

plot_confusion_matrix(Y_evaluate, predicted_eval, title="Evaluation Confusion Matrix")