import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Read datasets from the csv file
train_df = pd.read_csv('./train.csv')

# Data cleaning?

# Extract the X and Y  for training
X_train = train_df.iloc[:, :-1]
Y_train = train_df.iloc[:, -1]

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, Y_train)

# Predict Output
# predicted = model.predict()