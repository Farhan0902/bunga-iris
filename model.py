import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#random seed
seed = 42

# Load dataset
iris_df = pd.read_csv('data/iris.csv')
iris_df.sample(frac=1,random_state=seed)

# selecting features and target data
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Species']

# Splitting dataset into training and testing sets
# 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# Create an instance of RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy }')

# Save the trained model to a file
joblib.dump(clf, 'rf_model.sav')