import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
train_df = pd.read_csv('train.csv')  
test_df = pd.read_csv('test.csv')
gender_df = pd.read_csv('gender_submission.csv')

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
train_df = train_df[features + ["Survived"]]
test_df = test_df[features]

# Handle missing values
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
test_df["Embarked"].fillna(test_df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
train_df["Sex"] = train_df["Sex"].map({'male': 0, 'female': 1})
test_df["Sex"] = test_df["Sex"].map({'male': 0, 'female': 1})
train_df["Embarked"] = train_df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2})
test_df["Embarked"] = test_df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2})

# Split data into training and validation sets
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_df = test_df.values  # Convert to NumPy array before scaling
test_df = scaler.transform(test_df)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)
test_predictions = model.predict(test_df)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(f"Validation Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Create submission file
submission = pd.DataFrame({"PassengerId": gender_df["PassengerId"], "Survived": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
