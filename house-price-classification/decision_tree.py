import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("data.csv")

# Encode categorical columns
le_location = LabelEncoder()
le_category = LabelEncoder()

data['Location'] = le_location.fit_transform(data['Location'])
data['Category'] = le_category.fit_transform(data['Category'])

# Features and target
X = data[['Size', 'Bedrooms', 'Age', 'Parking', 'Location']]
y = data['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Decision Tree model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict new house
new_house = [[1250, 3, 4, 1, le_location.transform(['City'])[0]]]
prediction = model.predict(new_house)
print("Predicted Category:",
      le_category.inverse_transform(prediction)[0])
