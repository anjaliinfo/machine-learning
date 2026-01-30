import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data.csv")

# Encode categorical data
le_location = LabelEncoder()
le_category = LabelEncoder()

data['Location'] = le_location.fit_transform(data['Location'])
data['Category'] = le_category.fit_transform(data['Category'])

# Features and target
X = data[['Size', 'Bedrooms', 'Age', 'Parking', 'Location']]
y = data['Category']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Predict new house
new_house = [[1250, 3, 4, 1, le_location.transform(['City'])[0]]]
prediction = rf_model.predict(new_house)
print("Predicted Category:",
      le_category.inverse_transform(prediction)[0])
