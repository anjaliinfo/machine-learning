import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("data.csv")

# Features and label
X = data[['Size', 'Bedrooms']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Predict new house price
new_house = [[1250, 3]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])
