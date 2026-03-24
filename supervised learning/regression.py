from sklearn.linear_model import LinearRegression

# Input (Years of experience)
X = [[1],[2],[3],[4],[5]]

# Output (Salary)
y = [20000,25000,30000,35000,40000]

# Create model
model = LinearRegression()

# Train model
model.fit(X,y)

# Predict salary for 6 years experience
prediction = model.predict([[6]])

print(prediction)