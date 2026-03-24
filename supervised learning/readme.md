# Supervised Learning 

Supervised Learning is a type of Machine Learning (ML) where a model learns from data that already has the correct answers.

# Supervised Learning Types : 

# 1. Classification

    from sklearn.neighbors import KNeighborsClassifier
    
    # Input data
    X = [[1],[2],[3],[4]]
    # Output labels
    y = ['Spam','Spam','Not Spam','Not Spam']
    
    model = KNeighborsClassifier(n_neighbors=1)
    
    model.fit(X,y)
    
    print(model.predict([[2]]))

    <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/47e53a57-e9a6-47d7-9bb5-7d84c85c765b" />

  # 2. Regression

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

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b26f5d52-e269-41d4-a31c-a9784eb28221" />




