from sklearn.neighbors import KNeighborsClassifier

# Input data
X = [[1],[2],[3],[4]]
# Output labels
y = ['Spam','Spam','Not Spam','Not Spam']

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X,y)

print(model.predict([[2]]))