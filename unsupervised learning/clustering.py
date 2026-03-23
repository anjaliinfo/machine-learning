from sklearn.cluster import KMeans

# Customer data (Age, Spending)
X = [
    [20,2000],
    [22,2500],
    [35,8000],
    [40,8500]
]

model = KMeans(n_clusters=2)

model.fit(X)

print(model.labels_)