from sklearn.decomposition import PCA
import numpy as np

X = np.array([
    [1,2],
    [3,4],
    [5,6]
])

pca = PCA(n_components=1)

result = pca.fit_transform(X)

print(result)