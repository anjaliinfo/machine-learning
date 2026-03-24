# Unsupervised Learning
Model learns from only input data

# Unsupervised Learning types:

# 1. Clustering

<img width="611" height="72" alt="image" src="https://github.com/user-attachments/assets/5fe4c6be-fa96-4fe3-9a46-cd8700ee73d8" /> 

# Customer data. age  spending
    from sklearn.cluster import KMeans

    X = [
        [20,2000],
        [22,2500],
        [35,8000],
        [40,8500]
    ]
    
    model = KMeans(n_clusters=2)
    
    model.fit(X)
    
    print(model.labels_)
  
# 2. Dimensionality reduction
    
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

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/32deb185-0b03-4e74-ad5a-b7569ebb59d7" />


