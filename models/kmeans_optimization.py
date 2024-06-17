import numpy as np
from sklearn.metrics import pairwise_distances_argmin 
"""
Modelo implementado siguiendo como guia el video: https://www.youtube.com/watch?v=5w5iUbTlpMQ&t=626s
El video fue usado como base junto con el código del Colab realizado en clase el día 27 de Mayo de 2024: https://colab.research.google.com/drive/1l_lnIbGiX_pOCeCsH_wCJCZTs9jXDzXJ?usp=sharing
"""

class KMeansClustering:

    def  __init__(self, n_clusters=3, random_state=None, umbral=0.0001):
        self.n_clusters = n_clusters
        self.centroids =  None
        self.random_state = random_state
        self.umbral = umbral

    def fit(self, X, max_iterations=200):

        if self.random_state is not None:
            np.random.seed(self.random_state)
    
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.n_clusters, X.shape[1]))

        for _ in range(max_iterations):
            # Optimización
            y = pairwise_distances_argmin(X, self.centroids)

            cluster_indices = []

            for i in range(self.n_clusters):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])

                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            
            if np.max(self.centroids - np.array(cluster_centers)) < self.umbral:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y

    def predict(self, X_new):
        return pairwise_distances_argmin(X_new, self.centroids)