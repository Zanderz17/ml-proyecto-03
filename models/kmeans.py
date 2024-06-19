import numpy as np
import matplotlib.pyplot as plt
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

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=200):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.centroids = np.random.randn(self.n_clusters, X.shape[1])

        for _ in range(max_iterations):
            y = []

            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

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
        clusters = []
        for data_point in X_new:
            distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
            cluster_num = np.argmin(distances)
            clusters.append(cluster_num)
        return np.array(clusters)