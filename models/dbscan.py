from scipy.spatial import KDTree
"""
Implementación basada en el Colab realizado el día 29 de mayo: https://colab.research.google.com/drive/1GySU0Z5etcSk93DvQ9K46aM1ijUK64f-?usp=sharing
"""
class DBSCAN:
    def __init__(self, radio, vecinos_min):
        self.radio = radio
        self.vecinos_min = vecinos_min
        self.labels = None
        self.tree = None
    
    def fit(self, data):
        self.search_tree = KDTree(data)
        self.assignment = [None] * len(data)
        num_clusters = 0
        
        for index in range(len(data)):
            if self.assignment[index] is not None:
                continue
            
            current_sample = data[index]
            nearby_points = self.search_tree.query_ball_point(current_sample, self.radio)
            
            if len(nearby_points) < self.vecinos_min:
                self.assignment[index] = -1  
                continue
            
            num_clusters += 1
            active_cluster = num_clusters
            self.assignment[index] = active_cluster
            exploration_set = set(nearby_points)
            
            while exploration_set:
                next_index = exploration_set.pop()
                
                if self.assignment[next_index] == -1:
                    self.assignment[next_index] = active_cluster
                
                if self.assignment[next_index] is not None:
                    continue
                
                next_sample = data[next_index]
                expanded_neighbors = self.search_tree.query_ball_point(next_sample, self.radio)
                self.assignment[next_index] = active_cluster
                
                if len(expanded_neighbors) >= self.vecinos_min:
                    exploration_set.update(expanded_neighbors)
        
        return self.assignment
