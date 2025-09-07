import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_classification  # Changed to make_classification for imbalanced dataset
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def incremental_kmeans(X, k):
    m = X.shape[0]
    if m == 0:
        raise ValueError("Dataset is empty")
    
    k = min(10, k) 
    c = [np.mean(X, axis=0)]
    t = 1
    
    while t < k:
        best_inertia = float('inf')
        best_centers = None
        
        sample_size = min(200, m)
        indices = np.random.choice(m, sample_size, replace=False)
        for p in indices:
            init_centers = c + [X[p]]
            current_centers = np.array(init_centers)
            
            distances = pairwise_distances(X, current_centers)
            labels = np.argmin(distances, axis=1)
            
            new_centers = []
            for i in range(t + 1):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    new_centers.append(current_centers[i])
            
            new_distances = pairwise_distances(X, np.array(new_centers))
            inertia = np.sum(np.min(new_distances, axis=1) ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = new_centers
        
        if best_centers is not None:
            c = best_centers
            t += 1
    
    return np.array(c)

def incremental_smote(X, y, irt=2.0):
    minority = X[y == 0]
    majority = X[y == 1]
    N = len(minority)
    J = len(majority)
    
    Y = []
    k = J - N
    F = []
    
    centers = incremental_kmeans(X, k)
    distances = pairwise_distances(X, centers)
    labels = np.argmin(distances, axis=1)
    
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_minority = np.sum(y[cluster_indices] == 0)
        cluster_majority = np.sum(y[cluster_indices] == 1)
        imbalance_ratio = (cluster_majority + 1) / (cluster_minority + 1)
        if imbalance_ratio < irt:
            F.append(i)
    
    w_f_list = []
    for f in F:
        cluster_indices = np.where(labels == f)[0]
        minority_indices = cluster_indices[y[cluster_indices] == 0]
        if len(minority_indices) == 0:
            continue
        cluster_data = X[minority_indices]
        cluster_center = centers[f]
        distances = pairwise_distances(cluster_data, [cluster_center], metric='euclidean')
        w_f = np.sum(distances) / len(cluster_data) if len(cluster_data) > 0 else 0
        w_f_list.append(w_f)
    
    total_w = np.sum(w_f_list)
    weights = [w_f / total_w if total_w > 0 else 0 for w_f in w_f_list]
    sorted_indices = np.argsort(weights)
    F = [F[i] for i in sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    
    for f, weight in zip(F, weights):
        num_generation = int(k * weight) + 1
        if k - num_generation < 0:
            num_generation = k
        k -= num_generation
        
        cluster_indices = np.where(labels == f)[0]
        minority_indices = cluster_indices[y[cluster_indices] == 0]
        if len(minority_indices) == 0:
            continue
        cluster_data = X[minority_indices]
        count_f = len(cluster_data)
        
        a = 1 / (int(num_generation / count_f) + 2)
        for i in range(1, num_generation + 1):
            alpha = a * (i / count_f)
            j = i % count_f
            center = centers[f]
            data_point = cluster_data[j]
            new_point = center + (data_point - center) * alpha
            Y.append(new_point)
    
    Y = np.array(Y)
    
    if len(Y) > 0:
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X, y)
        y_pred = knn.predict(Y)
        distances_to_majority = pairwise_distances(Y, X[y == 1])
        min_distances = np.min(distances_to_majority, axis=1)
        Y = Y[(y_pred == 0) & (min_distances > 0.5)]
    else:
        Y = np.array([])
    
    return Y, centers

if __name__ == "__main__":

    X, y = make_classification(n_samples=200, n_features=5, n_informative=2, n_redundant=0, n_clusters_per_class=1, weights=[0.3, 0.7], flip_y=0.01, random_state=100)
    
    Y, centers = incremental_smote(X, y, irt=2.0)
    
    X_synthetic = Y
    y_synthetic = np.zeros(len(Y))
    X_new = np.vstack([X, X_synthetic]) if len(Y) > 0 else X
    y_new = np.hstack([y, y_synthetic]) if len(Y) > 0 else y
    
    print(f"Original dataset size: {X.shape}, Minority: {np.sum(y == 0)}, Majority: {np.sum(y == 1)}")
    print(f"Number of synthetic samples: {len(Y)}")
    
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Minority (Original)')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Majority')
    if len(Y) > 0:
        plt.scatter(X_new[len(X):][:, 0], X_new[len(X):][:, 1], c='green', marker='x', label='Synthetic Minority')
    plt.scatter(centers[:, 0], centers[:, 1], c='yellow', marker='*', s=200, label='Cluster Centers')
    plt.legend()
    plt.title("Incremental SMOTE Results on Imbalanced Synthetic Dataset")
    plt.show()