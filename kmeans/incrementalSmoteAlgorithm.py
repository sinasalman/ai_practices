import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

def incremental_kmeans(X, k):
    """
    Implements Algorithm 1: Incremental K-Means without library KMeans.
    
    Parameters:
    - X: The dataset (numpy array of shape (m, features))
    - k: Number of clusters
    
    Returns:
    - centers: The centers of the clusters
    """
    m = X.shape[0]
    if m == 0:
        raise ValueError("Dataset is empty")
    
    # Limit k to avoid long computation
    k = min(10, k)  # To optimize and prevent hanging
    
    # Step 2: Initialize first cluster center as mean of all data
    c = [np.mean(X, axis=0)]
    t = 1
    
    # Step 4: While t < k
    while t < k:
        best_inertia = float('inf')
        best_centers = None
        
        # Step 5-6: For each p in 1 to m (sample to optimize)
        sample_size = min(100, m)  # Limit loop size to prevent hanging
        indices = np.random.choice(m, sample_size, replace=False)
        for p in indices:
            # Step 7: Initial centers: current centers + X[p]
            init_centers = c + [X[p]]
            current_centers = np.array(init_centers)
            
            # Assign points to nearest center
            distances = pairwise_distances(X, current_centers)
            labels = np.argmin(distances, axis=1)
            
            # Compute new centers
            new_centers = []
            for i in range(t + 1):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    new_centers.append(current_centers[i])  # Keep old center if cluster is empty
            
            # Compute inertia
            new_distances = pairwise_distances(X, np.array(new_centers))
            inertia = np.sum(np.min(new_distances, axis=1) ** 2)
            
            # Step 8: If better (lower inertia)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = new_centers
        
        # Step 9-11: Update centers if found better
        if best_centers is not None:
            c = best_centers
            t += 1
    
    return np.array(c)

def incremental_smote(X, y, irt=2.0):
    """
    Implements Algorithm 2: Incremental SMOTE with Incremental K-Means.
    
    Parameters:
    - X: Input data (numpy array)
    - y: Labels (numpy array, 0 for minority, 1 for majority)
    - irt: Imbalance ratio threshold
    
    Returns:
    - Y: The set of oversampled minority instances
    """
    
    # Separate minority and majority
    minority = X[y == 0]
    majority = X[y == 1]
    N = len(minority)  # |N|
    J = len(majority)  # |J|
    
    # Step 1: Y <- empty
    Y = []
    
    # Step 2: k = |J| - |N|
    k = J - N
    
    # Step 3: F <- empty
    F = []
    
    # Step 4: clusters <- INCREMENTAL K-MEANS(X, k)
    centers = incremental_kmeans(X, k)
    
    # Get labels by assigning each point to nearest center
    distances = pairwise_distances(X, centers)
    labels = np.argmin(distances, axis=1)
    
    # Step 5-8: For each cluster c
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue  # Skip empty clusters
        cluster_minority = np.sum(y[cluster_indices] == 0)
        cluster_majority = np.sum(y[cluster_indices] == 1)
        
        # Step 6: imbalance ratio
        imbalance_ratio = (cluster_majority + 1) / (cluster_minority + 1)
        
        # Step 7: If < irt
        if imbalance_ratio < irt:
            F.append(i)  # Step 8: Add to F
    
    # Steps 9-11: For f in F, compute w_f
    w_f_list = []
    for f in F:
        cluster_indices = np.where(labels == f)[0]
        cluster_data = X[cluster_indices]
        cluster_center = centers[f]
        
        # Step 10: w_f = sum of distances from the center / count(f)
        distances = pairwise_distances(cluster_data, [cluster_center], metric='euclidean')
        w_f = np.sum(distances) / len(cluster_data)
        w_f_list.append(w_f)
    
    # Step 11: total_w = sum w_f
    total_w = np.sum(w_f_list)
    
    # Steps 12-13: Compute weight(f) = w_f / total_w
    weights = [w_f / total_w if total_w > 0 else 0 for w_f in w_f_list]
    
    # Step 14: Sort weight(f) values in ascending order
    sorted_indices = np.argsort(weights)
    F = [F[i] for i in sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    
    # Step 15-19: For f in F (sorted)
    for f, weight in zip(F, weights):
        # Step 16: numberofgeneration(f) = floor(k * weight(f)) + 1
        num_generation = int(k * weight) + 1
        
        # Step 17-18: If k - num < 0, numberofgeneration(f) = k
        if k - num_generation < 0:
            num_generation = k
        
        # Step 19: k = k - numberofgeneration(f)
        k -= num_generation
        
        # Steps 20-26: Generate synthetic points
        cluster_indices = np.where(labels == f)[0]
        cluster_data = X[cluster_indices]
        count_f = len(cluster_data)
        
        # Step 21: a = 1 / (floor(num_gen / count(f)) + 2)
        a = 1 / (int(num_generation / count_f) + 2)
        
        # Step 22: For i=1 to numberofgeneration(f)
        for i in range(1, num_generation + 1):
            # Step 23: alpha = a * (i / count(f))
            alpha = a * (i / count_f)
            
            # Step 24: j = i mod count(f)
            j = i % count_f
            
            # Step 25: newpoint = center + (j-th data - center) * alpha
            center = centers[f]
            data_point = cluster_data[j]
            new_point = center + (data_point - center) * alpha
            
            # Step 26: Add to Y
            Y.append(new_point)
    
    # Step 27: Return Y
    return np.array(Y)

# Example Usage with Breast Cancer Dataset
if __name__ == "__main__":
    # Load breast cancer dataset
    data = load_wine()
    X = data.data
    y = data.target  # 0: malignant (minority?), 1: benign
    
    # Apply Incremental SMOTE
    Y = incremental_smote(X, y, irt=2.0)
    
    # For visualization, combine with original data (not part of Algorithm 2)
    X_synthetic = Y
    y_synthetic = np.zeros(len(Y))
    X_new = np.vstack([X, X_synthetic])
    y_new = np.hstack([y, y_synthetic])
    
    # Print results
    print(f"Original dataset size: {X.shape}, Minority: {np.sum(y == 0)}, Majority: {np.sum(y == 1)}")
    print(f"Number of synthetic samples: {len(Y)}")
    
    # Visualize results (using first 2 features for simplicity)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Minority (Original)')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Majority')
    plt.scatter(X_new[len(X):][:, 0], X_new[len(X):][:, 1], c='green', marker='x', label='Synthetic Minority')
    plt.legend()
    plt.title("Incremental SMOTE Results on Breast Cancer Dataset")
    plt.show()