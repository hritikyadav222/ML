# EXPERIMENT 8: DBSCAN CLUSTERING
#
# Aim:
# To implement DBSCAN for identifying clusters and noise points.
#
# Theory:
# DBSCAN is a density-based clustering algorithm. Epsilon (eps) defines the
# neighbourhood radius, while min_samples defines the minimum number of points
# required to form a dense region. Points labelled -1 are considered noise.
#
# Algorithm:
# 1. Import the required libraries.
# 2. Generate a sample dataset.
# 3. Select eps and min_samples.
# 4. Apply DBSCAN and obtain cluster labels.
# 5. Count clusters and noise points.
# 6. Visualize the clusters.
#
# Output:
# Displays cluster labels, cluster count, noise count and a graph.
#
# Result:
# DBSCAN identifies dense clusters and noise points.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate sample data
X, _ = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=42
)

# Apply DBSCAN
model = DBSCAN(eps=0.2, min_samples=5)
labels = model.fit_predict(X)

# Count clusters and noise
unique_labels = set(labels)
cluster_count = len(unique_labels - {-1})
noise_count = int(np.sum(labels == -1))

print("Cluster Labels:", sorted(unique_labels))
print("Number of Clusters:", cluster_count)
print("Number of Noise Points:", noise_count)

# Visualize clusters
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    s=25
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")
plt.show()

# Core points meet the density requirement.
# Border points are near a core point but do not meet that requirement themselves.
# Noise points do not belong to any cluster and are labelled -1.
#
# Result: DBSCAN groups dense points and identifies noise.