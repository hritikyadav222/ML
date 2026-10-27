# EXPERIMENT 6: DIMENSIONALITY REDUCTION
#
# Aim:
# To implement PCA, SVD and LDA for dimensionality reduction.
#
# Theory:
# PCA reduces dimensions by preserving directions of maximum variance.
# SVD decomposes a matrix into U, singular values and V-transpose.
# LDA uses class labels to find features that separate classes.
#
# Algorithm:
# 1. Import libraries and load the Iris dataset.
# 2. Scale the features.
# 3. Apply PCA to reduce features to two dimensions.
# 4. Apply SVD to the scaled feature matrix.
# 5. Apply LDA using the class labels.
# 6. Display reduced data shapes and PCA variance.
# 7. Plot the reduced data for comparison.
#
# Output:
# Displays transformed shapes, explained variance and three plots.
#
# Result:
# PCA, SVD and LDA are implemented and compared.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply SVD
U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)
X_svd = U[:, :2] * S[:2]

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Display results
print("Original Data Shape:", X.shape)
print("PCA Reduced Shape:", X_pca.shape)
print("SVD Reduced Shape:", X_svd.shape)
print("LDA Reduced Shape:", X_lda.shape)
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total PCA Variance Retained:",
      sum(pca.explained_variance_ratio_))

# Plot PCA
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA")
plt.show()

# Plot SVD
plt.figure()
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap="viridis")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("SVD")
plt.show()

# Plot LDA
plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="viridis")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA")
plt.show()

# Result: The three methods reduce the dataset to two dimensions.