import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# Load a small subset of the Iris dataset
iris = load_iris()
data = iris.data[:6]

# ---------- Proximity Matrix Function ----------
def proximity_matrix(data):
    n = data.shape[0]
    proximity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(data[i] - data[j])
            proximity_matrix[i, j] = distance
            proximity_matrix[j, i] = distance
    return proximity_matrix

# ---------- Dendrogram Plot Function ----------
def plot_dendrogram(data, method):
    linkage_matrix = linkage(data, method=method)
    dendrogram(linkage_matrix)
    plt.title(f'Dendrogram - {method.capitalize()} Linkage')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

# ---------- Output ----------
print("Proximity Matrix:")
print(proximity_matrix(data))

# Plot Dendrograms
plot_dendrogram(data, 'single')
plot_dendrogram(data, 'complete')
