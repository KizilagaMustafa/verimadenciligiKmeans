import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Örnek veri oluşturma bloğu
np.random.seed(42)
X = np.concatenate([np.random.normal(0, 1, (50, 2)),
                    np.random.normal(5, 1, (50, 2)),
                    np.random.normal(10, 1, (50, 2))])

# K-means modelini oluşturma bloğu
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Küme merkezleri ve kümeler
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Veri ve küme merkezlerini görselleştirme bloğu
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='w', label='Veri Noktaları')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Küme Merkezleri')
plt.title('KMeans Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.show()



