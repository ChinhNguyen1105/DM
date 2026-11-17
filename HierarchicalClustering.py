
import numpy as np


class PureHierarchicalClustering:
    def __init__(self, n_clusters=2, linkage="single"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def _euclidean_dist(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def _cluster_distance(self, c1, c2):
        distances = [
            self._euclidean_dist(p1, p2)
            for p1 in c1
            for p2 in c2
        ]

        if self.linkage == "single":
            return min(distances)
        elif self.linkage == "complete":
            return max(distances)
        elif self.linkage == "average":
            return np.mean(distances)
        else:
            raise ValueError("linkage must be 'single', 'complete' or 'average'")

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        n_samples = len(X)

        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError("Invalid number of clusters")

        clusters = {i: [X[i]] for i in range(n_samples)}
        cluster_indices = {i: [i] for i in range(n_samples)}

        while len(clusters) > self.n_clusters:
            keys = list(clusters.keys())
            min_dist = float("inf")
            to_merge = (None, None)

            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    k1, k2 = keys[i], keys[j]

                    dist = self._cluster_distance(
                        clusters[k1],
                        clusters[k2]
                    )

                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (k1, k2)

            k1, k2 = to_merge

            clusters[k1].extend(clusters[k2])
            cluster_indices[k1].extend(cluster_indices[k2])

            del clusters[k2]
            del cluster_indices[k2]

        self.labels_ = np.zeros(n_samples, dtype=int)

        for label_id, indices in enumerate(cluster_indices.values()):
            self.labels_[indices] = label_id

        return self



import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


class LibraryHierarchicalClustering:
    def __init__(self, n_clusters=2, linkage="ward"):
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    @property
    def labels_(self):
        return self.model.labels_

    def plot_dendrogram(self, X, linkage="ward"):
        X = np.asarray(X, dtype=float)

        Z = sch.linkage(X, method=linkage)

        plt.figure(figsize=(8, 5))
        sch.dendrogram(Z)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Data Samples")
        plt.ylabel("Euclidean Distance")
        plt.show()

if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # 1. Đọc dữ liệu
    df = pd.read_csv(
        r'D:\KhaiPhaDuLieu\TX2_KhaiPha\Data\diabetes.txt'
    )

    # 2. Lấy Features
    X = df.drop(columns=['Outcome'])

    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. Khởi tạo mô hình
    model1 = PureHierarchicalClustering(
        n_clusters=2,
        linkage="single"
    )

    model2 = LibraryHierarchicalClustering(
        n_clusters=2,
        linkage="single"
    )

    # 5. Huấn luyện
    model1.fit(X)
    model2.fit(X)

    # 6. Hiển thị kết quả
    result = pd.DataFrame({
        'Pure Hierarchical': model1.labels_,
        'Library Hierarchical': model2.labels_
    })

    print(result)

    # 7. Vẽ Dendrogram
    model2.plot_dendrogram(X, linkage="single")