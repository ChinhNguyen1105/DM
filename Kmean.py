
import numpy as np
import pandas as pd


class PureKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=float)

        np.random.seed(self.random_state)

        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples")

        random_indices = np.random.choice(
            n_samples,
            self.n_clusters,
            replace=False
        )

        self.cluster_centers_ = X[random_indices].copy()

        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
                axis=2
            )

            labels = np.argmin(distances, axis=1)

            new_centers = np.array([
                X[labels == k].mean(axis=0)
                if np.any(labels == k)
                else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])

            center_shift = np.linalg.norm(
                new_centers - self.cluster_centers_
            )

            self.cluster_centers_ = new_centers
            self.labels_ = labels

            if center_shift < self.tol:
                break

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=float)

        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
            axis=2
        )

        return np.argmin(distances, axis=1)



from sklearn.cluster import KMeans as SklearnKMeans


class LibraryKMeans:
    def __init__(self, n_clusters=2, max_iter=300, random_state=42):
        self.model = SklearnKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init=10
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def cluster_centers_(self):
        return self.model.cluster_centers_

    @property
    def labels_(self):
        return self.model.labels_

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
    model1 = PureKMeans(n_clusters=2)
    model2 = LibraryKMeans(n_clusters=2)

    # 5. Huấn luyện
    model1.fit(X)
    model2.fit(X)

    # 6. Kết quả
    result = pd.DataFrame({
        'Pure KMeans': model1.labels_,
        'Library KMeans': model2.labels_
    })

    print(result)

    print("\nPure Centers:")
    print(model1.cluster_centers_)

    print("\nLibrary Centers:")
    print(model2.cluster_centers_)