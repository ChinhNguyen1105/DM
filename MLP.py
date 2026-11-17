import numpy as np
from sklearn.metrics import accuracy_score


class PureMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, a):
        return a * (1.0 - a)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        error_output = output - y
        delta_output = error_output * self._sigmoid_derivative(output)

        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * self._sigmoid_derivative(self.a1)

        self.W2 -= self.lr * np.dot(self.a1.T, delta_output) / m
        self.b2 -= self.lr * np.sum(delta_output, axis=0, keepdims=True) / m

        self.W1 -= self.lr * np.dot(X.T, delta_hidden) / m
        self.b1 -= self.lr * np.sum(delta_hidden, axis=0, keepdims=True) / m

    def fit(self, X, y, epochs=2000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        output = self.forward(np.asarray(X, dtype=float))

        if output.shape[1] == 1:
            return (output >= 0.5).astype(int).ravel()

        return np.argmax(output, axis=1)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)



from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class LibraryMLP:
    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    ):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 1. Đọc dữ liệu
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    df = pd.read_csv(
        r'D:\KhaiPhaDuLieu\TX2_KhaiPha\Data\diabetes.txt',
        sep=r'[\s,]+',
        header=None,
        names=columns,
        engine='python'
    )

    # 2. Chia X, y
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Khởi tạo mô hình
    model1 = PureMLP(
        input_size=X_train.shape[1],
        hidden_size=8,
        output_size=1
    )

    model2 = LibraryMLP()

    # 5. Huấn luyện
    model1.fit(X_train, y_train, epochs=2000)
    model2.fit(X_train, y_train)

    # 6. Đánh giá
    result = pd.DataFrame([
        ['Pure MLP', model1.evaluate(X_test, y_test)],
        ['Library MLP', model2.evaluate(X_test, y_test)]
    ], columns=['Model', 'Accuracy'])

    print(result)