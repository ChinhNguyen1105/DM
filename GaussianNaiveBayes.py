import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class PureGaussianNB:
    def __init__(self):
        self.classes = []
        self.p_prior = {}
        self.means = {}
        self.stds = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]

            self.p_prior[c] = len(X_c) / len(X)
            self.means[c] = np.mean(X_c, axis=0)
            self.stds[c] = np.std(X_c, axis=0, ddof=0)

        return self

    def gaussian_pdf(self, x, mean, std):
        var = np.maximum(std ** 2, 1e-9)

        exponent = np.exp(
            -((x - mean) ** 2) / (2 * var)
        )

        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for sample in X:
            scores = {}

            for c in self.classes:
                likelihoods = self.gaussian_pdf(
                    sample,
                    self.means[c],
                    self.stds[c]
                )

                scores[c] = (
                    np.log(self.p_prior[c])
                    + np.sum(np.log(likelihoods))
                )

            predictions.append(max(scores, key=scores.get))

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class LibraryGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.model = GaussianNB(
            var_smoothing=var_smoothing
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

if __name__ == '__main__':
    df = pd.read_csv('D:\KhaiPhaDuLieu\TX2_KhaiPha\Data\diabetes.txt')
    x_train = df.loc[:,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
    x_sample = pd.DataFrame(
       [ {'Pregnancies': 10
            , 'Glucose': 110
            , 'BloodPressure': 80
            , 'SkinThickness': 30
            , 'Insulin': 5
            , 'BMI': 31
            , 'DiabetesPedigreeFunction': 0.627
            , 'Age': 45}]
    )
    y_train = df.loc[:,'Outcome']

    model1 = PureGaussianNB()
    model2 = LibraryGaussianNB()


    model1.fit(x_train,y_train )
    model2.fit(x_train,y_train )

    result = pd.DataFrame(
        data=[
            ['model1', model1.predict(x_sample)[0]],
            ['model2', model2.predict(x_sample)[0]]
        ],
        columns=['model', 'predict']
    )

    print(result)