import math
import pandas as pd

class ID3pure:
    def __init__(self):
        self.tree = None
        self.target_col = None

    # 1. Tính Entropy
    def calc_entropy(self, y):
        counts = y.value_counts()
        return -sum(
            (count / len(y)) * math.log2(count / len(y))
            for count in counts
        )

    # 2. Xây dựng cây quyết định
    def build_tree(self, df):
        y = df[self.target_col]

        # Dừng nếu tập dữ liệu thuần nhất
        if len(y.unique()) == 1:
            return y.iloc[0]

        # Dừng nếu không còn thuộc tính
        features = [col for col in df.columns if col != self.target_col]
        if not features:
            return y.mode().iloc[0]

        # Tìm thuộc tính có Information Gain lớn nhất
        base_entropy = self.calc_entropy(y)
        best_gain, best_feat = -1, None

        for feat in features:
            weighted_entropy = sum(
                (len(sub) / len(df)) * self.calc_entropy(sub[self.target_col])
                for _, sub in df.groupby(feat)
            )

            gain = base_entropy - weighted_entropy

            if gain > best_gain:
                best_gain, best_feat = gain, feat

        # Phân nhánh đệ quy
        tree = {best_feat: {}}

        for val, sub_df in df.groupby(best_feat):
            sub_clean = sub_df.drop(columns=[best_feat])
            tree[best_feat][val] = self.build_tree(sub_clean)

        return tree

    # 3. Huấn luyện mô hình
    def fit(self, df, target_col):
        self.target_col = target_col
        self.tree = self.build_tree(df)
        return self

    # 4. Dự đoán
    def predict_one(self, sample, tree=None):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return tree

        feat = next(iter(tree))
        val = sample.get(feat)

        if val in tree[feat]:
            return self.predict_one(sample, tree[feat][val])

        return None


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class ID3:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_tree(self):
        return self.model.tree_


if __name__ == '__main__':
    df = pd.read_csv(
        r'D:\KhaiPhaDuLieu\TX2_KhaiPha\bai so 2. cay quyet dinh\on\PlayTennis.csv'
    )

    model1 = ID3pure()
    model1.fit(df, 'PlayTennis')

    model2 = ID3()

    X = pd.get_dummies(df.drop(columns=['PlayTennis']))
    y = df['PlayTennis']

    model2.fit(X, y)

    x_sample = pd.Series({
        'Outlook': 'Sunny',
        'Temperature': 'Cool',
        'Humidity': 'High',
        'Wind': 'Strong'
    })

    result = pd.DataFrame([
        ['ID3 Pure', model1.predict_one(x_sample.to_dict())],
        ['ID3 Library', model2.predict(pd.get_dummies(
            pd.DataFrame([x_sample])
        ).reindex(columns=X.columns, fill_value=0))[0]]
    ], columns=['Model', 'Prediction'])

    print(result)