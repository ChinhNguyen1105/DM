import pandas as pd


class DataFrameNaiveBayes:

    def __init__(self):
        self.p_prior = {}
        self.p_cond = {}
        self.classes = []

    def fit(self, df, target_col):
        total_samples = len(df)
        self.classes = df[target_col].unique()

        features = [
            col for col in df.columns
            if col != target_col
        ]

        class_counts = df[target_col].value_counts()

        self.p_prior = class_counts / total_samples

        for c in self.classes:
            self.p_cond[c] = {}

            df_c = df[df[target_col] == c]
            n_c = len(df_c)

            for col in features:
                val_counts = df_c[col].value_counts()
                self.p_cond[c][col] = (val_counts / n_c).to_dict()

        print(self.p_cond)

    def predict(self, sample):
        scores = {}

        for c in self.classes:
            prob = self.p_prior[c]

            for col, val in sample.items():
                cond_prob = self.p_cond[c].get(
                    col, {}
                ).get(val, 0.0)

                prob *= cond_prob

            scores[c] = prob

        print("Điểm số tính được cho các lớp:", scores)
        return max(scores, key=scores.get)