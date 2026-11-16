import pandas as pd

class DataFrameNaiveBayes:

    def __init__(self):
        self.p_prior = {}  # Xác suất tiên nghiệm P(c)
        self.p_cond = {}   # Xác suất có điều kiện P(x_j | c)
        self.classes = []

    def fit(self, df, target_col):
        """
        Huấn luyện trực tiếp trên Pandas DataFrame.
        """

        total_samples = len(df)
        self.classes = df[target_col].unique()

        features = [
            col for col in df.columns
            if col != target_col
        ]

        # 1. Tính xác suất tiên nghiệm P(c)
        class_counts = df[target_col].value_counts()

        self.p_prior = class_counts / total_samples


        # 2. Tính xác suất có điều kiện P(x_j = v | c)
        for c in self.classes:
            self.p_cond[c] = {}
            # Lọc các dòng thuộc lớp c
            df_c = df[df[target_col] == c]
            n_c = len(df_c)

            for col in features:
                # Đếm tần suất xuất hiện
                val_counts = df_c[col].value_counts()
                # Tính xác suất có điều kiện
                self.p_cond[c][col] = (val_counts / n_c).to_dict()
        print(self.p_cond)

    def predict(self, sample):
        """
        Dự đoán nhãn cho một mẫu mới.
        Sample có thể là dict hoặc pd.Series.
        """
        scores = {}

        for c in self.classes:
            # Bắt đầu với xác suất tiên nghiệm P(c)
            prob = self.p_prior[c]
            # Nhân xác suất có điều kiện của từng thuộc tính
            for col, val in sample.items():
                cond_prob = self.p_cond[c].get(
                    col, {}
                ).get(val, 0.0)
                prob *= cond_prob
            scores[c] = prob

        print("Điểm số tính được cho các lớp:", scores)
        return max(scores, key=scores.get)








