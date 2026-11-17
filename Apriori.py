from itertools import combinations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


class AprioriPure:
    def __init__(self, min_support=0.01, min_conf=0.5):
        self.min_support = min_support
        self.min_conf = min_conf
        self.frequent_itemsets = {}
        self.rules = []
        self.n_transactions = 0

    def get_support_count(self, itemset, dataset):
        return int(dataset[list(itemset)].all(axis=1).sum())

    def fit(self, dataset):
        self.n_transactions = len(dataset)
        self.frequent_itemsets = {}
        self.rules = []

        if self.n_transactions == 0:
            raise ValueError("Dataset khong duoc rong.")

        if not 0 < self.min_support <= 1:
            raise ValueError("min_support phai nam trong (0, 1].")

        min_count = self.min_support * self.n_transactions

        items = sorted(dataset.columns.tolist())

        current_frequent = [
            [item] for item in items
            if self.get_support_count([item], dataset) >= min_count
        ]

        for itemset in current_frequent:
            self.frequent_itemsets[tuple(itemset)] = (
                self.get_support_count(itemset, dataset)
            )

        while current_frequent:
            candidates = []

            for l1, l2 in combinations(current_frequent, 2):
                if l1[:-1] == l2[:-1]:
                    candidate = sorted(set(l1) | set(l2))

                    if len(candidate) == len(l1) + 1:
                        if candidate not in candidates:
                            candidates.append(candidate)

            next_frequent = []

            for candidate in candidates:
                subsets = combinations(candidate, len(candidate) - 1)

                if not all(
                    tuple(subset) in self.frequent_itemsets
                    for subset in subsets
                ):
                    continue

                sup = self.get_support_count(candidate, dataset)

                if sup >= min_count:
                    next_frequent.append(candidate)
                    self.frequent_itemsets[tuple(candidate)] = sup

            current_frequent = next_frequent

        return self.frequent_itemsets

    def generate_association_rules(self):
        self.rules = []

        for itemset, sup_xy in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = tuple(
                        item for item in itemset
                        if item not in antecedent
                    )

                    sup_x = self.frequent_itemsets.get(antecedent, 0)
                    sup_y = self.frequent_itemsets.get(consequent, 0)

                    if sup_x == 0 or sup_y == 0:
                        continue

                    confidence = sup_xy / sup_x
                    lift = confidence / (sup_y / self.n_transactions)

                    if confidence >= self.min_conf:
                        self.rules.append({
                            "Antecedent": antecedent,
                            "Consequent": consequent,
                            "Support Count": sup_xy,
                            "Support": sup_xy / self.n_transactions,
                            "Confidence": confidence,
                            "Lift": lift
                        })

        return self.rules

    def get_frequent_itemsets(self):
        return pd.DataFrame([
            {
                "Itemset": itemset,
                "Support Count": count,
                "Support": count / self.n_transactions
            }
            for itemset, count in self.frequent_itemsets.items()
        ])

    def get_rules(self):
        return pd.DataFrame(self.rules)


class AprioriImport:
    def __init__(self, min_support=0.01, min_conf=0.5):
        self.min_support = min_support
        self.min_conf = min_conf
        self.frequent_itemsets = pd.DataFrame()
        self.rules = pd.DataFrame()
        self.dataset_encoded = None

    def fit(self, dataset):
        if len(dataset) == 0:
            raise ValueError("Dataset khong duoc rong.")

        self.dataset_encoded = dataset.astype(bool)

        self.frequent_itemsets = apriori(
            self.dataset_encoded,
            min_support=self.min_support,
            use_colnames=True
        )

        return self.frequent_itemsets

    def generate_association_rules(self):
        if self.frequent_itemsets.empty:
            self.rules = pd.DataFrame()
            return self.rules

        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_conf
        )

        return self.rules

    def get_frequent_itemsets(self):
        if self.frequent_itemsets.empty:
            return pd.DataFrame()

        result = self.frequent_itemsets.copy()

        result["Support Count"] = (
            result["support"] * len(self.dataset_encoded)
        ).round().astype(int)

        return result.rename(columns={
            "itemsets": "Itemset",
            "support": "Support"
        })[["Itemset", "Support Count", "Support"]]

    def get_rules(self):
        if self.rules.empty:
            return pd.DataFrame()

        return self.rules[[
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]].rename(columns={
            "antecedents": "Antecedent",
            "consequents": "Consequent",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift"
        })


if __name__ == "__main__":
    from mlxtend.preprocessing import TransactionEncoder

    df = pd.read_csv(
        'D:/KhaiPhaDuLieu/TX2_KhaiPha/bai so 1. Luat ket hop/store_data.csv',
        header=None
    )

    transactions = [
        row.dropna().tolist()
        for _, row in df.iterrows()
    ]

    te = TransactionEncoder()

    data = pd.DataFrame(
        te.fit_transform(transactions),
        columns=te.columns_
    )

    min_support = 50 / len(data)
    min_conf = 0.5

    model1 = AprioriPure(min_support, min_conf)
    model2 = AprioriImport(min_support, min_conf)

    model1.fit(data)
    model1.generate_association_rules()

    model2.fit(data)
    model2.generate_association_rules()

    print("=== APRIORI PURE ===")
    print(model1.get_frequent_itemsets())
    print(model1.get_rules())

    print("\n=== APRIORI MLXTEND ===")
    print(model2.get_frequent_itemsets())
    print(model2.get_rules())
