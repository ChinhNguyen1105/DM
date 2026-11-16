from itertools import combinations
import pandas as pd


class Apriori:
    def __init__(self, min_count=2, min_conf=0.5):
        self.min_count = min_count
        self.min_conf = min_conf
        self.frequent_itemsets = {}
        self.rules = []

    # 1. Tính Support Count
    def get_support_count(self, itemset, dataset):
        return sum(
            set(itemset).issubset(set(transaction))
            for transaction in dataset
        )

    # 2. Sinh tập mục phổ biến
    def fit(self, dataset):
        items = sorted(set(
            item for transaction in dataset
            for item in transaction
        ))

        # Sinh L1
        current_frequent = [
            [item] for item in items
            if self.get_support_count([item], dataset) >= self.min_count
        ]

        self.frequent_itemsets = {}

        for itemset in current_frequent:
            self.frequent_itemsets[tuple(itemset)] = (
                self.get_support_count(itemset, dataset)
            )

        # Sinh Lk (k >= 2)
        while current_frequent:
            candidates = []
            n = len(current_frequent)

            # Bước Join
            for i in range(n):
                for j in range(i + 1, n):
                    l1, l2 = current_frequent[i], current_frequent[j]

                    if l1[:-1] == l2[:-1]:
                        candidate = sorted(set(l1) | set(l2))

                        if len(candidate) == len(l1) + 1:
                            if candidate not in candidates:
                                candidates.append(candidate)

            # Bước Prune và kiểm tra Support
            next_frequent = []

            for candidate in candidates:
                sup = self.get_support_count(candidate, dataset)

                if sup >= self.min_count:
                    next_frequent.append(candidate)
                    self.frequent_itemsets[tuple(candidate)] = sup

            current_frequent = next_frequent

        return self.frequent_itemsets

    # 3. Sinh luật kết hợp
    def generate_association_rules(self):
        self.rules = []

        for itemset, sup_xy in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            # Sinh các tập con làm vế trái
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = tuple(
                        item for item in itemset
                        if item not in antecedent
                    )

                    sup_x = self.frequent_itemsets[antecedent]
                    conf = sup_xy / sup_x

                    if conf >= self.min_conf:
                        self.rules.append({
                            "Antecedent": antecedent,
                            "Consequent": consequent,
                            "Support Count": sup_xy,
                            "Confidence": conf
                        })

        return self.rules

    # 4. Chuyển tập phổ biến sang DataFrame
    def get_frequent_itemsets(self):
        return pd.DataFrame([
            {
                "Itemset": itemset,
                "Support Count": count
            }
            for itemset, count in self.frequent_itemsets.items()
        ])

    # 5. Chuyển luật kết hợp sang DataFrame
    def get_rules(self):
        return pd.DataFrame(self.rules)
