
import pandas as pd
from itertools import combinations


class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None


class AssociationRules:
    def __init__(self, min_confidence=0.5, min_lift=1.0):
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.result = None

    def fit(self, frequent_items):
        support_dict = {
            frozenset(row["itemsets"]): row["support"]
            for _, row in frequent_items.iterrows()
        }

        rules = []

        for itemset, support in support_dict.items():
            if len(itemset) < 2:
                continue

            for size in range(1, len(itemset)):
                for antecedent in combinations(itemset, size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent

                    support_a = support_dict.get(antecedent, 0)
                    support_b = support_dict.get(consequent, 0)

                    if support_a == 0 or support_b == 0:
                        continue

                    confidence = support / support_a
                    lift = confidence / support_b

                    if (
                        confidence >= self.min_confidence
                        and lift >= self.min_lift
                    ):
                        rules.append({
                            "antecedents": set(antecedent),
                            "consequents": set(consequent),
                            "support": support,
                            "confidence": confidence,
                            "lift": lift
                        })

        self.result = pd.DataFrame(
            rules,
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        )

        if not self.result.empty:
            self.result = (
                self.result
                .sort_values(
                    ["confidence", "lift"],
                    ascending=False
                )
                .reset_index(drop=True)
            )

        return self.result

    def get_rules(self):
        return self.result


class FPGrowthPure:
    def __init__(
        self,
        min_support=0.5,
        min_confidence=0.5,
        min_lift=1.0
    ):
        self.min_support = min_support
        self.freq_items = {}
        self.result = None
        self.rule_model = AssociationRules(
            min_confidence,
            min_lift
        )
        self.rules = None

    def update_header(self, header, item, target):
        if header[item][1] is None:
            header[item][1] = target
            header[item][2] = target
        else:
            header[item][2].node_link = target
            header[item][2] = target

    def update_tree(self, items, node, header, count):
        item = items[0]

        if item in node.children:
            child = node.children[item]
            child.count += count
        else:
            child = FPNode(item, count, node)
            node.children[item] = child

            self.update_header(header, item, child)

        if len(items) > 1:
            self.update_tree(items[1:], child, header, count)

    def create_tree(self, dataset, min_count):
        item_counts = {}

        for trans, count in dataset.items():
            for item in trans:
                item_counts[item] = item_counts.get(item, 0) + count

        header = {
            item: [count, None, None]
            for item, count in item_counts.items()
            if count >= min_count
        }

        if not header:
            return None, None

        root = FPNode("Null", 1, None)

        for trans, count in dataset.items():
            local = {
                item: header[item][0]
                for item in trans if item in header
            }

            if local:
                ordered = [
                    item for item, _ in sorted(
                        local.items(),
                        key=lambda x: (x[1], x[0]),
                        reverse=True
                    )
                ]

                self.update_tree(ordered, root, header, count)

        return root, header

    def ascend_tree(self, node, path):
        while node.parent is not None:
            path.append(node.item)
            node = node.parent

    def find_prefix_path(self, node):
        paths = {}

        while node is not None:
            path = []
            self.ascend_tree(node, path)

            if len(path) > 1:
                prefix = tuple(path[1:])
                paths[prefix] = paths.get(prefix, 0) + node.count

            node = node.node_link

        return paths

    def mine_tree(self, header, min_count, prefix):
        items = sorted(header, key=lambda x: header[x][0])

        for item in items:
            new_prefix = prefix | {item}
            self.freq_items[frozenset(new_prefix)] = header[item][0]

            paths = self.find_prefix_path(header[item][1])
            _, cond_header = self.create_tree(paths, min_count)

            if cond_header is not None:
                self.mine_tree(cond_header, min_count, new_prefix)

    def fit(self, df):
        if len(df) == 0:
            raise ValueError("Dataset khong duoc rong.")

        if not 0 < self.min_support <= 1:
            raise ValueError("min_support phai nam trong (0, 1].")

        min_count = self.min_support * len(df)
        dataset = {}

        columns = df.columns.tolist()

        for row in df.itertuples(index=False, name=None):
            trans = tuple(
                columns[i]
                for i, value in enumerate(row)
                if value
            )

            if trans:
                dataset[trans] = dataset.get(trans, 0) + 1

        self.freq_items = {}

        _, header = self.create_tree(dataset, min_count)

        if header is not None:
            self.mine_tree(header, min_count, set())

        results = [
            {
                "itemsets": set(itemset),
                "support": count / len(df),
                "support_count": count
            }
            for itemset, count in self.freq_items.items()
        ]

        self.result = pd.DataFrame(
            results,
            columns=["itemsets", "support", "support_count"]
        )

        if not self.result.empty:
            self.result = (
                self.result
                .sort_values("support", ascending=False)
                .reset_index(drop=True)
            )

        self.rules = None

        return self.result

    def generate_rules(self):
        if self.result is None:
            raise ValueError("Hay chay fit() truoc.")

        self.rules = self.rule_model.fit(self.result)

        return self.rules

    def get_frequent_items(self):
        return self.result

    def get_rules(self):
        return self.rules

from mlxtend.frequent_patterns import fpgrowth, association_rules

class FPGrowthLibrary:
    def __init__(
        self,
        min_support=0.5,
        min_confidence=0.5,
        min_lift=1.0
    ):
        self.min_support = min_support
        self.result = None
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = None

    def fit(self, df):
        if len(df) == 0:
            raise ValueError("Dataset khong duoc rong.")

        self.result = fpgrowth(
            df.astype(bool),
            min_support=self.min_support,
            use_colnames=True
        )

        self.result["support_count"] = (
            self.result["support"] * len(df)
        ).round().astype(int)

        self.result = (
            self.result
            .sort_values("support", ascending=False)
            .reset_index(drop=True)
        )

        self.rules = None
        return self.result

    def generate_rules(self):
        if self.result is None:
            raise ValueError("Hay chay fit() truoc.")

        if self.result.empty:
            self.rules = pd.DataFrame(
                columns=[
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            )
        else:
            self.rules = association_rules(
                self.result,
                metric="confidence",
                min_threshold=self.min_confidence
            )
            self.rules = self.rules[
                self.rules["lift"] >= self.min_lift
            ].reset_index(drop=True)

        return self.rules

    def get_frequent_items(self):
        return self.result

    def get_rules(self):
        return self.rules

from mlxtend.preprocessing import TransactionEncoder
if __name__ == "__main__":
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

    model1 = FPGrowthPure(min_support, 0.5, 1.0)
    model2 = FPGrowthLibrary(min_support, 0.5, 1.0)

    model1.fit(data)
    model1.generate_rules()

    model2.fit(data)
    model2.generate_rules()

    print("=== FP PURE ===")
    print(model1.get_frequent_items())
    print(model1.get_rules())

    print("\n=== FP MLXTEND ===")
    print(model2.get_frequent_items())
    print(model2.get_rules())