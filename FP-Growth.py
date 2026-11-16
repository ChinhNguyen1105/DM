
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth


class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None


class FPGrowthPure:
    def __init__(self, min_support=0.5):
        self.min_support = min_support
        self.freq_items = {}
        self.result = None

    def update_header(self, node, target):
        while node.node_link is not None:
            node = node.node_link
        node.node_link = target

    def update_tree(self, items, node, header, count):
        item = items[0]

        if item in node.children:
            node.children[item].count += count
        else:
            node.children[item] = FPNode(item, count, node)

        child = node.children[item]

        if header[item][1] is None:
            header[item][1] = child
        else:
            self.update_header(header[item][1], child)

        if len(items) > 1:
            self.update_tree(items[1:], child, header, count)

    def create_tree(self, dataset, min_count):
        header = {}

        for trans, count in dataset.items():
            for item in trans:
                header[item] = header.get(item, 0) + count

        header = {
            item: [count, None]
            for item, count in header.items()
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
        if node.parent is not None:
            path.append(node.item)
            self.ascend_tree(node.parent, path)

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
            raise ValueError("Dataset không được rỗng.")

        min_count = self.min_support * len(df)
        dataset = {}

        for _, row in df.iterrows():
            trans = tuple(sorted(
                col for col in df.columns if row[col]
            ))

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

        return self.result

    def predict(self):
        return self.result


class FPGrowthLibrary:
    def __init__(self, min_support=0.5):
        self.min_support = min_support
        self.result = None

    def fit(self, df):
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

        return self.result

    def predict(self):
        return self.result
