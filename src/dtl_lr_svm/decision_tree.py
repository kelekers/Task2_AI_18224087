import numpy as np

class Node:
    __slots__ = (
        "feature_index", "threshold", "left", "right",
        "value", "n_samples", "class_counts", "impurity"
    )

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.n_samples = None
        self.class_counts = None
        self.impurity = None

    def is_leaf(self):
        return self.value is not None

class DecisionTreeCART:

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        random_state=None,
    ):
        assert criterion in ("gini", "entropy"), "criterion harus 'gini' atau 'entropy'"
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        self.root_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.feature_names_ = None
        self._sample_weight_map = None

    @staticmethod
    def _gini(weighted_counts, total_weight):
        if total_weight <= 0:
            return 0.0
        p = weighted_counts / total_weight
        return 1.0 - np.sum(p ** 2)

    @staticmethod
    def _entropy(weighted_counts, total_weight):
        if total_weight <= 0:
            return 0.0
        p = weighted_counts / total_weight
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _impurity(self, weighted_counts, total_weight):
        if self.criterion == "gini":
            return self._gini(weighted_counts, total_weight)
        return self._entropy(weighted_counts, total_weight)

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.feature_names_ = (
            feature_names if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])]
        )

        sample_weight = self._compute_sample_weight(y)

        self.root_ = self._grow_tree(X, y, sample_weight, depth=0)

        if self.ccp_alpha > 0:
            self._cost_complexity_prune(X, y, sample_weight)

        return self

    def _compute_sample_weight(self, y):
        n_samples = len(y)
        if self.class_weight is None:
            weight_map = {c: 1.0 for c in self.classes_}
        elif self.class_weight == "balanced":
            weight_map = {}
            for c in self.classes_:
                count_c = np.sum(y == c)
                weight_map[c] = n_samples / (self.n_classes_ * count_c)
        elif isinstance(self.class_weight, dict):
            weight_map = self.class_weight
        else:
            raise ValueError("class_weight harus None, 'balanced', atau dict")

        self._sample_weight_map = weight_map
        sample_weight = np.array([weight_map[label] for label in y], dtype=np.float64)
        return sample_weight

    def _weighted_class_counts(self, y, sample_weight):
        counts = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            counts[i] = sample_weight[y == c].sum()
        return counts

    def _grow_tree(self, X, y, sample_weight, depth):
        node = Node()
        n_samples = sample_weight.sum()
        class_counts = self._weighted_class_counts(y, sample_weight)
        node.n_samples = n_samples
        node.class_counts = class_counts
        node.impurity = self._impurity(class_counts, n_samples)

        majority_class = self.classes_[np.argmax(class_counts)]

        if (
            node.impurity == 0.0
            or depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.value = majority_class
            return node

        split = self._best_split(X, y, sample_weight, node.impurity, n_samples)

        if split is None:
            node.value = majority_class
            return node

        feature_index, threshold, left_mask, right_mask, impurity_decrease = split

        if impurity_decrease < self.min_impurity_decrease:
            node.value = majority_class
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow_tree(
            X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1
        )
        node.right = self._grow_tree(
            X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1
        )
        return node

    def _best_split(self, X, y, sample_weight, parent_impurity, n_samples):
        n_features = X.shape[1]
        best_gain = self.min_impurity_decrease
        best_gain = -np.inf
        best_split = None

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[label] for label in y])
        onehot_w = np.zeros((len(y), self.n_classes_))
        onehot_w[np.arange(len(y)), y_idx] = sample_weight

        total_counts = onehot_w.sum(axis=0)

        for feature_index in range(n_features):
            values = X[:, feature_index]
            sort_order = np.argsort(values, kind="mergesort")
            sorted_values = values[sort_order]
            sorted_onehot_w = onehot_w[sort_order]
            sorted_weight = sample_weight[sort_order]

            cum_left_counts = np.cumsum(sorted_onehot_w, axis=0)
            cum_left_weight = np.cumsum(sorted_weight)

            diff_mask = np.empty(len(sorted_values), dtype=bool)
            diff_mask[:-1] = sorted_values[:-1] != sorted_values[1:]
            diff_mask[-1] = False

            candidate_idx = np.where(diff_mask)[0]
            if len(candidate_idx) == 0:
                continue

            w_left = cum_left_weight[candidate_idx]
            w_right = n_samples - w_left

            valid = (w_left >= self.min_samples_leaf) & (w_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            candidate_idx = candidate_idx[valid]
            w_left = w_left[valid]
            w_right = w_right[valid]
            counts_left = cum_left_counts[candidate_idx]
            counts_right = total_counts - counts_left

            impurity_left = self._impurity_vectorized(counts_left, w_left)
            impurity_right = self._impurity_vectorized(counts_right, w_right)

            weighted_impurity = (w_left / n_samples) * impurity_left + \
                                 (w_right / n_samples) * impurity_right
            gains = parent_impurity - weighted_impurity

            local_best = np.argmax(gains)
            if gains[local_best] > best_gain:
                idx = candidate_idx[local_best]
                threshold = (sorted_values[idx] + sorted_values[idx + 1]) / 2.0
                left_mask = values <= threshold
                right_mask = ~left_mask
                best_gain = gains[local_best]
                best_split = (feature_index, threshold, left_mask, right_mask, best_gain)

        return best_split

    def _impurity_vectorized(self, weighted_counts_matrix, total_weight):
        total_weight_safe = np.where(total_weight <= 0, 1.0, total_weight)
        p = weighted_counts_matrix / total_weight_safe[:, None]

        if self.criterion == "gini":
            impurity = 1.0 - np.sum(p ** 2, axis=1)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_p = np.where(p > 0, np.log2(p), 0.0)
            impurity = -np.sum(p * log_p, axis=1)

        impurity = np.where(total_weight <= 0, 0.0, impurity)
        return impurity

    def _cost_complexity_prune(self, X, y, sample_weight):
        def prune_node(node):
            if node.is_leaf():
                return 1

            n_left_leaves = prune_node(node.left)
            n_right_leaves = prune_node(node.right)
            n_leaves_subtree = n_left_leaves + n_right_leaves

            total = node.class_counts.sum()
            majority_weight = node.class_counts.max()
            error_as_leaf = (total - majority_weight)

            error_subtree = self._subtree_error(node)

            cost_as_leaf = error_as_leaf + self.ccp_alpha * 1
            cost_subtree = error_subtree + self.ccp_alpha * n_leaves_subtree

            if cost_as_leaf <= cost_subtree:
                majority_class = self.classes_[np.argmax(node.class_counts)]
                node.value = majority_class
                node.left = None
                node.right = None
                return 1

            return n_leaves_subtree

        prune_node(self.root_)

    def _subtree_error(self, node):
        if node.is_leaf():
            total = node.class_counts.sum()
            majority_weight = node.class_counts.max()
            return total - majority_weight
        return self._subtree_error(node.left) + self._subtree_error(node.right)

    def _predict_proba_single(self, x, node):
        if node.is_leaf():
            total = node.class_counts.sum()
            if total == 0:
                return np.ones(self.n_classes_) / self.n_classes_
            return node.class_counts / total
        if x[node.feature_index] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        return self._predict_proba_single(x, node.right)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probs = np.array([self._predict_proba_single(x, self.root_) for x in X])
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        if self.n_classes_ == 2:
            positive_idx = list(self.classes_).index(1) if 1 in self.classes_ else 1
            return (probs[:, positive_idx] >= threshold).astype(int)
        return self.classes_[np.argmax(probs, axis=1)]

    def feature_importances(self):
        importances = np.zeros(len(self.feature_names_))

        def traverse(node):
            if node.is_leaf():
                return
            gain = node.impurity * node.n_samples
            if node.left is not None:
                gain -= node.left.impurity * node.left.n_samples
            if node.right is not None:
                gain -= node.right.impurity * node.right.n_samples
            importances[node.feature_index] += gain
            traverse(node.left)
            traverse(node.right)

        traverse(self.root_)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        return dict(zip(self.feature_names_, importances))

    def get_depth(self):
        def depth(node):
            if node.is_leaf():
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        return depth(self.root_)

    def get_n_leaves(self):
        def count(node):
            if node.is_leaf():
                return 1
            return count(node.left) + count(node.right)
        return count(self.root_)

    def export_text(self, node=None, depth=0, prefix=""):
        if node is None:
            node = self.root_
        indent = "  " * depth
        if node.is_leaf():
            total = node.class_counts.sum()
            dist = ", ".join(
                f"{c}: {cnt:.1f}" for c, cnt in zip(self.classes_, node.class_counts)
            )
            return f"{indent}{prefix}Leaf -> class={node.value} (n={total:.1f}, dist=[{dist}])\n"

        fname = self.feature_names_[node.feature_index]
        text = f"{indent}{prefix}[{fname} <= {node.threshold:.4f}] (n={node.n_samples:.1f})\n"
        text += self.export_text(node.left, depth + 1, prefix="├─ True: ")
        text += self.export_text(node.right, depth + 1, prefix="└─ False: ")
        return text
