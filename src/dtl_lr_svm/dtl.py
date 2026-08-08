import itertools
import numpy as np

try:
    from .decision_tree import Node, DecisionTreeCART
except ImportError:  # pragma: no cover - fallback for direct script execution
    from decision_tree import Node, DecisionTreeCART

class AdvancedNode(Node):
    __slots__ = ("is_categorical", "left_categories")

    def __init__(self):
        super().__init__()
        self.is_categorical = False
        self.left_categories = None

class DecisionTreeCARTAdvanced(DecisionTreeCART):
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
        categorical_features=None,
        max_categorical_subset_size=10,
    ):
        assert criterion in ("gini", "entropy", "gain_ratio", "chi2"), \
            "criterion harus salah satu dari: 'gini', 'entropy', 'gain_ratio', 'chi2'"

        super().__init__(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )
        self.criterion = criterion
        self.categorical_features = set(categorical_features) if categorical_features else set()
        self.max_categorical_subset_size = max_categorical_subset_size

    def _impurity(self, weighted_counts, total_weight):
        if self.criterion in ("gini",):
            return self._gini(weighted_counts, total_weight)
        return self._entropy(weighted_counts, total_weight)

    def _impurity_vectorized(self, weighted_counts_matrix, total_weight):
        if self.criterion in ("gini",):
            total_weight_safe = np.where(total_weight <= 0, 1.0, total_weight)
            p = weighted_counts_matrix / total_weight_safe[:, None]
            impurity = 1.0 - np.sum(p ** 2, axis=1)
            return np.where(total_weight <= 0, 0.0, impurity)

        total_weight_safe = np.where(total_weight <= 0, 1.0, total_weight)
        p = weighted_counts_matrix / total_weight_safe[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(p > 0, np.log2(p), 0.0)
        impurity = -np.sum(p * log_p, axis=1)
        return np.where(total_weight <= 0, 0.0, impurity)

    @staticmethod
    def _split_information(w_left, w_right, n_samples):
        p_left = w_left / n_samples
        p_right = w_right / n_samples
        si = np.zeros_like(p_left)
        mask_left = p_left > 0
        mask_right = p_right > 0
        si = np.where(mask_left, -p_left * np.log2(np.where(mask_left, p_left, 1)), 0.0)
        si = si + np.where(mask_right, -p_right * np.log2(np.where(mask_right, p_right, 1)), 0.0)
        return si

    def _chi2_score(self, counts_left, counts_right, w_left, w_right, n_samples, total_counts):
        expected_left = np.outer(w_left, total_counts) / n_samples
        expected_right = np.outer(w_right, total_counts) / n_samples

        expected_left_safe = np.where(expected_left <= 0, 1e-10, expected_left)
        expected_right_safe = np.where(expected_right <= 0, 1e-10, expected_right)

        chi2 = np.sum((counts_left - expected_left) ** 2 / expected_left_safe, axis=1) + \
               np.sum((counts_right - expected_right) ** 2 / expected_right_safe, axis=1)
        return chi2

    def _best_split(self, X, y, sample_weight, parent_impurity, n_samples):
        n_features = X.shape[1]
        best_score = -np.inf
        best_split = None

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[label] for label in y])
        onehot_w = np.zeros((len(y), self.n_classes_))
        onehot_w[np.arange(len(y)), y_idx] = sample_weight
        total_counts = onehot_w.sum(axis=0)

        for feature_index in range(n_features):
            values = X[:, feature_index]

            if feature_index in self.categorical_features:
                result = self._best_categorical_split(
                    values, y, sample_weight, onehot_w, total_counts,
                    parent_impurity, n_samples
                )
            else:
                result = self._best_numeric_split(
                    values, onehot_w, sample_weight, total_counts,
                    parent_impurity, n_samples
                )

            if result is None:
                continue

            score, threshold_or_categories, left_mask, right_mask, is_categorical = result
            if score > best_score:
                best_score = score
                best_split = (feature_index, is_categorical, threshold_or_categories,
                              left_mask, right_mask, score)

        if best_split is None:
            return None

        feature_index, is_categorical, thr_or_cat, left_mask, right_mask, score = best_split
        counts_left = self._weighted_class_counts(y[left_mask], sample_weight[left_mask])
        counts_right = self._weighted_class_counts(y[right_mask], sample_weight[right_mask])
        w_left = sample_weight[left_mask].sum()
        w_right = sample_weight[right_mask].sum()
        impurity_left = self._impurity(counts_left, w_left)
        impurity_right = self._impurity(counts_right, w_right)
        weighted_impurity = (w_left / n_samples) * impurity_left + (w_right / n_samples) * impurity_right
        impurity_decrease = parent_impurity - weighted_impurity

        return (feature_index, is_categorical, thr_or_cat, left_mask, right_mask, impurity_decrease)

    def _score_gain(self, parent_impurity, w_left, w_right, impurity_left, impurity_right, n_samples):
        weighted_impurity = (w_left / n_samples) * impurity_left + (w_right / n_samples) * impurity_right
        return parent_impurity - weighted_impurity

    def _best_numeric_split(self, values, onehot_w, sample_weight, total_counts, parent_impurity, n_samples):
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
            return None

        w_left = cum_left_weight[candidate_idx]
        w_right = n_samples - w_left
        valid = (w_left >= self.min_samples_leaf) & (w_right >= self.min_samples_leaf)
        if not np.any(valid):
            return None

        candidate_idx = candidate_idx[valid]
        w_left = w_left[valid]
        w_right = w_right[valid]
        counts_left = cum_left_counts[candidate_idx]
        counts_right = total_counts - counts_left

        impurity_left = self._impurity_vectorized(counts_left, w_left)
        impurity_right = self._impurity_vectorized(counts_right, w_right)
        gains = parent_impurity - ((w_left / n_samples) * impurity_left + (w_right / n_samples) * impurity_right)

        if self.criterion == "gain_ratio":
            si = self._split_information(w_left, w_right, n_samples)
            si_safe = np.where(si <= 1e-10, 1e-10, si)
            scores = gains / si_safe
        elif self.criterion == "chi2":
            scores = self._chi2_score(counts_left, counts_right, w_left, w_right, n_samples, total_counts)
        else:
            scores = gains

        local_best = np.argmax(scores)
        idx = candidate_idx[local_best]
        threshold = (sorted_values[idx] + sorted_values[idx + 1]) / 2.0
        left_mask = values <= threshold
        right_mask = ~left_mask
        return (scores[local_best], threshold, left_mask, right_mask, False)

    def _best_categorical_split(self, values, y, sample_weight, onehot_w, total_counts,
                                 parent_impurity, n_samples):
        unique_cats = np.unique(values)
        if len(unique_cats) <= 1:
            return None

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        positive_class_idx = label_to_idx.get(1, self.n_classes_ - 1)

        cat_positive_rate = []
        for cat in unique_cats:
            mask = values == cat
            w_total = sample_weight[mask].sum()
            w_positive = onehot_w[mask, positive_class_idx].sum()
            rate = w_positive / w_total if w_total > 0 else 0.0
            cat_positive_rate.append(rate)

        order = np.argsort(cat_positive_rate)
        sorted_cats = unique_cats[order]
        n_cats = len(sorted_cats)
        cat_to_sorted_idx = {cat: i for i, cat in enumerate(sorted_cats)}
        row_cat_idx = np.array([cat_to_sorted_idx[v] for v in values])

        per_cat_counts = np.zeros((n_cats, self.n_classes_))
        per_cat_weight = np.zeros(n_cats)
        for i in range(n_cats):
            mask = row_cat_idx == i
            per_cat_weight[i] = sample_weight[mask].sum()
            per_cat_counts[i] = onehot_w[mask].sum(axis=0)

        cum_left_counts = np.cumsum(per_cat_counts, axis=0)[:-1]
        cum_left_weight = np.cumsum(per_cat_weight)[:-1]
        w_left = cum_left_weight
        w_right = n_samples - w_left

        valid = (w_left >= self.min_samples_leaf) & (w_right >= self.min_samples_leaf)
        if not np.any(valid):
            return None

        counts_left_valid = cum_left_counts[valid]
        counts_right_valid = total_counts - counts_left_valid
        w_left_valid = w_left[valid]
        w_right_valid = w_right[valid]

        impurity_left = self._impurity_vectorized(counts_left_valid, w_left_valid)
        impurity_right = self._impurity_vectorized(counts_right_valid, w_right_valid)
        gains = parent_impurity - ((w_left_valid / n_samples) * impurity_left +
                                    (w_right_valid / n_samples) * impurity_right)

        if self.criterion == "gain_ratio":
            si = self._split_information(w_left_valid, w_right_valid, n_samples)
            si_safe = np.where(si <= 1e-10, 1e-10, si)
            scores = gains / si_safe
        elif self.criterion == "chi2":
            scores = self._chi2_score(counts_left_valid, counts_right_valid, w_left_valid, w_right_valid, n_samples, total_counts)
        else:
            scores = gains

        local_best = np.argmax(scores)
        valid_indices = np.where(valid)[0]
        split_point = valid_indices[local_best]

        best_left_categories = set(sorted_cats[:split_point + 1].tolist())
        best_left_mask = np.isin(values, list(best_left_categories))
        best_right_mask = ~best_left_mask
        best_score = scores[local_best]

        return (best_score, best_left_categories, best_left_mask, best_right_mask, True)

    def _grow_tree(self, X, y, sample_weight, depth):
        node = AdvancedNode()
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

        feature_index, is_categorical, thr_or_cat, left_mask, right_mask, impurity_decrease = split

        if impurity_decrease < self.min_impurity_decrease:
            node.value = majority_class
            return node

        node.feature_index = feature_index
        node.is_categorical = is_categorical
        if is_categorical:
            node.left_categories = thr_or_cat
            node.threshold = None
        else:
            node.threshold = thr_or_cat
            node.left_categories = None

        node.left = self._grow_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)
        return node

    def _predict_proba_single(self, x, node):
        if node.is_leaf():
            total = node.class_counts.sum()
            if total == 0:
                return np.ones(self.n_classes_) / self.n_classes_
            return node.class_counts / total

        if getattr(node, "is_categorical", False):
            goes_left = x[node.feature_index] in node.left_categories
        else:
            goes_left = x[node.feature_index] <= node.threshold

        if goes_left:
            return self._predict_proba_single(x, node.left)
        return self._predict_proba_single(x, node.right)

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
        if getattr(node, "is_categorical", False):
            cats_str = "{" + ", ".join(str(c) for c in sorted(node.left_categories)) + "}"
            text = f"{indent}{prefix}[{fname} in {cats_str}] (n={node.n_samples:.1f})\n"
        else:
            text = f"{indent}{prefix}[{fname} <= {node.threshold:.4f}] (n={node.n_samples:.1f})\n"
        text += self.export_text(node.left, depth + 1, prefix="├─ True: ")
        text += self.export_text(node.right, depth + 1, prefix="└─ False: ")
        return text

    def _cost_complexity_prune(self, X, y, sample_weight):
        total_weight_root = self.root_.class_counts.sum()

        def prune_pass(node):
            if node.is_leaf():
                return 1, False

            n_left_leaves, pruned_left = prune_pass(node.left)
            n_right_leaves, pruned_right = prune_pass(node.right)
            n_leaves_subtree = n_left_leaves + n_right_leaves
            any_pruned = pruned_left or pruned_right

            total = node.class_counts.sum()
            majority_weight = node.class_counts.max()
            error_as_leaf = (total - majority_weight) / total_weight_root

            error_subtree = self._subtree_error(node) / total_weight_root

            cost_as_leaf = error_as_leaf + self.ccp_alpha * 1
            cost_subtree = error_subtree + self.ccp_alpha * n_leaves_subtree

            if cost_as_leaf <= cost_subtree:
                majority_class = self.classes_[np.argmax(node.class_counts)]
                node.value = majority_class
                node.left = None
                node.right = None
                node.is_categorical = False
                node.left_categories = None
                node.threshold = None
                return 1, True

            return n_leaves_subtree, any_pruned

        max_iterations = 200
        for _ in range(max_iterations):
            _, any_pruned = prune_pass(self.root_)
            if not any_pruned:
                break


def cv_cost_complexity_pruning(
    tree_class, X, y, ccp_alphas, cv_splitter, macro_f1_fn,
    tree_kwargs=None, feature_names=None, n_repeats=1,
):
    tree_kwargs = tree_kwargs or {}

    splits = list(cv_splitter)

    records = []
    for alpha in ccp_alphas:
        fold_scores = []
        for train_idx, val_idx in splits:
            clf = tree_class(ccp_alpha=alpha, **tree_kwargs)
            clf.fit(X[train_idx], y[train_idx], feature_names=feature_names)
            pred = clf.predict(X[val_idx])
            fold_scores.append(macro_f1_fn(y[val_idx], pred))

        fold_scores = np.array(fold_scores)
        records.append({
            "ccp_alpha": alpha,
            "macro_f1_cv_mean": fold_scores.mean(),
            "macro_f1_cv_std": fold_scores.std(),
            "n_folds": len(fold_scores),
        })

    best_record = max(records, key=lambda r: r["macro_f1_cv_mean"])
    best_alpha = best_record["ccp_alpha"]

    return records, best_alpha
