import numpy as np

class StandardScalerScratch:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class RBFSampler:
    def __init__(self, gamma=1.0, n_components=200, random_state=42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.weights_ = rng.normal(0, np.sqrt(2 * self.gamma), size=(n_features, self.n_components))
        self.offset_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        projection = X @ self.weights_ + self.offset_
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class LinearSVMScratch:
    def __init__(self, C=1.0, learning_rate=0.01, n_epochs=200, batch_size=256,
                 class_weight=None, random_state=42, tol=1e-5):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.random_state = random_state
        self.tol = tol

    def _compute_sample_weight(self, y):
        classes = np.unique(y)
        n_samples = len(y)
        if self.class_weight is None:
            return np.ones(n_samples)
        elif self.class_weight == "balanced":
            weight_map = {}
            for c in classes:
                weight_map[c] = n_samples / (len(classes) * np.sum(y == c))
            return np.array([weight_map[label] for label in y])
        elif isinstance(self.class_weight, dict):
            return np.array([self.class_weight[label] for label in y])
        raise ValueError("class_weight harus None, 'balanced', atau dict")

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        y_signed = np.where(y == 1, 1.0, -1.0)
        sample_weight = self._compute_sample_weight(y)

        self.w_ = np.zeros(n_features)
        self.b_ = 0.0

        self.loss_history_ = []

        indices = np.arange(n_samples)
        prev_loss = np.inf

        for epoch in range(self.n_epochs):
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y_signed[batch_idx]
                wb = sample_weight[batch_idx]

                margin = yb * (Xb @ self.w_ + self.b_)
                mask = margin < 1

                grad_w = self.w_.copy()
                if np.any(mask):
                    grad_w -= self.C * (wb[mask, None] * yb[mask, None] * Xb[mask]).sum(axis=0) / len(batch_idx)
                    grad_b = -self.C * np.sum(wb[mask] * yb[mask]) / len(batch_idx)
                else:
                    grad_b = 0.0

                self.w_ -= self.learning_rate * grad_w
                self.b_ -= self.learning_rate * grad_b

            margin_full = y_signed * (X @ self.w_ + self.b_)
            hinge = np.maximum(0, 1 - margin_full)
            loss = 0.5 * np.dot(self.w_, self.w_) + self.C * np.mean(sample_weight * hinge)
            self.loss_history_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def decision_function(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X, threshold=0.0):
        scores = self.decision_function(X)
        return (scores >= threshold).astype(int)

    def predict_proba_sigmoid(self, X):
        scores = self.decision_function(X)
        proba_pos = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - proba_pos, proba_pos])

class RBFSVMScratch:
    def __init__(self, C=1.0, gamma=0.1, n_components=200, learning_rate=0.01,
                 n_epochs=200, batch_size=256, class_weight=None, random_state=42):
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        self.sampler_ = RBFSampler(gamma=self.gamma, n_components=self.n_components,
                                    random_state=self.random_state)
        X_transformed = self.sampler_.fit_transform(X)

        self.linear_svm_ = LinearSVMScratch(
            C=self.C, learning_rate=self.learning_rate, n_epochs=self.n_epochs,
            batch_size=self.batch_size, class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.linear_svm_.fit(X_transformed, y)
        self.loss_history_ = self.linear_svm_.loss_history_
        return self

    def decision_function(self, X):
        X_transformed = self.sampler_.transform(X)
        return self.linear_svm_.decision_function(X_transformed)

    def predict(self, X, threshold=0.0):
        scores = self.decision_function(X)
        return (scores >= threshold).astype(int)

    def predict_proba_sigmoid(self, X):
        scores = self.decision_function(X)
        proba_pos = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - proba_pos, proba_pos])