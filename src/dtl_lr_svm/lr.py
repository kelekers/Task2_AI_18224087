import numpy as np

class LogisticRegressionScratch:
    def __init__(self, C=1.0, method="newton", learning_rate=0.1, n_iter=200,
                 tol=1e-6, class_weight=None, fit_intercept=True, random_state=42):
        self.C = C
        self.method = method
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.random_state = random_state

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

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        Xb = self._add_intercept(X)
        n_samples, n_features = Xb.shape
        sample_weight = self._compute_sample_weight(y)

        self.theta_ = np.zeros(n_features)
        self.loss_history_ = []
        lam = 1.0 / self.C

        for it in range(self.n_iter):
            z = Xb @ self.theta_
            p = self._sigmoid(z)
            grad = Xb.T @ (sample_weight * (p - y)) / n_samples
            reg_grad = np.zeros(n_features)
            reg_grad[1:] = (lam / n_samples) * self.theta_[1:]
            grad += reg_grad

            eps = 1e-10
            log_loss = -np.mean(sample_weight * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            reg_loss = (lam / (2 * n_samples)) * np.sum(self.theta_[1:] ** 2)
            total_loss = log_loss + reg_loss
            self.loss_history_.append(total_loss)

            if self.method == "newton":
                W = sample_weight * p * (1 - p)
                H = (Xb.T * W) @ Xb / n_samples
                reg_H = np.eye(n_features) * (lam / n_samples)
                reg_H[0, 0] = 0.0
                H += reg_H
                try:
                    delta = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, grad, rcond=None)[0]
                self.theta_ -= delta
            else:
                self.theta_ -= self.learning_rate * grad

            if it > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ = self.theta_[0]
            self.coef_ = self.theta_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta_

        return self

    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p1 = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


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
