import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_hat):
    return -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)).mean()

def normalize(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    return (X - X_mean) / X_std, X_mean, X_std

def softmax(X):
    return np.exp(X) / np.exp(X).sum(axis=1).reshape(-1, 1)

def impute(X, strategy='mean', vec=None):
    imp_vec = vec
    if strategy == 'mean':
        if imp_vec is None:
            imp_vec = np.nanmean(X, axis=0)
    else:
        raise ValueError(f'Unrecognized strategy: {strategy}')
    
    
    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xi = np.where(np.isnan(Xi), imp_vec[i], Xi)
        X[:, i] = Xi

    return X, imp_vec

class BinaryLR:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def predict(self, X):
        return sigmoid(X @ self.w + self.b) > 0.5

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

class LogisticRegression:
    def __init__(self, lr=.05, verbose=False, seed=None):
        self.lr = lr
        self.verbose = verbose
        self.seed = seed
        self.classifiers = []
    
    def fit(self, X, y, n_epochs=3, batch_size=-1):
        self.unique_targets = np.unique(y)
        
        self.classifiers = []
        if self.verbose:
            print(f"# of targets: {len(self.unique_targets)}")
        if 'values' in dir(X):
            X = X.values
        n, m = X.shape
        batch_size = n if batch_size == -1 else batch_size
        n_batches = int(n // batch_size)
        if batch_size < n:
            n_batches += 1
        if self.verbose:
            print("n_batches:", n_batches)
        for i_clf in range(len(self.unique_targets)):
            if self.verbose:
                print(f"Training classifier #{i_clf+1}")
            target = np.where(y == self.unique_targets[i_clf], 1, 0)
            w, b = self._initializer(m)
            for epoch in range(n_epochs):
                losses = []
                for it in range(n_batches):
                    if it < n_batches - 1:
                        Xb = X[it * batch_size:(it + 1) * batch_size, :]
                        yb = target[it * batch_size:(it + 1) * batch_size]
                    else:
                        Xb = X[it * batch_size:]
                        yb = target[it * batch_size:]
                    if len(Xb) == 0:
                        continue
                    yh = sigmoid(Xb @ w + b)
                    if len(yh.shape) > 1:
                        yh.ravel()
                    losses.append(log_loss(yb, yh))
                    err = (yh - yb).reshape(-1, 1)
                    dw = (err * Xb).mean(axis=0)
                    db = err.mean(axis=0)
                    
                    w -= self.lr * dw
                    b -= self.lr * db
                if self.verbose:
                    print(f"[{epoch}/{n_epochs}]: mean loss: {np.mean(losses)}")
            self.classifiers.append(BinaryLR(w, b))
    
    def predict_proba(self, X):
        preds = np.zeros((len(X), len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            y_pred = clf.predict_proba(X)
            preds[:, i] = y_pred
        return softmax(preds)
    
    def predict(self, X):
        return np.array([self.unique_targets[c] for c in np.argmax(self.predict_proba(X), axis=1)])
            
    def _initializer(self, n_features, scale=.01):
        if self.seed is not None:
            np.random.seed(self.seed)

        w = np.random.randn(n_features) * scale
        b = np.zeros(1)
        return w, b

def accuracy(y_true, y_pred):
    return ((y_true.values == y_pred)).sum() / len(y_true)
