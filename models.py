import numpy as np
from sklearn.linear_model import LogisticRegression


def build_logistic_model(random_state=42):
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2500,
        solver="liblinear",
        random_state=random_state,
    )


def fit_surrogate(X, y, labeled_idx, random_state=42):
    model = build_logistic_model(random_state=random_state)
    idx = np.asarray(labeled_idx, dtype=int)
    model.fit(X[idx], y[idx])
    return model
