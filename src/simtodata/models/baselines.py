"""Non-neural baseline classifiers wrapping sklearn pipelines."""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_baseline(name: str):
    """Create a baseline classifier pipeline.

    Args:
        name: 'logistic_regression' or 'gradient_boosting'.

    Returns:
        sklearn Pipeline with StandardScaler and classifier.
    """
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
