"""
classify.py
-----------
Train and evaluate classifiers on band-power EEG features.
Compares SVM, Logistic Regression, and Random Forest.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline


# ── 1. Define classifiers ────────────────────────────────────────────────────

CLASSIFIERS = {
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]),
}

LABEL_NAMES = ["Left Hand", "Right Hand"]


# ── 2. Cross-validated comparison ────────────────────────────────────────────

def compare_classifiers(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """
    Run stratified k-fold CV for each classifier and print accuracy scores.

    Returns
    -------
    results : dict  {name: mean_accuracy}
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'='*50}")
    print(f"Cross-validation results  ({cv}-fold)")
    print(f"{'='*50}")

    for name, pipeline in CLASSIFIERS.items():
        scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name:25s}  {scores.mean():.3f} ± {scores.std():.3f}")

    return results


# ── 3. Full train + evaluate on held-out split ───────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str = "SVM (RBF)",
    test_size: float = 0.2,
) -> None:
    """
    Train the chosen classifier on 80 % of data, evaluate on the rest,
    and save a confusion matrix plot.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    pipeline = CLASSIFIERS[classifier_name]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n{'='*50}")
    print(f"Hold-out evaluation  [{classifier_name}]")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {classifier_name}")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved → results/confusion_matrix.png")


# ── 4. Bar chart of CV results ────────────────────────────────────────────────

def plot_comparison(results: dict) -> None:
    """Bar chart comparing mean CV accuracies."""
    names  = list(results.keys())
    scores = list(results.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, scores, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean CV Accuracy")
    ax.set_title("Classifier Comparison — Motor Imagery EEG")
    ax.bar_label(bars, fmt="%.3f", padding=5)
    plt.tight_layout()
    plt.savefig("results/classifier_comparison.png", dpi=150)
    plt.close()
    print("Saved → results/classifier_comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load pre-extracted features (run preprocess.py first or import directly)
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_raw, preprocess, make_epochs, extract_features

    os.makedirs("results", exist_ok=True)

    raw    = load_raw()
    raw    = preprocess(raw)
    epochs = make_epochs(raw)
    X, y   = extract_features(epochs)

    results = compare_classifiers(X, y)
    plot_comparison(results)
    train_and_evaluate(X, y, classifier_name="SVM (RBF)")