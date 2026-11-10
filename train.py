"""Train the champion model and export it with metadata.

Replace the dataset/candidate list with your own capstone model if needed;
the API only requires `models/model.joblib` (a fitted estimator exposing
`predict_proba`) and `models/metadata.json`.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("models")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=SEED
    )

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }

    scores = {}
    for name, est in candidates.items():
        scores[name] = cross_val_score(est, X_train, y_train, cv=5, scoring="roc_auc").mean()
        print(f"{name:20s} CV ROC-AUC = {scores[name]:.4f}")

    champion_name = max(scores, key=scores.get)
    champion = candidates[champion_name].fit(X_train, y_train)
    proba = champion.predict_proba(X_test)[:, 1]
    pred = champion.predict(X_test)
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    print(f"Champion: {champion_name} | test metrics: {metrics}")

    joblib.dump(champion, OUT / "model.joblib")
    metadata = {
        "model_name": champion_name,
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_names": list(data.feature_names),
        "n_features": len(data.feature_names),
        "classes": {"0": str(data.target_names[0]), "1": str(data.target_names[1])},
        "test_metrics": metrics,
        "example_input": [float(v) for v in X_test[0]],
        "examples": {
            "malignant": [float(v) for v in X_test[list(y_test).index(0)]],
            "benign": [float(v) for v in X_test[list(y_test).index(1)]],
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved models/model.joblib and models/metadata.json")


if __name__ == "__main__":
    main()
