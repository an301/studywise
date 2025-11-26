"""
Train a tiny posture classification model (neutral / slouch / lean).

Usage:
    python backend/train_posture_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> int:
    csv_path = os.path.join("data", "posture_data.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run posture_collect_data.py first.")
        return 1

    df = pd.read_csv(csv_path)

    df = df[df["label"].isin(["neutral", "slouch", "lean"])]

    if len(df) < 50:
        print("WARNING: very little data. Collect more samples for better model.")
    print(f"Loaded {len(df)} labeled frames.")

    X = df[["neck_deg", "ear_deg", "shoulder_y"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(multi_class="multinomial", max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("backend/models", exist_ok=True)
    model_path = os.path.join("backend", "models", "posture_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
