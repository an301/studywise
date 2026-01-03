"""
Improved Posture Model Training v2.

Features:
- More features for better discrimination
- Random Forest for better non-linear boundaries
- Cross-validation
- Feature importance analysis
- Confusion matrix visualization

Usage:
    python backend/prototypes/train_posture_model_v2.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> int:
    csv_path = os.path.join("data", "posture_data_v2.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run posture_collect_data_v2.py first to collect new data.")
        return 1
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows")
    
    # Filter valid labels
    valid_labels = ["neutral", "slouch", "lean", "away"]
    df = df[df["label"].isin(valid_labels)]
    print(f"After filtering: {len(df)} labeled frames")
    
    # Show class distribution
    print("\n=== CLASS DISTRIBUTION ===")
    print(df["label"].value_counts())
    
    # Check for imbalance
    counts = df["label"].value_counts()
    if counts.max() / counts.min() > 3:
        print("\nWARNING: Classes are imbalanced! Consider collecting more data for minority classes.")
    
    # Feature columns (excluding timestamp and label)
    feature_cols = ["neck_angle", "ear_angle", "nose_dist_norm", 
                    "ear_dist_norm", "nose_forward_norm", "shoulder_y_norm",
                    "head_tilt", "roll_deg"]
    
    # Handle 'away' class - it has sentinel values (-1)
    # For away, we'll keep the -1 values as they're distinctive
    
    # Check for missing features
    print("\n=== FEATURE AVAILABILITY ===")
    for col in feature_cols:
        if col in df.columns:
            valid = (df[col] != -1).sum()
            print(f"  {col}: {valid}/{len(df)} valid ({100*valid/len(df):.1f}%)")
        else:
            print(f"  {col}: MISSING FROM DATA")
    
    # Prepare features
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        print("ERROR: No feature columns found in data!")
        return 1
    
    X = df[available_cols].values
    y = df["label"].values
    
    print(f"\nUsing {len(available_cols)} features: {available_cols}")
    
    # Replace any NaN with -1 (sentinel)
    X = np.nan_to_num(X, nan=-1.0)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Model: Random Forest (better for non-linear boundaries)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",  # Handle imbalanced classes
            random_state=42,
            n_jobs=-1,
        ))
    ])
    
    # Cross-validation
    print("\n=== CROSS-VALIDATION ===")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\n=== CONFUSION MATRIX ===")
    labels = ["away", "lean", "neutral", "slouch"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Pretty print confusion matrix
    print(f"{'':>10}", end="")
    for l in labels:
        print(f"{l:>10}", end="")
    print()
    for i, row_label in enumerate(labels):
        print(f"{row_label:>10}", end="")
        for j, val in enumerate(cm[i]):
            print(f"{val:>10}", end="")
        print()
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, labels=labels))
    
    # Feature importance (from Random Forest)
    print("\n=== FEATURE IMPORTANCE ===")
    rf_model = model.named_steps["clf"]
    importances = rf_model.feature_importances_
    for col, imp in sorted(zip(available_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {col:>20}: {imp:.3f} {bar}")
    
    # Save model
    os.makedirs("backend/models", exist_ok=True)
    model_path = os.path.join("backend", "models", "posture_model_v2.joblib")
    
    # Save model along with feature column names
    model_data = {
        "model": model,
        "feature_cols": available_cols,
        "classes": list(model.classes_),
    }
    joblib.dump(model_data, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Summary
    accuracy = (y_pred == y_test).mean()
    print(f"\n=== SUMMARY ===")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"Model: Random Forest (100 trees, max_depth=10)")
    print(f"Features: {len(available_cols)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

