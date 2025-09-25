<<<<<<< HEAD
# train_failure_model.py
import argparse
import json
import joblib
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def parse_args():
    p = argparse.ArgumentParser(description="Train a model to predict asset failure probability")
    p.add_argument("--csv", required=True, help="Path to the dataset")
    p.add_argument("--target", default="Failure", help="Target column")
    p.add_argument("--id_cols", nargs="*", default=["asset_id", "geometry"], help="Columns to drop")
    p.add_argument("--model_out", default="models/failure_model.joblib", help="Path to save model")
    p.add_argument("--metrics_out", default="models/failure_metrics.json", help="Path to save metrics")
    return p.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded dataset with {len(df)} rows")

    # Features and target
    X = df.drop(columns=[args.target] + [c for c in args.id_cols if c in df.columns], errors="ignore")
    y = df[args.target]

    # Add derived age feature
    current_year = datetime.now().year
    if "installation_year" in X.columns:
        X["age"] = current_year - X["installation_year"]

    # Identify categorical and numerical columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "classification_report": classification_report(y_val, y_pred, output_dict=True)
    }

    # Save
    joblib.dump(pipeline, args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete!")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
=======
# train_failure_model.py
import argparse
import json
import joblib
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def parse_args():
    p = argparse.ArgumentParser(description="Train a model to predict asset failure probability")
    p.add_argument("--csv", required=True, help="Path to the dataset")
    p.add_argument("--target", default="Failure", help="Target column")
    p.add_argument("--id_cols", nargs="*", default=["asset_id", "geometry"], help="Columns to drop")
    p.add_argument("--model_out", default="models/failure_model.joblib", help="Path to save model")
    p.add_argument("--metrics_out", default="models/failure_metrics.json", help="Path to save metrics")
    return p.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded dataset with {len(df)} rows")

    # Features and target
    X = df.drop(columns=[args.target] + [c for c in args.id_cols if c in df.columns], errors="ignore")
    y = df[args.target]

    # Add derived age feature
    current_year = datetime.now().year
    if "installation_year" in X.columns:
        X["age"] = current_year - X["installation_year"]

    # Identify categorical and numerical columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "classification_report": classification_report(y_val, y_pred, output_dict=True)
    }

    # Save
    joblib.dump(pipeline, args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete!")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
>>>>>>> e1b9a867411dbb77ab32dd9e3d5dd674e5971692
