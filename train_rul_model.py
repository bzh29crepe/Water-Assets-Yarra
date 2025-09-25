<<<<<<< HEAD
# train_rul_model.py
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

def parse_args():
    p = argparse.ArgumentParser(description="Train a model to predict Remaining Useful Life (RUL)")
    p.add_argument("--csv", required=True, help="Path to the asset dataset CSV")
    p.add_argument("--target", default="Remaining_Years", help="Target column name")
    p.add_argument("--id_cols", nargs="*", default=["asset_id", "geometry"], help="Columns to drop")
    p.add_argument("--model_out", default="models/rul_model.joblib", help="Path to save model")
    p.add_argument("--metrics_out", default="models/rul_metrics.json", help="Path to save metrics")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- Load Data ----
    df = pd.read_csv(args.csv)
    print(f"Loaded dataset with {len(df)} rows")

    # ---- Prepare Features and Target ----
    X = df.drop(columns=[args.target] + [c for c in args.id_cols if c in df.columns], errors="ignore")
    y = df[args.target]

    # Add derived age feature
    current_year = datetime.now().year
    if "installation_year" in X.columns:
        X["age"] = current_year - X["installation_year"]

    # Identify categorical and numerical columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")

    # ---- Split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # ---- Preprocessor ----
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

    # ---- Model ----
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1
    )

    # ---- Pipeline ----
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # ---- Train ----
    pipeline.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = pipeline.predict(X_val)
    metrics = {
        "MAE": float(mean_absolute_error(y_val, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "R2": float(r2_score(y_val, y_pred))
    }

    # ---- Save Artifacts ----
    joblib.dump(pipeline, args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model training complete!")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")

if __name__ == "__main__":
    main()
=======
# train_rul_model.py
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

def parse_args():
    p = argparse.ArgumentParser(description="Train a model to predict Remaining Useful Life (RUL)")
    p.add_argument("--csv", required=True, help="Path to the asset dataset CSV")
    p.add_argument("--target", default="Remaining_Years", help="Target column name")
    p.add_argument("--id_cols", nargs="*", default=["asset_id", "geometry"], help="Columns to drop")
    p.add_argument("--model_out", default="models/rul_model.joblib", help="Path to save model")
    p.add_argument("--metrics_out", default="models/rul_metrics.json", help="Path to save metrics")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- Load Data ----
    df = pd.read_csv(args.csv)
    print(f"Loaded dataset with {len(df)} rows")

    # ---- Prepare Features and Target ----
    X = df.drop(columns=[args.target] + [c for c in args.id_cols if c in df.columns], errors="ignore")
    y = df[args.target]

    # Add derived age feature
    current_year = datetime.now().year
    if "installation_year" in X.columns:
        X["age"] = current_year - X["installation_year"]

    # Identify categorical and numerical columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")

    # ---- Split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # ---- Preprocessor ----
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

    # ---- Model ----
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1
    )

    # ---- Pipeline ----
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # ---- Train ----
    pipeline.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = pipeline.predict(X_val)
    metrics = {
        "MAE": float(mean_absolute_error(y_val, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "R2": float(r2_score(y_val, y_pred))
    }

    # ---- Save Artifacts ----
    joblib.dump(pipeline, args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model training complete!")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")

if __name__ == "__main__":
    main()
>>>>>>> e1b9a867411dbb77ab32dd9e3d5dd674e5971692
