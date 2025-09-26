import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import requests
from pathlib import Path
from datetime import datetime

# =============================================================================
# Streamlit Page Config
# =============================================================================
st.set_page_config(
    page_title="Failure Probability",
    layout="wide"
)

st.session_state.update({"__streamlit_page_name__": "Failure Probability"})

# =============================================================================
# Config
# =============================================================================
RUL_MODEL_URL = "https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/rul_model.joblib"
FAILURE_MODEL_URL = "https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/failure_model.joblib"

RUL_LOCAL_MODEL_PATH = Path("rul_model.joblib")
FAILURE_LOCAL_MODEL_PATH = Path("failure_model.joblib")

DATA_PATH = Path("data/yarra_assets_unknown.csv")

# =============================================================================
# Helper: Download model from Hugging Face
# =============================================================================
@st.cache_resource
def download_model_from_hf(url, local_path):
    """Download model from Hugging Face if not available locally."""
    if not local_path.exists():
        st.write(f"Downloading model from Hugging Face: {url}")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            st.write(f"Model successfully downloaded ({size_mb:.2f} MB)")
        else:
            st.error(f"Error downloading model: HTTP {response.status_code}")
            st.stop()

    return joblib.load(local_path)

# =============================================================================
# Load Dataset
# =============================================================================
@st.cache_data
def load_assets():
    """Load the raw dataset."""
    return pd.read_csv(DATA_PATH)

# =============================================================================
# Preprocess for Prediction
# =============================================================================
def preprocess_data(df):
    """
    Prepare the dataset by computing additional features like 'age'.
    """
    current_year = datetime.now().year

    # Compute 'age' if missing
    if "age" not in df.columns:
        df["age"] = current_year - df["installation_year"]

    return df

# =============================================================================
# Predict Remaining Useful Life (RUL)
# =============================================================================
def predict_rul(df, rul_model):
    """
    Use the RUL model to predict Remaining Useful Life and
    add it to the dataframe as 'Remaining_Years'.
    """
    feature_cols = [
        "asset_name", "asset_type", "material", "diameter",
        "installation_year", "status", "network_type",
        "length", "depth", "zone", "pressure_rating",
        "location", "age"
    ]

    # Check for missing columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for RUL prediction: {missing_cols}")
        return df

    X = df[feature_cols]
    df["Remaining_Years"] = rul_model.predict(X)

    return df

# =============================================================================
# Predict Failure Probability
# =============================================================================
def predict_failure(df, failure_model):
    """
    Use the failure model to predict failure probabilities.
    Requires 'Remaining_Years' from RUL predictions.
    """
    feature_cols = [
        "asset_name", "asset_type", "material", "diameter",
        "installation_year", "status", "network_type",
        "length", "depth", "zone", "pressure_rating",
        "location", "age", "Remaining_Years"
    ]

    # Check for missing columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for Failure prediction: {missing_cols}")
        return df

    X = df[feature_cols]
    df["Failure_Prob"] = failure_model.predict_proba(X)[:, 1]

    return df

# =============================================================================
# Load Models
# =============================================================================
@st.cache_resource
def load_models():
    rul_model = download_model_from_hf(RUL_MODEL_URL, RUL_LOCAL_MODEL_PATH)
    failure_model = download_model_from_hf(FAILURE_MODEL_URL, FAILURE_LOCAL_MODEL_PATH)
    return rul_model, failure_model

# =============================================================================
# Streamlit Layout
# =============================================================================
st.title("Asset Failure Probability Dashboard")
st.markdown("""
This dashboard first **predicts the Remaining Useful Life (RUL)** of each asset,
then uses that prediction along with other features to **predict failure probability**.

The data comes from `yarra_assets_unknown.csv` and goes through two prediction stages:
1. RUL Prediction → `Remaining_Years`
2. Failure Prediction → `Failure_Prob`
""")

# ---- Load data and models ----
df = load_assets()
df = preprocess_data(df)
rul_model, failure_model = load_models()

# ---- Generate RUL predictions ----
df = predict_rul(df, rul_model)

# ---- Generate failure predictions ----
df = predict_failure(df, failure_model)

# =============================================================================
# Sidebar Filters
# =============================================================================
st.sidebar.header("Filters")

zones = st.sidebar.multiselect(
    "Zone",
    options=df["zone"].unique(),
    default=df["zone"].unique()
)

types = st.sidebar.multiselect(
    "Asset Type",
    options=df["asset_type"].unique(),
    default=df["asset_type"].unique()
)

min_failure = float(df["Failure_Prob"].min())
max_failure = float(df["Failure_Prob"].max())

failure_range = st.sidebar.slider(
    "Filter by Failure Probability",
    min_value=min_failure,
    max_value=max_failure,
    value=(min_failure, max_failure),
    step=0.01
)

# ---- Filter dataframe ----
filtered_df = df[
    (df["zone"].isin(zones)) &
    (df["asset_type"].isin(types)) &
    (df["Failure_Prob"].between(failure_range[0], failure_range[1]))
]

# =============================================================================
# KPIs
# =============================================================================
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Assets", len(filtered_df))
col2.metric("Average Failure Probability", f"{filtered_df['Failure_Prob'].mean():.2f}")
col3.metric("Average Remaining Life (Years)", f"{filtered_df['Remaining_Years'].mean():.1f}")

# =============================================================================
# Top 5 Assets with Highest Failure Probability
# =============================================================================
st.subheader("Top 5 Assets with Highest Failure Probability")

top_5 = filtered_df.sort_values(by="Failure_Prob", ascending=False).head(5)
st.table(top_5[[
    "asset_id", "asset_type", "material", "Remaining_Years", "Failure_Prob", "zone", "status"
]])

st.markdown("""
These assets should be **prioritized for preventive maintenance** or **immediate inspection** due to their high risk of failure.
""")

# =============================================================================
# Map Visualization
# =============================================================================
st.subheader("Geospatial View of Assets by Failure Probability")

# Split geometry into lat/lon
coords = filtered_df["geometry"].str.split(",", expand=True).astype(float)
filtered_df["lat"] = coords[0]
filtered_df["lon"] = coords[1]

map_fig = px.scatter_mapbox(
    filtered_df,
    lat="lat",
    lon="lon",
    color="Failure_Prob",
    hover_data=["asset_id", "asset_type", "material", "Remaining_Years", "Failure_Prob"],
    color_continuous_scale="Reds",
    title="Failure Probability by Location",
    zoom=10
)
map_fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(map_fig, use_container_width=True)

# =============================================================================
# Full Table
# =============================================================================
st.subheader("Detailed Asset Table")
st.dataframe(
    filtered_df[
        [
            "asset_id", "asset_name", "asset_type", "material",
            "installation_year", "Remaining_Years", "Failure_Prob",
            "zone", "status", "location", "age"
        ]
    ].sort_values("Failure_Prob", ascending=False),
    use_container_width=True
)

# =============================================================================
# Histogram of Failure Probability
# =============================================================================
st.subheader("Failure Probability Distribution")

hist_fig = px.histogram(
    filtered_df,
    x="Failure_Prob",
    nbins=20,
    title="Distribution of Failure Probabilities",
    labels={"Failure_Prob": "Failure Probability"}
)
st.plotly_chart(hist_fig, use_container_width=True)

st.markdown("""
Assets with a **high probability of failure** should be inspected and scheduled for **preventive maintenance** to avoid costly breakdowns.
""")
