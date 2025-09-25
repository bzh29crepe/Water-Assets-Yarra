import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import requests
from pathlib import Path

# ---- Config ----
FAILURE_MODEL_ID = "1vCpGmY0y2cCsmSkr4ZJVHK196qqXKLwD"

# ---- Helper function to download the model from Google Drive ----
@st.cache_resource
def load_model_from_gdrive(file_id, local_filename):
    """Télécharge un modèle depuis Google Drive et le charge avec joblib."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    local_path = Path(local_filename)

    # Télécharger le fichier seulement s'il n'existe pas déjà
    if not local_path.exists():
        st.write("Téléchargement du modèle depuis Google Drive...")
        response = requests.get(url)
        if response.status_code == 200:
            local_path.write_bytes(response.content)
            st.write(f"Modèle téléchargé avec succès ({os.path.getsize(local_path)} octets)")
        else:
            st.error(f"Erreur lors du téléchargement. Code HTTP: {response.status_code}")
            st.stop()

    # Charger le modèle
    return joblib.load(local_path)

# ---- Load Data and Model ----
@st.cache_data
def load_assets():
    base_path = Path(__file__).resolve().parent.parent
    csv_path = "data/yarra_assets.csv"
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model():
    return load_model_from_gdrive(FAILURE_MODEL_ID, "failure_model.joblib")

# ---- Streamlit Layout ----
st.title("Asset Failure Probability")

# Load data and model
df = load_assets()
model = load_model()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
zones = st.sidebar.multiselect("Zone", df["zone"].unique(), default=df["zone"].unique())
types = st.sidebar.multiselect("Asset Type", df["asset_type"].unique(), default=df["asset_type"].unique())

filtered_df = df[df["zone"].isin(zones) & df["asset_type"].isin(types)]

# ---- Predict probabilities ----
X = filtered_df.drop(columns=["Failure", "asset_id", "geometry"], errors="ignore")
current_year = pd.Timestamp.now().year
X["age"] = current_year - X["installation_year"]

filtered_df["Failure_Prob"] = model.predict_proba(X)[:, 1]

# ---- KPIs ----
st.subheader("Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Assets", len(filtered_df))
col2.metric("Average Failure Probability", f"{filtered_df['Failure_Prob'].mean():.2f}")

# ---- Map ----
st.subheader("Map of Assets by Failure Probability")
coords = filtered_df["geometry"].str.split(",", expand=True).astype(float)
filtered_df["lat"] = coords[0]
filtered_df["lon"] = coords[1]

map_fig = px.scatter_mapbox(
    filtered_df,
    lat="lat",
    lon="lon",
    color="Failure_Prob",
    hover_data=["asset_id", "asset_type", "material", "Failure_Prob"],
    color_continuous_scale="Reds",
    title="Failure Probability by Location",
    zoom=10
)
map_fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(map_fig, use_container_width=True)

# ---- Table ----
st.subheader("Detailed Asset Table")
st.dataframe(
    filtered_df[
        ["asset_id", "asset_type", "material", "installation_year",
         "Failure_Prob", "zone", "status"]
    ].sort_values("Failure_Prob", ascending=False),
    use_container_width=True
)
