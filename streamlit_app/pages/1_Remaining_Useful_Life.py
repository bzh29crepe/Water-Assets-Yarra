import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import requests
from pathlib import Path

# ---- Config ----
HUGGINGFACE_MODEL_URL = "https://huggingface.co/louislb1302/failure_model.joblib/resolve/main/rul_model.joblib"

st.set_page_config(
    page_title="Remaining Useful Life",
)

st.session_state.update({"__streamlit_page_name__": "Remaining Useful Life"})


def download_model_from_hf(url, local_filename):
    local_path = Path(local_filename)
    if not local_path.exists():
        st.write("Downloading the model from Hugging Face...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        size_mb = local_path.stat().st_size / (1024 * 1024)
        st.write(f"Model successfully downloaded ({size_mb:.2f} MB)")
    return local_path

# ---- Load Data ----
@st.cache_data
def load_assets():
    csv_path = Path("data/yarra_assets.csv")
    return pd.read_csv(csv_path)

# ---- Load Model ----
@st.cache_resource
def load_model():
    model_path = download_model_from_hf(HUGGINGFACE_MODEL_URL, "rul_model.joblib")
    return joblib.load(model_path)

# ---- Streamlit Layout ----
st.set_page_config(page_title="Remaining Useful Life Dashboard", layout="wide")

st.title("Water Asset Remaining Useful Life (RUL) Dashboard")
st.markdown("This dashboard predicts and visualizes the **remaining useful life** of water infrastructure assets.")

# Load data and model
df = load_assets()
model = load_model()

# ---- Sidebar filters ----
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

min_rul = float(df["Remaining_Years"].min())
max_rul = float(df["Remaining_Years"].max())

rul_range = st.sidebar.slider(
    "Filter by Remaining Life (Years)",
    min_value=min_rul,
    max_value=max_rul,
    value=(min_rul, max_rul),
    step=0.5
)

# ---- Filter the dataframe ----
filtered_df = df[
    (df["zone"].isin(zones)) &
    (df["asset_type"].isin(types)) &
    (df["Remaining_Years"].between(rul_range[0], rul_range[1]))
]

# ---- Display KPIs ----
st.subheader("Key Performance Indicators")

total_assets = len(filtered_df)
avg_rul = filtered_df["Remaining_Years"].mean()

col1, col2 = st.columns(2)
col1.metric("Total Assets", total_assets)
col2.metric("Average Remaining Life (Years)", f"{avg_rul:.1f}")

# ---- Top 5 Assets ----
st.subheader("Top 5 Assets with the Smallest Remaining Life")
top_5 = filtered_df.sort_values(by="Remaining_Years", ascending=True).head(5)
st.table(top_5[["asset_id", "asset_type", "material", "Remaining_Years", "zone", "status"]])

st.markdown("""
These assets should be **prioritized for inspection or replacement** as they have the **lowest remaining lifespan**.
""")

# ---- Map View ----
st.subheader("Geospatial View of Assets")

# Split geometry into lat/lon
coords = filtered_df["geometry"].str.split(",", expand=True).astype(float)
filtered_df["lat"] = coords[0]
filtered_df["lon"] = coords[1]

map_fig = px.scatter_mapbox(
    filtered_df,
    lat="lat",
    lon="lon",
    color="Remaining_Years",
    size="diameter",
    hover_data=["asset_id", "asset_type", "material", "Remaining_Years"],
    color_continuous_scale="RdYlGn",
    title="Assets by Remaining Useful Life",
    zoom=10
)
map_fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(map_fig, use_container_width=True)

# ---- Full Table ----
st.subheader("Full Asset Table")
st.dataframe(filtered_df[[
    "asset_id", "asset_type", "material", "installation_year",
    "Remaining_Years", "zone", "status"
]], use_container_width=True)

# ---- Histogram ----
st.subheader("Remaining Life Distribution")
hist_fig = px.histogram(
    filtered_df,
    x="Remaining_Years",
    nbins=20,
    title="Distribution of Remaining Useful Life"
)
st.plotly_chart(hist_fig, use_container_width=True)

st.markdown("Assets with **low remaining life** should be prioritized for replacement or inspection.")
