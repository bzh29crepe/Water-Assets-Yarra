import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---- Load Model and Data ----
@st.cache_data
def load_assets():
    return pd.read_csv("../data/yarra_assets.csv")

@st.cache_resource
def load_model():
    return joblib.load("../models/rul_model.joblib")

# ---- Streamlit Layout ----
st.set_page_config(page_title="Remaining Useful Life Dashboard", layout="wide")

st.title("Water Asset Remaining Useful Life (RUL) Dashboard")
st.markdown("This dashboard predicts and visualizes the **remaining useful life** of water infrastructure assets.")

# Load data and model
df = load_assets()
model = load_model()

# ---- Sidebar filters ----
st.sidebar.header("Filters")

# Zone filter
zones = st.sidebar.multiselect(
    "Zone",
    options=df["zone"].unique(),
    default=df["zone"].unique()
)

# Asset Type filter
types = st.sidebar.multiselect(
    "Asset Type",
    options=df["asset_type"].unique(),
    default=df["asset_type"].unique()
)

# Remaining Life slider filter
min_rul = float(df["Remaining_Years"].min())
max_rul = float(df["Remaining_Years"].max())

rul_range = st.sidebar.slider(
    "Filter by Remaining Life (Years)",
    min_value=min_rul,
    max_value=max_rul,
    value=(min_rul, max_rul),
    step=0.5
)

# Apply filters
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

# ---- Top 5 Assets with Smallest Remaining Life ----
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

# ---- Table ----
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
