import streamlit as st
from pathlib import Path

# ---- CONFIGURATION ----
st.set_page_config(page_title="Water Asset Dashboard", layout="wide")

# ---- BARRE DE NAVIGATION ----
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .nav-container {
        display: flex;
        justify-content: flex-start;
        gap: 20px;
        margin-bottom: 20px;
    }
    .nav-button {
        padding: 8px 16px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #f9f9f9;
        color: #333;
        text-decoration: none;
        font-weight: 500;
    }
    .nav-button:hover {
        background-color: #e6e6e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Menu ----
st.title("Water Infrastructure Asset Dashboard")

# readme.md
readme_path = Path("README.md")
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    st.markdown(readme_content, unsafe_allow_html=True)
else:
    st.warning("README.md file not found.")
