import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.title("⬇️ Download Results")

files = {
    "Clustered Customers": "clustered_customers.csv",
    "RFM Segmentation": "rfm_segmented.csv",
    "SPaK Statistics": "spak_cluster_statistics.csv"
}

for name, file in files.items():
    path = os.path.join(DATA_DIR, file)
    with open(path, "rb") as f:
        st.download_button(name, f, file_name=file)
