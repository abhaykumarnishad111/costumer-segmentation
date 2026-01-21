import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.title("📌 Project Overview")

df = pd.read_csv(os.path.join(DATA_DIR, "clustered_customers.csv"))
rfm = pd.read_csv(os.path.join(DATA_DIR, "rfm_segmented.csv"))

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", df["CustomerID"].nunique())
col2.metric("KMeans Clusters", df["KMeans_Cluster"].nunique())
col3.metric("RFM Segments", rfm["RFM_Segment"].nunique())

st.success("All models executed successfully")
