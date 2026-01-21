import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.title("🔍 Cluster Visualization")

df = pd.read_csv(os.path.join(DATA_DIR, "clustered_customers.csv"))

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
X = StandardScaler().fit_transform(df[features])

pca = PCA(n_components=2)
df["PC1"], df["PC2"] = pca.fit_transform(X).T

cluster_type = st.selectbox(
    "Select Cluster Type",
    ["KMeans_Cluster", "DBSCAN_Cluster", "AE_Cluster"]
)

fig = px.scatter(
    df, x="PC1", y="PC2",
    color=cluster_type,
    title="Cluster Distribution (PCA)"
)

st.plotly_chart(fig, use_container_width=True)
