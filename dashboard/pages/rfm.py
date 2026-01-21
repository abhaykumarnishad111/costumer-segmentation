import streamlit as st
import pandas as pd
import plotly.express as px
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.title("💰 RFM Analysis")

rfm = pd.read_csv(os.path.join(DATA_DIR, "rfm_segmented.csv"))

st.dataframe(rfm.head())

fig = px.scatter(
    rfm,
    x="Recency",
    y="Monetary",
    size="Frequency",
    color="RFM_Segment",
    title="RFM Segmentation"
)

st.plotly_chart(fig, use_container_width=True)
