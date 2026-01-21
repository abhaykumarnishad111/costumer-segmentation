import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.title("📊 SPaK Statistical Analysis")

stats = pd.read_csv(os.path.join(DATA_DIR, "spak_cluster_statistics.csv"))
st.dataframe(stats)
