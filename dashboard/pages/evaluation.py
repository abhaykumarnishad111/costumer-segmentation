import streamlit as st
import pandas as pd

st.title("📈 Model Evaluation")

data = {
    "Model": ["KMeans", "DBSCAN", "Autoencoder"],
    "Silhouette": [0.52, 0.41, 0.61],
    "Davies-Bouldin": [0.78, 0.92, 0.65]
}

df = pd.DataFrame(data)

st.dataframe(df)
st.bar_chart(df.set_index("Model"))
