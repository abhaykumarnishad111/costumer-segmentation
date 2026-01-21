import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "Mall_Customers.csv"))

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Save safely
df.to_csv(os.path.join(DATA_DIR, "clustered_customers.csv"), index=False)

print("✅ K-Means clustering completed")
