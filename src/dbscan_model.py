import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "clustered_customers.csv"))

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]]
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.6, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

df.to_csv(os.path.join(DATA_DIR, "clustered_customers.csv"), index=False)
print("✅ DBSCAN clustering completed")
