import pandas as pd
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "clustered_customers.csv"))

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_clusters(labels, name):
    if len(set(labels)) <= 1:
        print(f"❌ {name}: Only one cluster found")
        return

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print(f"\n{name} Evaluation:")
    print(f"Silhouette Score : {sil:.3f}")
    print(f"Davies-Bouldin   : {db:.3f}")

# -----------------------------
# RUN EVALUATIONS
# -----------------------------
evaluate_clusters(df["KMeans_Cluster"], "K-Means")
evaluate_clusters(df["DBSCAN_Cluster"], "DBSCAN")
evaluate_clusters(df["AE_Cluster"], "Autoencoder + KMeans")

print("\n✅ Evaluation completed successfully")
