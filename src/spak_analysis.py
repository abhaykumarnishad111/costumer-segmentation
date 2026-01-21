import pandas as pd
import os
from scipy.stats import f_oneway, chi2_contingency

# -----------------------------------
# PATH SETUP
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# -----------------------------------
# LOAD DATA
# -----------------------------------
cluster_df = pd.read_csv(os.path.join(DATA_DIR, "clustered_customers.csv"))
rfm_df = pd.read_csv(os.path.join(DATA_DIR, "rfm_segmented.csv"))

print("✅ Data loaded successfully")

# -----------------------------------
# 1️⃣ DESCRIPTIVE STATISTICS (SPaK STEP 1)
# -----------------------------------
print("\n--- DESCRIPTIVE STATISTICS BY KMEANS CLUSTER ---")
desc_stats = cluster_df.groupby("KMeans_Cluster").mean(numeric_only=True)
print(desc_stats)

# -----------------------------------
# 2️⃣ ANOVA TEST (SPaK STEP 2)
# -----------------------------------
def anova_test(df, cluster_col, feature):
    groups = [
        df[df[cluster_col] == c][feature]
        for c in df[cluster_col].unique()
    ]
    return f_oneway(*groups)

features_to_test = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

print("\n--- ANOVA TEST RESULTS (KMeans Clusters) ---")
for feature in features_to_test:
    f_stat, p_value = anova_test(cluster_df, "KMeans_Cluster", feature)
    print(f"{feature}: F={f_stat:.3f}, p={p_value:.5f}")

# -----------------------------------
# 3️⃣ CHI-SQUARE TEST (SPaK STEP 3)
# -----------------------------------
# Gender vs Cluster relationship
contingency = pd.crosstab(
    cluster_df["Gender"],
    cluster_df["KMeans_Cluster"]
)

chi2, p, dof, expected = chi2_contingency(contingency)

print("\n--- CHI-SQUARE TEST (Gender vs KMeans Cluster) ---")
print(f"Chi2={chi2:.3f}, p-value={p:.5f}")

# -----------------------------------
# 4️⃣ RFM STATISTICAL VALIDATION
# -----------------------------------
print("\n--- RFM STATISTICS ---")
print(rfm_df[["Recency", "Frequency", "Monetary"]].describe())

# -----------------------------------
# 5️⃣ SAVE SPaK RESULTS
# -----------------------------------
desc_stats.to_csv(os.path.join(DATA_DIR, "spak_cluster_statistics.csv"))

print("\n✅ SPaK analysis completed successfully")
print("📁 Output saved: spak_cluster_statistics.csv")
