import pandas as pd
import os

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "rfm_data.csv"))
print("Columns found:", list(df.columns))

# -----------------------------
# DATE CONVERSION
# -----------------------------
df["PurchaseDate"] = pd.to_datetime(df["PurchaseDate"])

# -----------------------------
# RFM CALCULATION
# -----------------------------
snapshot_date = df["PurchaseDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "PurchaseDate": lambda x: (snapshot_date - x.max()).days,
    "OrderID": "count",
    "TransactionAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# -----------------------------
# SAFE RFM SCORING
# -----------------------------
def safe_qcut(series, q, labels):
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except ValueError:
        return pd.cut(series.rank(method="first"), bins=q, labels=labels)

rfm["R_Score"] = safe_qcut(rfm["Recency"], 4, [4, 3, 2, 1])
rfm["F_Score"] = safe_qcut(rfm["Frequency"], 4, [1, 2, 3, 4])
rfm["M_Score"] = safe_qcut(rfm["Monetary"], 4, [1, 2, 3, 4])

# -----------------------------
# RFM SEGMENT
# -----------------------------
rfm["RFM_Segment"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
rfm.to_csv(os.path.join(DATA_DIR, "rfm_segmented.csv"))

print("✅ RFM segmentation completed successfully")
print(rfm.head())
