import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "Mall_Customers.csv"))

print("Original Columns:", list(df.columns))

# -----------------------------
# HANDLE CATEGORICAL DATA
# -----------------------------
if "Gender" in df.columns:
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

# -----------------------------
# SELECT FEATURES
# -----------------------------
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
X = df[features]

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# SAVE PREPROCESSED DATA
# -----------------------------
processed_df = pd.DataFrame(X_scaled, columns=features)
processed_df["CustomerID"] = df["CustomerID"]

processed_df.to_csv(os.path.join(DATA_DIR, "preprocessed_customers.csv"), index=False)

print("✅ Preprocessing completed successfully")
print("📁 Saved: preprocessed_customers.csv")
