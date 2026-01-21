import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "clustered_customers.csv"))

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]]
X_scaled = StandardScaler().fit_transform(X)

inp = Input(shape=(X_scaled.shape[1],))
encoded = Dense(8, activation="relu")(inp)
decoded = Dense(X_scaled.shape[1])(encoded)

autoencoder = Model(inp, decoded)
encoder = Model(inp, encoded)

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=16, verbose=0)

latent = encoder.predict(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
df["AE_Cluster"] = kmeans.fit_predict(latent)

df.to_csv(os.path.join(DATA_DIR, "clustered_customers.csv"), index=False)
print("✅ Autoencoder clustering completed")
