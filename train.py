# ==========================================
# Week 1: Log Anomaly Detection - Model Training
# ==========================================

import os
import pickle
from utils import parse_log, featurize
from sklearn.ensemble import IsolationForest

# --------------------------
# File paths
# --------------------------
TRAIN_FILE = "logs/train_logs.txt"          # Training logs
MODEL_FILE = "models/anomaly_model.pkl"     # Path to save trained model

# --------------------------
# Read and parse training logs
# --------------------------
lines = [l.strip() for l in open(TRAIN_FILE).readlines() if l.strip()]
features = [featurize(parse_log(l)) for l in lines if parse_log(l)]

# --------------------------
# Check if features are empty
# --------------------------
if not features:
    raise ValueError("🚨 Training data empty! Please add valid logs to train_logs.txt")

# Debug prints to verify features
print("DEBUG: Number of features =", len(features))
print("DEBUG: First feature =", features[0])

# --------------------------
# Train Isolation Forest model
# --------------------------
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(features)

# --------------------------
# Save trained model
# --------------------------
os.makedirs("models", exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Week 1: Model trained and saved successfully!")
