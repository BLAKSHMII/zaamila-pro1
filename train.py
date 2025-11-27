import os
import pickle
import yaml
from utils import parse_log, featurize
from sklearn.ensemble import IsolationForest

# =========================
# Week 1: Default training file
# =========================
TRAIN_FILE = "logs/train_logs.txt"
MODEL_FILE = "models/anomaly_model.pkl"

# =========================
# Week 4: Override using YAML config
# =========================
cfg = yaml.safe_load(open("config.yaml"))
TRAIN_FILE = cfg.get("train_file", TRAIN_FILE)   # fallback to default
MODEL_FILE = cfg.get("model_file", MODEL_FILE)

# =========================
# Read and featurize logs
# =========================
lines = [l.strip() for l in open(TRAIN_FILE).readlines() if l.strip()]
features = [featurize(parse_log(l)) for l in lines if parse_log(l)]

if not features:
    raise ValueError("🚨 Training data empty! Add valid logs")

# Debug
print("DEBUG: Number of features =", len(features))
print("DEBUG: First feature =", features[0])

# =========================
# Train Isolation Forest
# =========================
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(features)

# =========================
# Save model
# =========================
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("✅ Week 1 & 4: Model trained and saved successfully!")
