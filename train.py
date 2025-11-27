import os, pickle
from utils import parse_log, featurize
from sklearn.ensemble import IsolationForest
import yaml

# ---------- Week 4 config ----------
cfg = yaml.safe_load(open("config.yaml"))
TRAIN_FILE = cfg.get("train_file", "logs/train_logs.txt")
MODEL_FILE = cfg.get("model_file", "models/anomaly_model.pkl")

# ---------- Week 1 training ----------
lines = [l.strip() for l in open(TRAIN_FILE).readlines() if l.strip()]
features = [featurize(parse_log(l)) for l in lines if parse_log(l)]

if not features:
    raise ValueError("🚨 Training data empty! Add valid logs to train_logs.txt")

print("DEBUG: Number of features =", len(features))
print("DEBUG: First feature =", features[0])

model = IsolationForest(contamination=0.02, random_state=42)
model.fit(features)

os.makedirs("models", exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("✅ Week 1: Model trained and saved using config from Week 4!")
