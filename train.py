
import os, pickle
from utils import parse_log, featurize
from sklearn.ensemble import IsolationForest

TRAIN_FILE = "logs/train_logs.txt"
MODEL_FILE = "models/anomaly_model.pkl"

lines = [l.strip() for l in open(TRAIN_FILE).readlines() if l.strip()]
features = [featurize(parse_log(l)) for l in lines if parse_log(l)]

if not features:
    raise ValueError("Training data empty! please addd")

model = IsolationForest(contamination=0.02, random_state=42)
model.fit(features)

os.makedirs("models", exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Week 1: Model trained and saved!")
