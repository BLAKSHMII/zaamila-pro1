from utils import parse_log, featurize
from sklearn.ensemble import IsolationForest
import pickle

lines = open("logs/train_logs.txt").read().splitlines()

features = []
for l in lines:
    parsed = parse_log(l)
    if parsed:
        features.append(featurize(parsed))

print("DEBUG: Number of features =", len(features))
print("DEBUG: First feature =", features[0] if features else "None")

if not features:
    raise ValueError("🚨 Training data is empty! Check train_logs.txt")
# Load training logs
lines = open("logs/train_logs.txt").read().splitlines()
features = [featurize(parse_log(l)) for l in lines if parse_log(l)]

# Train model
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(features)

# Save model
pickle.dump(model, open("models/anomaly_model.pkl", "wb"))
print("Week 1: Model trained and saved!")
