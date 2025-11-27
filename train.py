import pickle
from sklearn.ensemble import IsolationForest
from utils import parse_log, featurize

# Load training logs
lines = open("logs/train_logs.txt").read().splitlines()
features = [featurize(parse_log(l)) for l in lines if parse_log(l)]

# Train model
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(features)

# Save model
pickle.dump(model, open("models/anomaly_model.pkl", "wb"))
print("Week 1: Model trained and saved!")
#======================================================