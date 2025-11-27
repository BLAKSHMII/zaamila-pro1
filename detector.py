import time, pickle
from utils import parse_log, featurize

LOG_FILE = "logs/access.log"
MODEL_FILE = "models/anomaly_model.pkl"

model = pickle.load(open(MODEL_FILE, "rb"))

def tail_file(path):
    with open(path,"r") as f:
        f.seek(0,2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            yield line

for line in tail_file(LOG_FILE):
    parsed = parse_log(line)
    if parsed:
        feat = [featurize(parsed)]
        result = model.predict(feat)[0]
        if result == -1:
            print("🚨 Anomaly Detected:", line.strip())
        else:
            print("OK:", line.strip())
#week-3
from alert import send_slack

if result == -1:
    print("🚨 Anomaly Detected:", line.strip())
    send_slack(f"🚨 Anomaly detected: {line.strip()}")
