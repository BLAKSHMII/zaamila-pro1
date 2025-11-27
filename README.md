Sentinel: Real-Time Log Anomaly Detection
Project Overview

Sentinel is a Python-based system to monitor live logs, detect anomalies, and send alerts. It uses:

Log Parsing & Feature Extraction (Regex + numerical features)

Isolation Forest for unsupervised anomaly detection

Real-time log tailing for live monitoring

Slack alerts for anomaly notification

Configurable via YAML

Project Folder Structure
sentinel/
├── logs/
│   ├── train_logs.txt      # Training logs (normal traffic)
│   └── access.log          # Live logs for detection
├── models/                 # Trained IsolationForest model
├── utils.py                # Log parser & featurizer
├── train.py                # Model training / retraining script
├── detector.py             # Real-time anomaly detection
├── alert.py                # Slack alert integration
├── testlog_generator.py    # Simulates live logs
├── config.yaml             # Configuration file
└── requirements.txt        # Python dependencies

Installation
git clone <repo-url>
cd sentinel
pip install -r requirements.txt


requirements.txt example:

scikit-learn
requests
PyYAML
watchdog

Week-Wise Implementation
Week 1 — Parse Logs & Train Model

Script: train.py

Utils: utils.py (parse_log + featurize functions)

Training logs: logs/train_logs.txt

Steps:

python train.py


Expected Output:

✅ Week 1: Model trained and saved successfully!


Parses logs, extracts features, and trains an Isolation Forest model.

Saves model to models/anomaly_model.pkl.

Week 2 — Real-Time Detection

Script: detector.py

Simulated logs: testlog_generator.py writes to logs/access.log

Run live simulation:

Terminal 1: Generate logs

python testlog_generator.py


Terminal 2: Detect anomalies

python detector.py


Expected Output:

OK: 10.0.0.1 - - [27/Nov/2025] "GET /home HTTP/1.1" 200 500
🚨 Anomaly Detected: 10.0.0.9 - - [27/Nov/2025] "GET /admin HTTP/1.1" 500 50


Tail logs in real-time

Detects anomalies (-1 in Isolation Forest)

Week 3 — Slack Alerts

Script: alert.py

Update: detector.py integrates send_slack()

When anomaly detected:

🚨 Anomaly Detected: 10.0.0.9 - - [27/Nov/2025] "GET /admin HTTP/1.1" 500 50


Alert is sent to your configured Slack webhook.

Week 4 — Config + Retraining

Config file: config.yaml

log_file: logs/access.log
train_file: logs/train_logs.txt
model_file: models/anomaly_model.pkl
sleep_interval: 0.2
slack_webhook: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"


Update train.py to read YAML for retraining.

Run retraining:

python train.py


Expected Output:

✅ Week 4: Model retrained and saved successfully!


Configurable logs, model path, sleep interval, Slack webhook.

Supports performance testing and live retraining.

Summary Table
Week	Task
1	Parse logs, extract features, train IsolationForest model
2	Real-time log tailing + anomaly detection
3	Slack alert integration for anomalies
4	Config YAML + retraining + performance testing
Notes

Make sure train_logs.txt contains enough normal logs to avoid empty vocabulary or 1D feature errors.

Use proper Slack webhook URL in config.yaml.

testlog_generator.py simulates anomalies (~10% probability) for testing.
