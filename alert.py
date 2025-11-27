import requests

def send_slack(msg, webhook="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"):
    try:
        requests.post(webhook, json={"text": msg})
    except Exception as e:
        print("⚠️ Slack alert failed:", e)
