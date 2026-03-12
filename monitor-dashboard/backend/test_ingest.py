# test_ingest.py
import time, requests
url = "http://127.0.0.1:5000/ingest"
r = requests.post(url, json={
    "timestamp": int(time.time()),
    "people": 0, "temp": 0, "hum": 0
}, timeout=2)
print(r.status_code, r.text)
