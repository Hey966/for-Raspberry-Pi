# dashboard_client.py
# -*- coding: utf-8 -*-
import time
import requests

DASHBOARD_URL = "http://127.0.0.1:5000/ingest"  # 若在別台機器，改成伺服器 IP 或 ngrok https
AUTH_TOKEN = ""                                  # 若 app.py 有設 AUTH_TOKEN，填同一組；否則留空
HEADERS = {"X-Auth-Token": AUTH_TOKEN} if AUTH_TOKEN else {}

# 最小冷卻（秒）：避免短時間內洗頻
PUSH_MIN_INTERVAL_SEC = 1.0
_last_push_ts = 0.0
_last_people = None

def push_people_count(count: int, temp: float = 0.0, hum: float = 0.0, force=False, timeout=2) -> bool:
    """立即推送一筆人數（可含 temp/hum）。"""
    payload = {
        "timestamp": int(time.time()),
        "people": int(count),
        "temp": float(temp),
        "hum": float(hum)
    }
    try:
        r = requests.post(DASHBOARD_URL, json=payload, headers=HEADERS, timeout=timeout)
        ok = r.ok and r.json().get("ok")
        if not ok:
            print("[INGEST][WARN]", r.status_code, r.text)
        return bool(ok)
    except Exception as e:
        print("[INGEST][ERR]", e)
        return False

def push_people_count_throttled(count: int, temp: float = 0.0, hum: float = 0.0) -> bool:
    """只在人數變化、且超過最小間隔時才推送。"""
    global _last_push_ts, _last_people
    now = time.time()
    if _last_people is None or count != _last_people or (now - _last_push_ts) >= PUSH_MIN_INTERVAL_SEC:
        ok = push_people_count(count, temp=temp, hum=hum)
        if ok:
            _last_people = count
            _last_push_ts = now
        return ok
    return False
