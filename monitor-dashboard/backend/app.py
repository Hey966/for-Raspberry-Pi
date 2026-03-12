# app.py
# -*- coding: utf-8 -*-
import os
import time
import json
import random
import threading
from collections import deque
from queue import Queue

from flask import Flask, Response, jsonify, request, send_from_directory
try:
    from flask_cors import CORS
except Exception:
    CORS = None  # 沒裝也不影響：同源下不必用 CORS

# ========== 環境設定 ==========
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")          # 若留空，/ingest 不驗證
ENABLE_FAKE = int(os.environ.get("ENABLE_FAKE", "1"))  # 1=內建模擬資料、0=關閉
STATIC_DIR = os.environ.get("STATIC_DIR", "../frontend")

# ========== Flask ==========
app = Flask(__name__, static_folder=STATIC_DIR)
if CORS:
    CORS(app)

# ========== 共享狀態 ==========
clients = []                     # 每個 SSE 客戶端一個 Queue
history = deque(maxlen=3600)     # 最近 3600 筆（約 1 小時）
latest_data = None
lock = threading.Lock()

# ========== 假資料（可關閉） ==========
def generate_fake_data():
    now = int(time.time())
    return {
        "timestamp": now,
        "temp": round(20 + random.uniform(-2, 4), 2),
        "hum": round(40 + random.uniform(-5, 5), 2),
        "people": random.randint(0, 6),
    }

def push_data(data: dict):
    """寫入最新值/歷史，並推送給所有 SSE clients。"""
    global latest_data
    with lock:
        latest_data = data
        history.append(data)
    for q in list(clients):
        try:
            q.put(data)
        except Exception:
            pass

# ========== API ==========
@app.route("/api/data")
def api_data():
    with lock:
        if latest_data is None:
            return jsonify(generate_fake_data())
        return jsonify(latest_data)

@app.route("/api/history")
def api_history():
    with lock:
        return jsonify(list(history))

# ========== SSE ==========
def sse_stream(q: Queue):
    try:
        with lock:
            if latest_data:
                yield f"data: {json.dumps(latest_data, ensure_ascii=False)}\n\n"
        while True:
            msg = q.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
    except GeneratorExit:
        return

@app.route("/stream")
def stream():
    q = Queue()
    clients.append(q)
    return Response(sse_stream(q), mimetype="text/event-stream")

# ========== Ingest（外部送資料） ==========
@app.route("/ingest", methods=["POST"])
def ingest():
    if AUTH_TOKEN:
        token = request.headers.get("X-Auth-Token", "")
        if token != AUTH_TOKEN:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        data = request.get_json(force=True)
        # 只強制要 timestamp & people；temp/hum 改為可選
        required = {"timestamp", "people"}
        if not required.issubset(data.keys()):
            return jsonify({"ok": False, "error": f"missing fields, need {sorted(required)}"}), 400

        # 正規化型別
        data["timestamp"] = int(data["timestamp"])
        data["people"] = int(data["people"])
        data["temp"] = float(data.get("temp", 0.0))
        data["hum"] = float(data.get("hum", 0.0))

        push_data(data)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ========== 健康檢查 ==========
@app.route("/health")
def health():
    return jsonify({"ok": True, "clients": len(clients)})

# ========== 前端靜態檔案 ==========
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/favicon.ico")
def favicon():
    return Response(b"", mimetype="image/x-icon")  # 避免 404 噴 log

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# ========== 背景模擬發送 ==========
def background_fake_publisher():
    while True:
        try:
            push_data(generate_fake_data())
        except Exception as e:
            print("[FAKE_PUB][WARN]", e)
        time.sleep(1)

if __name__ == "__main__":
    if ENABLE_FAKE:
        t = threading.Thread(target=background_fake_publisher, daemon=True)
        t.start()
    print(f"* Serving on http://{HOST}:{PORT}  (SSE /stream, API /api/data, POST /ingest)")
    if AUTH_TOKEN:
        print("* Ingest protected with X-Auth-Token")
    app.run(host=HOST, port=PORT, threaded=True)
