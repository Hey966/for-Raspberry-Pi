# -*- coding: utf-8 -*-
"""
server.py
結合：
- LINE Bot Webhook + 綁定/班級指令 + 請假系統
- 人臉辨識簽到伺服器 + 出勤網頁（含分班管理＋遲到顯示＋班級遲到名單）
"""

import os, json, atexit, subprocess, time, requests, shutil, queue
from pathlib import Path
from urllib.parse import urljoin
from collections import defaultdict

from flask import Flask, request, abort, jsonify, redirect

# === 以此檔所在資料夾為工作目錄（.env / users.json / encodings.npz 同層） ===
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# === 小工具：遮罩長度用，方便 debug ===
def _safe_len(v):
    return 0 if not v else len(v)

def _mask(v, keep=4):
    if not v:
        return "(empty)"
    return v[:keep] + "*" * max(0, len(v) - keep)

# === 讀取 .env（若存在）＋偵錯 ===
dotenv_loaded = False
try:
    from dotenv import load_dotenv, dotenv_values
    dotenv_path = BASE_DIR / ".env"
    print(f"[ENV] target: {dotenv_path} exists={dotenv_path.exists()}")
    if dotenv_path.exists():
        print("[ENV] keys in .env:", list(dotenv_values(dotenv_path, encoding="utf-8").keys()))
    load_dotenv(dotenv_path, override=True, encoding="utf-8")
    dotenv_loaded = True
    print("[ENV] loaded .env =", dotenv_loaded)
except Exception as e:
    print("[ENV][WARN] python-dotenv 未載入：", e)

# === OpenMP / ONNX 臨時設定（要在科學套件 import 前） ===
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# ===== LINE Bot v3 =====
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage
)
from linebot.v3.exceptions import InvalidSignatureError
try:
    from linebot.v3.messaging.exceptions import ApiException
except Exception:
    try:
        from linebot.v3.exceptions import ApiException
    except Exception:
        ApiException = Exception

# ==== 人臉辨識/科學套件 ====
import cv2
import numpy as np
import datetime
import threading
import platform
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ============================================================
# 公用小工具
# ============================================================
def _env_bool(name, default=False):
    v = os.environ.get(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _clean(s: str) -> str:
    if not s:
        return ""
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(z, "")
    return s.strip()

# ============================================================
# users.json & 請假資料 檔案操作
# ============================================================
USERS_JSON = BASE_DIR / "users.json"
LEAVE_JSON = BASE_DIR / "leave_requests.json"

def load_users():
    try:
        return json.loads(USERS_JSON.read_text("utf-8"))
    except FileNotFoundError:
        return {"_by_user_id": {}, "_by_name": {}}
    except Exception as e:
        print("[USERS][ERROR] 讀取失敗", e)
        return {"_by_user_id": {}, "_by_name": {}}

def save_users(data):
    try:
        tmp = USERS_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(USERS_JSON)
        return True
    except Exception as e:
        print("[USERS][ERROR] 寫入失敗", e)
        return False

def bind_user(user_id: str, name: str):
    """
    綁定 LINE userId ↔ 姓名
    回傳:
      ("new", None)
      ("same", old_name)
      ("update", old_name)
      ("invalid", None)
    """
    name = _clean(name)
    if not name:
        return ("invalid", None)

    data   = load_users()
    by_uid = data.setdefault("_by_user_id", {})
    by_name = data.setdefault("_by_name", {})

    if user_id in by_uid:
        old_rec  = by_uid.get(user_id, {})
        old_name = _clean(old_rec.get("name", ""))
        old_cls  = _clean(old_rec.get("class", ""))

        if old_name == name:
            return ("same", old_name)

        # 名字不同 → 更新（移除舊 name 映射，確保同一個 LINE 帳號只有一個名字）
        if old_name and by_name.get(old_name) == user_id:
            del by_name[old_name]

        new_rec = {"name": name}
        if old_cls:
            new_rec["class"] = old_cls
        by_uid[user_id] = new_rec
        by_name[name] = user_id
        save_users(data)
        return ("update", old_name)

    # 第一次綁定
    new_rec = {"name": name}
    by_uid[user_id] = new_rec
    by_name[name] = user_id
    save_users(data)
    return ("new", None)

def get_class_for_name(name: str) -> str:
    """
    依姓名查詢班級（從 users.json _by_user_id 裡的 class）
    找不到就回傳空字串
    """
    name = _clean(name)
    data = load_users()
    by_name = data.get("_by_name", {})
    by_uid = data.get("_by_user_id", {})
    uid = by_name.get(name)
    if not uid:
        return ""
    rec = by_uid.get(uid, {})
    return _clean(rec.get("class", ""))

# ---- 請假資料 ----
def load_leaves():
    try:
        return json.loads(LEAVE_JSON.read_text("utf-8"))
    except FileNotFoundError:
        return {"requests": []}
    except Exception as e:
        print("[LEAVE][ERROR] 讀取失敗", e)
        return {"requests": []}

def save_leaves(data):
    try:
        tmp = LEAVE_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(LEAVE_JSON)
        return True
    except Exception as e:
        print("[LEAVE][ERROR] 寫入失敗", e)
        return False

def add_leave_request(user_id, name, phone, reason, leave_time):
    data = load_leaves()
    reqs = data.setdefault("requests", [])
    req_id = str(int(time.time() * 1000))  # 用時間戳當 id
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    req = {
        "id": req_id,
        "user_id": user_id,
        "name": _clean(name),
        "phone": _clean(phone),
        "reason": reason.strip(),
        "leave_time": leave_time.strip(),
        "status": "pending",        # pending / approved / rejected
        "review_comment": "",
        "created_at": now_str,
        "reviewed_at": "",
    }
    reqs.append(req)
    save_leaves(data)
    print(f"[LEAVE] 新請假申請 id={req_id} name={name}")
    return req

def update_leave_status(req_id, status, comment=""):
    data = load_leaves()
    reqs = data.setdefault("requests", [])
    target = None
    for r in reqs:
        if r.get("id") == req_id:
            target = r
            break
    if not target:
        print("[LEAVE] 找不到 id =", req_id)
        return None

    target["status"] = status
    target["review_comment"] = comment.strip()
    target["reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_leaves(data)
    print(f"[LEAVE] 更新狀態 id={req_id} -> {status} comment={comment}")
    return target

# ============================================================
# Flask / LINE 基本設定
# ============================================================
PORT = int(os.environ.get("PORT", 5001))
HOST = os.environ.get("HOST", "0.0.0.0")
CHANNEL_ACCESS_TOKEN = (os.environ.get("CHANNEL_ACCESS_TOKEN") or "").strip()
CHANNEL_SECRET       = (os.environ.get("CHANNEL_SECRET") or "").strip()

print("[CONFIG] SECRET len =", _safe_len(CHANNEL_SECRET), "value:", _mask(CHANNEL_SECRET))
print("[CONFIG] TOKEN  len =", _safe_len(CHANNEL_ACCESS_TOKEN), "value:", _mask(CHANNEL_ACCESS_TOKEN))

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    print("[HINT] 檢查：1) .env 與 server.py 同層、不是 .env.txt；2) 無空格/引號/Bearer；3) UTF-8；4) 已安裝 python-dotenv")
    raise SystemExit("[FATAL] 缺少 CHANNEL_SECRET 或 CHANNEL_ACCESS_TOKEN。")

app = Flask(__name__)
handler = WebhookHandler(CHANNEL_SECRET)
config  = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

# ============================================================
# 自動啟動 ngrok（本機除錯用）
# ============================================================
def _find_ngrok_exe():
    p = (os.environ.get("NGROK") or "").strip().strip('"')
    if p and os.path.isfile(p):
        return p
    p2 = shutil.which("ngrok")
    if p2:
        return p2
    for cand in (
        r"C:\tools\ngrok\ngrok.exe",
        r"C:\ngrok\ngrok.exe",
        "/usr/local/bin/ngrok",
        "/usr/bin/ngrok",
    ):
        if os.path.isfile(cand):
            return cand
    return None

def _kill_ngrok_silent():
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/IM", "ngrok.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["pkill", "-f", "ngrok"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass

def _probe_public_url(timeout=25):
    api = "http://127.0.0.1:4040/api/tunnels"
    end = time.time() + timeout
    last_err = None
    while time.time() < end:
        try:
            r = requests.get(api, timeout=2)
            if r.ok:
                data = r.json()
                for t in data.get("tunnels", []):
                    pub = t.get("public_url", "")
                    if pub.startswith("https://"):
                        return pub
                for t in data.get("tunnels", []):
                    pub = t.get("public_url", "")
                    if pub:
                        return pub
        except Exception as e:
            last_err = e
        time.sleep(0.8)
    raise RuntimeError(f"無法從 4040 取得 public URL：{last_err}")

def start_ngrok_if_needed(local_host="127.0.0.1", port=5001, webhook_path="/webhook"):
    """
    本機：START_NGROK=1 則啟動 ngrok
    Render：若偵測到 RENDER=true/1，則強制停用 ngrok
    """
    on_render = _env_bool("RENDER", False) or bool(os.environ.get("RENDER_EXTERNAL_URL"))
    start_ngrok = _env_bool("START_NGROK", True) and not on_render
    if not start_ngrok:
        reason = "Render 環境" if on_render else "START_NGROK=0"
        print(f"[NGROK] 跳過啟動（{reason}）")
        return None

    exe = _find_ngrok_exe()
    if not exe:
        print("[NGROK][ERROR] 找不到 ngrok，可在 .env 設 NGROK=完整路徑")
        return None

    region = (os.environ.get("NGROK_REGION") or "").strip() or None
    extra  = (os.environ.get("NGROK_ARGS") or "").strip() or None

    _kill_ngrok_silent()
    cmd = [exe, "http", f"http://{local_host}:{port}"]
    if region:
        cmd += ["--region", region]
    if extra:
        cmd += extra.split()

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    atexit.register(lambda: (proc.poll() is None) and proc.terminate())

    try:
        public_url = _probe_public_url(timeout=25)
        full = urljoin(public_url + "/", webhook_path.lstrip("/"))
        print(f"[NGROK] public url: {public_url}")
        print(f"[NGROK] Webhook：{full}")
        try:
            if os.name == "nt":
                subprocess.run(
                    f'echo {full}| clip',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("[NGROK] 已將 Webhook URL 複製到剪貼簿。")
        except Exception:
            pass
        return public_url
    except Exception as e:
        print("[NGROK][WARN]", e)
        print("[NGROK][HINT] 打開 http://127.0.0.1:4040 檢查 ngrok 狀態。")
        return None

# ============================================================
# LINE 推播佇列（給人臉簽到 / 遲到掃描 / 請假結果 用）
# ============================================================
push_q = queue.Queue(maxsize=256)

def push_async(uid, text):
    try:
        push_q.put_nowait((uid, text))
    except queue.Full:
        print("[PUSH] 佇列已滿，丟棄推播")

def _push_worker():
    print("[PUSH] 背景推播執行緒啟動")
    with ApiClient(config) as api_client:
        api = MessagingApi(api_client)
        while True:
            try:
                uid, text = push_q.get()
                try:
                    api.push_message(
                        PushMessageRequest(to=uid, messages=[TextMessage(text=text)])
                    )
                    print(f"[PUSH] to={uid} OK")
                except ApiException as e:
                    print("[PUSH][ERR] status=", getattr(e, "status", None),
                          "body=", getattr(e, "body", None))
            except Exception as e:
                print("[PUSH][ERR] worker exception:", e)
            finally:
                try:
                    push_q.task_done()
                except Exception:
                    pass

threading.Thread(target=_push_worker, daemon=True).start()

# ============================================================
# 人臉伺服器設定（原出勤系統）
# ============================================================
IS_WINDOWS = (os.name == "nt")
SCRIPT_DIR = BASE_DIR

ENC_PATH    = SCRIPT_DIR / "encodings.npz"
CONFIG_DIR  = SCRIPT_DIR / "checkin"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

MEMBERS_JSON   = CONFIG_DIR / "members.json"

# 辨識門檻
COS_THRESHOLD          = 0.30
COS_SECOND_BEST_MARGIN = 0.03

# 遲到時間設定
LATE_CUTOFF_STR = os.environ.get("LATE_CUTOFF", "08:00")

def _parse_time_hhmm(s, fallback="08:00"):
    try:
        hh, mm = s.split(":")
        return datetime.time(int(hh), int(mm))
    except Exception:
        fh, fm = fallback.split(":")
        return datetime.time(int(fh), int(fm))

LATE_CUTOFF = _parse_time_hhmm(LATE_CUTOFF_STR, "08:00")

# TTS（Windows）
SPEAK_ON_SUCCESS = True
VOICE_RATE       = 0
USE_BEEP_TTS     = True
last_tts         = 0.0
TTS_COOLDOWN     = 1.0

def tts(text):
    if not IS_WINDOWS:
        return
    now = time.time()
    global last_tts
    if now - last_tts < TTS_COOLDOWN:
        return
    last_tts = now

    try:
        import winsound
        winsound.MessageBeep(-1)
    except Exception:
        pass

# 名單載入（名字 → userId，讀 users.json + members.json）
def _load_members_map():
    mapping = {}

    if USERS_JSON.exists():
        try:
            d = json.loads(USERS_JSON.read_text("utf-8"))
            for k, v in (d.get("_by_name") or {}).items():
                if isinstance(v, str) and (v.startswith("U") or v.startswith("C")):
                    mapping[_clean(k)] = _clean(v)
        except Exception as e:
            print("[USERS_JSON][ERR]", e)

    if MEMBERS_JSON.exists():
        try:
            d = json.loads(MEMBERS_JSON.read_text("utf-8-sig"))
            for k, v in d.items():
                if isinstance(v, str) and (v.startswith("U") or v.startswith("C")):
                    mapping.setdefault(_clean(k), _clean(v))
        except Exception as e:
            print("[MEMBERS_JSON][ERR]", e)

    print(f"[INFO] 名單載入完成，總共 {len(mapping)} 筆")
    return mapping

NAME_TO_UID = _load_members_map()

# 簽到紀錄（今天）
last_checkin_time   = defaultdict(lambda: datetime.datetime.min)  # name -> datetime
late_notice_sent_on = {}  # name -> date

def _is_today(dt):
    return (dt.date() == datetime.date.today())

def line_text_checkin(name, ts):
    return f"✅ 簽到成功\n姓名：{name}\n時間：{ts:%Y-%m-%d %H:%M:%S}"

def line_text_late(name, ts):
    return f"⚠️ 遲到通知（截止 {LATE_CUTOFF.hour:02d}:{LATE_CUTOFF.minute:02d}）\n姓名：{name}\n時間：{ts:%Y-%m-%d %H:%M:%S}"

# 今天出勤統計（修正：超過截止時間未簽到也算遲到）
def get_today_attendance():
    rows = []
    now = datetime.datetime.now()
    now_time = now.time()

    # 重新載入名單（避免 users.json / members.json 變更後沒重開）
    global NAME_TO_UID
    NAME_TO_UID = _load_members_map()

    for name in sorted(NAME_TO_UID.keys()):
        ts = last_checkin_time[name]

        if _is_today(ts):
            status   = "checked_in"
            time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            # ⚠️ 這裡用 >=：到截止時間就算遲到
            is_late  = (ts.time() >= LATE_CUTOFF)
        else:
            status   = "not_checked"
            time_str = ""
            # 若現在時間已經超過截止時間，且還沒簽到 → 也視為「遲到（未簽到）」
            if now_time > LATE_CUTOFF:
                is_late = True
            else:
                is_late = False  # 尚未到截止時間，不標遲到

        cls = get_class_for_name(name)

        # ⭐ debug：印出每個人實際被判斷的狀態
        print(
            f"[ATTEND] name={name}, status={status}, checkin_ts={ts}, "
            f"cutoff={LATE_CUTOFF}, is_late={is_late}"
        )

        rows.append({
            "name": name,
            "status": status,
            "time_str": time_str,
            "is_late": is_late,
            "class": cls,
        })
    return rows

# 載入人臉 encodings
if not ENC_PATH.exists():
    raise FileNotFoundError(f"找不到 encodings.npz：{ENC_PATH}")

data = np.load(ENC_PATH, allow_pickle=True)
KNOWN_ENCODINGS = data["encodings"].astype("float32")
KNOWN_NAMES     = data["names"]

if KNOWN_ENCODINGS.ndim != 2 or KNOWN_ENCODINGS.shape[1] != 512:
    raise RuntimeError("encodings 維度不是 (N, 512)，請確認 build_embeddings.py")

def l2_normalize(x):
    return x / (np.linalg.norm(x) + 1e-9)

KNOWN_ENCODINGS = np.array([l2_normalize(e) for e in KNOWN_ENCODINGS], dtype="float32")
print(f"[INFO] 載入人臉庫：{KNOWN_ENCODINGS.shape[0]} 筆")

# InsightFace
use_cuda = ("CUDAExecutionProvider" in ort.get_available_providers())
print("[INFO] ONNX providers:", ort.get_available_providers())
print("[INFO] 使用 CUDA:", use_cuda)

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=(0 if use_cuda else -1), det_size=(640, 640))

# 可選：faiss 加速
try:
    import faiss
    use_faiss = True
    faiss_index = faiss.IndexFlatIP(512)
    faiss_index.add(KNOWN_ENCODINGS)
    print("[INFO] 使用 faiss 做相似度搜尋")
except Exception as e:
    use_faiss = False
    print("[INFO] 無法使用 faiss，改用 numpy，相似度會慢一點：", e)

def decide_name(enc):
    if use_faiss:
        D, I = faiss_index.search(enc.reshape(1, -1), 5)
        sims = D[0]
        idxs = I[0]

        best = None
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            nm = str(KNOWN_NAMES[idx])
            if (best is None) or (sim > best[1]):
                best = (nm, float(sim))

        if best is None:
            return "Unknown", None

        best_name, best_sim = best
        if best_sim >= COS_THRESHOLD:
            return best_name, best_sim
        else:
            return "Unknown", best_sim
    else:
        sims = KNOWN_ENCODINGS @ enc
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        best_name = str(KNOWN_NAMES[idx])
        if best_sim >= COS_THRESHOLD:
            return best_name, best_sim
        else:
            return "Unknown", best_sim

# ============================================================
# Flask 路由
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# ----- LINE Webhook -----
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK", 200

# ----- 測試推播（用姓名查 users.json） -----
@app.route("/push", methods=["GET"])
def push_to_name():
    name = (request.args.get("name") or "").strip()
    text = (request.args.get("text") or "測試訊息").strip()
    if not name:
        return "缺少 ?name=參數", 400
    users = load_users()
    user_id = users.get("_by_name", {}).get(name)
    if not user_id:
        return f"找不到此姓名的綁定：{name}", 404
    try:
        with ApiClient(config) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(to=user_id, messages=[TextMessage(text=text)])
            )
        return f"Push 成功 → {name} ({user_id})：{text}", 200
    except ApiException as e:
        return f"Push 失敗 status={getattr(e,'status',None)}, body={getattr(e,'body',None)}", 500

# ----- 人臉辨識簽到 -----
@app.route("/recognize", methods=["POST"])
def recognize():
    """
    樹莓派每次丟一張圖：
    - 找最大那張臉
    - 比對名字
    - 若 name != Unknown 且今天第一次 → 推播簽到成功
    """
    file = request.files.get("image")
    if not file:
        print("[DEBUG] /recognize 沒收到 image 檔")
        return jsonify({"name": "Unknown", "sim": None}), 200

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        print("[DEBUG] /recognize 影像解碼失敗")
        return jsonify({"name": "Unknown", "sim": None}), 200

    h, w, _ = img.shape
    print(f"[DEBUG] 收到影像尺寸: {w}x{h}")

    faces = face_app.get(img)
    print(f"[DEBUG] InsightFace 偵測到臉數量: {len(faces)}")

    if not faces:
        return jsonify({"name": "Unknown", "sim": None}), 200

    # 最大臉
    def area(f):
        l, t, r, b = f.bbox
        return (r - l) * (b - t)

    f = max(faces, key=area)
    enc = f.normed_embedding.astype("float32")
    enc = l2_normalize(enc)

    name, sim = decide_name(enc)
    print(f"[DEBUG] 最佳匹配：name={name}, sim={sim}")

    now = datetime.datetime.now()

    # 簽到成功推播（僅限今天第一次）
    if name != "Unknown":
        first_today = not _is_today(last_checkin_time[name])
        uid = NAME_TO_UID.get(_clean(name), "")

        if first_today and uid:
            push_async(uid, line_text_checkin(name, now))
            last_checkin_time[name] = now
            tts("簽到成功")

    return jsonify({"name": name, "sim": float(sim) if sim is not None else None}), 200

# ----- 今日出勤 JSON -----
@app.route("/attendance/today.json", methods=["GET"])
def attendance_today_json():
    rows = get_today_attendance()
    return jsonify({
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "items": rows
    })

# ----- 今日出勤管理網頁 -----
@app.route("/attendance/today", methods=["GET"])
def attendance_today_page():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    cutoff = LATE_CUTOFF.strftime("%H:%M")

    html = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <title>__TODAY__ 出勤列表</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      background: radial-gradient(circle at top left,#e0f2fe 0,#f9fafb 40%,#f3f4f6 100%);
      color:#111827;
    }
    .page {
      min-height: 100vh;
      display:flex;
      justify-content:center;
      align-items:flex-start;
      padding: 24px 12px;
    }
    .card {
      width:100%;
      max-width:1080px;
      background:#ffffffcc;
      backdrop-filter: blur(10px);
      padding:24px 24px 20px;
      border-radius:20px;
      box-shadow:
        0 18px 40px rgba(15,23,42,0.12),
        0 1px 2px rgba(15,23,42,0.08);
      border:1px solid rgba(148,163,184,0.35);
    }
    .card-header {
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:12px;
      margin-bottom:16px;
    }
    .title-block h1 {
      font-size: 22px;
      margin:0 0 4px 0;
      display:flex;
      align-items:center;
      gap:8px;
    }
    .title-pill {
      font-size:11px;
      padding:2px 8px;
      border-radius:999px;
      background:#eff6ff;
      color:#1d4ed8;
      border:1px solid #bfdbfe;
    }
    .subtitle {
      font-size:13px;
      color:#6b7280;
      line-height:1.5;
    }
    .subtitle strong { color:#374151; }
    .chip-row {
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      margin-top:6px;
    }
    .chip {
      background:#f3f4f6;
      padding:6px 10px;
      border-radius:999px;
      font-size:12px;
      display:flex;
      align-items:center;
      gap:4px;
      color:#4b5563;
    }
    .chip strong {
      font-size:12px;
      color:#111827;
    }
    .chip.badge-ok {
      background:#ecfdf3;
      color:#15803d;
    }
    .chip.badge-miss {
      background:#fef2f2;
      color:#b91c1c;
    }
    .meta {
      text-align:right;
      font-size:12px;
      color:#9ca3af;
      margin-top:4px;
    }
    .meta span {
      display:inline-flex;
      align-items:center;
      gap:4px;
    }
    .meta-dot {
      width:6px;
      height:6px;
      border-radius:999px;
      background:#22c55e;
      box-shadow:0 0 0 3px rgba(34,197,94,0.25);
    }

    .summary-grid {
      display:grid;
      grid-template-columns: repeat(4,minmax(0,1fr));
      gap:10px;
      margin:10px 0 18px;
    }
    .summary-card {
      background:#f9fafb;
      border-radius:14px;
      padding:10px 12px;
      border:1px solid #e5e7eb;
    }
    .summary-label {
      font-size:11px;
      color:#6b7280;
      margin-bottom:4px;
      display:flex;
      justify-content:space-between;
      align-items:center;
    }
    .summary-value {
      font-size:18px;
      font-weight:700;
      color:#111827;
    }
    .summary-tag-ok {
      font-size:11px;
      color:#166534;
      background:#dcfce7;
      border-radius:999px;
      padding:1px 6px;
    }
    .summary-tag-miss {
      font-size:11px;
      color:#b91c1c;
      background:#fee2e2;
      border-radius:999px;
      padding:1px 6px;
    }
    .summary-tag-rate {
      font-size:11px;
      color:#1d4ed8;
      background:#e0edff;
      border-radius:999px;
      padding:1px 6px;
    }

    .class-summary-card {
      margin-bottom: 18px;
      background:#ffffff;
      border-radius:14px;
      border:1px solid #e5e7eb;
      padding:10px 12px;
    }
    .class-summary-title {
      font-size:13px;
      font-weight:600;
      color:#111827;
      margin-bottom:6px;
      display:flex;
      justify-content:space-between;
      align-items:center;
    }
    .class-summary-title span {
      font-size:11px;
      color:#6b7280;
    }
    .class-summary-table {
      width:100%;
      border-collapse:collapse;
      font-size:12px;
    }
    .class-summary-table th,
    .class-summary-table td {
      padding:6px 8px;
      border-bottom:1px solid #f3f4f6;
      text-align:left;
      white-space:nowrap;
    }
    .class-summary-table th {
      color:#6b7280;
      font-weight:600;
      background:#f9fafb;
    }
    .class-summary-table tr:hover {
      background:#eef2ff;
      cursor:pointer;
    }
    .class-name-cell {
      font-weight:600;
      color:#111827;
    }
    .class-rate-good {
      color:#166534;
      font-weight:600;
    }
    .class-rate-bad {
      color:#b91c1c;
      font-weight:600;
    }

    .toolbar {
      display:flex;
      flex-wrap:wrap;
      gap:12px;
      align-items:center;
      justify-content:space-between;
      margin-bottom:8px;
    }
    .toolbar-left {
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      align-items:center;
    }
    .field {
      display:flex;
      align-items:center;
      gap:6px;
      font-size:12px;
      color:#4b5563;
    }
    .field input[type="text"],
    .field select {
      padding:6px 9px;
      border-radius:8px;
      border:1px solid #d1d5db;
      font-size:13px;
      outline:none;
      min-width:140px;
      background:white;
    }
    .field input[type="checkbox"] {
      transform: scale(1.1);
    }
    .field input[type="text"]:focus,
    .field select:focus {
      border-color:#4f46e5;
      box-shadow:0 0 0 1px rgba(79,70,229,0.4);
    }
    .toolbar-right {
      font-size:12px;
      color:#6b7280;
      display:flex;
      align-items:center;
      gap:8px;
    }
    .refresh-btn {
      padding:4px 10px;
      border-radius:999px;
      border:1px solid #4f46e5;
      background:#eef2ff;
      font-size:12px;
      color:#4f46e5;
      cursor:pointer;
    }
    .refresh-btn:hover {
      background:#e0e7ff;
    }

    .table-wrap {
      margin-top:6px;
      border-radius:14px;
      overflow:hidden;
      border:1px solid #e5e7eb;
      background:#ffffff;
    }
    table {
      width:100%;
      border-collapse: collapse;
      font-size:13px;
    }
    thead {
      background:#f9fafb;
    }
    th, td {
      padding:9px 10px;
      text-align:left;
      white-space:nowrap;
    }
    th {
      font-weight:600;
      color:#4b5563;
      border-bottom:1px solid #e5e7eb;
    }
    th.sortable {
      cursor:pointer;
      user-select:none;
    }
    th.sortable:hover {
      background:#eef2ff;
    }
    th .sort-indicator {
      font-size:11px;
      margin-left:4px;
      color:#9ca3af;
    }
    tbody tr:nth-child(even) {
      background:#f9fafb;
    }
    tbody tr:hover {
      background:#eef2ff;
    }
    .name-cell {
      display:flex;
      align-items:center;
      gap:8px;
    }
    .name-avatar {
      width:22px;
      height:22px;
      border-radius:999px;
      background:linear-gradient(135deg,#60a5fa,#4f46e5);
      color:white;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:11px;
      font-weight:600;
      flex-shrink:0;
    }
    .status-pill {
      display:inline-flex;
      align-items:center;
      gap:5px;
      padding:2px 9px;
      border-radius:999px;
      font-size:12px;
      font-weight:500;
    }
    .status-ok {
      background:#e0f7ec;
      color:#166534;
    }
    .status-dot-ok {
      width:8px;height:8px;border-radius:999px;background:#22c55e;
    }
    .status-miss {
      background:#fef2f2;
      color:#b91c1c;
    }
    .status-dot-miss {
      width:8px;height:8px;border-radius:999px;background:#ef4444;
    }
    .time-cell {
      font-variant-numeric: tabular-nums;
      color:#4b5563;
    }
    .time-cell.empty {
      color:#d1d5db;
      font-style:italic;
    }

    @media (max-width:768px) {
      .card {
        padding:18px 14px 16px;
      }
      .card-header {
        flex-direction:column;
        align-items:flex-start;
      }
      .summary-grid {
        grid-template-columns: repeat(2,minmax(0,1fr));
      }
      .class-summary-card {
        padding:8px 10px;
      }
      .class-summary-table th,
      .class-summary-table td {
        padding:5px 6px;
      }
      .toolbar {
        flex-direction:column;
        align-items:flex-start;
        gap:8px;
      }
      .toolbar-right {
        align-self:flex-end;
      }
      th, td {
        padding:7px 8px;
      }
    }
    @media (max-width:480px) {
      .summary-grid {
        grid-template-columns: repeat(1,minmax(0,1fr));
      }
      .field input[type="text"],
      .field select {
        min-width:0;
        width:100%;
      }
      .toolbar-right {
        align-self:flex-start;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <div class="card-header">
        <div class="title-block">
          <h1>
            出勤列表
            <span class="title-pill">__TODAY__</span>
          </h1>
          <div class="subtitle">
            截止時間：<strong>__CUTOFF__</strong><br>
            只顯示已在 <code>users.json</code> / <code>members.json</code> 建立的名單。狀態：
            <div class="chip-row">
              <div class="chip badge-ok">🟢 <strong>已簽到</strong></div>
              <div class="chip badge-miss">🔴 <strong>未簽到</strong></div>
            </div>
          </div>
        </div>
        <div class="meta">
          <span><span class="meta-dot"></span> 出勤伺服器運行中</span>
        </div>
      </div>

      <div class="summary-grid" id="summary">
      </div>

      <!-- 班級統計區塊 -->
      <div id="classSummary" class="class-summary-card">
      </div>

      <div class="toolbar">
        <div class="toolbar-left">
          <div class="field">
            <span>搜尋姓名：</span>
            <input type="text" id="searchInput" placeholder="輸入關鍵字..." />
          </div>
          <div class="field">
            <span>狀態篩選：</span>
            <select id="statusFilter">
              <option value="all">全部</option>
              <option value="checked_in">只看已簽到</option>
              <option value="not_checked">只看未簽到</option>
            </select>
          </div>
          <div class="field">
            <span>班級篩選：</span>
            <select id="classFilter">
              <option value="all">全部</option>
            </select>
          </div>
          <div class="field">
            <label>
              <input type="checkbox" id="lateOnly" />
              只顯示遲到
            </label>
          </div>
        </div>
        <div class="toolbar-right">
          <span id="rowCountLabel">顯示筆數：0</span>
          <button id="refreshBtn" class="refresh-btn">重新整理</button>
        </div>
      </div>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th class="sortable" data-key="name">
                姓名
                <span class="sort-indicator" id="sortIndicator-name"></span>
              </th>
              <th class="sortable" data-key="class">
                班級
                <span class="sort-indicator" id="sortIndicator-class"></span>
              </th>
              <th class="sortable" data-key="status">
                狀態
                <span class="sort-indicator" id="sortIndicator-status"></span>
              </th>
              <th class="sortable" data-key="time_str">
                簽到時間
                <span class="sort-indicator" id="sortIndicator-time_str"></span>
              </th>
              <th>
                是否遲到
              </th>
            </tr>
          </thead>
          <tbody id="tbody">
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const API_URL = "/attendance/today.json";
    let rawItems = [];
    let currentSortKey = "name";
    let currentSortDir = "asc";

    function fetchAttendance() {
      fetch(API_URL)
        .then(res => res.json())
        .then(data => {
          rawItems = data.items || [];
          renderClassOptions(rawItems);
          renderAll();
        })
        .catch(err => {
          console.error("載入出勤資料失敗:", err);
        });
    }

    function renderClassOptions(allItems) {
      const sel = document.getElementById("classFilter");
      if (!sel) return;

      const set = new Set();
      allItems.forEach(it => {
        if (it.class) set.add(it.class);
      });

      const classes = Array.from(set).sort();
      let html = '<option value="all">全部</option>';
      classes.forEach(c => {
        html += `<option value="${c}">${c}</option>`;
      });
      sel.innerHTML = html;
    }

    function renderAll() {
      const searchValue = document.getElementById("searchInput").value.trim().toLowerCase();
      const statusFilter = document.getElementById("statusFilter").value;
      const classFilter  = document.getElementById("classFilter").value;
      const lateOnlyEl   = document.getElementById("lateOnly");
      const lateOnly     = lateOnlyEl && lateOnlyEl.checked;

      let items = rawItems.slice();

      // 搜尋
      if (searchValue) {
        items = items.filter(it => (it.name || "").toLowerCase().includes(searchValue));
      }

      // 狀態篩選
      if (statusFilter !== "all") {
        items = items.filter(it => it.status === statusFilter);
      }

      // 班級篩選
      if (classFilter !== "all") {
        items = items.filter(it => (it.class || "") === classFilter);
      }

      // 只顯示遲到（包含：已簽到遲到＋未簽到但已超過截止時間）
      if (lateOnly) {
        items = items.filter(it => it.is_late === true);
      }

      // 排序
      items.sort((a, b) => {
        let va = a[currentSortKey] || "";
        let vb = b[currentSortKey] || "";

        if (currentSortKey === "time_str") {
          if (!va && !vb) return 0;
          if (!va) return 1;
          if (!vb) return -1;
        }

        if (typeof va === "string") va = va.toLowerCase();
        if (typeof vb === "string") vb = vb.toLowerCase();

        if (va < vb) return currentSortDir === "asc" ? -1 : 1;
        if (va > vb) return currentSortDir === "asc" ? 1 : -1;
        return 0;
      });

      renderTableBody(items);
      renderSummary(rawItems);
      renderClassSummary(rawItems);
      renderRowCount(items);
      renderSortIndicators();
    }

    function getInitials(name) {
      if (!name) return "?";
      if (name.length <= 2) return name;
      return name.slice(-2);
    }

    function renderTableBody(items) {
      const tbody = document.getElementById("tbody");
      tbody.innerHTML = "";

      items.forEach(row => {
        const tr = document.createElement("tr");

        // 姓名＋小頭像
        const tdName = document.createElement("td");
        const nameWrap = document.createElement("div");
        nameWrap.className = "name-cell";

        const avatar = document.createElement("div");
        avatar.className = "name-avatar";
        avatar.textContent = getInitials(row.name || "");

        const nameText = document.createElement("span");
        nameText.textContent = row.name || "";

        nameWrap.appendChild(avatar);
        nameWrap.appendChild(nameText);
        tdName.appendChild(nameWrap);
        tr.appendChild(tdName);

        // 班級
        const tdClass = document.createElement("td");
        tdClass.textContent = row.class || "未設定";
        tr.appendChild(tdClass);

        // 狀態
        const tdStatus = document.createElement("td");
        const statusSpan = document.createElement("span");
        if (row.status === "checked_in") {
          statusSpan.className = "status-pill status-ok";
          statusSpan.innerHTML = '<span class="status-dot-ok"></span><span>已簽到</span>';
        } else {
          statusSpan.className = "status-pill status-miss";
          statusSpan.innerHTML = '<span class="status-dot-miss"></span><span>未簽到</span>';
        }
        tdStatus.appendChild(statusSpan);
        tr.appendChild(tdStatus);

        // 時間
        const tdTime = document.createElement("td");
        tdTime.className = "time-cell";
        if (row.time_str) {
          tdTime.textContent = row.time_str;
        } else {
          tdTime.textContent = "尚未簽到";
          tdTime.classList.add("empty");
        }
        tr.appendChild(tdTime);

        // 是否遲到
        const tdLate = document.createElement("td");
        if (row.status === "checked_in") {
          const lateSpan = document.createElement("span");
          if (row.is_late === true) {
            lateSpan.className = "status-pill status-miss";
            lateSpan.innerHTML = '<span class="status-dot-miss"></span><span>遲到</span>';
          } else {
            lateSpan.className = "status-pill status-ok";
            lateSpan.innerHTML = '<span class="status-dot-ok"></span><span>準時</span>';
          }
          tdLate.appendChild(lateSpan);
        } else {
          if (row.is_late === true) {
            const lateSpan = document.createElement("span");
            lateSpan.className = "status-pill status-miss";
            lateSpan.innerHTML = '<span class="status-dot-miss"></span><span>遲到（未簽到）</span>';
            tdLate.appendChild(lateSpan);
          } else {
            tdLate.textContent = "—";
            tdLate.className = "time-cell empty";
          }
        }
        tr.appendChild(tdLate);

        tbody.appendChild(tr);
      });
    }

        function renderClassSummary(allItems) {
      const container = document.getElementById("classSummary");
      if (!container) return;

      // 依班級彙總
      const classMap = new Map();
      allItems.forEach(it => {
        const cls = it.class || "未設定";
        if (!classMap.has(cls)) {
          classMap.set(cls, {
            className: cls,
            total: 0,
            checked: 0,
            late: 0,
            absent: 0,
          });
        }
        const rec = classMap.get(cls);
        rec.total += 1;

        if (it.status === "checked_in") {
          rec.checked += 1;
          if (it.is_late === true) {
            rec.late += 1;
          }
        } else {
          rec.absent += 1;
          if (it.is_late === true) {
            rec.late += 1;
          }
        }
      });

      const rows = Array.from(classMap.values()).sort((a, b) => {
        // 班級排序：未設定放最後
        if (a.className === "未設定") return 1;
        if (b.className === "未設定") return -1;
        return a.className.localeCompare(b.className, "zh-Hant");
      });

      let tableHtml = `
        <div class="class-summary-title">
          <div>班級統計</div>
          <span>點選某一班級，可快速套用下方列表的「班級篩選」。</span>
        </div>
        <table class="class-summary-table">
          <thead>
            <tr>
              <th>班級</th>
              <th>總人數</th>
              <th>已簽到</th>
              <th>遲到</th>
              <th>未簽到</th>
              <th>出勤率</th>
            </tr>
          </thead>
          <tbody>
      `;

      rows.forEach(rec => {
        const rate = rec.total === 0 ? 0 : Math.round(rec.checked * 1000 / rec.total) / 10;
        const rateClass = rate >= 90 ? "class-rate-good" : "class-rate-bad";
        tableHtml += `
          <tr data-class="${rec.className}">
            <td class="class-name-cell">${rec.className}</td>
            <td>${rec.total}</td>
            <td>${rec.checked}</td>
            <td>${rec.late}</td>
            <td>${rec.absent}</td>
            <td class="${rateClass}">${rate}%</td>
          </tr>
        `;
      });

      tableHtml += `
          </tbody>
        </table>
      `;
      container.innerHTML = tableHtml;

      // 點某一列 → 切換班級篩選
      const rowsEl = container.querySelectorAll("tbody tr[data-class]");
      rowsEl.forEach(tr => {
        tr.addEventListener("click", () => {
          const cls = tr.getAttribute("data-class");
          const sel = document.getElementById("classFilter");
          if (!sel) return;
          if (cls === "未設定") {
            sel.value = "all";
          } else {
            sel.value = cls;
          }
          renderAll();
        });
      });
    }

    function renderRowCount(items) {
      const label = document.getElementById("rowCountLabel");
      label.textContent = `顯示筆數：${items.length}`;
    }

    function renderSortIndicators() {
      const keys = ["name", "class", "status", "time_str"];
      keys.forEach(k => {
        const span = document.getElementById("sortIndicator-" + k);
        if (!span) return;
        if (k === currentSortKey) {
          span.textContent = currentSortDir === "asc" ? "▲" : "▼";
        } else {
          span.textContent = "";
        }
      });
    }

    function setupEvents() {
      document.getElementById("searchInput").addEventListener("input", () => {
        renderAll();
      });

      document.getElementById("statusFilter").addEventListener("change", () => {
        renderAll();
      });

      document.getElementById("classFilter").addEventListener("change", () => {
        renderAll();
      });

      document.getElementById("lateOnly").addEventListener("change", () => {
        renderAll();
      });

      const refreshBtn = document.getElementById("refreshBtn");
      refreshBtn.addEventListener("click", () => {
        fetchAttendance();
      });

      const ths = document.querySelectorAll("th.sortable");
      ths.forEach(th => {
        th.addEventListener("click", () => {
          const key = th.getAttribute("data-key");
          if (key === currentSortKey) {
            currentSortDir = (currentSortDir === "asc") ? "desc" : "asc";
          } else {
            currentSortKey = key;
            currentSortDir = "asc";
          }
          renderAll();
        });
      });
    }

    setupEvents();
    fetchAttendance();
  </script>
</body>
</html>
"""
    html = html.replace("__TODAY__", today_str).replace("__CUTOFF__", cutoff)
    return html

# ----- 請假 JSON 列表 -----
@app.route("/leave/list.json", methods=["GET"])
def leave_list_json():
    data = load_leaves()
    return jsonify(data)

# ----- 請假管理網頁（美化＋篩選） -----
@app.route("/leave", methods=["GET"])
def leave_page():
    data = load_leaves()
    reqs = data.get("requests", [])
    # 依建立時間新到舊
    reqs = sorted(reqs, key=lambda r: r.get("created_at", ""), reverse=True)

    total = len(reqs)
    pending_count  = sum(1 for r in reqs if r.get("status") == "pending")
    approved_count = sum(1 for r in reqs if r.get("status") == "approved")
    rejected_count = sum(1 for r in reqs if r.get("status") == "rejected")

    rows_html = ""
    for r in reqs:
        rid   = r.get("id", "")
        name  = r.get("name", "")
        phone = r.get("phone", "")
        reason = r.get("reason", "")
        ltime  = r.get("leave_time", "")
        status = r.get("status", "pending")
        comment = r.get("review_comment", "")
        created_at = r.get("created_at", "")
        reviewed_at = r.get("reviewed_at", "")

        if status == "pending":
            status_text = "待審核"
            status_class = "status-pending"
        elif status == "approved":
            status_text = "已批准"
            status_class = "status-approved"
        else:
            status_text = "已駁回"
            status_class = "status-rejected"

        if status == "pending":
            action_html = f"""
              <div class="action-btns">
                <form method="post" action="/leave/approve" class="inline-form">
                  <input type="hidden" name="id" value="{rid}">
                  <button type="submit" class="btn btn-approve">批准</button>
                </form>
                <form method="post" action="/leave/reject" class="inline-form">
                  <input type="hidden" name="id" value="{rid}">
                  <input type="text" name="comment" placeholder="駁回原因（選填）" class="comment-input">
                  <button type="submit" class="btn btn-reject">駁回</button>
                </form>
              </div>
            """
        else:
            extra = ""
            if status == "rejected" and comment:
                extra = f"（原因：{comment}）"
            if reviewed_at:
                extra += f"<br><span class='review-time'>審核時間：{reviewed_at}</span>"
            action_html = f"<span class='status-label {status_class}'>{status_text}</span><div class='status-extra'>{extra}</div>"

        rows_html += f"""
        <tr data-status="{status}" data-name="{name}">
          <td>{rid}</td>
          <td>{name}</td>
          <td>{phone}</td>
          <td>{reason}</td>
          <td>{ltime}</td>
          <td>{created_at}</td>
          <td><span class="status-chip {status_class}">{status_text}</span></td>
          <td>{action_html}</td>
        </tr>
        """

    html = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>請假審核</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      background:#f3f4f6;
      padding:20px;
      margin:0;
    }}
    h1 {{
      margin-top:0;
      margin-bottom:8px;
    }}
    .sub {{
      margin:0 0 16px;
      font-size:13px;
      color:#6b7280;
    }}
    .summary-cards {{
      display:grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap:10px;
      margin-bottom:16px;
    }}
    .summary-card {{
      background:white;
      border-radius:14px;
      padding:10px 12px;
      border:1px solid #e5e7eb;
    }}
    .summary-label {{
      font-size:11px;
      color:#6b7280;
      margin-bottom:4px;
    }}
    .summary-value {{
      font-size:18px;
      font-weight:700;
    }}
    .summary-value.pending {{ color:#b45309; }}
    .summary-value.approved {{ color:#15803d; }}
    .summary-value.rejected {{ color:#b91c1c; }}

    .toolbar {{
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      align-items:center;
      margin-bottom:10px;
    }}
    .toolbar .field {{
      font-size:13px;
      color:#4b5563;
      display:flex;
      align-items:center;
      gap:6px;
    }}
    .toolbar input[type="text"],
    .toolbar select {{
      padding:5px 8px;
      border-radius:8px;
      border:1px solid #d1d5db;
      font-size:13px;
      background:white;
      outline:none;
    }}
    .toolbar input[type="text"]:focus,
    .toolbar select:focus {{
      border-color:#4f46e5;
      box-shadow:0 0 0 1px rgba(79,70,229,0.4);
    }}

    .table-wrap {{
      border-radius:14px;
      overflow:hidden;
      border:1px solid #e5e7eb;
      background:white;
    }}

    table {{
      border-collapse:collapse;
      width:100%;
      background:white;
      font-size:13px;
    }}
    th, td {{
      border-bottom:1px solid #e5e7eb;
      padding:6px 8px;
      font-size:13px;
      vertical-align:top;
    }}
    th {{
      background:#f9fafb;
      white-space:nowrap;
    }}
    tr:nth-child(even) td {{
      background:#f9fafb;
    }}
    tr:hover td {{
      background:#eef2ff;
    }}

    .status-chip {{
      display:inline-block;
      padding:2px 8px;
      border-radius:999px;
      font-size:11px;
      font-weight:600;
    }}
    .status-pending {{
      background:#fef3c7;
      color:#92400e;
    }}
    .status-approved {{
      background:#dcfce7;
      color:#166534;
    }}
    .status-rejected {{
      background:#fee2e2;
      color:#b91c1c;
    }}
    .action-btns {{
      display:flex;
      flex-wrap:wrap;
      gap:4px;
      align-items:center;
    }}
    .inline-form {{
      display:inline-flex;
      align-items:center;
      gap:4px;
      margin:0;
    }}
    .btn {{
      padding:4px 8px;
      border-radius:999px;
      border:none;
      font-size:12px;
      cursor:pointer;
    }}
    .btn-approve {{
      background:#16a34a;
      color:white;
    }}
    .btn-approve:hover {{
      background:#15803d;
    }}
    .btn-reject {{
      background:#ef4444;
      color:white;
    }}
    .btn-reject:hover {{
      background:#b91c1c;
    }}
    .comment-input {{
      border-radius:8px;
      border:1px solid #d1d5db;
      padding:3px 6px;
      font-size:12px;
      min-width:140px;
    }}
    .comment-input:focus {{
      border-color:#4f46e5;
      outline:none;
      box-shadow:0 0 0 1px rgba(79,70,229,0.4);
    }}
    .status-label {{
      font-size:13px;
    }}
    .status-extra {{
      font-size:11px;
      color:#6b7280;
      margin-top:2px;
    }}
    .review-time {{
      font-size:11px;
      color:#9ca3af;
    }}

    @media (max-width:768px) {{
      .summary-cards {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      th, td {{
        font-size:12px;
      }}
    }}
    @media (max-width:480px) {{
      .summary-cards {{
        grid-template-columns: repeat(1, minmax(0, 1fr));
      }}
      .toolbar {{
        flex-direction:column;
        align-items:flex-start;
      }}
      .comment-input {{
        min-width:0;
        width:100%;
      }}
    }}
  </style>
</head>
<body>
  <h1>請假審核</h1>
  <p class="sub">下方可依姓名與狀態篩選，對「待審核」案件進行批准或駁回，結果會以 LINE 通知當事人。</p>

  <div class="summary-cards">
    <div class="summary-card">
      <div class="summary-label">總申請數</div>
      <div class="summary-value">{total}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">待審核</div>
      <div class="summary-value pending">{pending_count}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">已批准</div>
      <div class="summary-value approved">{approved_count}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">已駁回</div>
      <div class="summary-value rejected">{rejected_count}</div>
    </div>
  </div>

  <div class="toolbar">
    <div class="field">
      <span>搜尋姓名：</span>
      <input type="text" id="searchName" placeholder="輸入姓名關鍵字">
    </div>
    <div class="field">
      <span>狀態篩選：</span>
      <select id="statusFilter">
        <option value="all">全部</option>
        <option value="pending">只看待審核</option>
        <option value="approved">只看已批准</option>
        <option value="rejected">只看已駁回</option>
      </select>
    </div>
  </div>

  <div class="table-wrap">
    <table id="leaveTable">
      <thead>
        <tr>
          <th>編號</th>
          <th>姓名</th>
          <th>手機號碼</th>
          <th>事由</th>
          <th>請假時間</th>
          <th>申請時間</th>
          <th>狀態</th>
          <th>操作 / 結果</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <script>
    function applyFilters() {{
      const nameInput = document.getElementById("searchName").value.trim().toLowerCase();
      const statusFilter = document.getElementById("statusFilter").value;
      const rows = document.querySelectorAll("#leaveTable tbody tr");

      rows.forEach(tr => {{
        const rowStatus = tr.getAttribute("data-status");
        const rowName = (tr.getAttribute("data-name") || "").toLowerCase();

        let ok = true;

        if (nameInput && !rowName.includes(nameInput)) {{
          ok = false;
        }}

        if (statusFilter !== "all" && rowStatus !== statusFilter) {{
          ok = false;
        }}

        tr.style.display = ok ? "" : "none";
      }});
    }}

    document.getElementById("searchName").addEventListener("input", applyFilters);
    document.getElementById("statusFilter").addEventListener("change", applyFilters);
  </script>
</body>
</html>
"""
    return html

# ----- 請假批准 -----
@app.route("/leave/approve", methods=["POST"])
def leave_approve():
    req_id = (request.form.get("id") or "").strip()
    req = update_leave_status(req_id, "approved")
    if req and req.get("user_id"):
        msg = "請假結果：批准"
        push_async(req["user_id"], msg)
    return redirect(request.referrer or "/leave")

# ----- 請假駁回 -----
@app.route("/leave/reject", methods=["POST"])
def leave_reject():
    req_id = (request.form.get("id") or "").strip()
    comment = (request.form.get("comment") or "").strip()
    req = update_leave_status(req_id, "rejected", comment)
    if req and req.get("user_id"):
        if comment:
            msg = f"請假結果：駁回\n原因：{comment}"
        else:
            msg = "請假結果：駁回"
        push_async(req["user_id"], msg)
    return redirect(request.referrer or "/leave")

# ============================================================
# 每日遲到掃描：超過時間還沒簽到的人 → 推遲到通知
# ============================================================
def _late_scan_worker():
    print(f"[LATE-SCAN] 背景遲到掃描啟動，每天 {LATE_CUTOFF.strftime('%H:%M')} 執行一次")
    while True:
        now = datetime.datetime.now()
        today = now.date()
        target_dt = datetime.datetime.combine(today, LATE_CUTOFF)
        if now >= target_dt:
            target_dt = target_dt + datetime.timedelta(days=1)

        while True:
            now = datetime.datetime.now()
            delta = (target_dt - now).total_seconds()
            if delta <= 0:
                break
            time.sleep(min(delta, 60))

        try:
            now = datetime.datetime.now()
            today = now.date()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

            global NAME_TO_UID
            NAME_TO_UID = _load_members_map()

            print(f"[LATE-SCAN] {ts_str} 開始掃描遲到名單...")

            for name, uid in NAME_TO_UID.items():
                if not uid:
                    continue

                if _is_today(last_checkin_time[name]):
                    continue

                if late_notice_sent_on.get(name) == today:
                    continue

                text = line_text_late(name, now)
                push_async(uid, text)
                late_notice_sent_on[name] = today
                print(f"[LATE-SCAN] 推播遲到通知給：{name} ({uid})")

        except Exception as e:
            print("[LATE-SCAN][ERR]", e)

threading.Thread(target=_late_scan_worker, daemon=True).start()

# ============================================================
# LINE 訊息處理（綁定 / 班級 / 請假）
# ============================================================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    user_id = getattr(event.source, "user_id", None)
    text = (event.message.text or "").strip()
    reply_text = None
    print(f"[EVENT] userId={user_id}, text={text}")

    # 1. 查詢目前綁定
    if text == "查詢":
        users = load_users()
        rec = users.get("_by_user_id", {}).get(user_id, {})
        name = rec.get("name")
        cls  = rec.get("class")
        if name and cls:
            reply_text = f"目前已綁定：{name}（{cls}）✅"
        elif name:
            reply_text = f"目前已綁定：{name} ✅\n若要設定班級：班級 你的班級"
        else:
            reply_text = "尚未綁定，請傳：綁定 你的名字"

    # 2. 綁定姓名
    elif text.startswith("綁定"):
        name = text.replace("綁定", "", 1).strip()
        status, old = bind_user(user_id, name)

        if status == "invalid":
            reply_text = "❌ 請用：綁定 你的名字\n例如：綁定 郭昶邑"
        elif status == "new":
            reply_text = f"已綁定：{name} ✅"
        elif status == "same":
            reply_text = f"你已經綁定為「{name}」囉 ✅"
        elif status == "update":
            reply_text = f"已更新綁定：由「{old}」改為「{name}」 ✅"
        else:
            reply_text = "綁定時發生錯誤，請再試一次。"

    # 3. 設定班級
    elif text.startswith("班級"):
        cls = text.replace("班級", "", 1).strip()
        if not cls:
            reply_text = "❌ 請用：班級 你的班級\n例如：班級 一年甲班"
        else:
            data = load_users()
            by_uid = data.setdefault("_by_user_id", {})
            rec = by_uid.setdefault(user_id, {})
            name = rec.get("name")
            rec["class"] = cls
            save_users(data)
            if name:
                reply_text = f"已為 {name} 設定班級：{cls} ✅"
            else:
                reply_text = f"已設定班級：{cls} ✅\n（建議再用：綁定 你的名字）"

    # 4. 請假：先給填寫教學
    elif text == "請假":
        reply_text = (
            "請輸入請假資訊（用逗號分隔）：\n"
            "格式：\n"
            "請假填寫 姓名,手機號碼,事由,請假時間\n\n"
            "例如：\n"
            "請假填寫 郭小明,0912345678,家庭因素,2025-11-25 08:00-12:00"
        )

    # 5. 接收請假內容
    elif text.startswith("請假填寫"):
        payload = text.replace("請假填寫", "", 1).strip()
        parts = [p.strip() for p in payload.split(",") if p.strip()]
        if len(parts) != 4:
            reply_text = (
                "❌ 格式錯誤，請用：\n"
                "請假填寫 姓名,手機號碼,事由,請假時間\n\n"
                "例如：\n"
                "請假填寫 郭小明,0912345678,家庭因素,2025-11-25 08:00-12:00"
            )
        else:
            name, phone, reason, leave_time = parts
            add_leave_request(user_id, name, phone, reason, leave_time)
            reply_text = (
                "已發送請假需求 ✅\n"
                f"姓名：{name}\n"
                f"手機：{phone}\n"
                f"事由：{reason}\n"
                f"時間：{leave_time}\n"
                "審核結果將以 LINE 通知你。"
            )

    # 6. 其他訊息
    else:
        users = load_users()
        rec = users.get("_by_user_id", {}).get(user_id, {})
        name = rec.get("name")
        if not name:
            reply_text = "您好，需要綁定請輸入：綁定 你的名字\n若要請假請輸入：請假"
        else:
            reply_text = "目前支援指令：\n1. 查詢\n2. 綁定 你的名字\n3. 班級 你的班級\n4. 請假"

    # === 回覆訊息 ===
    try:
        with ApiClient(config) as api_client:
            api = MessagingApi(api_client)
            try:
                api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply_text)],
                    )
                )
                print(f"[LINE] Reply 成功 → {user_id}: {reply_text}")
            except ApiException as e:
                print("[LINE][ERROR][reply] status=", getattr(e,"status",None),
                      "body=", getattr(e,"body",None))
                if user_id:
                    try:
                        api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[TextMessage(text=f"(fallback) {reply_text}")],
                            )
                        )
                        print(f"[LINE] Push 成功 → {user_id}: {reply_text}")
                    except ApiException as e2:
                        print("[LINE][ERROR][push] status=", getattr(e2,"status",None),
                              "body=", getattr(e2,"body",None))
    except Exception as e:
        print("[LINE][ERROR] 外層錯誤:", e)

# ============================================================
# 進入點
# ============================================================
if __name__ == "__main__":
    public_url = start_ngrok_if_needed(local_host="127.0.0.1", port=PORT, webhook_path="/webhook")
    if public_url:
        print("[提示] 到 LINE Developers 貼上：", f"{public_url}/webhook")
        print("      並確保 Use webhook = ON，再按 Verify。")
    print(f"[FLASK] http://127.0.0.1:{PORT}  /  http://{HOST}:{PORT}")
    print(" 辨識端點      ：http://0.0.0.0:%d/recognize" % PORT)
    print(" 今日出勤 JSON ：http://0.0.0.0:%d/attendance/today.json" % PORT)
    print(" 今日出勤網頁 ：http://0.0.0.0:%d/attendance/today" % PORT)
    print(" 請假管理網頁 ：http://0.0.0.0:%d/leave" % PORT)
    app.run(host=HOST, port=PORT, debug=False)
