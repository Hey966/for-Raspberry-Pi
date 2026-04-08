# -*- coding: utf-8 -*-
"""
server.py
結合：
- LINE Bot Webhook + 綁定/班級指令 + 請假系統
- 人臉辨識簽到伺服器 + 出勤網頁（含分班管理＋遲到顯示＋班級遲到名單）
"""

import os, json, atexit, subprocess, time, requests, shutil, queue, re
from pathlib import Path
from urllib.parse import urljoin
from collections import defaultdict

from flask import Flask, request, abort, jsonify, redirect, render_template  # ★ 多了 render_template

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

        # 名字不同 → 更新
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
    依辨識名稱或顯示名稱查班級
    """
    name = _clean(name)
    data = load_users()

    by_name = data.get("_by_name", {})
    by_uid = data.get("_by_user_id", {})

    # 1) 用顯示名稱找有 LINE 的人
    uid = by_name.get(name)
    if uid:
        rec = by_uid.get(uid, {})
        cls = _clean(rec.get("class", ""))
        if cls:
            return cls

    # 2) 用 face_name 找有 LINE 的人
    for uid, rec in (by_uid or {}).items():
        candidate = _clean(rec.get("face_name", rec.get("name", "")))
        if candidate == name:
            return _clean(rec.get("class", ""))

    # 3) 用顯示名稱找 _web_users
    web_users = data.get("_web_users", {})
    rec = web_users.get(name, {})
    cls = _clean(rec.get("class", ""))
    if cls:
        return cls

    # 4) 用 face_name 找 _web_users
    for display_name, rec in (web_users or {}).items():
        candidate = _clean(rec.get("face_name", rec.get("name", "")))
        if candidate == name:
            return _clean(rec.get("class", ""))

    return ""

def get_order_for_name(name: str) -> int:
    """
    依辨識名稱或顯示名稱查排序欄位 order
    找不到就回傳很大的數字，排最後
    """
    name = _clean(name)
    data = load_users()

    by_name = data.get("_by_name", {})
    by_uid = data.get("_by_user_id", {})

    # 1) 用顯示名稱找有 LINE 的人
    uid = by_name.get(name)
    if uid:
        rec = by_uid.get(uid, {})
        return int(rec.get("order", 999999))

    # 2) 用 face_name 找有 LINE 的人
    for uid, rec in (by_uid or {}).items():
        candidate = _clean(rec.get("face_name", rec.get("name", "")))
        if candidate == name:
            return int(rec.get("order", 999999))

    # 3) 用顯示名稱找 _web_users
    web_users = data.get("_web_users", {})
    rec = web_users.get(name)
    if rec:
        return int(rec.get("order", 999999))

    # 4) 用 face_name 找 _web_users
    for display_name, rec in (web_users or {}).items():
        candidate = _clean(rec.get("face_name", rec.get("name", "")))
        if candidate == name:
            return int(rec.get("order", 999999))

    return 999999


def get_display_name(face_name: str) -> str:
    """
    依辨識名稱(face_name)找正式顯示名稱(name)
    找不到就直接回傳 face_name
    """
    face_name = _clean(face_name)
    data = load_users()

    # 先找有 LINE 的人
    for uid, rec in (data.get("_by_user_id") or {}).items():
        candidate = _clean(rec.get("face_name", rec.get("name", "")))
        if candidate == face_name:
            return _clean(rec.get("name", face_name))

    # 再找 _web_users
    for display_name, rec in (data.get("_web_users") or {}).items():
        candidate = _clean(rec.get("face_name", rec.get("name", "")))
        if candidate == face_name:
            return _clean(rec.get("name", display_name))

    return face_name

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

    # ★ 根據姓名查 users.json 的班級（沒有就空字串）
    cls = get_class_for_name(name)

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
        # ★ 新增：當下的班級一起存進請假資料
        "class": cls,
    }
    reqs.append(req)
    save_leaves(data)
    print(f"[LEAVE] 新請假申請 id={req_id} name={name} class={cls}")
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

def _extract_date_from_leave_time(leave_time: str):
    """
    嘗試從 leave_time 字串中抓出一個 YYYY-MM-DD 當成請假日期：
    - 先試前 10 個字元
    - 再用 regex 找第一個 yyyy-mm-dd
    都失敗就回傳 None
    """
    if not leave_time:
        return None
    s = leave_time.strip()

    # 1) 先試前 10 個字
    if len(s) >= 10:
        candidate = s[:10]
        try:
            return datetime.date.fromisoformat(candidate)
        except Exception:
            pass

    # 2) 用正則找第一個 yyyy-mm-dd
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(0))
        except Exception:
            pass

    return None

def has_approved_leave_today(user_id, name, date_obj: datetime.date) -> bool:
    """
    檢查此 user 今天是否有「已批准」請假：
    - 優先用 user_id 比對
    - 退而求其次用姓名比對
    - leave_time 預期格式：'YYYY-MM-DD 08:00-12:00' 之類
      → 只認第一個 YYYY-MM-DD
    """
    data = load_leaves()
    clean_name = _clean(name)

    for r in data.get("requests", []):
        if r.get("status") != "approved":
            continue

        # 比對 user 身分
        if r.get("user_id") != user_id and _clean(r.get("name", "")) != clean_name:
            continue

        lt = (r.get("leave_time") or "").strip()
        if not lt:
            continue

        d = _extract_date_from_leave_time(lt)
        if d is None:
            continue

        if d == date_obj:
            return True

    return False

# ============================================================
# Flask / LINE 基本設定
# ============================================================
# 預設改成 5000，比較符合你想用 http://192.168.0.xx:5000 的習慣
PORT = int(os.environ.get("PORT", 5000))
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
    """
    推播背景執行緒：
    - 如果 payload 是 str → 一則訊息
    - 如果 payload 是 list/tuple → 多則訊息，彼此間隔 1 秒
    """
    print("[PUSH] 背景推播執行緒啟動")
    with ApiClient(config) as api_client:
        api = MessagingApi(api_client)
        while True:
            try:
                uid, payload = push_q.get()
                try:
                    # 情況 A：單一字串（一般簽到、請假結果、測試推播）
                    if isinstance(payload, str):
                        api.push_message(
                            PushMessageRequest(
                                to=uid,
                                messages=[TextMessage(text=payload)]
                            )
                        )
                        print(f"[PUSH] to={uid} OK (single)")

                    # 情況 B：多則訊息 list / tuple（例如遲到通知三段）
                    elif isinstance(payload, (list, tuple)):
                        for i, txt in enumerate(payload):
                            api.push_message(
                                PushMessageRequest(
                                    to=uid,
                                    messages=[TextMessage(text=str(txt))]
                                )
                            )
                            print(f"[PUSH] to={uid} OK part {i+1}/{len(payload)}")
                            if i + 1 < len(payload):
                                time.sleep(1.0)  # 每則間隔 1 秒

                    else:
                        print("[PUSH][WARN] 未知 payload 型別：", type(payload))

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

    # 讀取 users.json (LINE 綁定使用者 + _web_users)
    if USERS_JSON.exists():
        try:
            d = json.loads(USERS_JSON.read_text("utf-8"))

            # 1) LINE 綁定使用者
            for k, v in (d.get("_by_name") or {}).items():
                if isinstance(v, str) and (v.startswith("U") or v.startswith("C")):
                    mapping[_clean(k)] = _clean(v)

            # 2) 只顯示在網頁、沒有 LINE 的使用者
            for display_name, rec in (d.get("_web_users") or {}).items():
                face_name = _clean(rec.get("face_name", display_name))
                mapping.setdefault(face_name, None)

        except Exception as e:
            print("[USERS_JSON][ERR]", e)

    # 讀取 members.json (舊版沒有 LINE 的學生，保留相容)
    if MEMBERS_JSON.exists():
        try:
            d = json.loads(MEMBERS_JSON.read_text("utf-8-sig"))
            for k, v in d.items():
                mapping.setdefault(_clean(k), None)
        except Exception as e:
            print("[MEMBERS_JSON][ERR]", e)

    print(f"[INFO] 名單載入完成，總共 {len(mapping)} 筆")
    return mapping

NAME_TO_UID = _load_members_map()

# 簽到紀錄（今天）
last_checkin_time   = defaultdict(lambda: datetime.datetime.min)  # name -> datetime
# 記錄「最後一次遲到通知時間」，用來做每 5 分鐘重發
late_notice_last_ts = {}  # name -> datetime

def _is_today(dt):
    return (dt.date() == datetime.date.today())

def line_text_checkin(name, ts):
    return f"✅ 簽到成功\n姓名：{name}\n時間：{ts:%Y-%m-%d %H:%M:%S}"

def line_text_late(name, ts):
    """
    遲到通知 → 回傳三段文字：
    1) ⚠️ 遲到通知（截止 HH:MM）
    2) 姓名：xxx
    3) 時間：YYYY-MM-DD HH:MM:SS
    """
    cutoff_str = f"{LATE_CUTOFF.hour:02d}:{LATE_CUTOFF.minute:02d}"
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    return [
        
        f"⚠️ 遲到通知（截止 {cutoff_str}）",
        f"姓名：{name}",
        f"時間：{ts_str}",
    ]

# 今天出勤統計
def get_today_attendance():
    """
    回傳今天每個學生的出勤狀態：
      status:
        - "checked_in"  已簽到
        - "not_checked" 尚未簽到 / 未出現
        - "on_leave"    已批准請假
      is_late:
        - 只對「已簽到的人」有意義；請假一律 False
      has_leave:
        - True 表示今天有批准請假
    """
    rows = []
    now = datetime.datetime.now()
    now_time = now.time()
    today = now.date()

    # 重新載入名單，避免 users.json / members.json 變更沒重啟
    global NAME_TO_UID
    NAME_TO_UID = _load_members_map()

    for name in NAME_TO_UID.keys():
        uid = NAME_TO_UID.get(name)
        ts = last_checkin_time[name]

        # 先看今天是否有批准請假
        has_leave = False
        if uid:
            try:
                has_leave = has_approved_leave_today(uid, name, today)
            except Exception as e:
                print("[ATTEND][LEAVE][ERR]", name, e)

        if has_leave:
            status = "on_leave"
            time_str = ""
            is_late = False
        else:
            if _is_today(ts):
                status = "checked_in"
                time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                # 已簽到才看遲到（> 截止時間才算遲到）
                is_late = (ts.time() > LATE_CUTOFF)
            else:
                status = "not_checked"
                time_str = ""
                # 未到但已超過截止時間 → 視為遲到（未簽到）
                is_late = (now_time > LATE_CUTOFF)

        cls = get_class_for_name(name)

        rows.append({
            "name": get_display_name(name),
            "status": status,
            "time_str": time_str,
            "is_late": is_late,
            "class": cls,
            "has_leave": has_leave,
            "order": get_order_for_name(name)
        })

    rows.sort(key=lambda r: (r.get("order", 999999), r.get("name", "")))
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
# 簽到：改成「丟到背景執行」
# ============================================================
def handle_checkin_in_background(name, now):
    """
    把「今天第一次簽到 → 推播 + beep」丟到背景 thread 做，
    避免未來 TTS 改成真人語音時卡住 HTTP 回應。
    """
    if not name or name == "Unknown":
        return

    try:
        # 判斷今天是否已經簽到過
        first_today = not _is_today(last_checkin_time[name])
    except KeyError:
        # dict 裡沒有這個名字 → 當作今天第一次
        first_today = True

    uid = NAME_TO_UID.get(_clean(name), "")

    if first_today:
        try:
            # 先更新本機簽到時間（有沒有 LINE 都要記錄）
            last_checkin_time[name] = now

            # 有 LINE 才推播
            if uid:
                push_async(uid, line_text_checkin(name, now))

            # 簽到成功 beep 一聲
            tts("簽到成功")
        except Exception as e:
            print("[CHECKIN][ERR] 背景簽到處理失敗:", e)


# ============================================================
# Flask 路由
# ============================================================

# ----- 首頁 -----
@app.route("/", methods=["GET"])
def index_page():
    # 今天日期 & 出勤截止時間
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    cutoff = LATE_CUTOFF.strftime("%H:%M")

    # --- 請假統計 ---
    data = load_leaves()
    reqs = data.get("requests", [])
    leave_total   = len(reqs)
    leave_pending = sum(1 for r in reqs if r.get("status") == "pending")
    leave_approved = sum(1 for r in reqs if r.get("status") == "approved")
    leave_rejected = sum(1 for r in reqs if r.get("status") == "rejected")

    # --- 出勤統計 ---
    rows = get_today_attendance()
    att_total   = len(rows)
    att_checked = sum(1 for r in rows if r.get("status") == "checked_in")
    att_late    = sum(1 for r in rows if r.get("is_late"))

    # ★ 把 index.html 換成 home.html
    return render_template(
        "home.html",
        today_str=today_str,
        cutoff=cutoff,
        leave_total=leave_total,
        leave_pending=leave_pending,
        leave_approved=leave_approved,
        leave_rejected=leave_rejected,
        att_total=att_total,
        att_checked=att_checked,
        att_late=att_late,
    )


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

# ----- 人臉辨識簽到（多人版 + 向下相容） -----
@app.route("/recognize", methods=["POST"])
def recognize():
    """
    樹莓派每次丟一張圖：
    - 找出所有人臉
    - 每張臉都比對名字
    - 回傳 faces: [{name, sim, bbox=[x,y,w,h]}, ...]
    - 順便附上一個 main_name / main_sim（最大臉），給舊版 client 用
    - 對每個 name != Unknown，在背景 thread 做「今天第一次簽到」推播
    """
    start = time.time()

    file = request.files.get("image")
    if not file:
        print("[DEBUG] /recognize 沒收到 image 檔")
        return jsonify({"faces": [], "name": "Unknown", "sim": None}), 200

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        print("[DEBUG] /recognize 影像解碼失敗")
        return jsonify({"faces": [], "name": "Unknown", "sim": None}), 200

    h, w, _ = img.shape
    print(f"[DEBUG] 收到影像尺寸: {w}x{h}")

    faces = face_app.get(img)
    print(f"[DEBUG] InsightFace 偵測到臉數量: {len(faces)}")

    if not faces:
        cost = time.time() - start
        print("[DEBUG] 無人臉, cost = {:.3f}s".format(cost))
        return jsonify({"faces": [], "name": "Unknown", "sim": None}), 200

    now = datetime.datetime.now()
    result_faces = []

    # 先決定哪一張當「主臉」（面積最大）
    def area(f):
        l, t, r, b = f.bbox
        return (r - l) * (b - t)

    main_f = max(faces, key=area)
    main_name = "Unknown"
    main_sim = None

    for f in faces:
        l, t, r, b = f.bbox
        x = int(l)
        y = int(t)
        w_box = int(r - l)
        h_box = int(b - t)

        enc = f.normed_embedding.astype("float32")
        enc = l2_normalize(enc)

        face_name, sim = decide_name(enc)
        display_name = get_display_name(face_name)

        print(f"[DEBUG] 最佳匹配：face_name={face_name}, display_name={display_name}, sim={sim}, bbox=({x},{y},{w_box},{h_box})")

        result_faces.append({
            "name": display_name,
            "face_name": face_name,
            "sim": float(sim) if sim is not None else None,
            "bbox": [x, y, w_box, h_box],
        })

        # 設定 main_name / main_sim
        if f is main_f:
            main_name = display_name
            main_sim = sim

        # 針對每個有名字的人 → 丟到背景做「今天第一次簽到」
        if face_name != "Unknown":
            threading.Thread(
                target=handle_checkin_in_background,
                args=(face_name, now),
                daemon=True
            ).start()

    cost = time.time() - start
    print("[DEBUG] /recognize 多人版 cost = {:.3f}s, faces = {}".format(cost, len(result_faces)))

    # ★ 關鍵：同時回傳 faces（多人）和 name/sim（主臉）給舊版 client 使用
    return jsonify({
        "faces": result_faces,
        "name": main_name,
        "sim": float(main_sim) if main_sim is not None else None
    }), 200

# ----- 今日出勤 JSON -----
@app.route("/attendance/today.json", methods=["GET"])
def attendance_today_json():
    rows = get_today_attendance()
    return jsonify({
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "items": rows
    })
    
# ===== 測試用：強制簽到 / 取消簽到（給 CMD / curl 用） =====
@app.route("/test/checkin", methods=["GET"])
def test_checkin():
    """
    測試用：強制把某個名字標記為「今天已簽到」
    用法：
      /test/checkin?name=林廣至
      /test/checkin?name=林廣至&time=2025-12-03T08:05:00
    """
    name = _clean(request.args.get("name", ""))
    if not name:
        return "缺少 ?name= 參數", 400

    ts_str = request.args.get("time", "")
    if ts_str:
        try:
            ts = datetime.datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.datetime.now()
    else:
        ts = datetime.datetime.now()

    # 把這個名字的 last_checkin_time 改成指定時間
    last_checkin_time[name] = ts
    return f"{name} 已標記為簽到，時間 {ts}", 200


@app.route("/test/reset_checkin", methods=["GET"])
def test_reset_checkin():
    """
    測試用：把某個名字重置為「今天沒簽到」
    用法：
      /test/reset_checkin?name=林廣至
    """
    name = _clean(request.args.get("name", ""))
    if not name:
        return "缺少 ?name= 參數", 400

    # 重置成 datetime.min → get_today_attendance() 會當作沒簽到
    last_checkin_time[name] = datetime.datetime.min
    # 順便把遲到通知時間清掉，避免影響遲到推播測試
    late_notice_last_ts.pop(name, None)

    return f"{name} 已重置為未簽到狀態", 200


# ----- 今日出勤管理網頁（改用 template） -----
@app.route("/attendance/today", methods=["GET"])
def attendance_today_page():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    cutoff = LATE_CUTOFF.strftime("%H:%M")
    # 對應 templates/attendance_today.html
    return render_template("attendance_today.html",
                           today_str=today_str,
                           cutoff=cutoff)

# ----- 請假 JSON 列表 -----
@app.route("/leave/list.json", methods=["GET"])
def leave_list_json():
    data = load_leaves()
    return jsonify(data)

# ----- 請假管理網頁（改用 template） -----
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

    # ★ 幫每一筆請假記錄補上 class_name 給模板使用
    for r in reqs:
        # 先用請假資料裡存的 class，沒有的話再用姓名去 users.json 查
        cls = (r.get("class") or get_class_for_name(r.get("name", "")))
        r["class_name"] = cls or ""

    # 對應 templates/leave.html
    return render_template(
        "leave.html",
        total=total,
        pending_count=pending_count,
        approved_count=approved_count,
        rejected_count=rejected_count,
        requests=reqs,
    )

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

# ----- 校務行事曆 -----
@app.route("/calendar", methods=["GET"])
def calendar_page():
    """
    校務行事曆頁面（對應 templates/calendar.html）
    支援查詢：
      /calendar?m=2025-09&q=段考
    """

    # ---- 月份選單（你之後可改成自動） ----
    month_options = [
        {"value": "2025-08", "label": "2025 年 8 月"},
        {"value": "2025-09", "label": "2025 年 9 月"},
        {"value": "2025-10", "label": "2025 年 10 月"},
        {"value": "2025-11", "label": "2025 年 11 月"},
    ]

    default_month = month_options[0]["value"]
    current_month = request.args.get("m") or default_month
    keyword = (request.args.get("q") or "").strip().lower()

    # ---- 目前用「寫死」的校務行事（之後可改讀 JSON） ----
    all_events = [
        {
            "date": "2025-09-01",
            "weekday": "一",
            "title": "開學日",
            "target": "全校",
            "note": "",
            "important": True,
            "month": "2025-09",
        },
        {
            "date": "2025-09-09",
            "weekday": "二",
            "title": "導師會議",
            "target": "導師",
            "note": "放學後於會議室召開",
            "important": False,
            "month": "2025-09",
        },
        {
            "date": "2025-09-25",
            "weekday": "四",
            "title": "第一次段考",
            "target": "全校學生",
            "note": "詳見考程表",
            "important": True,
            "month": "2025-09",
        },
    ]

    # ---- 依月份過濾 ----
    events = [e for e in all_events if e["month"] == current_month]

    # ---- 依 keyword 過濾 ----
    if keyword:
        def match(e):
            text = (
                (e.get("title") or "") +
                (e.get("note") or "") +
                (e.get("target") or "")
            ).lower()
            return keyword in text

        events = [e for e in events if match(e)]

    # ---- 排序 ----
    events.sort(key=lambda e: e["date"])

    # ---- 找月份中文名 ----
    current_month_label = ""
    for m in month_options:
        if m["value"] == current_month:
            current_month_label = m["label"]
            break

    return render_template(
        "calendar.html",
        events=events,
        month_options=month_options,
        current_month=current_month,
        current_month_label=current_month_label,
        keyword=keyword,
    )


# ============================================================
# 每日遲到掃描：
#  - 僅在「超過 LATE_CUTOFF」後才會判定遲到
#  - 有批准請假者：完全不發遲到通知
#  - 每人每隔 REMIND_MIN 分鐘重發一次通知，直到簽到或隔天
# ============================================================
REMIND_MIN = 5  # 每幾分鐘提醒一次

def _late_scan_worker():
    print(f"[LATE-SCAN] 背景遲到掃描啟動，超過 {LATE_CUTOFF.strftime('%H:%M')} 後啟動提醒，每 {REMIND_MIN} 分鐘檢查一次")

    global late_notice_last_ts

    while True:
        try:
            now = datetime.datetime.now()
            today = now.date()
            now_time = now.time()

            # 稍微降低頻率：每 60 秒掃描一次即可
            sleep_sec = 60

            # 還沒到截止時間 → 先睡一輪
            if now_time <= LATE_CUTOFF:
                time.sleep(sleep_sec)
                continue

            # 每次掃描前刷新名單
            global NAME_TO_UID
            NAME_TO_UID = _load_members_map()

            print(f"[LATE-SCAN] {now.strftime('%Y-%m-%d %H:%M:%S')} 掃描遲到名單中...")

            for name, uid in NAME_TO_UID.items():
                if not uid:
                    continue

                # 今天已簽到 → 不發遲到通知
                if _is_today(last_checkin_time[name]):
                    continue

                # 今天有批准請假 → 不發遲到通知
                if has_approved_leave_today(uid, name, today):
                    print(f"[LATE-SCAN] {name} 今日有批准請假，略過遲到通知")
                    continue

                # 查看上次通知時間（若是前一天要重置）
                last_ts = late_notice_last_ts.get(name)
                if last_ts is not None and last_ts.date() != today:
                    last_ts = None

                # 若從未通知過，或距離上次已超過 REMIND_MIN 分鐘 → 發通知
                need_notify = False
                if last_ts is None:
                    need_notify = True
                else:
                    delta_min = (now - last_ts).total_seconds() / 60.0
                    if delta_min >= REMIND_MIN:
                        need_notify = True

                if not need_notify:
                    continue

                text_list = line_text_late(name, now)  # list: [訊息1, 訊息2, 訊息3]
                push_async(uid, text_list)
                late_notice_last_ts[name] = now
                print(f"[LATE-SCAN] 推播遲到通知給：{name} ({uid})")

        except Exception as e:
            print("[LATE-SCAN][ERR]", e)

        # 最後再睡一輪
        time.sleep(60)

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
    print(f"[FLASK] 首頁           ：http://127.0.0.1:{PORT}  /  http://{HOST}:{PORT}")
    print(" 辨識端點      ：http://0.0.0.0:%d/recognize" % PORT)
    print(" 今日出勤 JSON ：http://0.0.0.0:%d/attendance/today.json" % PORT)
    print(" 今日出勤網頁 ：http://0.0.0.0:%d/attendance/today" % PORT)
    print(" 請假管理網頁 ：http://0.0.0.0:%d/leave" % PORT)
    app.run(host=HOST, port=PORT, debug=False)
