# -*- coding: utf-8 -*-
# === 環境變數要在科學套件 import 前 ===
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("ORT_NUM_THREADS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL","3")
os.environ.setdefault("OMP_THREAD_LIMIT","1")

import cv2, numpy as np, time, datetime, threading, queue, base64, subprocess, json, requests, itertools, shutil
from pathlib import Path
from collections import defaultdict

# ================= 路徑與檔案 =================
SCRIPT_DIR = Path(__file__).resolve().parent
ENC_PATH   = Path(r"C:\Users\jack0\Desktop\for Raspberry Pi\python3.13.7\encodings.npz")  # 由 build_embeddings.py 產生（多向量）
USERS_JSON = SCRIPT_DIR / "users.json"
CONFIG_DIR = Path(os.environ.get("LINE_CONFIG_DIR", SCRIPT_DIR / "checkin")); CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MEMBERS_JSON = CONFIG_DIR / "members.json"
LINE_TOKEN_TXT = CONFIG_DIR / "line_token.txt"

# ================= 顯示／影像參數（流暢） =================
DISPLAY_SCALE = 0.6
SCALE         = 0.65        # 輸入縮小倍率（偵測端降載）
DET_SIZE      = (640, 640)  # InsightFace 偵測大小（越小越快）
USE_CLAHE     = True
CLAHE_DARK_Y  = 85          # 偏暗才做 CLAHE
BLUR_VAR_MIN  = 70.0        # 太糊就跳過偵測
MIN_FACE_W    = 60          # 縮小後的最小臉寬/高（像素）

# ================= 門檻（ArcFace cosine） =================
COS_THRESHOLD          = 0.40   # 讓 sim≈0.42 穩定過線
COS_SECOND_BEST_MARGIN = 0.05   # 第一名與第二名需差距
COS_NEIGHBOR_LIMIT     = 0.33   # 備用（未使用 FAISS 時）

# ================= 流程頻率 =================
DETECT_EVERY_N   = 5        # 每 N 幀做一次「偵測+編碼」
MAX_PROC_FPS     = 12       # 最大處理 FPS（丟幀保即時）
NAME_HOLD_SEC    = 0.40     # 名稱持有（防跳名）

# ================= 追蹤器 =================
TRACKER_TYPE = "KCF"        # KCF 輕量，CSRT 穩但較慢
TRACK_LOST_TOLERANCE = 8

# ================= 通知 / 儀表板（預設關）=================
LATE_CUTOFF_STR = os.environ.get("LATE_CUTOFF","08:00")  # 單人晚到判斷（辨識到時用）
ENABLE_INGEST = os.environ.get("ENABLE_INGEST","0") not in ("0","false","False")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL","http://127.0.0.1:5000/ingest")
AUTH_TOKEN    = os.environ.get("DASHBOARD_AUTH","")
HEADERS = {"X-Auth-Token": AUTH_TOKEN} if AUTH_TOKEN else {}

# ===== 定時遲到掃描設定 =====
LATE_SCAN_ENABLED = os.environ.get("LATE_SCAN_ENABLED","1") not in ("0","false","False")
LATE_SCAN_TIME_STR = os.environ.get("LATE_SCAN_TIME", "08:00")   # 每天（或工作日）掃描的截止時間
LATE_SCAN_SCOPE = os.environ.get("LATE_SCAN_SCOPE","weekday").lower()  # "weekday" or "daily"

# ================= 語音（預設開啟；改為無阻塞 Beep） =================
SPEAK_ON_SUCCESS = os.environ.get("SPEAK_ON_SUCCESS","1") not in ("0","false","False")
VOICE_RATE = int(os.environ.get("TTS_RATE", 0))
# 若要改回 PowerShell 語音，將下行設為 False
USE_BEEP_TTS = True
last_tts = 0.0
TTS_COOLDOWN = 1.0   # 仍保留最低冷卻，避免連珠音效

# ================== PIL 中文繪字 ==================
from PIL import ImageFont, ImageDraw, Image

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",     # 微軟正黑
    r"C:\Windows\Fonts\msjhbd.ttc",   # 微軟正黑粗
    r"C:\Windows\Fonts\mingliu.ttc",  # 細明體
]
def _pick_font(size):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return None

FONT_LABEL = _pick_font(32)   # 人名
FONT_INFO  = _pick_font(22)   # 面板/輔助資訊
USE_PIL_EVERY_K = 1           # 這個不再用來切換 OpenCV，維持相容

def draw_texts_cn(img_bgr, items, frame_idx: int):
    if not items: return img_bgr
    # 回到原本邏輯：只要有 font，就全部用 PIL 畫，避免 ? 號
    use_pil = all(it.get("font") is not None for it in items if it)
    if use_pil:
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for it in items:
            if not it: continue
            text = str(it.get("text",""))
            x, y = it.get("org",(0,0))
            b,g,r = it.get("bgr",(255,255,255))
            font  = it.get("font")
            draw.text((x,y), text, font=font, fill=(r,g,b))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        for it in items:
            if not it: continue
            text = str(it.get("text",""))
            x, y = it.get("org",(0,0))
            bgr  = it.get("bgr",(255,255,255))
            cv2.putText(img_bgr, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)
        return img_bgr

# ================= 小工具：TTS/LINE/… =================
import base64, subprocess, shutil

def _powershell_path():
    for exe in ("powershell.exe", "pwsh.exe", "powershell"):
        p = shutil.which(exe)
        if p: return p
    return "powershell"

def _powershell_say(text, rate=0, volume=100):
    safe = (text or "").replace("'", "''")
    ps = f"""
Add-Type -AssemblyName System.Speech | Out-Null;
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$s.Rate = {max(-10,min(10,int(rate)))};
$s.Volume = {max(0,min(100,int(volume)))};
$s.Speak('{safe}');
"""
    enc = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    try:
        subprocess.Popen([_powershell_path(),"-NoProfile","-NonInteractive","-WindowStyle","Hidden","-EncodedCommand",enc],
                         creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    except Exception as e:
        try:
            import winsound; winsound.Beep(880, 120)
        except Exception:
            print(f"[SPEAK] PowerShell 語音失敗：{e}")

def tts(text):
    if not SPEAK_ON_SUCCESS: return
    global last_tts
    now = time.time()
    if (now - last_tts) < TTS_COOLDOWN: return
    last_tts = now
    if USE_BEEP_TTS:
        def _beep():
            try:
                import winsound
                winsound.MessageBeep(-1)
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()
    else:
        threading.Thread(target=_powershell_say, args=(text, VOICE_RATE, 100), daemon=True).start()

def _clean(s:str)->str:
    if s is None: return ""
    for z in ("\u200b","\u200c","\u200d","\ufeff"): s = s.replace(z,"")
    return s.strip().replace("\r","").replace("\n","").replace("\t","")

def _get_line_token():
    tok = os.environ.get("CHANNEL_ACCESS_TOKEN") or (LINE_TOKEN_TXT.read_text("utf-8", errors="ignore").splitlines()[0] if LINE_TOKEN_TXT.exists() else "")
    tok = _clean(tok)
    if not tok or not tok.isascii(): raise RuntimeError("缺少或非法 CHANNEL_ACCESS_TOKEN")
    return tok

# ---- 非阻塞推播：背景佇列 ----
push_q = queue.Queue(maxsize=256)
ingest_q = queue.Queue(maxsize=256)

def _push_to_user(uid: str, text: str)->bool:
    if not uid or not uid.startswith(("U","C")) or not uid.isascii(): return False
    try:
        tok = _get_line_token()
        r = requests.post("https://api.line.me/v2/bot/message/push",
                          headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},
                          json={"to":uid,"messages":[{"type":"text","text":text}]}, timeout=6)
        return 200 <= r.status_code < 300
    except Exception as e:
        print(f"[PUSH][ERR] {e}")
        return False

def _push_worker():
    while True:
        try:
            uid, text = push_q.get()
            _ = _push_to_user(uid, text)
        except Exception as e:
            print(f"[PUSH-WORKER][ERR] {e}")
        finally:
            try: push_q.task_done()
            except Exception: pass

threading.Thread(target=_push_worker, daemon=True).start()

def push_async(uid: str, text: str):
    try:
        push_q.put_nowait((uid, text))
    except queue.Full:
        print("[PUSH] 佇列滿，丟棄一則推播。")

def push_people_count_throttled(count:int, temp:float=0.0, hum:float=0.0):
    if not ENABLE_INGEST or not DASHBOARD_URL.startswith("http"): return True
    try:
        ingest_q.put_nowait({"timestamp":int(time.time()),"people":int(count),"temp":float(temp),"hum":float(hum)})
    except queue.Full:
        pass
    return True

def _ingest_worker():
    while True:
        payload = ingest_q.get()
        try:
            r = requests.post(DASHBOARD_URL, json=payload, headers=HEADERS, timeout=2)
        except Exception:
            pass
        finally:
            try: ingest_q.task_done()
            except Exception: pass

threading.Thread(target=_ingest_worker, daemon=True).start()

def _parse_time_hhmm(s, fallback="08:00"):
    try:
        hh,mm = s.split(":"); return datetime.time(int(hh), int(mm))
    except:
        fh, fm = fallback.split(":")
        return datetime.time(int(fh), int(fm))

def variance_of_laplacian(gray): return cv2.Laplacian(gray, cv2.CV_64F).var()

# ================= 時間 / 簽到狀態 =================
LATE_CUTOFF = _parse_time_hhmm(LATE_CUTOFF_STR, "08:00")
LATE_SCAN_TIME = _parse_time_hhmm(LATE_SCAN_TIME_STR, "08:00")
last_checkin_time = defaultdict(lambda: datetime.datetime.min)
late_notice_sent_on = {}

def _is_today(dt): 
    return (dt and dt.date()==datetime.date.today())

# === 定時遲到掃描 ===
def _late_scan_worker():
    if not LATE_SCAN_ENABLED:
        print("[INFO] 定時遲到掃描：已關閉（LATE_SCAN_ENABLED=0）")
        return
    scope = LATE_SCAN_SCOPE
    print(f"[INFO] 定時遲到掃描啟動：{scope} {LATE_SCAN_TIME.strftime('%H:%M')}")

    while True:
        now = datetime.datetime.now()
        next_dt = datetime.datetime.combine(now.date(), LATE_SCAN_TIME)
        if now >= next_dt:
            next_dt = next_dt + datetime.timedelta(days=1)
        if scope == "weekday":
            while next_dt.weekday() >= 5:
                next_dt += datetime.timedelta(days=1)

        while True:
            now = datetime.datetime.now()
            delta = (next_dt - now).total_seconds()
            if delta <= 0: 
                break
            time.sleep(min(delta, 60))

        try:
            NAME_TO_UID = _load_members_map()
            today = datetime.date.today()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            checked_today = {name for name, t in last_checkin_time.items() if _is_today(t)}
            candidates = [n for n in NAME_TO_UID.keys() if n not in checked_today]

            for name in candidates:
                if late_notice_sent_on.get(name) == today:
                    continue
                uid = NAME_TO_UID.get(_clean(name))
                if uid:
                    text = (f"⚠️ 遲到通知（截止 {LATE_SCAN_TIME.hour:02d}:{LATE_SCAN_TIME.minute:02d}）\n"
                            f"姓名：{name}\n"
                            f"時間：{ts}")
                    push_async(uid, text)
                    late_notice_sent_on[name] = today
                else:
                    print(f"[LATE-SCAN] 找不到 {name} 的 userId，略過。")
        except Exception as e:
            print(f"[LATE-SCAN][ERR] {e}")

# ================= 名單載入 =================
def _load_members_map():
    mapping = {}
    if USERS_JSON.exists():
        try:
            data = json.loads(USERS_JSON.read_text("utf-8"))
            for k,v in (data.get("_by_name") or {}).items():
                k2=_clean(k); v2=_clean(v)
                if k2 and v2 and v2.startswith(("U","C")): mapping[k2]=v2
        except Exception: pass
    if MEMBERS_JSON.exists():
        try:
            data = json.loads(MEMBERS_JSON.read_text("utf-8-sig"))
            for k,v in data.items():
                k2=_clean(k); v2=_clean(v)
                if k2 and v2 and v2.startswith(("U","C")): mapping.setdefault(k2,v2)
        except Exception: pass
    return mapping

# ================= 載入人臉庫 =================
data = np.load(ENC_PATH, allow_pickle=True)
KNOWN_ENCODINGS = data["encodings"].astype("float32")
KNOWN_NAMES     = data["names"]
def l2_normalize(x, axis=1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n
if KNOWN_ENCODINGS.ndim!=2 or KNOWN_ENCODINGS.shape[1]!=512:
    raise RuntimeError("encodings 維度非 512（ArcFace），請用 build_embeddings.py 重建")
KNOWN_ENCODINGS = l2_normalize(KNOWN_ENCODINGS)

# ================= InsightFace =================
from insightface.app import FaceAnalysis
import onnxruntime as ort
use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=(0 if use_cuda else -1), det_size=DET_SIZE)

# ================= 檢索（FAISS Top-K 投票） =================
use_faiss = True
try:
    import faiss
    faiss_index = faiss.IndexFlatIP(KNOWN_ENCODINGS.shape[1])
    faiss_index.add(KNOWN_ENCODINGS.astype("float32"))
except Exception:
    use_faiss = False

def decide_name_for_encoding(enc, th=COS_THRESHOLD):
    if use_faiss:
        import faiss
        K = 5
        D,I = faiss_index.search(enc.reshape(1,-1).astype("float32"), K)
        sims=D[0]; idxs=I[0]
        cand={}
        for sim,idx in zip(sims,idxs):
            if int(idx)<0: continue
            nm=str(KNOWN_NAMES[int(idx)]); cand.setdefault(nm,[]).append(float(sim))
        if cand:
            ranked=sorted(((nm,float(np.mean(v))) for nm,v in cand.items()), key=lambda x:-x[1])
            best_name,best_sim=ranked[0]
            second_sim = ranked[1][1] if len(ranked)>1 else -1.0
            if best_sim>=th and ((second_sim<0) or (best_sim-second_sim)>=COS_SECOND_BEST_MARGIN):
                return best_name,best_sim
            return "Unknown",best_sim
        return "Unknown",None
    else:
        sims = KNOWN_ENCODINGS @ enc
        order = np.argsort(-sims)
        best_idx=int(order[0]); best_sim=float(sims[best_idx]); best_name=str(KNOWN_NAMES[best_idx])
        second_sim=None
        for j in order[1:]:
            if str(KNOWN_NAMES[int(j)])!=best_name:
                second_sim=float(sims[int(j)]); break
        if best_sim>=th and (second_sim is None or (best_sim-second_sim)>=COS_SECOND_BEST_MARGIN):
            return best_name,best_sim
        return "Unknown",best_sim

# ================= 前處理 / 追蹤器 =================
def preprocess_small(bgr_small):
    if USE_CLAHE:
        yuv=cv2.cvtColor(bgr_small, cv2.COLOR_BGR2YUV)
        y,u,v=cv2.split(yuv)
        if float(np.mean(y))<CLAHE_DARK_Y:
            cl=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); y=cl.apply(y)
            bgr_small = cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YUV2BGR)
    return bgr_small

def _create_tracker():
    t = (TRACKER_TYPE or "KCF").upper()
    try:
        if t=="KCF":
            return cv2.legacy.TrackerKCF_create()
        elif t=="CSRT":
            return cv2.legacy.TrackerCSRT_create()
    except Exception:
        pass
    if t=="KCF" and hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if t=="CSRT" and hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.TrackerMIL_create()

# ================= 名單 / 背景任務 =================
NAME_TO_UID = _load_members_map()
threading.Thread(target=_late_scan_worker, daemon=True).start()

# ================= 擷取執行緒 =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
assert cap.isOpened(), "無法開啟攝影機"

cv2.setUseOptimized(True); cv2.setNumThreads(1)
frame_q = queue.Queue(maxsize=1)
stop_flag = False

def grabber():
    while not stop_flag:
        ok,frame = cap.read()
        if not ok: continue
        if not frame_q.empty():
            try: frame_q.get_nowait()
            except queue.Empty: pass
        frame_q.put(frame)

threading.Thread(target=grabber, daemon=True).start()

print("按 q 離開，] / [ 調門檻，t 測試推播，s 測試語音")
last_proc = 0.0
frame_idx = 0
trackers = {}
track_names = {}
name_hold = {}
tid_gen = itertools.count(1)

# === 推播文字 ===
def line_text_checkin(name: str, ts: datetime.datetime) -> str:
    return (f"✅ 簽到成功\n"
            f"姓名：{name}\n"
            f"時間：{ts:%Y-%m-%d %H:%M:%S}")

def line_text_late(name: str, ts: datetime.datetime) -> str:
    return (f"⚠️ 遲到通知（截止 {LATE_CUTOFF.hour:02d}:{LATE_CUTOFF.minute:02d}）\n"
            f"姓名：{name}\n"
            f"時間：{ts:%Y-%m-%d %H:%M:%S}")

def draw_panel_texts(text_items, known_cnt, total_cnt):
    text_items.append({"text": f"已知: {known_cnt}",                  "org": (12, 28),  "bgr": (0,255,0),    "font": FONT_INFO})
    text_items.append({"text": f"陌生: {max(0,total_cnt-known_cnt)}","org": (12, 56),  "bgr": (42,42,165),  "font": FONT_INFO})
    text_items.append({"text": f"檢測間隔: {DETECT_EVERY_N}",        "org": (12, 84),  "bgr": (255,255,255),"font": FONT_INFO})
    text_items.append({"text": f"門檻: {COS_THRESHOLD:.2f}",         "org": (12, 112), "bgr": (255,255,255),"font": FONT_INFO})
    text_items.append({"text": f"逾時掃描: {'開' if LATE_SCAN_ENABLED else '關'} {LATE_SCAN_TIME.strftime('%H:%M')} ({'平日' if LATE_SCAN_SCOPE=='weekday' else '每天'})",
                       "org": (12, 140), "bgr": (255,255,0), "font": FONT_INFO})

# ================= 主迴圈 =================
while True:
    try:
        frame = frame_q.get(timeout=1)
    except queue.Empty:
        if stop_flag: break
        continue

    frame_idx += 1
    now = time.time()
    if (now - last_proc) < (1.0/max(MAX_PROC_FPS,1)):
        disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("Face Recognition (Full)", disp)
        if (cv2.waitKey(1) & 0xFF)==ord('q'): break
        continue
    last_proc = now

    s = float(SCALE); inv = 1.0/s
    small = cv2.resize(frame, None, fx=s, fy=s)
    small = preprocess_small(small)

    do_detect = (frame_idx % DETECT_EVERY_N == 0) or (len(trackers)==0)
    if do_detect:
        if variance_of_laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)) < BLUR_VAR_MIN:
            do_detect = False

    boxes_small, names_this, sims_this = [], [], []

    if do_detect:
        faces = app.get(small)
        for f in faces:
            l,t,r,b = map(int, f.bbox)
            if (r-l)<MIN_FACE_W or (b-t)<MIN_FACE_W: continue
            boxes_small.append( (t,r,b,l) )
            emb = f.normed_embedding.astype("float32")
            emb = emb / (np.linalg.norm(emb)+1e-9)
            nm, sim = decide_name_for_encoding(emb, th=COS_THRESHOLD)
            names_this.append(nm); sims_this.append(sim)

        trackers.clear(); track_names.clear()
        for (t,r,b,l),nm in zip(boxes_small,names_this):
            x,y,w,h = int(l*inv), int(t*inv), int((r-l)*inv), int((b-t)*inv)
            if w<=0 or h<=0: continue
            trk = _create_tracker(); trk.init(frame,(x,y,w,h))
            tid = next(tid_gen)
            trackers[tid] = [trk,0]
            track_names[tid] = nm

    else:
        remove=[]
        for tid,(trk,lost) in list(trackers.items()):
            ok,box = trk.update(frame)
            if not ok:
                lost += 1; trackers[tid][1]=lost
                if lost>=TRACK_LOST_TOLERANCE: remove.append(tid)
                continue
            x,y,w,h = box
            if w<=0 or h<=0:
                lost += 1; trackers[tid][1]=lost
                if lost>=TRACK_LOST_TOLERANCE: remove.append(tid)
                continue
            boxes_small.append( (int(y*s), int((x+w)*s), int((y+h)*s), int(x*s)) )
            names_this.append( track_names.get(tid,"Unknown") )
            sims_this.append(None)
        for tid in remove:
            trackers.pop(tid,None); track_names.pop(tid,None)

    known_cnt = 0
    total_cnt = len(names_this)
    now_ts = datetime.datetime.now()

    text_items = []
    for i,((t,r,b,l), nm) in enumerate(zip(boxes_small, names_this)):
        key = i
        ts_hold = time.time()
        rec = name_hold.get(key)
        use_nm = nm
        if rec and rec["name"]!=nm:
            if (ts_hold - rec["since"]) < NAME_HOLD_SEC: use_nm = rec["name"]
            else: name_hold[key] = {"name":nm,"since":ts_hold}
        elif not rec:
            name_hold[key] = {"name":nm,"since":ts_hold}

        color = (0,255,0) if use_nm!="Unknown" else (42,42,165)
        T,R,B,L = int(t*inv), int(r*inv), int(b*inv), int(l*inv)
        cv2.rectangle(frame,(L,T),(R,B),color,2)

        text_items.append({"text": use_nm, "org": (L, max(0, T-12)), "bgr": color, "font": FONT_LABEL})
        if i<len(sims_this) and sims_this[i] is not None:
            text_items.append({"text": f"sim={sims_this[i]:.2f}", "org": (L, min(B+18, frame.shape[0]-8)), "bgr": (255,255,0), "font": FONT_INFO})

        if use_nm!="Unknown":
            known_cnt += 1
            if not _is_today(last_checkin_time[use_nm]):
                uid_map = _load_members_map()
                uid = uid_map.get(_clean(use_nm),"")
                if uid:
                    push_async(uid, line_text_checkin(use_nm, now_ts))
                cutoff_dt = now_ts.replace(hour=LATE_CUTOFF.hour, minute=LATE_CUTOFF.minute, second=0, microsecond=0)
                if now_ts > cutoff_dt and uid:
                    push_async(uid, line_text_late(use_nm, now_ts))
                last_checkin_time[use_nm] = now_ts
                tts("簽到成功")

    push_people_count_throttled(known_cnt)
    draw_panel_texts(text_items, known_cnt, total_cnt)
    frame = draw_texts_cn(frame, text_items, frame_idx)

    disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Face Recognition (Full)", disp)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    elif key==ord(']'):
        COS_THRESHOLD = min(0.90, COS_THRESHOLD+0.01); print("[TUNE] COS_THRESHOLD =", round(COS_THRESHOLD,2))
    elif key==ord('['):
        COS_THRESHOLD = max(0.10, COS_THRESHOLD-0.01); print("[TUNE] COS_THRESHOLD =", round(COS_THRESHOLD,2))
    elif key==ord('t'):
        try:
            m = _load_members_map()
            test_name = next(iter(m.keys()))
            push_async(m[test_name],
                f"🔔 測試推播\n姓名：{test_name}\n時間：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
            ); print("[TEST] line_test_push 已送入佇列")
        except StopIteration:
            print("[TEST] 名單為空（先用 LINE 對 bot 傳：連結 你的名字）")
    elif key==ord('s'):
        tts("簽到成功")

# 收尾
stop_flag = True
cap.release()
cv2.destroyAllWindows()
