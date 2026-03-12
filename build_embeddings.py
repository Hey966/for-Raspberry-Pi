# build_embeddings.py 〔精簡完整版〕
import os, sys, cv2, numpy as np, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--faces", type=str, default=None, help="人臉資料夾（每人一子資料夾）")
parser.add_argument("--out",   type=str, default=r"C:\Users\jack0\Desktop\for Raspberry Pi\python3.13.7\encodings.npz")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
FACES_DIR = Path(args.faces) if args.faces else (SCRIPT_DIR / "faces")
OUT_NPZ   = Path(args.out)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

from insightface.app import FaceAnalysis
import onnxruntime as ort

provs = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in provs

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=(0 if use_cuda else -1), det_size=(960, 960))

def l2_normalize(x, axis=1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def collect_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def embed_one_image(img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    emb = f.normed_embedding.astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

def main():
    if not FACES_DIR.exists():
        print(f"[ERR] 找不到資料夾：{FACES_DIR}")
        sys.exit(1)

    all_enc, all_names = [], []
    persons = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])
    if not persons:
        print("[ERR] faces/ 底下沒有人名資料夾"); sys.exit(1)

    for person_dir in persons:
        person = person_dir.name.strip()
        imgs = collect_images(person_dir)
        if not imgs:
            print(f"[WARN] {person} 沒有影像，略過"); continue

        ok_cnt = 0
        for p in imgs:
            try:
                img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None: 
                    print(f"[WARN] 讀圖失敗：{p}"); continue
                emb = embed_one_image(img)
                if emb is None or emb.shape[0] != 512:
                    print(f"[WARN] 未偵測到臉或維度錯誤：{p.name}"); continue
                all_enc.append(emb)          # ← 不取均值，全保留
                all_names.append(person)
                ok_cnt += 1
            except Exception as e:
                print(f"[WARN] {person}/{p.name} 失敗：{e}")

        print(f"[OK] {person}: 有效向量 {ok_cnt} 筆")

    if not all_enc:
        print("[ERR] 無任何有效向量"); sys.exit(1)

    enc_arr = np.stack(all_enc, axis=0).astype("float32")
    names_arr = np.array(all_names, dtype=object)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, encodings=enc_arr, names=names_arr)

    print(f"[DONE] 保存：{OUT_NPZ}")
    print(f"[INFO] enc count = {enc_arr.shape[0]}, dim = {enc_arr.shape[1]} (應為 512)")

if __name__ == "__main__":
    main()
