# build_embeddings.py
# 產生 encodings.npz（給 server.py 用）

import os
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# 設定路徑
SCRIPT_DIR = Path(__file__).resolve().parent
FACES_DIR  = SCRIPT_DIR / "faces"
ENC_PATH   = SCRIPT_DIR / "encodings.npz"

def l2_normalize(x):
    return x / (np.linalg.norm(x) + 1e-9)

def main():
    if not FACES_DIR.exists():
        raise FileNotFoundError(f"找不到人臉資料夾：{FACES_DIR}")

    # 初始化 InsightFace（buffalo_l 模型）
    # ctx_id=-1 代表使用 CPU，若有 GPU 可改為 0
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    all_encodings = []
    all_names = []

    # 遍歷 faces 資料夾下的每個子資料夾
    for person_dir in sorted(FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name  # 資料夾名即為人名
        print(f"\n[INFO] 處理人物：{name}")

        # 搜尋圖片檔案
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            img_files.extend(person_dir.glob(ext))

        if not img_files:
            print(f"  ⚠️ 找不到任何圖片，略過")
            continue

        for img_path in img_files:
            # --- 修正：解決 Windows 中文路徑讀取失敗問題 ---
            try:
                # 使用 numpy 先讀取原始位元組，再解碼成圖片
                raw_data = np.fromfile(str(img_path), dtype=np.uint8)
                img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  ⚠️ 讀取時發生錯誤：{img_path.name} ({e})")
                continue

            if img is None:
                print(f"  ⚠️ 讀取失敗（格式不支援）：{img_path.name}")
                continue

            # 偵測人臉
            faces = app.get(img)
            if not faces:
                print(f"  ⚠️ {img_path.name} 沒偵測到臉，略過")
                continue

            # 若一張圖有多張臉，取面積最大者
            def area(f):
                l, t, r, b = f.bbox
                return (r - l) * (b - t)
            
            target_face = max(faces, key=area)

            # 提取並正規化 embedding
            enc = target_face.normed_embedding.astype("float32")
            enc = l2_normalize(enc)

            all_encodings.append(enc)
            all_names.append(name)

            print(f"  ✔ {img_path.name} -> 成功產生 Embedding")

    # 檢查是否有產出
    if not all_encodings:
        raise RuntimeError("❌ 沒有成功產生任何 embedding，請確認 faces 資料夾內是否有清楚的人臉照片。")

    # 轉為 numpy 格式並儲存
    all_encodings = np.stack(all_encodings, axis=0).astype("float32")
    all_names = np.array(all_names)

    np.savez(ENC_PATH, encodings=all_encodings, names=all_names)
    print("-" * 50)
    print(f"[DONE] 任務完成！")
    print(f"共處理 {len(set(all_names))} 個人，總計 {all_encodings.shape[0]} 筆特徵向量。")
    print(f"檔案已儲存至：{ENC_PATH}")

if __name__ == "__main__":
    main()