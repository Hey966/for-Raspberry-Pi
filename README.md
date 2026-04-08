# for-Raspberry-Pi

以 **Hey966/for-Raspberry-Pi** 專案內容為基礎整理的完整說明文件。  
本專案是一套結合 **人臉辨識簽到、本機管理頁面、LINE 綁定／通知、請假系統、雲端後端與 Google 試算表同步** 的智慧考勤系統。  
本 README 也加入了 **Windows CMD 啟動教學（可設定截止時間）**，方便在 **不修改原始 `server.py`** 的前提下切換遲到截止時間。  

> 本 README 內容是依照你提供的詳細教學文件整理而成。原始內容可參考你上傳的教學檔 fileciteturn1file0

---

## 1. 專案總覽

這份專案不是單純的一支人臉辨識程式，而是一套把以下功能串起來的完整系統：

- 人臉辨識簽到
- 本機管理網頁
- LINE Bot 綁定與通知
- 請假申請與審核資料
- 雲端後端 API
- Google 試算表同步

### 系統分工

- **本機端**：攝影機取像、即時辨識、簽到記錄、管理網頁  
- **資料端**：`users.json`、`leave_requests.json`、`encodings.npz`  
- **通知端**：LINE Bot 綁定姓名、查詢、推播通知  
- **雲端端**：接收 check-in、同步 Google 試算表、提供遠端 webhook  

---

## 2. 專案資料夾與檔案用途

| 路徑 / 檔案 | 用途說明 |
|---|---|
| `build_embeddings.py` | 建立人臉特徵檔 `encodings.npz`，把 `faces` 資料夾中的照片轉成可比對的向量 |
| `realtime_recognizer.py` | 即時人臉辨識主程式，負責攝影機讀取、辨識、簽到、顯示、通知 |
| `python3.12.12/server.py` | 整合版 Flask 伺服器，包含 LINE webhook、綁定、請假、出勤頁面與班級管理 |
| `python3.12.12/server_V0.py` | 較舊版本伺服器，可作為回退或比對用 |
| `python3.12.12/users.json` | 本機使用者資料，保存姓名、LINE userId、班級、排序、`face_name` 等資料 |
| `python3.12.12/leave_requests.json` | 請假申請資料 |
| `python3.12.12/encodings.npz` | 已建立的人臉特徵向量 |
| `facecheck-backend/app.py` | 雲端後端，處理 webhook、check-in、morning scan、Google Sheets 同步 |
| `linebot_app/app.py` | 精簡版 LINE Bot，主要做基本連結與查詢 |
| `linebot_app/users.json` | LINE Bot 專用綁定資料 |
| `service-account.json` | Google 服務帳號金鑰，用於試算表或雲端服務 |
| `linebot_app/.env` / `python3.12.12/.env` | 環境變數設定檔，通常放 token、secret、網址、金鑰 |

---

## 3. 系統整體流程

1. 先收集每個人的人臉照片，依人名建立資料夾  
2. 執行 `build_embeddings.py`，把照片轉成 `encodings.npz`  
3. 執行 `realtime_recognizer.py` 或 `server.py`，透過攝影機即時辨識  
4. 辨識成功後，以 `face_name` 對應到正式姓名、班級、排序與 LINE userId  
5. 系統記錄簽到資料，必要時推送 LINE 訊息，並可同步到雲端後端或 Google 試算表  
6. 管理者可透過網頁查看出勤、遲到、班級名單與請假資料  

---

## 4. 主要模組詳細介紹

### 4-1 `build_embeddings.py`：建立人臉特徵庫

這支程式負責掃描人臉照片資料夾，把每張照片送進 InsightFace 模型，取得 **512 維人臉特徵向量**，再寫成 `encodings.npz`。之後即時辨識時直接拿這個檔案做比對，不必每次重新計算所有註冊照片。

#### 功能重點
- 輸入：`faces` 資料夾（每個人一個子資料夾）
- 輸出：`encodings.npz`，內含 `encodings` 與 `names`
- 使用 `insightface.app.FaceAnalysis`
- 不是只保留每人一筆平均值，而是保留所有有效向量，提高辨識穩定度

---

### 4-2 `realtime_recognizer.py`：即時辨識核心

這支是整個系統的辨識主程式，負責：

- 開啟攝影機
- 抓取畫面
- 做前處理
- 偵測臉
- 抽人臉向量
- 與資料庫比對
- 顯示結果
- 簽到與推播

#### 功能重點
- 讀取 `encodings.npz`
- 使用 cosine similarity 判斷是否為同一人
- 可設定最佳分數門檻與第一名、第二名差距
- 支援名稱暫留、追蹤器、音效／語音、LINE 推播與背景佇列
- 會結合 `users.json`，把辨識名稱轉成正式顯示姓名與班級資訊

#### 常見可調整參數
- `SCALE`
- `DET_SIZE`
- `COS_THRESHOLD`
- `COS_SECOND_BEST_MARGIN`
- `TRACKER_TYPE`
- `MAX_PROC_FPS`
- `DETECT_EVERY_N`

---

### 4-3 `python3.12.12/server.py`：整合管理伺服器

這支 `server.py` 是整個系統最重要的整合後台。  
它把以下功能包在同一個 Flask 應用中：

- LINE Bot Webhook
- 使用者綁定
- 班級管理
- 請假系統
- 人臉辨識簽到伺服器
- 出勤管理網頁

#### 功能重點
- 讀取 `.env`
- 維護 `users.json` 與 `leave_requests.json`
- 提供姓名綁定 `bind_user()`
- 根據辨識名稱找正式姓名 `get_display_name()`
- 根據人名查班級 `get_class_for_name()`
- 根據人名查排序 `get_order_for_name()`
- 提供網頁管理與 API 路由

> 如果你要做展示版或實際運作版，多半會改這支。

---

### 4-4 `facecheck-backend/app.py`：雲端後端

這支是雲端版 Flask 後端，適合部署到 Render 或其他雲端服務。

#### 功能重點
- 接收 `check-in`
- 處理 webhook
- 平日 08:00 未簽到提醒
- Google 試算表同步
- 提供健康檢查與推播介面

#### 常見路由
- `GET /users`
- `POST /checkin`
- `POST /cron/morning_scan`
- `GET /push`
- `GET /health`

---

### 4-5 `linebot_app/app.py`：LINE 精簡版

這支比較像測試版或輕量版，功能較單純。

#### 功能重點
- 處理「連結 姓名」
- 基本查詢
- 適合先做 LINE 綁定測試

如果要完整整合請假、班級、出勤，建議使用 `server.py`。

---

## 5. 資料檔詳細說明

### 5-1 `users.json`

`users.json` 是整個系統最重要的資料檔之一。

#### 常見結構
- `_by_user_id`：用 LINE userId 當 key，對應姓名、班級、排序、`face_name`
- `_by_name`：用姓名反查 userId
- `_web_users`：部分網頁端建立的人員資料

#### 常見欄位
- `name`
- `class`
- `order`
- `face_name`

---

### 5-2 `leave_requests.json`

這是請假資料庫。

#### 常見欄位
- `id`
- `user_id`
- `name`
- `phone`
- `reason`
- `leave_time`
- `status`
- `review_comment`
- `created_at`
- `reviewed_at`
- `class`

這表示請假流程已經不是單純留言，而是完整的「申請－審核」資料模型。

---

### 5-3 `encodings.npz`

`encodings.npz` 裡面通常至少有兩個陣列：

- `encodings`：人臉向量
- `names`：該向量對應的人名

辨識時會把即時畫面抽出的向量與這些向量做相似度比對。

---

## 6. 安裝與執行教學

### 6-1 下載專案

```bash
git clone https://github.com/Hey966/for-Raspberry-Pi.git
cd for-Raspberry-Pi
```

---

### 6-2 建議環境

- **Windows 開發機**：適合先調通程式與安裝模型
- **Raspberry Pi**：正式部署端，需確認相機、Python 版本、OpenCV 與 onnxruntime 相容性
- 若採用 **Anaconda**，建議獨立建立環境，避免套件衝突

---

### 6-3 套件安裝概念

依專案內容推測，至少會需要：

- Flask
- opencv-python
- numpy
- insightface
- onnxruntime
- line-bot-sdk
- python-dotenv
- requests
- gspread
- google-auth

---

### 6-4 建議執行順序

1. 先確認 `.env`、`users.json`、`service-account.json` 等檔案位置正確  
2. 先執行 `build_embeddings.py` 建立最新 `encodings.npz`  
3. 本機辨識模式：執行 `realtime_recognizer.py`  
4. 整合管理模式：執行 `python3.12.12/server.py`  
5. 雲端模式：部署 `facecheck-backend/app.py`  
6. LINE 測試模式：執行 `linebot_app/app.py`  

---

## 7. Windows CMD 啟動教學（可設定截止時間）

這個功能的目的是在 **不修改原本 `server.py`** 的前提下，透過 Windows 的 CMD 或批次檔先設定環境變數，再啟動伺服器。

### 適用情境
1. 早點名、晚點名或臨時活動需要不同截止時間  
2. 不想直接改 `server.py` 內的預設值  
3. 想保留原專案檔案不動，只靠啟動方式切換設定  

### 原理
`server.py` 會透過這種方式讀取環境變數：

```python
LATE_CUTOFF_STR = os.environ.get("LATE_CUTOFF", "08:00")
```

所以只要在啟動前先設定：

```cmd
set LATE_CUTOFF=23:00
```

程式就會把 `23:00` 當成遲到截止時間。  
如果沒有設定，才會退回預設值。

---

### 手動 CMD 啟動步驟

```cmd
conda activate base
conda activate camara
cd C:\Users\jack0\Desktop\for Raspberry Pi\python3.12.12
set LATE_CUTOFF=23:00
echo %LATE_CUTOFF%
python server.py
```

### 各指令說明

#### `conda activate base`
先啟用 Conda 基本環境，確保 `conda` 指令可正常切換。

#### `conda activate camara`
切換到專案環境 `camara`，讓 Flask、OpenCV、InsightFace 等套件能正常載入。

#### `cd C:\Users\jack0\Desktop\for Raspberry Pi\python3.12.12`
切換到 `server.py` 所在目錄。  
如果沒有切到正確目錄，執行 `python server.py` 時可能找不到 `.env`、`users.json` 或其他相對路徑檔案。

#### `set LATE_CUTOFF=23:00`
設定本次啟動要使用的截止時間。  
這個值只會在目前這個 CMD 視窗有效，關掉視窗後就失效。

#### `echo %LATE_CUTOFF%`
顯示目前設定值，確認 CMD 真的有把截止時間設進去。

#### `python server.py`
在已設定環境變數的狀態下啟動伺服器。

---

### 如何修改截止時間

只要改這一行即可：

```cmd
set LATE_CUTOFF=23:00
```

例如改成：

```cmd
set LATE_CUTOFF=08:00
```

或：

```cmd
set LATE_CUTOFF=18:30
```

建議使用 **24 小時制 `HH:MM`**。

---

### 建議新增 `.bat` 批次檔

如果不想每次手動輸入指令，可以在 `python3.12.12` 資料夾內新增 `start_with_cutoff.bat`：

```bat
@echo off
call conda activate base
call conda activate camara
cd /d C:\Users\jack0\Desktop\for Raspberry Pi\python3.12.12
set LATE_CUTOFF=23:00
echo LATE_CUTOFF=%LATE_CUTOFF%
python server.py
pause
```

#### 重點說明
- `cd /d`：可連磁碟機一起切換
- `pause`：避免視窗執行完就直接關掉，方便檢查錯誤訊息

---

### 互動版 `.bat`

如果希望每次啟動前都自己輸入時間：

```bat
@echo off
call conda activate base
call conda activate camara
cd /d C:\Users\jack0\Desktop\for Raspberry Pi\python3.12.12
set /p LATE_CUTOFF=請輸入截止時間（例如 23:00）： 
echo 目前截止時間為 %LATE_CUTOFF%
python server.py
pause
```

---

### 注意事項

1. `camara` 必須與你實際建立的 Conda 環境名稱相同  
2. 如果雙擊 bat 後顯示 `conda` 不是內部或外部命令，代表 bat 尚未正確載入 Conda  
3. 如果 `server.py` 是用相對路徑讀 `.env`、`users.json`、`leave_requests.json`，啟動前一定要先 `cd` 到正確資料夾  
4. `set` 只會改目前 CMD 工作階段的暫時值，不會永久修改系統環境變數  
5. 若改到 Raspberry Pi / Linux，寫法要從 `set` 改成 `export`

---

### 故障排除

#### 問題 1：`echo` 後看不到 `23:00`
可能是 `set` 沒成功執行，或前後多了空白。  
建議改寫成：

```cmd
set "LATE_CUTOFF=23:00"
```

#### 問題 2：`python server.py` 顯示找不到檔案
通常是 `cd` 位置不對，請先確認 `server.py` 是否真的位於 `python3.12.12` 資料夾中。

#### 問題 3：程式有啟動，但截止時間沒有變
代表 `server.py` 可能沒有在你預期的位置讀取 `LATE_CUTOFF`，或啟動的不是同一支 `server.py`。

---

### 最建議的用法

平常保留原本 `server.py` 不動，只新增一個 bat 啟動檔。  
若要切換不同截止時間，可以複製多個 bat，例如：

- `start_0800.bat`
- `start_1200.bat`
- `start_2300.bat`

這樣最直觀，也最不容易改壞原始專案。

---

## 8. 怎麼新增一個新的人臉使用者

1. 在 `faces` 資料夾下建立一個新的子資料夾  
2. 資料夾名稱通常就是辨識名  
3. 放入多張清楚、不同角度、不同光線的人臉照片  
4. 重新執行 `build_embeddings.py`  
5. 確認 `encodings.npz` 更新完成  
6. 若要與 LINE 或正式姓名對應，還要在 `users.json` 中補上 `name`、`face_name`、`class`、`order` 等欄位  

---

## 9. 怎麼修改系統

| 修改項目 | 修改位置與說明 |
|---|---|
| 修改辨識門檻 | 到 `realtime_recognizer.py` 調整 `COS_THRESHOLD`、`COS_SECOND_BEST_MARGIN` |
| 修改攝影機解析度與速度 | 調整 `SCALE`、`DET_SIZE`、`MAX_PROC_FPS`、`DETECT_EVERY_N` |
| 修改顯示名稱 | 到 `users.json` 設定 `face_name` 與正式 `name` 的對應 |
| 修改班級 | 在 `users.json` 每個人紀錄中加上 `class` |
| 修改名單排序 | 在 `users.json` 中調整 `order` |
| 修改遲到時間 | 改 `.env` 或程式中的 `LATE_CUTOFF`、`LATE_SCAN_TIME` |
| 修改 LINE Token / Secret | 改 `.env` 或 `line_token.txt` |
| 修改雲端試算表 | 改 `GOOGLE_SERVICE_ACCOUNT_JSON` 與 `GOOGLE_SHEET_ID` |
| 修改請假流程 | 改 `server.py` 與 `leave_requests.json` 的寫入／審核邏輯 |
| 修改 Windows 寫死路徑 | 把程式中的 `C:\Users\...` 改成相對路徑或 Linux 路徑 |

---

## 10. 從 Windows 改成 Raspberry Pi 的重點

目前專案中有明顯 Windows 路徑，例如：

```text
C:\Users\jack0\Desktop\...
```

這些都必須改掉。

### 需要特別注意的地方
- PIL 字型路徑若寫成 Windows 字型，例如 `C:\Windows\Fonts\msjh.ttc`，在 Raspberry Pi 上要改成 Linux 可用字型
- `winsound` 與 PowerShell 語音是 Windows 專用功能，Raspberry Pi 上要改成 `beep`、`aplay`、`espeak` 或直接關閉
- 相機裝置編號、OpenCV 後端、解析度在 Pi 上可能不同，要重新測試
- `insightface` 與 `onnxruntime` 在 Pi 上安裝難度較高，需確認版本與 CPU 架構是否相容

---

## 11. 安全與部署注意事項

你的專案內含：
- `.env`
- `service-account.json`
- LINE 設定相關檔案

這些都屬於敏感資訊。

### 建議
- 若要公開 GitHub，必須移除 token、secret、私鑰、Google 服務帳號金鑰
- `users.json` 與 `leave_requests.json` 含個資，正式環境應限制存取權限
- 若部署在公開網路，API 路由建議加上金鑰驗證與權限控管

---

## 12. 建議的實際維護方式

- **日常維護**：更新人臉照片、重建 `encodings.npz`
- **資料維護**：定期備份 `users.json` 與 `leave_requests.json`
- **辨識調校**：依環境光線調整門檻與解析度
- **部署維護**：把本機版、雲端版、LINE 版分開管理，避免一改全壞
- **版本管理**：重大改動前先備份 `server.py` 與 `users.json`

---

## 13. 總結

這份 `for-Raspberry-Pi` 專案是一個已經具備完整雛形的智慧考勤系統。  
它不是只有辨識，而是把：

- 註冊
- 即時辨識
- 綁定
- 通知
- 請假
- 雲端同步

全部串在一起。

真正要穩定使用時，最重要的不是一直加功能，而是先把：

- 路徑
- 環境
- 資料結構
- 部署方式
- 金鑰管理

整理乾淨。  
只要先把 `users.json`、`encodings.npz`、`.env`、Google / LINE 金鑰管理好，再把 Windows 專用寫法改成 Raspberry Pi 可用版本，這套系統就能更穩定地往正式版走。
