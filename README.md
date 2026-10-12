# laya-deploy

**在本機 GPU 部署 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)，
並以專為 laya 設計的「決策模式」發問。**

Laya 是非自回歸的「System 1」決策模型：給它一段狀態（文字或 JSON）和一組型別化問題
（`choice` / `noul` / `score`），一次前向傳遞就回傳每個選項的機率，不生成文字。
本專案提供本機 HTTP 服務（請求與回應採 Decisions 格式：`state` + `questions` → `answers`），
以及依 laya 讀題方式設計的 Python 決策函式庫：本機、免費、單次約 35 ms。

```
  你的程式 / curl / laya_deploy.mode
          │  POST /api/alpha/decisions   {"model", "state", "questions"}
          ▼
  ┌───────────────────────────────────────────┐
  │ laya-deploy（FastAPI，預設 127.0.0.1:8100）│
  │  驗證請求 → model 名稱對應 checkpoint       │
  │  laya.Router（english / multilingual /     │
  │  typed-decisions），GPU 推論以鎖序列化       │
  └───────────────────────────────────────────┘
          │  {"answers": {...}, "usage": {"cost": 0}, ...}
          ▼
```

## 先看結論（2026-10-12 實測，RTX 3060 Ti 8 GB）

| 項目 | 結果 |
|---|---|
| 延遲（3 個問題、暖機後、n=30） | multilingual 中位數 **35.4 ms**（p95 43.9）；english **49.3 ms**（p95 57.3） |
| 首次載入 | 約 7–11 秒（已下載時）；首次下載 english ≈ 808 MB、multilingual ≈ 647 MB |
| VRAM | english 1.57 GiB、multilingual 1.22 GiB |
| 決策模式 vs 長提示問法（english） | 元件比對 **29/33 vs 15/33**、風險分類 **21/24 vs 17/24**；Pac-Man 2/5 勝 vs 0/5 |
| 不可逆動作的唯一把關 | **不建議**，見「限制」 |

## 安裝（Windows）

需要 Python 3.10+、[uv](https://docs.astral.sh/uv/)。NVIDIA GPU 非必要（CPU 也能跑，每次數百毫秒）。

```powershell
git clone https://github.com/stevenke1981/laya-deploy
cd laya-deploy
uv sync --extra model          # 核心 + torch（CUDA 12.8 版）+ laya，約 3 GB
copy .env.example .env         # 視需要調整
uv run laya-deploy doctor      # 檢查 torch / CUDA / 設定，並實際推論一次（首次會下載模型）
```

`pyproject.toml` 已設定在 Windows / Linux 從 PyTorch 的 cu128 索引取 CUDA 版 torch
（PyPI 上的 Windows torch 只有 CPU 版）。核心依賴只有 `fastapi`、`uvicorn`、`httpx`；
不裝 `model` extra 也能用 `--fake` 測試接線。

## 使用

```powershell
uv run laya-deploy serve                    # 啟動伺服器（或 scripts\start-server.ps1）
uv run laya-deploy serve --fake             # 不載模型，固定答案，用來測接線
uv run laya-deploy ask                      # 本機直接推論一個範例問題
uv run laya-deploy ask --url http://127.0.0.1:8100/api/alpha/decisions --state state.json --questions q.json
uv run laya-deploy bench --models english multilingual -n 30
uv run laya-deploy pull --models english multilingual typed-decisions   # 預先下載
uv run laya-deploy eval                     # 決策模式 vs 長提示問法的準確度（需伺服器）
```

### HTTP API

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/api/alpha/decisions` | Decisions 格式：`{model, state, questions}` → `{answers, usage, ...}` |
| POST | `/v1/decisions` | 同上（別名） |
| GET | `/health` | 狀態、裝置、GPU、已載入的 checkpoint |

```powershell
curl.exe -s http://127.0.0.1:8100/api/alpha/decisions -H "Content-Type: application/json" -d '{
  "model": "laya/english",
  "state": {"subject": "Duplicate charge on invoice #4411"},
  "questions": {"dept": {"type": "choice", "instructions": "Which department?",
                         "criteria": {"billing": "refunds", "technical": "bugs"}}}}'
```

請求的 `model` 欄位決定 checkpoint：`english`、`multilingual`、`typed-decisions`、`auto`
（可帶前綴，如 `laya/english`、`convaiinnovations/laya:multilingual`；`convaiinnovations/laya` = english）。
其他值或未提供時，一律用 `LAYA_DEPLOY_DEFAULT_MODEL`。

錯誤以 `{"error": {"message", "code"}}` 回傳：格式不符 400、未授權 401、過大 413、推論失敗 500。

## laya 決策模式（本專案的做法）

給 laya 一般大型模型習慣的長提示（整段情境塞進 `state`、事實寫在 `instructions`、附「都不符合」選項、是非題用 `noul`），
效果很差。逐項實驗後（`laya-deploy eval`、Pac-Man），整理出 laya 讀題的方式，並做成 `laya_deploy.mode`：

1. **`state` 只放被判斷的對象**：一句意圖、一個動作、一段短語。不放整個視窗、任務背景或棋盤。
2. **候選放在選項裡**，每個選項的描述寫出「它和別人哪裡不同」。
3. **不加「都不符合」（`none` / `other`）選項**：改由程式設門檻，低於門檻就棄權。
4. **是／否判斷用兩個選項的 `choice`**，兩邊描述要對比（如 destructive：刪除、送出、付款… vs harmless：開啟、瀏覽、儲存…），不用 `noul`。
5. **一次呼叫只問一個對象**：同一次呼叫的多題共用一份 `state`，把各題事實塞進 `instructions` 幾乎無效。
6. **排序、平手、門檻由程式負責**；相同的短語只問一次（`Judge` 內建 LRU 快取）。

```python
from laya_deploy.client import HttpAsker
from laya_deploy.mode import Judge, ground, risk

judge = Judge(HttpAsker(model="english"))
v = await ground(judge, "the Save button in the dialog", {"e3": 'Button "Save"', "e4": 'Button "Cancel"'})
v.label, v.p                       # "e3", 0.948（實測）；低於 min_p（預設 0.5）時 label 為 None
await risk(judge, 'click Button "Delete all" in Notepad')   # P(destructive)，實測 0.983
await judge.classify("I was charged twice", {"billing": "invoices, refunds", "tech": "bugs, outages"})
```

### 評測：決策模式 vs 長提示問法（`laya-deploy eval`）

33 個元件比對（grounding）案例（Notepad、另存新檔、Gmail 風格、中文對話框、設定頁；其中 6 個目標不存在，正確答案是棄權）、
24 個動作風險案例。案例是手寫的小集合，數字只當煙霧測試看。2026-10-12 實測：

| checkpoint | 問法 | grounding 正確 | 找到存在的目標 | 選錯元件 | 風險正確 | 漏掉高風險 | 誤報 |
|---|---|---|---|---|---|---|---|
| english | **決策模式** | **29/33** | **24/27** | 3 | **21/24** | 3 | 0 |
| english | 長提示 | 15/33 | 9/27 | 0 | 17/24 | 5 | 2 |
| multilingual | **決策模式** | **23/33** | **20/27** | 4 | **18/24** | 5 | 1 |
| multilingual | 長提示 | 7/33 | 1/27 | 0 | 14/24 | 10 | 0 |

長提示問法「選錯 0」只是因為它幾乎都回答 `none`。

## 限制（據實）

- **不要讓 laya 當不可逆動作（刪除、送出、付款）的唯一把關**：決策模式下 english 仍漏掉 3/24 個高風險動作。
- 用長提示發問就拿不到決策模式的效果（見上表）；呼叫 HTTP API 時請照決策模式的六條原則組題，或直接用 `laya_deploy.mode`。
- multilingual 偏好 catch-all 選項，中文也不比 english 好；`score` 題最弱；機率預設過度自信，門檻要用自己的資料量。
- **改進方向**：用 `laya.train` 以自己的標註資料微調，或依題型做溫度校準（`laya.calibrate`）。

## 示範：laya 玩 Pac-Man

Pac-Man 遊戲規則與視窗移植自同作者的 decision-cua（MIT），有兩種問法：

- `--style native`（預設，決策模式）：每個方向化成兩句短語，危險短語（deadly／dangerous／risky／safe）與食物短語。
  laya 逐句判斷安全與否，再從「吃得到東西」的角度比對食物短語；程式保留最安全的方向，再挑食物最好的，平手時維持原方向。
  短語重複率高，一局約 13 次 laya 呼叫，其餘 250 步左右都走快取。
- `--style prompt`：每一步一個長提示 `choice`（整張棋盤 + 每個方向的長描述）。

```powershell
uv run laya-deploy serve                                   # 另開一個終端機
uv run laya-deploy pacman                                  # 視窗版（english、決策模式）
uv run laya-deploy pacman --style prompt                   # 長提示問法，對照用
uv run laya-deploy pacman --headless --games 5 --quiet     # 5 個 seed，並與啟發式策略比較
uv run laya-deploy pacman --direct                         # 不經伺服器，本行程直接載入 laya
uv run laya-deploy pacman --fake                           # 只用啟發式，不需模型
```

按鍵：空白鍵暫停、`r` 重來、`q` / `Esc` 離開。HUD 的來源標記：`laya`（這步有新呼叫）、`cache`、`forced`（只有一條路）、
`fallback`（呼叫失敗，改用啟發式）。決策模式下箭頭顯示各方向的安全度占比。

實測（2026-10-12，seed 0–4，每局上限 600 步）：

| 策略 | 勝場 | 平均分數 | 平均步數 |
|---|---|---|---|
| laya english，決策模式 | **2/5** | **1912** | 305.2 |
| laya multilingual，決策模式 | 0/5 | 1680 | 378.2 |
| laya english，長提示 | 0/5 | 118 | 64.6 |
| laya multilingual，長提示 | 0/5 | 180 | 110.0 |
| 啟發式（基準，不用模型） | 5/5 | 3100 | 297.8 |

Pac-Man 的事實本來就由程式算好，laya 在這裡的角色是把短語判讀成安全度與偏好；分數提升來自問法，不是 laya 更懂遊戲。

## Agent skill

`skills/laya-decisions/SKILL.md` 教 agent 如何確認伺服器、用決策模式發問、選 checkpoint、設門檻。
複製到 `~/.claude/skills/`（Claude Code）或 `~/.config/opencode/skills/`（OpenCode）即可使用。

## 安全

- 預設只綁 `127.0.0.1`。綁其他介面時若沒設 `LAYA_DEPLOY_API_KEY`，`serve` 會拒絕啟動。
- Bearer token 以常數時間比對；請求上限 1 MB、每次最多 64 題。
- `/health` 不需驗證，只回傳裝置與已載入模型名稱。

## 專案結構

```
src/laya_deploy/
  config.py     LAYA_DEPLOY_* 設定、.env 讀取、model 名稱 → checkpoint
  engine.py     LayaEngine（laya.Router，延遲載入 torch）與 FakeEngine
  server.py     FastAPI：/api/alpha/decisions、/v1/decisions、/health，請求驗證
  client.py     HttpAsker（打伺服器）/ EngineAsker（本行程直接推論）
  mode.py       laya 決策模式：Judge（classify / match / filter + 快取）、ground、risk
  eval_cases.py grounding 與風險的標註案例
  eval_cli.py   eval 子命令：決策模式 vs 長提示問法
  cli.py        serve / ask / bench / pull / doctor
  pacman_cli.py pacman 子命令（--style native|prompt）
  games/        Pac-Man 規則、決策模式與長提示策略、啟發式、Tk 視窗
skills/laya-decisions/SKILL.md
scripts/start-server.ps1
tests/          以 FakeEngine 測試，不需下載模型或 GPU
```

測試：`uv run --extra dev pytest -q`（CI 在 Windows / Linux、Python 3.10 / 3.12 / 3.13 上跑，只裝宣告的依賴）。

## 授權

本專案 MIT。Laya 模型權重為 Apache-2.0（convaiinnovations）。
