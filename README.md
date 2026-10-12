# laya-deploy

**在本機 GPU 部署 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)，
並提供與 OpenRouter Decisions API 相同格式的端點。**

Laya 是非自回歸的「System 1」決策模型：給它一段狀態（文字或 JSON）和一組型別化問題
（`choice` / `noul` / `score`），一次前向傳遞就回傳每個選項的機率，不生成文字。
它的問答格式和 [decision-cua](https://github.com/stevenke1981/decision-cua) 使用的 Microsoft-Decision-1
相同，所以本專案把它包成同樣的 HTTP 介面：本機、免費、單次約 35 ms。

```
  decision-cua / curl / 任何程式
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
| decision-cua 的 `DecisionClient` 直連 | 可用：回應格式通過其解析器，`cost` 為 0 |
| **當 decision-cua 的 Decision-1 替身** | **目前不建議**，見下方「限制」 |

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
```

### HTTP API

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/api/alpha/decisions` | 與 OpenRouter Decisions 相同的請求與回應格式 |
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
其他值（例如 decision-cua 預設送出的 `microsoft/microsoft-decision-1`）一律用 `LAYA_DEPLOY_DEFAULT_MODEL`。

錯誤以 `{"error": {"message", "code"}}` 回傳：格式不符 400、未授權 401、過大 413、推論失敗 500。

### 讓 decision-cua 使用本機 laya

```powershell
$env:DCUA_DECISION_URL   = "http://127.0.0.1:8100/api/alpha/decisions"
$env:DCUA_DECISION_MODEL = "laya/english"     # 見下方限制：english 明顯比 multilingual 好
uv run dcua ground "the Save button"
```

decision-cua 仍要求 `OPENROUTER_API_KEY` 非空；它會當作 Bearer 送來，若未設 `LAYA_DEPLOY_API_KEY` 伺服器會忽略它。

## 限制（據實）

在 decision-cua 的假桌面（`DCUA_BACKEND=fake`，5 個元件）上，用 `dcua ground` 對同一組目標比較：

| 決策後端 | 找對元件 | 風險判斷「Save」vs「Delete all」 |
|---|---|---|
| Microsoft-Decision-1（OpenRouter） | 5/5（英文、中文描述皆對） | 0.05 vs 0.95，區分清楚 |
| laya english | 1 ok + 1 low_confidence / 6 | 0.61 vs 0.66，幾乎無區分 |
| laya multilingual | 0 / 6 | 0.65 vs 0.66 |

原因：decision-cua 的 grounding 問題附一個 `none`（都不符合）選項；multilingual checkpoint 會強烈偏向它
（拿掉 `none` 時 3/3 正確），english 在短狀態下帶 `none` 也 3/3 正確，但 decision-cua 的完整狀態與指令較長，零樣本下就不穩。
Laya 官方也說明基礎 checkpoint 的零樣本準確度有限、機率預設過度自信、`noul` 可能跟著選項標籤走。

所以：

- **適合**：分類、路由、審核、工單分流等自訂問題；或做為 decision-cua 的離線實驗後端。
- **不適合**：直接取代 Decision-1 執行真實桌面操作（風險閘門會失效）。
- **改進方向**：用 `laya.train`（套件內建）以 decision-cua 的 `runs/*.jsonl` 記錄微調，
  或依每種問題型別做溫度校準（`laya.calibrate`）。

## 示範：laya 玩 Pac-Man

移植自 decision-cua 的 Pac-Man：每一步問 laya 一個 `choice`，選項只列合法方向，
描述附上「吃到豆子／最近的豆子幾步／最近的危險鬼幾步／可吃的鬼幾步」，畫面上箭頭長短就是機率。

```powershell
uv run laya-deploy serve                                   # 另開一個終端機
uv run laya-deploy pacman --model english                  # 視窗版
uv run laya-deploy pacman --headless --games 5 --quiet     # 5 個 seed，並與啟發式策略比較
uv run laya-deploy pacman --direct                         # 不經伺服器，本行程直接載入 laya
uv run laya-deploy pacman --fake                           # 只用啟發式，不需模型
```

按鍵：空白鍵暫停、`r` 重來、`q` / `Esc` 離開。呼叫失敗時該步改用啟發式並標 `fallback`。

實測（2026-10-12，seed 0–4，每局上限 600 步）：

| 策略 | 勝場 | 平均分數 | 平均步數 | 每步延遲 |
|---|---|---|---|---|
| laya english | 0/5 | 118 | 64.6 | 約 42 ms |
| laya multilingual | 0/5 | 180 | 110.0 | 約 37 ms |
| 啟發式（基準） | 5/5 | 3100 | 297.8 | 0 |

laya 零樣本很快送命：它沒有可靠地依照「危險鬼幾步」避開。這個示範的用途是驗證接線、延遲與機率顯示，
以及當作微調前後的對照組，不代表 laya 能玩好遊戲。

## Agent skill

`skills/laya-decisions/SKILL.md` 教 agent 如何確認伺服器、發問、選 checkpoint、寫好問題與設門檻。
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
  cli.py        serve / ask / bench / pull / doctor
  pacman_cli.py pacman 子命令
  games/        Pac-Man 規則、laya 與啟發式策略、Tk 視窗（移植自 decision-cua）
skills/laya-decisions/SKILL.md
scripts/start-server.ps1
tests/          以 FakeEngine 測試，不需下載模型或 GPU
```

測試：`uv run --extra dev pytest -q`（CI 在 Windows / Linux、Python 3.10 / 3.12 / 3.13 上跑，只裝宣告的依賴）。

## 授權

本專案 MIT。Laya 模型權重為 Apache-2.0（convaiinnovations）。
