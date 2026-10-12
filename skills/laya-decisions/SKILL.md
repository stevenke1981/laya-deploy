---
name: laya-decisions
description: Make fast, structured decisions (classify, route, triage, yes/no, match a description to candidates) with the local laya model in laya decision mode, served by laya-deploy at http://127.0.0.1:8100. Use when the user wants a cheap local "System 1" judgment with calibrated probabilities instead of free text, or asks to run/test laya, laya-deploy, or its Pac-Man demo.
---

# Local decisions with laya (laya-deploy)

laya never writes text. You send a **state** and typed **questions**; it returns a probability
for every option in one forward pass (about 35–50 ms on the local GPU). Your code owns the thresholds.

## 1. Make sure the server is up

```powershell
curl.exe -s http://127.0.0.1:8100/health          # {"status":"ok","loaded":[...]}
# not running? from the laya-deploy checkout:
uv run laya-deploy serve                           # or scripts\start-server.ps1
```

## 2. Ask

`POST http://127.0.0.1:8100/api/alpha/decisions` (Decisions format):

```json
{"model": "english",
 "state": {"subject": "Refund request", "body": "I was charged twice."},
 "questions": {
   "dept":    {"type": "choice", "instructions": "Which team handles this?",
               "criteria": {"billing": "invoices, refunds", "tech": "bugs, outages"}},
   "urgency": {"type": "score",  "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
   "churn":   {"type": "noul",   "instructions": "Does the user threaten to leave?"}}}
```

Read `answers.<key>`: `choice` + `probabilities` + `confidence`; `noul` (P(yes)); `score` +
`probabilities` keyed `"0","1",...`. One-off from the shell: `uv run laya-deploy ask --state s.json --questions q.json --url http://127.0.0.1:8100/api/alpha/decisions`.

## 3. Pick the checkpoint (`model`)

| model | use for |
|---|---|
| `english` | Default. Best in every test so far. |
| `multilingual` | Non-English text (100+ languages), long input (`LAYA_DEPLOY_MAX_LEN` up to 8192). Strongly biased toward a catch-all option such as `none`/`other`. |
| `typed-decisions` | Fine-tuned on four typed workflows; near chance elsewhere. |
| `auto` | Let laya route by detected language. |

## 4. Ask in laya decision mode

laya reads the `state` and matches it against the option descriptions. It mostly ignores facts
written into `instructions`. Shape every question like this (measured: grounding 29/33 vs 15/33
with long prompts, risk 21/24 vs 17/24, english checkpoint):

1. `state` = only the subject being judged: `{"step_intent": "the Save button"}`, `{"action": "click Button \"Send\""}`, a short phrase. No window dumps, no task background, no boards.
2. Candidates go in the options; each description says what sets it apart.
3. No catch-all option (`none`, `other`). Abstain in code: top probability below a threshold (0.5 is a fair start).
4. Yes/no = a two-option `choice` with contrasting descriptions (e.g. destructive: "deletes, sends, pays, resets…" vs harmless: "opens, views, saves, cancels…"). Do not use `noul`.
5. One subject per call. Several questions in one call share one `state`; per-item facts in `instructions` do not work.
6. Your code owns ordering, tie-breaks and thresholds. Turn numbers into a few coarse phrases ("dangerous, ghost very close") so judgments repeat and can be cached.

From Python, use the library instead of hand-writing requests:

```python
from laya_deploy.client import HttpAsker
from laya_deploy.mode import Judge, ground, risk
judge = Judge(HttpAsker(model="english"))      # caches identical calls
v = await ground(judge, "the Save button", {"e3": 'Button "Save"', "e4": 'Button "Cancel"'}, min_p=0.5)
p = await risk(judge, 'click Button "Delete all" in Notepad')        # P(destructive)
v = await judge.classify("I was charged twice", {"billing": "invoices, refunds", "tech": "bugs, outages"})
```

`uv run laya-deploy eval` re-measures decision mode against long-prompt questions on the shipped cases.

Other limits: `score` is the weakest type; more than ~20 options squeeze the token budget (split into a hierarchy).

## 5. Trust, but gate

- Probabilities are overconfident by default. Act only above a threshold you measured on real
  examples; below it, escalate to a stronger model or ask the user.
- Do **not** make laya the only gate for irreversible actions (delete, send, pay): even in decision mode
  it missed 3 of 24 risky actions. Add a human confirmation or a stronger check.
- Errors come back as `{"error": {"message", "code"}}` with HTTP 400/401/413/500; report them, do not retry blindly.

## Pac-Man demo (sanity check that laya is wired up)

```powershell
uv run laya-deploy pacman                                    # window, english, decision mode
uv run laya-deploy pacman --style prompt                     # long-prompt question, for comparison
uv run laya-deploy pacman --headless --games 5 --quiet       # laya vs heuristic over 5 seeds
uv run laya-deploy pacman --fake                             # heuristic only, no model
```
Measured over seeds 0–4: decision mode (english) 2/5 wins, mean score 1912; long prompt 0/5, 118; heuristic 5/5, 3100.
Keys: space = pause, r = restart, q/Esc = quit. HUD source tags: `laya` (new call), `cache`, `forced`, `fallback` (call failed, heuristic moved).
