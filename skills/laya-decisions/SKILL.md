---
name: laya-decisions
description: Make fast, structured decisions (classify, route, triage, yes/no, score on a scale) with the local laya model served by laya-deploy at http://127.0.0.1:8100. Use when the user wants a cheap local "System 1" judgment with calibrated probabilities instead of free text, or asks to run/test laya, laya-deploy, or its Pac-Man demo.
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

`POST http://127.0.0.1:8100/api/alpha/decisions` (same wire format as OpenRouter Decisions):

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
| `english` | English text. Best for option-matching in tests so far. |
| `multilingual` | Non-English text (100+ languages), long input (`LAYA_DEPLOY_MAX_LEN` up to 8192). Strongly biased toward a catch-all option such as `none`/`other`. |
| `typed-decisions` | Fine-tuned on four typed workflows; near chance elsewhere. |
| `auto` | Let laya route by detected language. |

## 4. Write questions laya can answer

- Put the facts in `state`, the decision in `instructions`, and make each option's description
  say **what is different** about it (see the Pac-Man prompt: "eats a pellet, nearest dangerous ghost 2 steps away").
- Avoid a catch-all option (`none`, `other`) unless you need it; if you do, gate on its probability yourself.
- Prefer a two-option `choice` over `noul`: `noul` can follow the label wording instead of the state.
- `score` is the weakest type. Use it for coarse levels only.
- Many options (> ~20) squeeze the token budget: split into a hierarchy.

## 5. Trust, but gate

- Probabilities are overconfident by default. Act only above a threshold you measured on real
  examples; below it, escalate to a stronger model or ask the user.
- Do **not** use laya as the safety/risk gate for real desktop actions (decision-cua): zero-shot it does not
  separate "Save" from "Delete all" (0.61 vs 0.66). Keep Microsoft-Decision-1 for that.
- Errors come back as `{"error": {"message", "code"}}` with HTTP 400/401/413/500; report them, do not retry blindly.

## Pac-Man demo (sanity check that laya is wired up)

```powershell
uv run laya-deploy pacman --model english                    # window: arrows show laya's probabilities
uv run laya-deploy pacman --headless --games 5 --quiet       # laya vs heuristic over 5 seeds
uv run laya-deploy pacman --fake                             # heuristic only, no model
```
Keys in the window: space = pause, r = restart, q/Esc = quit. A `fallback` tag means the call failed and the heuristic moved.
