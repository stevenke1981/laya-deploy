"""Command line: serve / doctor / ask / bench / pull / pacman / eval."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .config import AUTO, CHECKPOINTS, Settings, load_dotenv

REPO = "convaiinnovations/laya"
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}

SAMPLE_STATE = {
    "window": "記事本 - 另存新檔",
    "elements": ["e1 按鈕 儲存(S)", "e2 按鈕 取消", "e3 編輯框 檔案名稱: intro.txt"],
}
SAMPLE_QUESTIONS = {
    "target": {
        "type": "choice",
        "instructions": "Which element is the Save button of the dialog?",
        "criteria": {"e1": "button 儲存(S) (Save)", "e2": "button 取消 (Cancel)", "e3": "edit box for the file name"},
    },
    "risky": {"type": "noul", "instructions": "Would clicking Save delete or irreversibly overwrite user data?"},
    "progress": {"type": "score", "instructions": "How far along is saving the file?",
                 "criteria": ["not started", "dialog open", "saved"]},
}


def _engine(settings: Settings, fake: bool):
    from .engine import FakeEngine, LayaEngine

    return FakeEngine() if fake else LayaEngine(settings)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    s = Settings.from_env()
    if args.host:
        s = Settings(**{**s.__dict__, "host": args.host})
    if args.port:
        s = Settings(**{**s.__dict__, "port": args.port})
    if s.is_public_bind and not s.api_key:
        print(f"refusing to bind {s.host} without LAYA_DEPLOY_API_KEY (no auth on a public interface)",
              file=sys.stderr)
        return 2
    print(f"loading engine ({'fake' if args.fake else 'laya'}; preload={','.join(s.preload) or 'none'}) ...")
    engine = _engine(s, args.fake)
    print(json.dumps(engine.info(), ensure_ascii=False))
    print(f"decisions endpoint: http://{s.host}:{s.port}/api/alpha/decisions")
    uvicorn.run(create_app(s, engine), host=s.host, port=s.port, log_level=args.log_level)
    return 0


def _load_json_arg(value: str) -> Any:
    p = Path(value)
    text = p.read_text(encoding="utf-8") if p.is_file() else value
    return json.loads(text)


def cmd_ask(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    state = _load_json_arg(args.state) if args.state else SAMPLE_STATE
    questions = _load_json_arg(args.questions) if args.questions else SAMPLE_QUESTIONS
    if args.url:
        import httpx

        headers = {"Authorization": f"Bearer {s.api_key}"} if s.api_key else {}
        r = httpx.post(args.url, json={"model": args.model, "state": state, "questions": questions},
                       headers=headers, timeout=120)
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
        return 0 if r.status_code < 400 else 1
    engine = _engine(Settings(**{**s.__dict__, "preload": ()}), args.fake)
    t0 = time.perf_counter()
    res = engine.predict(state, questions, args.model)
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    engine = _engine(Settings(**{**s.__dict__, "preload": ()}), fake=False)
    print(json.dumps(engine.info(), ensure_ascii=False))
    for model in args.models:
        t0 = time.perf_counter()
        engine.predict(SAMPLE_STATE, SAMPLE_QUESTIONS, model)  # load + warm-up
        first = (time.perf_counter() - t0) * 1000
        times = []
        for _ in range(args.n):
            t = time.perf_counter()
            engine.predict(SAMPLE_STATE, SAMPLE_QUESTIONS, model)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
        print(f"{model:16s} first={first:8.0f} ms  median={statistics.median(times):6.1f} ms  "
              f"p95={p95:6.1f} ms  (3 questions, n={args.n})")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    for name in args.models:
        sub = SUBFOLDERS[name]
        root = ["config.json", "rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"]
        patterns = [f"{sub}/*"] if sub else root
        path = snapshot_download(REPO, allow_patterns=patterns)
        print(f"{name}: {path}{'/' + sub if sub else ''}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print(f"python   {sys.version.split()[0]}")
    try:
        import laya
        import torch
    except ImportError as exc:
        print(f"FAIL     model stack missing ({exc}); run: uv sync --extra model")
        return 1
    print(f"laya     {getattr(laya, '__version__', '?')}")
    print(f"torch    {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu      {torch.cuda.get_device_name(0)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    else:
        print("WARN     no CUDA device: inference runs on CPU (hundreds of ms per call)")
    s = Settings.from_env()
    print(f"settings host={s.host} port={s.port} default_model={s.default_model} preload={','.join(s.preload)}")
    if s.is_public_bind and not s.api_key:
        print("FAIL     public bind without LAYA_DEPLOY_API_KEY")
        ok = False
    model = s.default_model if s.default_model != AUTO else "multilingual"
    try:
        engine = _engine(Settings(**{**s.__dict__, "preload": ()}), fake=False)
        t0 = time.perf_counter()
        res = engine.predict(SAMPLE_STATE, SAMPLE_QUESTIONS, model)
        print(f"predict  {model}: target={res['answers']['target']['choice']} "
              f"({(time.perf_counter() - t0) * 1000:.0f} ms incl. load)")
    except Exception as exc:  # noqa: BLE001 - doctor reports every failure
        print(f"FAIL     predict: {type(exc).__name__}: {exc}")
        ok = False
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="laya-deploy", description="Local laya decision server")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="run the HTTP decisions server")
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--fake", action="store_true", help="no model: deterministic answers (wiring tests)")
    sp.add_argument("--log-level", default="info")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("ask", help="one decision, locally or against a running server")
    sp.add_argument("--state", help="JSON text or path to a .json file (default: a sample)")
    sp.add_argument("--questions", help="JSON text or path to a .json file (default: a sample)")
    sp.add_argument("--model", default=os.environ.get("LAYA_DEPLOY_DEFAULT_MODEL", "multilingual"),
                    choices=CHECKPOINTS + (AUTO,))
    sp.add_argument("--url", help="send to this server URL instead of loading the model here")
    sp.add_argument("--fake", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("bench", help="latency per checkpoint on this machine")
    sp.add_argument("--models", nargs="+", default=["multilingual"], choices=CHECKPOINTS)
    sp.add_argument("-n", type=int, default=30)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("pull", help="download checkpoints into the Hugging Face cache")
    sp.add_argument("--models", nargs="+", default=["multilingual"], choices=CHECKPOINTS)
    sp.set_defaults(func=cmd_pull)

    from .pacman_cli import add_parser as add_pacman

    add_pacman(sub)

    from .eval_cli import add_parser as add_eval

    add_eval(sub)

    sp = sub.add_parser("doctor", help="check torch/CUDA, settings, and run one prediction")
    sp.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
