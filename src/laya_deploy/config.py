"""Settings from environment variables (and an optional .env file).

Only the LAYA_DEPLOY_* names are read here. laya's own LAYA_* variables are left
to laya itself, so the two never fight over the same knob.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CHECKPOINTS = ("english", "multilingual", "typed-decisions")
AUTO = "auto"


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment wins."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        raise SystemExit(f"invalid {name}={raw!r}: must be an integer") from None
    if not lo <= n <= hi:
        raise SystemExit(f"invalid {name}={n}: must be within {lo}-{hi}")
    return n


def _checkpoint(name: str, default: str, allow_auto: bool) -> str:
    raw = os.environ.get(name, "").strip().lower() or default
    allowed = CHECKPOINTS + ((AUTO,) if allow_auto else ())
    if raw not in allowed:
        raise SystemExit(f"invalid {name}={raw!r}: choose one of {', '.join(allowed)}")
    return raw


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8100
    device: str = ""  # "" = auto (cuda if available)
    default_model: str = "multilingual"  # used when a request names no laya checkpoint
    preload: tuple[str, ...] = ("multilingual",)
    max_loaded: int = 2
    api_key: str = ""  # empty = no auth (only safe on 127.0.0.1)
    max_len: int = 0  # 0 = checkpoint default
    model_label: str = "laya"  # reported back as response "model" prefix

    @classmethod
    def from_env(cls) -> Settings:
        preload_raw = os.environ.get("LAYA_DEPLOY_PRELOAD", "multilingual").strip().lower()
        preload: tuple[str, ...] = ()
        if preload_raw not in {"", "none", "0", "false"}:
            preload = CHECKPOINTS if preload_raw == "all" else tuple(
                p.strip() for p in preload_raw.split(",") if p.strip()
            )
            bad = [p for p in preload if p not in CHECKPOINTS]
            if bad:
                raise SystemExit(f"invalid LAYA_DEPLOY_PRELOAD entries: {', '.join(bad)}")
        return cls(
            host=os.environ.get("LAYA_DEPLOY_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_int("LAYA_DEPLOY_PORT", 8100, 1, 65535),
            device=os.environ.get("LAYA_DEPLOY_DEVICE", "").strip(),
            default_model=_checkpoint("LAYA_DEPLOY_DEFAULT_MODEL", "multilingual", allow_auto=True),
            preload=preload,
            max_loaded=_int("LAYA_DEPLOY_MAX_LOADED", 2, 1, len(CHECKPOINTS)),
            api_key=os.environ.get("LAYA_DEPLOY_API_KEY", "").strip(),
            max_len=_int("LAYA_DEPLOY_MAX_LEN", 0, 0, 8192),
        )

    @property
    def is_public_bind(self) -> bool:
        return self.host not in {"127.0.0.1", "localhost", "::1"}


def resolve_checkpoint(requested: str | None, default: str) -> str:
    """Map a request's `model` field to a laya checkpoint name (or "auto").

    decision-cua sends its configured model id (e.g. "microsoft/microsoft-decision-1"),
    so anything that does not name a laya checkpoint falls back to `default`.
    Accepted: "english", "laya/multilingual", "convaiinnovations/laya:typed-decisions", "auto", ...
    """
    if not requested:
        return default
    r = requested.strip().lower()
    for sep in ("/", ":"):
        r = r.rsplit(sep, 1)[-1] if sep in r else r
    if r in CHECKPOINTS or r == AUTO:
        return r
    if r == "laya":
        return "english"
    return default
