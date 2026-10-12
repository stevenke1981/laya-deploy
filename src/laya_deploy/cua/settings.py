"""Settings for laya computer use, from LAYA_CUA_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..client import DEFAULT_URL


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise SystemExit(f"invalid {name}={raw!r}: must be a number") from None
    if not lo <= v <= hi:
        raise SystemExit(f"invalid {name}={v}: must be within {lo}-{hi}")
    return v


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CuaSettings:
    backend: str = "auto"              # auto | windows | fake
    laya_url: str = DEFAULT_URL        # a running `laya-deploy serve`
    laya_model: str = "english"        # english did best on grounding and risk
    api_key: str = ""                  # LAYA_DEPLOY_API_KEY, if the server requires one
    ground_min_p: float = 0.5          # below this, grounding abstains (status low_confidence / no_match)
    risk_threshold: float = 0.5        # laya P(destructive) at or above this needs confirmation
    auto_confirm_risky: bool = False   # DANGEROUS: skip human confirmation
    click_mode: str = "mouse"          # mouse | invoke | auto
    settle_s: float = 0.5              # wait after an action before re-reading the screen
    outline_limit: int = 60            # elements shown by observe()

    @classmethod
    def from_env(cls) -> CuaSettings:
        backend = os.environ.get("LAYA_CUA_BACKEND", "auto").strip().lower() or "auto"
        if backend not in {"auto", "windows", "fake"}:
            raise SystemExit(f"invalid LAYA_CUA_BACKEND={backend!r}: choose auto, windows or fake")
        click_mode = os.environ.get("LAYA_CUA_CLICK_MODE", "mouse").strip().lower() or "mouse"
        if click_mode not in {"mouse", "invoke", "auto"}:
            raise SystemExit(f"invalid LAYA_CUA_CLICK_MODE={click_mode!r}: choose mouse, invoke or auto")
        return cls(
            backend=backend,
            laya_url=os.environ.get("LAYA_CUA_URL", DEFAULT_URL).strip() or DEFAULT_URL,
            laya_model=os.environ.get("LAYA_CUA_MODEL", "english").strip() or "english",
            api_key=os.environ.get("LAYA_DEPLOY_API_KEY", "").strip(),
            ground_min_p=_float("LAYA_CUA_GROUND_MIN_P", 0.5, 0.0, 1.0),
            risk_threshold=_float("LAYA_CUA_RISK_THRESHOLD", 0.5, 0.0, 1.0),
            auto_confirm_risky=_bool("LAYA_CUA_AUTO_CONFIRM_RISKY", False),
            click_mode=click_mode,
            settle_s=_float("LAYA_CUA_SETTLE_S", 0.5, 0.0, 10.0),
            outline_limit=int(_float("LAYA_CUA_OUTLINE_LIMIT", 60, 5, 500)),
        )
