from __future__ import annotations

import pytest

from laya_deploy.config import Settings, load_dotenv


def test_defaults_are_local_and_multilingual(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("LAYA_DEPLOY_"):
            monkeypatch.delenv(k)
    s = Settings.from_env()
    assert s.host == "127.0.0.1" and s.port == 8100
    assert s.default_model == "multilingual" and s.preload == ("multilingual",)
    assert not s.is_public_bind


def test_preload_all_and_none(monkeypatch):
    monkeypatch.setenv("LAYA_DEPLOY_PRELOAD", "all")
    assert len(Settings.from_env().preload) == 3
    monkeypatch.setenv("LAYA_DEPLOY_PRELOAD", "none")
    assert Settings.from_env().preload == ()


@pytest.mark.parametrize("name,value", [
    ("LAYA_DEPLOY_PORT", "abc"),
    ("LAYA_DEPLOY_PORT", "70000"),
    ("LAYA_DEPLOY_DEFAULT_MODEL", "gpt"),
    ("LAYA_DEPLOY_PRELOAD", "english,bogus"),
])
def test_bad_values_exit_with_message(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exc:
        Settings.from_env()
    assert name in str(exc.value)


def test_public_bind_flag():
    assert Settings(host="0.0.0.0").is_public_bind


def test_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nLAYA_DEPLOY_PORT=9001\nLAYA_DEPLOY_HOST='0.0.0.0'\n", encoding="utf-8")
    monkeypatch.setenv("LAYA_DEPLOY_PORT", "8200")
    monkeypatch.delenv("LAYA_DEPLOY_HOST", raising=False)
    load_dotenv(env)
    assert Settings.from_env().port == 8200
    assert Settings.from_env().host == "0.0.0.0"
    monkeypatch.delenv("LAYA_DEPLOY_HOST")
