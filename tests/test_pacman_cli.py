import asyncio

import pytest

from laya_deploy import pacman_cli
from laya_deploy.cli import build_parser
from laya_deploy.client import AskError
from laya_deploy.engine import EngineError
from laya_deploy.games.pacman_agent import HeuristicPolicy


@pytest.mark.parametrize("flag,value", [("--steps", "0"), ("--games", "-1"), ("--delay", "-1")])
def test_invalid_budgets_are_rejected(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pacman", flag, value])


def test_incompatible_backends_are_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pacman", "--direct", "--fake"])


def test_direct_engine_failure_is_actionable(monkeypatch, capsys):
    def fail(args):
        raise EngineError("model missing")

    monkeypatch.setattr(pacman_cli, "_make_policy", fail)
    assert pacman_cli.cmd_pacman(build_parser().parse_args(["pacman", "--direct"])) == 1
    assert "model missing" in capsys.readouterr().err


def test_headless_model_failure_exits_nonzero_and_closes(monkeypatch, capsys):
    class FailedPolicy:
        name = "LAYA TEST"

        async def decide(self, game):
            raise AskError("model disconnected")

    class Asker:
        closed = False

        async def aclose(self):
            self.closed = True

    asker = Asker()
    monkeypatch.setattr(pacman_cli, "_make_policy", lambda args: (FailedPolicy(), asker))
    assert pacman_cli.cmd_pacman(build_parser().parse_args(["pacman", "--headless", "--quiet"])) == 1
    assert asker.closed and "model disconnected" in capsys.readouterr().err


def test_explicit_fake_runs_offline(capsys):
    assert pacman_cli.cmd_pacman(build_parser().parse_args([
        "pacman", "--fake", "--headless", "--steps", "3", "--quiet",
    ])) == 0
    assert "HEURISTIC" in capsys.readouterr().out


def test_gui_receives_same_policy_and_transport_closer(monkeypatch):
    from laya_deploy.games import pacman_tk

    policy = HeuristicPolicy()
    closed = []

    class Asker:
        async def aclose(self):
            closed.append(True)

    def window(actual, **kwargs):
        assert actual is policy
        asyncio.run(kwargs["closer"]())

    monkeypatch.setattr(pacman_cli, "_make_policy", lambda args: (policy, Asker()))
    monkeypatch.setattr(pacman_tk, "run_app", window)
    assert pacman_cli.cmd_pacman(build_parser().parse_args(["pacman"])) == 0
    assert closed == [True]


@pytest.mark.parametrize("style", ["native", "prompt"])
@pytest.mark.parametrize("kind", ["missing", "wrong_labels", "zero", "nan"])
def test_real_policies_reject_malformed_runtime_answers(monkeypatch, style, kind):
    from laya_deploy.games.pacman import PacmanGame

    class Asker:
        async def __call__(self, state, questions):
            key, question = next(iter(questions.items()))
            labels = tuple(question["criteria"])
            probabilities = {label: 1 / len(labels) for label in labels}
            if key != "ready":
                if kind == "missing":
                    return {"answers": {key: {"choice": labels[0]}}}
                probabilities = ({"unexpected": 1} if kind == "wrong_labels" else
                                 {label: float("nan") if kind == "nan" else 0 for label in labels})
            return {"answers": {key: {"choice": labels[0], "probabilities": probabilities}}}

    monkeypatch.setattr(pacman_cli, "HttpAsker", lambda *args: Asker())
    policy, _ = pacman_cli._make_policy(build_parser().parse_args(["pacman", "--style", style]))
    with pytest.raises(AskError):
        asyncio.run(policy.decide(PacmanGame()))


def test_fake_server_cannot_be_reported_as_real_model(monkeypatch, capsys):
    import httpx

    from laya_deploy.client import HttpAsker
    from laya_deploy.config import Settings
    from laya_deploy.engine import FakeEngine
    from laya_deploy.server import create_app

    engine = FakeEngine()
    server = create_app(Settings(), engine)
    monkeypatch.setattr(pacman_cli, "HttpAsker", lambda url, model, key: HttpAsker(
        url, model, key, transport=httpx.ASGITransport(app=server),
    ))
    result = pacman_cli.cmd_pacman(build_parser().parse_args([
        "pacman", "--headless", "--quiet", "--steps", "3",
    ]))
    assert result == 1 and len(engine.calls) == 1
    assert "fake engine" in capsys.readouterr().err


def test_direct_mode_uses_model_and_closes_without_http(monkeypatch, capsys):
    from laya_deploy import engine as engine_module
    from laya_deploy.engine import FakeEngine

    class TestEngine(FakeEngine):
        device, gpu = "test-device", "test-gpu"

        def __init__(self, settings):
            super().__init__()
            assert settings.preload == ()

        def predict(self, *args, **kwargs):
            result = super().predict(*args, **kwargs)
            # A scripted provider response for CLI wiring, not model acceptance.
            return {**result, "routing": {"model": "english"}}

    def no_http(*args, **kwargs):
        raise AssertionError("direct mode must not open an HTTP client")

    monkeypatch.setattr(engine_module, "LayaEngine", TestEngine)
    monkeypatch.setattr(pacman_cli, "HttpAsker", no_http)
    assert pacman_cli.cmd_pacman(build_parser().parse_args([
        "pacman", "--direct", "--headless", "--steps", "3", "--games", "2",
    ])) == 0
    output = capsys.readouterr().out
    assert "test-device (test-gpu)" in output and "fallbacks=0" in output
    assert "HEURISTIC (baseline)" in output


def test_step_output_and_zero_delay_are_supported(capsys):
    assert pacman_cli.cmd_pacman(build_parser().parse_args([
        "pacman", "--fake", "--headless", "--steps", "1", "--delay", "0",
    ])) == 0
    assert "0001" in capsys.readouterr().out


def test_window_startup_error_is_reported_without_traceback(monkeypatch, capsys):
    from laya_deploy.games import pacman_tk

    def fail(*args, **kwargs):
        raise pacman_tk.tk.TclError("no display")

    monkeypatch.setattr(pacman_tk, "run_app", fail)
    assert pacman_cli.cmd_pacman(build_parser().parse_args(["pacman", "--fake"])) == 1
    assert "no display" in capsys.readouterr().err
