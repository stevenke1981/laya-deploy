"""Pac-Man window lifecycle regressions without a display or live model."""
import asyncio
import threading
from concurrent.futures import Future

import pytest

from laya_deploy.games.pacman_agent import HeuristicPolicy, MoveDecision

pacman_tk = pytest.importorskip("laya_deploy.games.pacman_tk", reason="Tk support is not installed")


class Scheduler:
    """One-thread Tk-style event queue with controllable time."""

    def __init__(self):
        self.now = 0
        self.callbacks = {}
        self.sequence = 0
        self.destroyed = False

    def after(self, delay, callback):
        self.sequence += 1
        token = f"after-{self.sequence}"
        self.callbacks[token] = (self.now + delay, callback)
        return token

    def after_cancel(self, token):
        self.callbacks.pop(token, None)

    def run_next(self):
        token = min(self.callbacks, key=lambda item: self.callbacks[item][0])
        self.now, callback = self.callbacks.pop(token)
        callback()

    def run_until(self, deadline):
        while self.callbacks and min(value[0] for value in self.callbacks.values()) <= deadline:
            self.run_next()
        self.now = deadline

    def destroy(self):
        self.destroyed = True

    def title(self, value):
        self.window_title = value

    def resizable(self, *args):
        pass

    def protocol(self, *args):
        pass

    def bind(self, *args):
        pass

    def geometry(self, *args):
        pass

    def lift(self):
        pass

    def attributes(self, *args):
        pass

    def mainloop(self):
        pass


class RecordingCanvas:
    def __init__(self, *args, **kwargs):
        self.texts = []

    def pack(self):
        pass

    def delete(self, tag):
        self.texts = []

    def create_text(self, *args, **kwargs):
        self.texts.append(kwargs.get("text", ""))

    def create_line(self, *args, **kwargs):
        pass

    def create_oval(self, *args, **kwargs):
        pass

    def create_arc(self, *args, **kwargs):
        pass

    def create_rectangle(self, *args, **kwargs):
        pass

    def create_polygon(self, *args, **kwargs):
        pass


class ManualWorker:
    def __init__(self):
        self.pending = False
        self.submissions = []
        self.stopped = False

    def submit(self, coroutine):
        coroutine.close()
        future = Future()
        if not self.pending:
            future.set_result(MoveDecision("left", {"left": 0.8, "right": 0.2}, "laya", 5.0))
        self.submissions.append(future)
        return future

    def stop(self, closer=None):
        for future in self.submissions:
            future.cancel()
        self.stopped = True


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(pacman_tk.tk, "Canvas", RecordingCanvas)
    monkeypatch.setattr(pacman_tk, "_AsyncWorker", ManualWorker)
    window = pacman_tk.PacmanApp(Scheduler(), HeuristicPolicy(), delay_ms=400)
    yield window
    if not window.closed:
        window.close()


def wait_for_arrows(app):
    app.root.run_next()
    app.root.run_next()
    assert app.decision is not None and app.steps == 0


def test_pause_during_arrows_stops_motion_until_resume(app):
    wait_for_arrows(app)
    position = app.game.pac
    app.toggle_pause()
    app.root.run_until(2000)
    assert app.steps == 0 and app.game.pac == position
    assert "PAUSED" in app.canvas.texts

    app.toggle_pause()
    app.root.run_next()
    app.root.run_next()
    assert app.steps == 1 and len(app.worker.submissions) == 1


def test_restart_discards_delayed_move_from_previous_game(app):
    wait_for_arrows(app)
    old_game = app.game
    stale_callback = next(iter(app.root.callbacks.values()))[1]
    app.restart()
    stale_callback()
    assert app.game is not old_game and app.seed == 1
    assert app.steps == 0 and app.game.pac == (9, 15)
    assert app.decision is None and len(app.root.callbacks) == 1


def test_repeated_pause_resume_keeps_exactly_one_tick(app):
    for _ in range(4):
        app.toggle_pause()
        assert not app.root.callbacks
        app.toggle_pause()
        assert len(app.root.callbacks) == 1
    wait_for_arrows(app)
    app.root.run_next()
    assert app.steps == 1 and len(app.worker.submissions) == 1


def test_restart_discards_inflight_result_without_overlapping_inference(app):
    app.worker.pending = True
    app.root.run_next()
    pending = app.future
    old_game = app.game
    for _ in range(3):
        app.restart()
        app.root.run_next()
    assert not pending.cancelled() and len(app.worker.submissions) == 1
    assert app.game is not old_game and app.seed == 3 and app.steps == 0
    assert len(app.root.callbacks) == 1
    pending.set_result(MoveDecision("left", {"left": 1.0}, "laya"))
    app.root.run_next()
    assert app.steps == 0 and app.decision is None
    app.worker.pending = False
    wait_for_arrows(app)
    app.root.run_next()
    assert app.steps == 1 and len(app.worker.submissions) == 2


def test_restart_while_paused_waits_for_resume(app):
    wait_for_arrows(app)
    app.toggle_pause()
    app.restart()
    assert app.paused and not app.root.callbacks and app.steps == 0
    app.toggle_pause()
    wait_for_arrows(app)
    app.root.run_next()
    assert app.steps == 1


def test_close_cancels_pending_callback_and_inference(app):
    app.worker.pending = True
    app.root.run_next()
    pending = app.future
    app.close()
    app.close()
    assert pending.cancelled() and not app.root.callbacks
    assert app.worker.stopped and app.root.destroyed


def test_policy_error_is_visible_and_restart_recovers(app, caplog):
    app.worker.pending = True
    app.root.run_next()
    app.future.set_exception(RuntimeError("model unavailable"))
    app.root.run_next()
    assert "MODEL ERROR" in app.canvas.texts and app.paused
    assert "RuntimeError: model unavailable" in app.canvas.texts
    assert "R: retry  |  Q / Esc: quit" in app.canvas.texts
    assert "model unavailable" in caplog.text
    assert not app.root.callbacks
    app.toggle_pause()
    assert app.paused and not app.root.callbacks
    app.worker.pending = False
    app.restart()
    wait_for_arrows(app)
    app.root.run_next()
    assert app.steps == 1 and app.status == "" and not app.paused


def test_step_limit_stops_scheduling_and_restart_resumes(app):
    app.max_steps = 1
    wait_for_arrows(app)
    app.root.run_next()
    app.root.run_next()
    assert "STEP LIMIT" in app.canvas.texts and not app.root.callbacks
    app.restart()
    wait_for_arrows(app)
    app.root.run_next()
    assert app.steps == 1


def test_worker_shutdown_cancels_tasks_closes_transport_and_joins_thread():
    worker = pacman_tk._AsyncWorker()
    started, cancelled, closed = threading.Event(), threading.Event(), threading.Event()

    async def decision():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def closer():
        closed.set()

    async def cleanup():
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    worker.submit(decision())
    assert started.wait(2)
    try:
        worker.stop(closer)
        assert cancelled.is_set() and closed.is_set()
        assert not worker._thread.is_alive() and worker.loop.is_closed()
    finally:
        worker._thread.join(timeout=2)
        if not worker.loop.is_closed() and not worker._thread.is_alive():
            worker.loop.run_until_complete(cleanup())
            worker.loop.close()


def test_worker_closer_error_is_logged_and_loop_still_closes(caplog):
    worker = pacman_tk._AsyncWorker()

    async def closer():
        raise OSError("close failed")

    try:
        worker.stop(closer)
        assert "close failed" in caplog.text
        assert not worker._thread.is_alive() and worker.loop.is_closed()
    finally:
        worker._thread.join(timeout=2)
        if not worker.loop.is_closed() and not worker._thread.is_alive():
            worker.loop.close()


def test_worker_shutdown_awaits_cleanup_after_prior_future_cancellation():
    worker = pacman_tk._AsyncWorker()
    started, cleaned = threading.Event(), threading.Event()

    async def decision():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleaned.set()

    future = worker.submit(decision())
    assert started.wait(2)
    future.cancel()
    worker.stop()
    assert cleaned.is_set()
    assert not worker._thread.is_alive() and worker.loop.is_closed()


def test_pending_first_model_decision_has_visible_status(app):
    app.worker.pending = True
    app.policy.name = "LAYA ENGLISH NATIVE"
    app.root.run_next()
    assert "Checking Laya..." in app.canvas.texts and app.steps == 0


@pytest.mark.parametrize("failure", ["tk", "app", "mainloop", None])
def test_run_app_closes_resources_once_on_startup_failure_or_return(monkeypatch, failure):
    root, workers, closed = Scheduler(), [], []
    real_worker = pacman_tk._AsyncWorker

    def create_worker():
        worker = real_worker()
        workers.append(worker)
        return worker

    def fail(*args, **kwargs):
        raise RuntimeError(f"{failure} failed")

    async def closer():
        closed.append(True)

    monkeypatch.setattr(pacman_tk, "_AsyncWorker", create_worker)
    monkeypatch.setattr(pacman_tk.tk, "Tk", fail if failure == "tk" else lambda: root)
    monkeypatch.setattr(pacman_tk.tk, "Canvas", RecordingCanvas)
    if failure == "app":
        monkeypatch.setattr(pacman_tk, "PacmanApp", fail)
    if failure == "mainloop":
        monkeypatch.setattr(root, "mainloop", fail)
    try:
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                pacman_tk.run_app(HeuristicPolicy(), closer=closer)
        else:
            pacman_tk.run_app(HeuristicPolicy(), closer=closer)
        assert closed == [True]
        assert workers and all(worker.loop.is_closed() for worker in workers)
        assert failure == "tk" or root.destroyed
    finally:
        for worker in workers:
            if not worker.loop.is_closed():
                worker.stop()
