import signal
import subprocess

from experiments import campaign_12h


def test_supervisor_kills_stuck_worker_before_global_deadline(monkeypatch, tmp_path):
    class StuckProcess:
        pid = 4321
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                assert timeout == 7
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    process = StuckProcess()
    killed = []
    monkeypatch.setattr(campaign_12h.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(campaign_12h.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(campaign_12h.time, "time", lambda: 90)
    result = campaign_12h.execute("train", until=50, deadline=100, log=tmp_path / "worker.log")
    assert result == -9
    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_supervisor_preserves_successful_exit(monkeypatch, tmp_path):
    class Finished:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(campaign_12h.subprocess, "Popen", lambda *a, **k: Finished())
    assert campaign_12h.execute("benchmark", until=100, deadline=200, log=tmp_path / "ok.log") == 0


def test_json_publication_preserves_structured_state(tmp_path):
    path = tmp_path / "nested" / "status.json"
    payload = {"phase": "training", "deadline_unix": 123, "evaluations": []}
    campaign_12h.write_json(path, payload)
    assert campaign_12h.read_json(path) == payload
    assert not path.with_name("status.json.partial").exists()
