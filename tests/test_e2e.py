import json
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")

PORT = 8797


@pytest.fixture(scope="module")
def radio():
    proc = subprocess.Popen([sys.executable, "-m", "nordlys",
                             "--port", str(PORT), "--seed", "42", "--no-tts"])
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/status", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield proc
    proc.terminate()
    proc.wait(timeout=10)


def test_stream_is_audible_mp3(radio):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/stream.mp3", timeout=30) as r:
        data = b""
        while len(data) < 120_000:
            data += r.read(8192)
    dec = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "pipe:1"],
        input=data, capture_output=True, timeout=60)
    pcm = np.frombuffer(dec.stdout, dtype=np.float32)
    assert len(pcm) > 44100
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    assert rms > 1e-3, f"stream is silent, rms={rms}"


def test_mood_steering(radio):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/mood",
                                 data=json.dumps({"brightness": 0.95}).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5)
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/status", timeout=5) as r:
        st = json.loads(r.read())
    assert st["mood"]["target"]["brightness"] == 0.95
