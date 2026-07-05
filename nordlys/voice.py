import os
import shlex
import subprocess
import tempfile
import threading
import wave
from collections import deque

import numpy as np


def load_wav(path: str, sr: int) -> np.ndarray:
    with wave.open(path, "rb") as w:
        wsr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, ch).T
    if ch == 1:
        x = np.vstack([x, x])
    elif ch > 2:
        x = x[:2]
    if wsr != sr:
        m = int(x.shape[1] * sr / wsr)
        src = np.linspace(0.0, 1.0, x.shape[1])
        dst = np.linspace(0.0, 1.0, m)
        x = np.vstack([np.interp(dst, src, x[0]), np.interp(dst, src, x[1])])
    peak = float(np.abs(x).max())
    if peak > 1e-6:
        x = x * (0.7 / peak)
    return x.astype(np.float32)


class TTSBackend:
    def __init__(self, cmd_template: str):
        self.tokens = shlex.split(cmd_template)

    def synth(self, text: str) -> str:
        fd, out = tempfile.mkstemp(suffix=".wav", prefix="nordlys-tts-")
        os.close(fd)
        cmd = [out if t == "{out}" else text if t == "{text}" else t for t in self.tokens]
        subprocess.run(cmd, check=True, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out


class VoiceBus:
    GAP_SEC = 1.0

    def __init__(self, sr: int, spool_dir: str, tts_cmd: str | None):
        self.sr = sr
        self.spool_dir = spool_dir
        os.makedirs(os.path.join(spool_dir, "done"), exist_ok=True)
        self.tts = TTSBackend(tts_cmd) if tts_cmd else None
        self.queue: deque[np.ndarray] = deque()
        self.lock = threading.Lock()
        self.playing: np.ndarray | None = None
        self.pos = 0
        self.gap_left = 0

    def enqueue_wav(self, path: str):
        with self.lock:
            self.queue.append(load_wav(path, self.sr))

    def say(self, text: str):
        if self.tts is None:
            raise RuntimeError("no TTS backend configured")
        path = self.tts.synth(text)
        try:
            self.enqueue_wav(path)
        finally:
            os.unlink(path)

    def poll_spool(self):
        try:
            names = sorted(f for f in os.listdir(self.spool_dir) if f.endswith(".wav"))
        except FileNotFoundError:
            return
        for name in names:
            src = os.path.join(self.spool_dir, name)
            dst = os.path.join(self.spool_dir, "done", name)
            try:
                self.enqueue_wav(src)
                os.replace(src, dst)
            except Exception:
                os.replace(src, dst + ".bad")

    @property
    def queued(self) -> int:
        return len(self.queue) + (1 if self.playing is not None else 0)

    @property
    def speaking(self) -> bool:
        return self.playing is not None

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros((2, n), np.float32)
        i = 0
        with self.lock:
            while i < n:
                if self.gap_left > 0:
                    step = min(self.gap_left, n - i)
                    self.gap_left -= step
                    i += step
                    continue
                if self.playing is None:
                    if not self.queue:
                        break
                    self.playing = self.queue.popleft()
                    self.pos = 0
                take = min(self.playing.shape[1] - self.pos, n - i)
                out[:, i:i + take] = self.playing[:, self.pos:self.pos + take]
                self.pos += take
                i += take
                if self.pos >= self.playing.shape[1]:
                    self.playing = None
                    self.gap_left = int(self.GAP_SEC * self.sr)
        return out
