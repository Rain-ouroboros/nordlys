import queue
import subprocess
import threading

import numpy as np


class Broadcast:
    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self.listeners: dict[int, queue.Queue] = {}
        self.lock = threading.Lock()
        self._next = 0

    def add(self):
        q = queue.Queue(maxsize=self.maxsize)
        with self.lock:
            i = self._next
            self._next += 1
            self.listeners[i] = q
        return i, q

    def remove(self, i: int):
        with self.lock:
            self.listeners.pop(i, None)

    def publish(self, chunk: bytes):
        with self.lock:
            qs = list(self.listeners.values())
        for q in qs:
            try:
                q.put_nowait(chunk)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    pass

    @property
    def count(self) -> int:
        with self.lock:
            return len(self.listeners)


class Mp3Encoder:
    def __init__(self, sr: int, broadcast: Broadcast, bitrate: str = "128k"):
        self.sr, self.broadcast, self.bitrate = sr, broadcast, bitrate
        self.proc: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None
        self.closing = False
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             # tiny probe: emit mp3 from the first block instead of
             # buffering ~4.5 s of input before any output appears
             "-probesize", "32", "-analyzeduration", "0",
             "-f", "s16le", "-ar", str(self.sr), "-ac", "2", "-i", "pipe:0",
             "-f", "mp3", "-b:a", self.bitrate, "-flush_packets", "1", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.reader.start()

    def _read(self, proc):
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                return
            self.broadcast.publish(chunk)

    def feed(self, block: np.ndarray):
        pcm = (np.clip(block.T, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        try:
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            if not self.closing:
                self._start()
                self.proc.stdin.write(pcm)
                self.proc.stdin.flush()

    def close(self):
        self.closing = True
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
