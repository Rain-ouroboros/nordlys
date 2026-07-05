import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import BLOCK, LEAD_SEC, MP3_BITRATE, PORT, SPOOL_DIR, SR, TTS_CMD
from .encoder import Broadcast, Mp3Encoder
from .generator import MusicGenerator
from .mixer import Ducker
from .voice import VoiceBus

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class Station:
    def __init__(self, sr=SR, block=BLOCK, seed=None, spool_dir=SPOOL_DIR,
                 tts_cmd=TTS_CMD, bitrate=MP3_BITRATE):
        self.sr, self.block = sr, block
        self.gen = MusicGenerator(sr=sr, block=block, seed=seed)
        self.voice = VoiceBus(sr, spool_dir, tts_cmd)
        self.ducker = Ducker(sr)
        self.broadcast = Broadcast()
        self.encoder = Mp3Encoder(sr, self.broadcast, bitrate)
        self.started = time.time()
        self.on_air = False
        self.stop = threading.Event()

    def run_pipeline(self):
        self.on_air = True
        lead, produced, t0 = LEAD_SEC, 0.0, time.monotonic()
        while not self.stop.is_set():
            self.voice.poll_spool()
            music = self.gen.next_block()
            voice = self.voice.pull(self.block)
            self.encoder.feed(self.ducker.process(music, voice))
            produced += self.block / self.sr
            ahead = produced - (time.monotonic() - t0)
            if ahead > lead:
                time.sleep(ahead - lead)
        self.on_air = False

    def status(self) -> dict:
        st = self.gen.status()
        st.update(listeners=self.broadcast.count,
                  uptime_sec=int(time.time() - self.started),
                  on_air=self.on_air, speaking=self.voice.speaking,
                  voice_queued=self.voice.queued)
        return st


class Handler(BaseHTTPRequestHandler):
    station: Station = None  # bound by make_server
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    @property
    def route(self) -> str:
        # UI adds cache-buster query strings; match on the bare path
        return self.path.split("?", 1)[0]

    def do_GET(self):
        if self.route == "/status":
            self._json(self.station.status())
        elif self.route in ("/", "/index.html"):
            try:
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    body = f.read()
            except FileNotFoundError:
                body = b"<html><body>nordlys</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.route == "/stream.mp3":
            self._stream()
        else:
            self._json({"error": "not found"}, 404)

    def _stream(self):
        i, q = self.station.broadcast.add()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                self.wfile.write(q.get(timeout=30))
        except Exception:
            pass
        finally:
            self.station.broadcast.remove(i)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        if self.route == "/mood":
            self.station.gen.set_mood(**{k: v for k, v in data.items()
                                         if isinstance(v, (int, float))})
            self._json({"ok": True,
                        "target": self.station.gen.status()["mood"]["target"]})
        elif self.route == "/say":
            text = str(data.get("text", "")).strip()
            if not text:
                return self._json({"error": "empty text"}, 400)
            if self.station.voice.tts is None:
                return self._json({"error": "no TTS backend"}, 503)
            threading.Thread(target=self.station.voice.say, args=(text,),
                             daemon=True).start()
            self._json({"ok": True, "queued": self.station.voice.queued + 1})
        elif self.route == "/skip":
            self.station.gen.skip()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


def make_server(station: Station, port: int = PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"station": station})
    return ThreadingHTTPServer(("0.0.0.0", port), handler)
