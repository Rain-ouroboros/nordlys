# nordlys

**Live:** http://192.144.15.152:8769/ — stream at `/stream.mp3`, voice: Silero xenia v5_ru.

Agent-steered generative ambient radio. One Python process: procedural
ambient music (Carbon Based Lifeforms / Meg Bowles territory), an HTTP mp3
stream, and a voice bus that lets an agent talk over the music like a live
DJ — music ducks under speech and comes back.

No neural nets, no Icecast, no liquidsoap. numpy + ffmpeg. Runs comfortably
in a fraction of one core.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python -m nordlys                 # http://localhost:8799/
.venv/bin/python -m nordlys --port 9000 --seed 42 --no-tts
```

Requires `ffmpeg` in PATH; `espeak-ng` for the default TTS backend
(swap `TTS_CMD` in `nordlys/config.py` for anything that writes a wav —
Silero, piper, …; tokens `{text}` and `{out}` are substituted, no shell).

## API

| Endpoint      | Method | Body / response |
|---------------|--------|-----------------|
| `/`           | GET    | web UI (Nord palette, mood-driven aurora) |
| `/stream.mp3` | GET    | endless mp3 stream, 128 kbps |
| `/status`     | GET    | mood (current+target), scene, key, listeners, on_air, speaking |
| `/mood`       | POST   | `{"energy":0..1, "warmth":0..1, "density":0..1, "brightness":0..1}` — any subset; applied over ~45 s |
| `/say`        | POST   | `{"text":"..."}` — TTS → on air over ducked music |
| `/skip`       | POST   | crossfade to next scene now |

Agent steering examples:

```bash
curl -X POST localhost:8799/mood -d '{"energy":0.7,"brightness":0.8}'
curl -X POST localhost:8799/say  -d '{"text":"северное сияние над тундрой"}'
curl -X POST localhost:8799/skip -d '{}'
```

External TTS without configuring a backend: drop a `.wav` into
`spool/voice/` — it goes on air next block, then moves to `spool/voice/done/`.

## How the music works

Ingredients are designed, arrangement is generative:

- chords random-walk over consonant modes (lydian/dorian/pentatonic),
  keys drift along the circle of fifths every few scenes;
- four layers — pad, drone, sparse bells, noise texture — on
  incommensurate clocks (Eno phasing: the combination never repeats);
- everything runs through a long FFT convolution reverb;
- five scenes (drift/deep/aurora/tundra/dawn) crossfade every 5–15 min,
  plus a slow day/night arc from wall-clock hour;
- the mood vector morphs toward API targets over ~45 s — no jumps, ambient
  forbids them.

## Layout

```
nordlys/config.py     constants (port, block size, TTS command)
nordlys/theory.py     scales, chord walk, key drift
nordlys/mood.py       mood vector + smoothing
nordlys/synth.py      voices and the four layers
nordlys/reverb.py     overlap-add FFT convolution
nordlys/scene.py      scene definitions + scheduler
nordlys/generator.py  puts the music together, block by block
nordlys/voice.py      voice queue, spool pickup, TTS backend
nordlys/mixer.py      sidechain ducking
nordlys/encoder.py    ffmpeg mp3 pipe + listener fan-out
nordlys/server.py     HTTP: stream, control API, UI
nordlys/web/          the page
```

Tests: `.venv/bin/pytest` (unit + live E2E over a real socket).

## Deployment (as running on the server)

- Code: `/opt/nordlys`, own venv (numpy only), unit: `deploy/nordlys.service`
  (User=ouroboros, port 8769 — the port already open in the provider
  firewall; 8799 is blocked upstream).
- Voice: `nordlys/tts_silero.py` runs under `/opt/ouroboros/.venv/bin/python`
  (torch + silero v5_ru live there), speaker `xenia` — the same voice as the
  main Rain radio. Swap voice via `NORDLYS_SILERO_SPEAKER` or the whole
  backend via `NORDLYS_TTS_CMD`.
- Steering from the agent box is plain localhost HTTP:

```bash
curl -X POST localhost:8769/mood -d '{"warmth":0.8,"density":0.3}'
curl -X POST localhost:8769/say  -d '{"text":"..."}'   # xenia, ducked over music
curl -X POST localhost:8769/skip -d '{}'
```

- Ops: `systemctl status nordlys`, logs in `journalctl -u nordlys`.
  Restart is safe at any time: listeners reconnect, music state regenerates.
