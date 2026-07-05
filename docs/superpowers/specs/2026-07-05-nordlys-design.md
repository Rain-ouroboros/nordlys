# Nordlys — agent-steered ambient radio

Date: 2026-07-05
Status: approved (design discussed and approved in conversation)

## Purpose

A standalone online radio an AI agent can inhabit: the agent steers what plays,
speaks over the music, and listeners hear a continuous, beautiful, calm,
never-boring ambient stream (reference: Carbon Based Lifeforms, Meg Bowles).

Hard constraints:

1. Agent influences the broadcast (mood, scenes, skip, announcements).
2. Runs on 2 CPU / 4 GB RAM shared with the agent process — target budget
   ≤ 0.5 core, ≤ 300 MB RSS.
3. Agent can speak on air over the music (ducking, like a live DJ).

Non-goals: neural music generation in realtime, Icecast/liquidsoap stack,
multi-channel stations, persistence of listener state.

## Architecture

One Python process, three stages connected by in-process queues:

```
[generator] -> [mixer] -> [encoder/stream]      [control HTTP API]
 scenes,       music+voice  ffmpeg mp3 pipe ->    /status /mood /say /skip
 mood morph    ducking      fan-out to listeners  + static web UI
```

- **Generator** renders audio in 8-second stereo float32 blocks, always a few
  blocks ahead of playback. All synthesis is vectorized numpy.
- **Mixer** overlays the voice bus onto the music bus with a sidechain
  ducking envelope (music dips ~-10 dB, attack ~0.5 s, release ~1.5 s).
- **Encoder/stream**: blocks are piped to one persistent `ffmpeg` subprocess
  (s16le stdin → mp3 stdout, 128 kbps), mp3 chunks fan out to every connected
  HTTP listener via per-listener queues. Slow listeners drop chunks, never
  stall the pipeline.
- **Control API** runs in the same process (stdlib `http.server`,
  ThreadingHTTPServer, port 8799) and serves the web UI, the stream at
  `/stream.mp3`, and the JSON control endpoints.

Python 3.14, dependencies: numpy only (own venv). External binaries: ffmpeg
(present), espeak-ng (fallback TTS, present).

## Music engine

Layered generative ambient, not pure randomness — "ingredients are designed,
arrangement is generative":

- **theory.py** — pitch/scale layer. Modes limited to consonant sets
  (lydian, dorian, major/minor pentatonic). Chord progression generator:
  random walk over diatonic degrees with voice-leading (minimal pitch
  movement between chords). Slow key drift: every 2–4 scenes, modulate to a
  neighboring key (±1 accidental).
- **synth.py** — instrument layer, all block-based numpy:
  - *pad*: detuned sine/triangle stack per chord tone, slow low-pass filter
    sweep, long attack/release (5–15 s), chords change every 20–40 s;
  - *drone*: root+fifth deep sustained tone, subtle amplitude LFO;
  - *sequence*: sparse bell/pluck notes from the scale (exp-decay sine with
    harmonics), incommensurate loop length vs the pad (Eno phasing — the
    combination never literally repeats), note probability driven by
    `density`;
  - *texture*: filtered noise (wind/rain/shimmer) — filter color driven by
    `warmth`/`brightness`.
- **reverb.py** — FFT convolution (scipy-free, numpy only: overlap-add)
  with a synthetic impulse response (exponentially decaying noise, 4–8 s
  tail). Wet level high — ambient lives in the reverb.
- **scene.py** — a scene = layer mix + tempo-of-change parameters. Scenes
  run 5–15 min, then crossfade 30–60 s into the next. Scene selection
  respects the current mood vector. Day/night arc: a slow bias on
  brightness/density from wall-clock hour.
- **mood.py** — mood vector, all floats 0..1:
  `{energy, warmth, density, brightness}` + `key` (note name) + `scene`
  (optional named scene request). Targets set via API; actual values move
  toward targets over 30–60 s (exponential smoothing per block). Ambient
  forbids jumps.

Determinism: every stochastic component takes an explicit RNG; tests seed it.

## Voice / TTS

- **voice.py** — voice queue. Two entry points:
  1. `POST /say {"text": "...", "voice": "..."}` — text goes through the
     configured TTS backend to a wav;
  2. drop a `.wav` file into `spool/voice/` — picked up as-is (lets the
     agent use any external TTS, e.g. Silero in voice_pilot's venv).
- TTS backend = configurable shell command template
  (`{text}` → wav file path). Default: `espeak-ng -v ru -s 140 -w {out}`.
  Silero later = swap one config line, engine code untouched.
- Voice wavs are resampled to 44.1 kHz stereo, mixed onto the voice bus;
  ducking envelope is computed from voice-bus activity (not hardcoded
  timing).

## Control API

| Endpoint        | Method | Body / response |
|-----------------|--------|-----------------|
| `/status`       | GET    | JSON: current + target mood, scene name, uptime, listeners, now-playing description |
| `/mood`         | POST   | JSON: any subset of mood fields → new targets |
| `/say`          | POST   | `{"text": ...}` → queued announcement, returns queue position |
| `/skip`         | POST   | begin crossfade to next scene now |
| `/stream.mp3`   | GET    | endless mp3 stream |
| `/`             | GET    | web UI |

No auth (localhost pilot; on the server it sits behind the existing edge).

## Web UI

Single static HTML file, no build step, no external assets. Nord palette
(#2E3440 polar night, #D8DEE9 snow storm, #88C0D0/#81A1C1 frost, #B48EAD /
#A3BE8C aurora accents). Dark by default.

Elements: station name, play/pause button, "on air" pulse dot, current scene
+ mood readout (small bars or dots), listener count, and a full-width canvas
running a slow aurora animation whose colors/motion are driven by the live
mood vector from `/status` polling (5 s). When the agent speaks, UI shows a
"voice" indicator. Minimal: one screen, no scroll, no framework.

## Testing

TDD throughout (pytest, own venv):

- theory: generated notes ∈ scale; voice-leading distance bounded; key drift
  reaches only neighboring keys.
- synth/reverb: block shape (2, N) float32, no NaN, peak ≤ 1.0, non-silence;
  reverb tail decays.
- mood: smoothing converges, never overshoots, rate-limited.
- mixer: ducking attenuates music under voice, releases after.
- scene: seeded scheduler is deterministic; crossfade is equal-power.
- stream: fan-out drops for slow listeners; encoder restarts if ffmpeg dies.
- E2E: launch server, record 20 s of `/stream.mp3` via ffmpeg, assert valid
  mp3 + audible RMS; POST /mood changes /status targets; POST /say produces
  ducking dip in captured audio.

## Performance budget

- 44.1 kHz stereo, 8 s blocks: synthesis + FFT reverb well under 0.1 core.
- ffmpeg mp3 encode: ~3–5 % of a core.
- RSS: numpy + buffers, target < 300 MB (typ. ~120 MB).
- Generator thread priority: render-ahead buffer of 3 blocks absorbs GC/API
  pauses; underrun emits silence block + warning (stream never dies).

## Deployment

Pilot: local, `python -m nordlys` in own venv, port 8799. Server deploy
(later, separate task): same process under systemd next to the agent; the
agent steers via localhost HTTP.
