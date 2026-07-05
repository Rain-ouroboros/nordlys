import numpy as np

from .mood import Mood
from .theory import midi_hz

PAD_PARTIALS = ((1.0, 1.0), (0.997, 0.8), (2.003, 0.25), (3.0, 0.1))
DRONE_PARTIALS = ((1.0, 1.0), (0.5, 0.6), (1.5, 0.3))
PLUCK_PARTIALS = ((1.0, 1.0), (2.0, 0.4), (3.0, 0.15))


def _pan_lr(pan: float) -> tuple[float, float]:
    th = (pan + 1.0) * np.pi / 4
    return float(np.cos(th)), float(np.sin(th))


class Voice:
    """One note: summed detuned partials, pad (attack/release) or pluck (exp decay)."""

    def __init__(self, freq, sr, pan=0.0, partials=PAD_PARTIALS,
                 attack=6.0, release=8.0, pluck=None):
        self.freq, self.sr, self.partials = float(freq), sr, partials
        self.l, self.r = _pan_lr(pan)
        self.attack, self.release, self.pluck = attack, release, pluck
        self.phases = np.zeros(len(partials))
        self.level = 0.0            # pad envelope level
        self.releasing = False
        self.elapsed = 0.0          # pluck clock, seconds
        self.dead = False

    def release_now(self):
        self.releasing = True

    def _envelope(self, n: int) -> np.ndarray:
        dt = n / self.sr
        if self.pluck is not None:
            t = self.elapsed + np.arange(n) / self.sr
            env = np.exp(-t / self.pluck)
            self.elapsed += dt
            if self.elapsed > 8 * self.pluck:
                self.dead = True
            return env
        if self.releasing:
            end = max(0.0, self.level - dt / self.release)
        else:
            end = min(1.0, self.level + dt / self.attack)
        env = np.linspace(self.level, end, n, endpoint=False)
        self.level = end
        if self.releasing and self.level <= 0.0:
            self.dead = True
        return env

    def render(self, n: int, brightness: float) -> np.ndarray:
        t = np.arange(n) / self.sr
        mono = np.zeros(n)
        amp_hi = 0.25 + 0.75 * brightness
        for i, (ratio, amp) in enumerate(self.partials):
            a = amp if ratio <= 1.001 else amp * amp_hi
            w = 2 * np.pi * self.freq * ratio
            mono += a * np.sin(self.phases[i] + w * t)
            self.phases[i] = (self.phases[i] + w * n / self.sr) % (2 * np.pi)
        mono *= self._envelope(n) / sum(a for _, a in self.partials)
        return np.stack([mono * self.l, mono * self.r]).astype(np.float32)


class _VoicePool:
    def __init__(self):
        self.voices: list[tuple[Voice, int]] = []  # (voice, start offset in next block)

    def add(self, v: Voice, offset: int = 0):
        self.voices.append((v, offset))

    def render(self, n: int, brightness: float) -> np.ndarray:
        out = np.zeros((2, n), np.float32)
        keep = []
        for v, off in self.voices:
            if off >= n:
                keep.append((v, off - n))
                continue
            out[:, off:] += v.render(n - off, brightness)
            if not v.dead:
                keep.append((v, 0))
        self.voices = keep
        return out


class PadLayer:
    GAIN = 0.20

    def __init__(self, sr, rng):
        self.sr, self.rng = sr, rng
        self.pool = _VoicePool()
        self.notes: dict[int, Voice] = {}

    def set_chord(self, midis: list[int]):
        for m, v in list(self.notes.items()):
            if m not in midis:
                v.release_now()
                del self.notes[m]
        for m in midis:
            if m not in self.notes:
                v = Voice(midi_hz(m), self.sr, pan=float(self.rng.uniform(-0.6, 0.6)))
                self.notes[m] = v
                self.pool.add(v)

    def render(self, n: int, mood: Mood) -> np.ndarray:
        return self.pool.render(n, mood.brightness) * self.GAIN


class DroneLayer:
    GAIN = 0.16

    def __init__(self, sr, rng):
        self.sr, self.rng = sr, rng
        self.pool = _VoicePool()
        self.voice: Voice | None = None
        self.lfo_phase = 0.0

    def set_root(self, midi: int):
        if self.voice is not None:
            self.voice.release_now()
        self.voice = Voice(midi_hz(midi - 12), self.sr, partials=DRONE_PARTIALS,
                           attack=10.0, release=12.0)
        self.pool.add(self.voice)

    def render(self, n: int, mood: Mood) -> np.ndarray:
        out = self.pool.render(n, mood.brightness * 0.5)
        t = np.arange(n) / self.sr
        lfo = 0.85 + 0.15 * np.sin(self.lfo_phase + 2 * np.pi * 0.05 * t)
        self.lfo_phase = (self.lfo_phase + 2 * np.pi * 0.05 * n / self.sr) % (2 * np.pi)
        return (out * lfo.astype(np.float32)) * self.GAIN


class SequenceLayer:
    """Sparse bells on an incommensurate clock — Eno phasing vs the pad."""
    GAIN = 0.14

    def __init__(self, sr, rng):
        self.sr, self.rng = sr, rng
        self.pool = _VoicePool()
        self.scale: list[int] = []
        self.clock = 0
        self.next_event = int(sr * 2.0)

    def set_scale(self, midis: list[int]):
        self.scale = [m for m in midis if m >= 60] or list(midis)

    def render(self, n: int, mood: Mood) -> np.ndarray:
        end = self.clock + n
        while self.next_event < end and self.scale:
            off = self.next_event - self.clock
            if self.rng.random() < 0.25 + 0.6 * mood.density:
                m = int(self.rng.choice(self.scale))
                decay = 2.0 + 2.5 * (1.0 - mood.energy)
                self.pool.add(Voice(midi_hz(m), self.sr,
                                    pan=float(self.rng.uniform(-0.8, 0.8)),
                                    partials=PLUCK_PARTIALS, pluck=decay), off)
            gap = (3.7 + 6.3 * (1.0 - mood.density)) * (1.0 + self.rng.uniform(-0.35, 0.35))
            self.next_event += max(int(self.sr * gap), 1)
        self.clock = end
        return self.pool.render(n, mood.brightness) * self.GAIN


class TextureLayer:
    """Spectrally shaped noise wash; warmth darkens the slope."""
    GAIN = 0.05
    XFADE = 4096

    def __init__(self, sr, rng):
        self.sr, self.rng = sr, rng
        self.tail = np.zeros((2, self.XFADE), np.float32)
        self.lfo_phase = 0.0

    def render(self, n: int, mood: Mood) -> np.ndarray:
        m = n + self.XFADE
        white = self.rng.standard_normal((2, m))
        spec = np.fft.rfft(white, axis=1)
        freqs = np.fft.rfftfreq(m, 1 / self.sr)
        slope = 0.8 + 1.4 * mood.warmth
        shape = 1.0 / np.maximum(freqs, 20.0) ** (slope / 2)
        shape[freqs > 9000] = 0.0
        noise = np.fft.irfft(spec * shape, m, axis=1)
        noise /= max(np.abs(noise).max(), 1e-9)
        x = np.linspace(0, 1, self.XFADE, dtype=np.float32)
        noise[:, :self.XFADE] = noise[:, :self.XFADE] * x + self.tail * (1 - x)
        self.tail = noise[:, n:n + self.XFADE].astype(np.float32).copy()
        t = np.arange(n) / self.sr
        swell = 0.6 + 0.4 * np.sin(self.lfo_phase + 2 * np.pi * 0.013 * t)
        self.lfo_phase = (self.lfo_phase + 2 * np.pi * 0.013 * n / self.sr) % (2 * np.pi)
        return (noise[:, :n] * swell).astype(np.float32) * self.GAIN
