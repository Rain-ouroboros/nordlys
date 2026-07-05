import time
from dataclasses import replace

import numpy as np

from .config import BLOCK, SR
from .mood import FIELDS, Mood, MoodState
from .reverb import Reverb, make_ir
from .scene import SceneScheduler, daynight_bias
from .synth import DroneLayer, PadLayer, SequenceLayer, TextureLayer
from .theory import NOTE_NAMES, ChordWalk, Key, drift_key, scale_midi


def _biased(mood: Mood, *biases: dict) -> Mood:
    m = replace(mood)
    for bias in biases:
        for k, v in bias.items():
            if k in FIELDS:
                setattr(m, k, min(1.0, max(0.0, getattr(m, k) + v)))
    return m


class MusicGenerator:
    def __init__(self, sr=SR, block=BLOCK, seed=None, hour_fn=None):
        self.sr, self.block = sr, block
        self.hour_fn = hour_fn or (lambda: time.localtime().tm_hour)
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.key = Key(int(rng.integers(0, 12)), "lydian")
        self.walk = ChordWalk(self.key, rng)
        self.pad = PadLayer(sr, rng)
        self.drone = DroneLayer(sr, rng)
        self.seq = SequenceLayer(sr, rng)
        self.tex = TextureLayer(sr, rng)
        self.reverb = Reverb(make_ir(sr, min(5.0, block / sr * 4), rng), block)
        self.mood = MoodState()
        self.sched = SceneScheduler(rng)
        self.chord: list[int] = []
        self.chord_timer = 0.0
        self.drift_at = int(rng.integers(2, 5))
        self.blocks = 0

    def _tick_harmony(self, dt: float, mood: Mood):
        if self.sched.transitions >= self.drift_at:
            self.key = drift_key(self.key, self.rng)
            self.walk = ChordWalk(self.key, self.rng)
            self.drift_at = self.sched.transitions + int(self.rng.integers(2, 5))
            self.chord_timer = 0.0
        self.chord_timer -= dt
        if self.chord_timer <= 0:
            self.chord = self.walk.next_chord()
            self.pad.set_chord(self.chord)
            self.drone.set_root(48 + self.key.root)
            self.seq.set_scale(scale_midi(self.key, 60, 84))
            self.chord_timer = float(self.rng.uniform(20, 40)) / (0.6 + 0.8 * mood.energy)

    def next_block(self) -> np.ndarray:
        dt = self.block / self.sr
        self.sched.step(dt)
        mood = _biased(self.mood.step(dt), self.sched.mood_bias(),
                       daynight_bias(self.hour_fn()))
        self._tick_harmony(dt, mood)
        g = self.sched.gains()
        dry = (self.pad.render(self.block, mood) * g["pad"]
               + self.drone.render(self.block, mood) * g["drone"]
               + self.seq.render(self.block, mood) * g["seq"]
               + self.tex.render(self.block, mood) * g["texture"])
        wet = self.reverb.process(dry)
        out = dry * 0.5 + wet * 0.9
        peak = float(np.abs(out).max())
        if peak > 0.97:
            out *= 0.97 / peak
        self.blocks += 1
        return out.astype(np.float32)

    def set_mood(self, **kw):
        self.mood.set_targets(**kw)

    def skip(self):
        self.sched.skip()

    def status(self) -> dict:
        return {
            "mood": self.mood.as_dict(),
            "scene": self.sched.name,
            "key": f"{NOTE_NAMES[self.key.root]} {self.key.mode}",
            "chord": list(self.chord),
            "blocks": self.blocks,
        }
