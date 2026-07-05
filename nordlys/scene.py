import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    name: str
    pad: float
    drone: float
    seq: float
    texture: float
    mood_bias: dict


SCENES = [
    Scene("drift", 1.0, 0.7, 0.35, 0.5, {"density": -0.05}),
    Scene("deep", 0.6, 1.0, 0.15, 0.35, {"brightness": -0.15, "energy": -0.1}),
    Scene("aurora", 1.0, 0.5, 0.6, 0.3, {"brightness": 0.1}),
    Scene("tundra", 0.7, 0.8, 0.25, 0.9, {"warmth": -0.1}),
    Scene("dawn", 0.9, 0.6, 0.5, 0.45, {"energy": 0.1, "brightness": 0.05}),
]

_GAIN_FIELDS = ("pad", "drone", "seq", "texture")


def daynight_bias(hour: int) -> dict:
    # brightest ~15:00, darkest ~03:00
    x = math.cos((hour - 15) / 24 * 2 * math.pi)
    return {"brightness": 0.10 * x, "density": 0.06 * x}


class SceneScheduler:
    def __init__(self, rng, scenes=SCENES, min_dur=300.0, max_dur=900.0, fade=45.0):
        self.rng, self.scenes = rng, list(scenes)
        self.min_dur, self.max_dur, self.fade = min_dur, max_dur, fade
        self.current = self.scenes[int(rng.integers(len(self.scenes)))]
        self.next: Scene | None = None
        self.remaining = float(rng.uniform(min_dur, max_dur))
        self.fade_t = 0.0
        self.transitions = 0

    def _pick_next(self) -> Scene:
        others = [s for s in self.scenes if s.name != self.current.name]
        return others[int(self.rng.integers(len(others)))]

    def step(self, dt: float):
        if self.next is None:
            self.remaining -= dt
            if self.remaining <= 0:
                self.next = self._pick_next()
                self.fade_t = 0.0
        else:
            self.fade_t += dt
            if self.fade_t >= self.fade:
                self.current, self.next = self.next, None
                self.remaining = float(self.rng.uniform(self.min_dur, self.max_dur))
                self.transitions += 1

    def _mix(self) -> float:
        if self.next is None:
            return 0.0
        return min(1.0, self.fade_t / self.fade)

    def gains(self) -> dict:
        # Linear blend: scenes share the same layer instances, so this
        # interpolates a gain on one signal (equal-power would overshoot).
        x = self._mix()
        nxt = self.next or self.current
        return {f: getattr(self.current, f) * (1 - x) + getattr(nxt, f) * x
                for f in _GAIN_FIELDS}

    def mood_bias(self) -> dict:
        x = self._mix()
        nxt = self.next or self.current
        keys = set(self.current.mood_bias) | set(nxt.mood_bias)
        return {k: self.current.mood_bias.get(k, 0.0) * (1 - x) + nxt.mood_bias.get(k, 0.0) * x
                for k in keys}

    def skip(self):
        if self.next is None:
            self.remaining = 0.0

    @property
    def name(self) -> str:
        return f"{self.current.name}→{self.next.name}" if self.next else self.current.name
