import math
from dataclasses import dataclass, replace, asdict

FIELDS = ("energy", "warmth", "density", "brightness")


@dataclass
class Mood:
    energy: float = 0.4
    warmth: float = 0.6
    density: float = 0.4
    brightness: float = 0.4


class MoodState:
    def __init__(self, mood: Mood | None = None, tau: float = 45.0):
        self.current = mood if mood is not None else Mood()
        self.target = replace(self.current)
        self.tau = tau

    def set_targets(self, **kw):
        for k, v in kw.items():
            if k in FIELDS:
                setattr(self.target, k, min(1.0, max(0.0, float(v))))

    def step(self, dt: float) -> Mood:
        a = 1.0 - math.exp(-dt / self.tau)
        for k in FIELDS:
            cur, tgt = getattr(self.current, k), getattr(self.target, k)
            setattr(self.current, k, cur + (tgt - cur) * a)
        return self.current

    def as_dict(self) -> dict:
        return {"current": asdict(self.current), "target": asdict(self.target)}
