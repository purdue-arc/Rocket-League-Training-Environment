from dataclasses import dataclass

@dataclass(frozen=True)
class BallConfig:
  radius: float = 0.2032
  mass: float = 0.45
  restitution_floor: float = 0.65

@dataclass(frozen=True)
class SimConfig:
  timestep: float = 0.001