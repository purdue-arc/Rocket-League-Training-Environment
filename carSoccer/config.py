from dataclasses import dataclass

INCH = 0.0254                              # inch -> meter

@dataclass(frozen=True)
class BallConfig:
  radius: float = 8 * INCH                 # m
  mass: float = 0.45                       # kg
  restitution_floor: float = 0.65          # dimensionless, assumed

@dataclass(frozen=True)
class SimConfig:
  timestep: float = 0.001                  # s
  contact_duration: float = 0.010          # s, assumed; measure on the real ball later
  min_contact_steps: int = 10              # contact must last at least this many timesteps