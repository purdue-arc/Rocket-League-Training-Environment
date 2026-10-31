from dataclasses import dataclass

INCH = 0.0254                              # inch -> meter
FOOT = 12 * INCH                           # foot -> meter
MPH = 0.44704                              # mile per hour -> m/s

@dataclass(frozen=True)
class BallConfig:
  radius: float = 8 * INCH                 # m
  mass: float = 0.45                       # kg
  restitutionFloor: float = 0.65          # dimensionless, assumed

@dataclass(frozen=True)
class SimConfig:
  timestep: float = 0.001                  # s
  contactDuration: float = 0.010          # s, assumed; measure on the real ball later
  minContactSteps: int = 10              # contact must last at least this many timesteps

@dataclass(frozen=True)
class ContactConfig:
  restitution: float                       # dimensionless, assumed until measured
  duration: float = 0.010                  # s, assumed
  friction: float = 1.0                    # sliding coefficient, assumed
  torsional: float = 0.005                 # m, spin friction; used when condim = 6, assumed
  rolling: float = 0.0                     # m, rolling friction; used when condim = 6
  condim: int = 3                          # 3 sliding, 6 adds rolling resistance

@dataclass(frozen=True)
class ContactsConfig:
  ballFloor: ContactConfig = ContactConfig(restitution=0.65, condim=6, rolling=0.02)   # rolling assumed
  ballWall: ContactConfig = ContactConfig(restitution=0.6)                             # assumed
  ballCar: ContactConfig = ContactConfig(restitution=0.5)                              # assumed
  carWall: ContactConfig = ContactConfig(restitution=0.3, friction=0.5)                # assumed; also chassis-floor
  carCar: ContactConfig = ContactConfig(restitution=0.3, friction=0.5)                 # assumed

@dataclass(frozen=True)
class ArenaConfig:
  length: float = 30 * FOOT                # m, along x, goal to goal
  width: float = 20 * FOOT                 # m, along y
  wallHeight: float = 0.50                 # m, assumed
  wallThickness: float = 0.10              # m, collider thickness outside the field
  chamfer: float = 0.50                    # m, leg length of the 45 degree corner pieces, assumed
  goalWidth: float = 1.50                  # m, assumed
  goalHeight: float = 0.60                 # m, assumed
  goalDepth: float = 0.50                  # m, assumed
  floorFriction: float = 1.0               # sliding coefficient for wheels on the floor, assumed

@dataclass(frozen=True)
class CarConfig:
  length: float = 0.46                     # m, chassis collider, assumed
  width: float = 0.31                      # m, chassis collider, assumed
  height: float = 0.08                     # m, chassis collider, assumed
  mass: float = 3.5                        # kg, whole car, assumed
  wheelMass: float = 0.10                  # kg per wheel, assumed
  knuckleMass: float = 0.05                # kg per steering knuckle, assumed
  wheelRadius: float = 0.06                # m, assumed
  wheelbase: float = 0.30                  # m, front to rear axle, assumed
  track: float = 0.18                      # m, left to right wheel centres, assumed
  axleDrop: float = 0.03                   # m, wheel centres below the chassis centre, assumed
  maxSteer: float = 0.50                   # rad, centre steer angle at full lock, assumed
  topSpeed: float = 50 * MPH               # m/s, given
  stallTorque: float = 0.50                # N m per wheel, assumed
  steerKp: float = 10.0                    # N m / rad, steering servo stiffness, assumed
  steerDamping: float = 0.3                # N m s / rad, steering joint damping, assumed
  steerArmature: float = 0.001             # kg m^2, steering joint armature, assumed
  wheelDamping: float = 0.001              # N m s / rad, wheel bearing damping, assumed

@dataclass(frozen=True)
class VariantConfig:
  count: int = 8                           # car variants compiled at startup; variant 0 is the nominal car
  sizeScale: tuple[float, float] = (0.85, 1.15)      # dimensionless, scales chassis, wheelbase, track and axle drop, assumed
  wheelScale: tuple[float, float] = (0.85, 1.15)     # dimensionless, wheel radius, clamped to stay under the chassis, assumed
  massScale: tuple[float, float] = (0.7, 1.3)        # dimensionless, whole car, assumed
  motorScale: tuple[float, float] = (0.7, 1.3)       # dimensionless, stall torque, assumed
  seed: int = 0                            # variants are drawn once from this seed, so a run is reproducible

@dataclass(frozen=True)
class SceneConfig:
  ball: BallConfig = BallConfig()
  sim: SimConfig = SimConfig()
  arena: ArenaConfig = ArenaConfig()
  car: CarConfig = CarConfig()
  contacts: ContactsConfig = ContactsConfig()
  variants: VariantConfig = VariantConfig()
  carsPerTeam: int = 3                     # P

@dataclass(frozen=True)
class SensorConfig:
  """Tracking-system failure modes applied to the true state to make state.measured. All zero is a perfect sensor."""
  carPosNoise: float = 0.0                 # m, std of car position noise
  ballPosNoise: float = 0.0                # m, std of ball position noise
  carVelNoise: float = 0.0                 # m/s, std of car velocity noise
  ballVelNoise: float = 0.0                # m/s, std of ball velocity noise
  angvelNoise: float = 0.0                 # rad/s, std of angular velocity noise
  yawNoise: float = 0.0                    # rad, std of car heading noise
  dropStart: float = 0.0                   # per tick, chance a tracked object starts a dropout (holds its last value)
  dropEnd: float = 1.0                     # per tick, chance a dropout ends; bursts last about 1 / dropEnd ticks
  ballDropScale: float = 1.0               # ball starts dropouts this many times as often as a car
  spikeProb: float = 0.0                   # per tick per object, chance of a position spike
  spikeSize: float = 0.0                   # m, std of a spike
  lagTicks: int = 0                        # control ticks the measurement trails the truth
  lagJitterProb: float = 0.0               # per tick, chance a measurement arrives one tick later still
  maxLagTicks: int = 3                     # control ticks of history kept

  def isPerfect(self) -> bool:
    return self == SensorConfig()

  @classmethod
  def realistic(cls) -> "SensorConfig":
    """An overhead camera at 50 Hz, all assumed until fitted from real logs (guide Section 15)."""
    return cls(carPosNoise=0.01, ballPosNoise=0.02, carVelNoise=0.05, ballVelNoise=0.1, angvelNoise=0.1, yawNoise=0.02,
               dropStart=0.01, dropEnd=0.3, ballDropScale=2.0, spikeProb=0.002, spikeSize=0.3,
               lagTicks=1, lagJitterProb=0.1)

@dataclass(frozen=True)
class RandomizationConfig:
  """Per-match physics randomisation ranges; each value is drawn uniformly at match start."""
  floorFriction: tuple[float, float] = (0.7, 1.2)    # scale on ArenaConfig.floorFriction, assumed
  ballMass: tuple[float, float] = (0.85, 1.15)       # scale on BallConfig.mass, assumed
  restitution: float = 0.05                          # dimensionless, +- on every pairing's restitution
  contactFriction: tuple[float, float] = (0.8, 1.2)  # scale on every pairing's sliding friction, assumed
  motorStrength: tuple[float, float] = (0.85, 1.15)  # scale on each car's stall torque, assumed
  steerTrim: float = 0.03                            # rad, +- per-car steering offset, assumed
  latencyTicks: tuple[int, int] = (0, 2)             # control ticks, inclusive, assumed until radio round-trips are measured

@dataclass(frozen=True)
class RewardConfig:
  touch: float = 1.0                       # to a car each time it starts touching the ball
  goal: float = 10.0                       # to every car of the team that scores
  conceded: float = 10.0                   # taken from every car of the team that concedes

@dataclass(frozen=True)
class GameConfig:
  controlHz: float = 50.0                  # Hz, control ticks per second, assumed
  actionRepeat: int = 3                    # control ticks per decision
  matchSeconds: float = 60.0               # s
  latencyTicks: int = 0                    # control ticks of command delay, assumed until measured
  maxLatencyTicks: int = 3                 # control ticks, size of the latency ring buffer
  throttleDeadband: float = 0.05           # |u| below this produces no torque, assumed
  motorTau: float = 0.05                   # s, first-order motor lag, assumed
  steerRate: float = 6.0                   # rad/s, servo slew limit, assumed
  airDensity: float = 1.2                  # kg/m^3
  ballDrag: float = 0.47                   # drag coefficient of a sphere, assumed
  ballLift: float = 0.5                    # Magnus coefficient, assumed
  flippedUp: float = 0.3                   # car counts as flipped when its up axis z component is below this
  flipRecoverSeconds: float = 1.0          # s flipped before the car is set upright
  ballOutMargin: float = 1.0               # m outside the arena box before the ball counts as out
  sensors: SensorConfig = SensorConfig()  # perfect by default; SensorConfig.realistic() for training
