import mujoco, torch
from dataclasses import dataclass
from torch import Tensor
from carSoccer.engine.sim.scene.car import CarGeometry, WHEELS

TEAMS = ("blue", "orange")

def carNames(carsPerTeam: int) -> list[str]:
  """Car names in batch order, blue first: blue0, blue1, ..., orange0, ..."""
  return [f"{team}{i}" for team in TEAMS for i in range(carsPerTeam)]

@dataclass(frozen=True)
class Layout:
  """Every model index the upper layers need, as tensors, so nothing is looked up by name inside the step."""
  carsPerTeam: int                           # P
  carQpos: Tensor                            # (2P,) qpos address of each car's free joint (pos 3 + quat 4)
  carQvel: Tensor                            # (2P,) qvel address (lin 3 + ang 3, angular in body frame)
  steerQpos: Tensor                          # (2P, 2) left and right steering joint qpos addresses
  steerAct: Tensor                           # (2P, 2) left and right steering actuator ids
  driveAct: Tensor                           # (2P, D) drive actuator ids, D = driven wheels
  ballQpos: int
  ballQvel: int
  ballBody: int
  floorGeom: int
  carBodies: list[list[int]]                 # body ids of each car, for mass randomisation
  touchSensor: Tensor                        # (2P,) sensordata address of each car's ball-contact flag
  pairIds: dict[str, list[int]]              # pairing name -> pair indices
  geometry: list[CarGeometry]                # per variant; index 0 is nominal

def buildLayout(model: mujoco.MjModel, carsPerTeam: int, pairNames: dict[str, list[str]], geometry: list[CarGeometry]) -> Layout:
  def obj(kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
      raise KeyError(f"no {kind.name} named {name!r}")
    return i

  J, A, B, S, P = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_BODY,
                   mujoco.mjtObj.mjOBJ_SENSOR, mujoco.mjtObj.mjOBJ_PAIR)
  names = carNames(carsPerTeam)
  long = lambda rows: torch.tensor(rows, dtype=torch.long)

  def bodiesOf(root):
    return [b for b in range(model.nbody) if model.body_rootid[b] == root]

  ballJoint = obj(J, "ball")

  return Layout(
    carsPerTeam=carsPerTeam,
    carQpos=long([model.jnt_qposadr[obj(J, n)] for n in names]),
    carQvel=long([model.jnt_dofadr[obj(J, n)] for n in names]),
    steerQpos=long([[model.jnt_qposadr[obj(J, f"{n}Steer{s}")] for s in "LR"] for n in names]),
    steerAct=long([[obj(A, f"{n}Steer{s}") for s in "LR"] for n in names]),
    driveAct=long([[obj(A, f"{n}Drive{w}") for w in WHEELS] for n in names]),
    ballQpos=int(model.jnt_qposadr[ballJoint]),
    ballQvel=int(model.jnt_dofadr[ballJoint]),
    ballBody=obj(B, "ball"),
    floorGeom=obj(mujoco.mjtObj.mjOBJ_GEOM, "floor"),
    carBodies=[bodiesOf(obj(B, f"{n}Chassis")) for n in names],
    touchSensor=long([model.sensor_adr[obj(S, f"{n}Touch")] for n in names]),
    pairIds={group: [obj(P, p) for p in pairs] for group, pairs in pairNames.items()},
    geometry=geometry,
  )
