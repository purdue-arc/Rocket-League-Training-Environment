import torch
from dataclasses import dataclass
from typing import Protocol
from tensordict import TensorDict
from torch import Tensor

# GameState: a TensorDict in the world frame, batch [N]
#   car.pos (N, 2P, 3)  car.quat (N, 2P, 4)  car.vel / car.angvel (N, 2P, 3, world frame)
#   car.steer (N, 2P)   car.uprightTime (N, 2P) s flipped
#   ball.pos / ball.vel / ball.angvel (N, 3)
#   score (N, 2) blue, orange    clock (N,) env steps since this match's kickoff
#   params.variant / params.latency (N,)
#   measured: car and ball as the sensors report them
GameState = TensorDict
StepEvents = TensorDict

def emptyGameState(numWorlds: int, carsPerTeam: int, device: torch.device) -> GameState:
  N, C = numWorlds, 2 * carsPerTeam
  f = dict(dtype=torch.float32, device=device)
  car = TensorDict({
    "pos": torch.zeros(N, C, 3, **f), "quat": torch.zeros(N, C, 4, **f),
    "vel": torch.zeros(N, C, 3, **f), "angvel": torch.zeros(N, C, 3, **f),
    "steer": torch.zeros(N, C, **f), "uprightTime": torch.zeros(N, C, **f),
  }, batch_size=[N])
  ball = TensorDict({"pos": torch.zeros(N, 3, **f), "vel": torch.zeros(N, 3, **f), "angvel": torch.zeros(N, 3, **f)}, batch_size=[N])
  params = TensorDict({
    "variant": torch.zeros(N, dtype=torch.long, device=device),
    "latency": torch.zeros(N, dtype=torch.long, device=device),
  }, batch_size=[N])

  return TensorDict({
    "car": car, "ball": ball, "params": params,
    "score": torch.zeros(N, 2, dtype=torch.long, device=device),
    "clock": torch.zeros(N, dtype=torch.long, device=device),
    "measured": TensorDict({"car": car.clone(), "ball": ball.clone()}, batch_size=[N]),
  }, batch_size=[N])

def emptyEvents(numWorlds: int, carsPerTeam: int, device: torch.device) -> StepEvents:
  N, C = numWorlds, 2 * carsPerTeam
  b = dict(dtype=torch.bool, device=device)
  return TensorDict({
    "goal": torch.zeros(N, 2, **b), "ballTouch": torch.zeros(N, C, **b), "ballOut": torch.zeros(N, **b),
    "kickoff": torch.zeros(N, **b), "flipped": torch.zeros(N, C, **b),
    "done": torch.zeros(N, **b), "truncated": torch.zeros(N, **b),
  }, batch_size=[N])

def maskedAssign(dst: TensorDict, src: TensorDict, mask: Tensor) -> None:
  """dst = src where mask (N,) is True, leaf by leaf, in place; no per-world loop and no sync."""
  for key in src.keys(include_nested=True, leaves_only=True):
    d = dst.get(key)
    m = mask.view(-1, *[1] * (d.dim() - 1))
    dst.set_(key, torch.where(m, src.get(key), d))

@dataclass(frozen=True)
class EngineCaps:
  teleport: bool                           # set_state places objects instantly
  trueState: bool                          # state holds the noise-free truth
  parallelWorlds: bool                     # numWorlds can exceed 1

class Engine(Protocol):
  numWorlds: int                           # SimEngine: N; RealEngine: 1
  device: torch.device
  state: GameState                         # true state (sim) + .measured (what sensors report)
  capabilities: EngineCaps

  def step(self, carActions: Tensor) -> StepEvents:
    """carActions: (N, 2P, 2) = [throttle, steer] per car, blue first, in [-1, 1]. One control tick."""
  def setState(self, mask: Tensor, target: GameState) -> None:
    """Sim: teleport masked worlds. Real: drive cars to the targets, wait for the ball, then return."""
  def setParams(self, mask: Tensor, params: dict[str, Tensor]) -> None:
    """Sim: per-world physics parameters. Real: ignored."""
