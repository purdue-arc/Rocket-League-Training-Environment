import math, torch
from tensordict import TensorDict
from torch import Tensor, nn
from carSoccer.components.obs.builders import ObsSpec

ENTITY_KEYS = ("self", "mates", "opponents", "ball", "goals", "game")   # actor input, in this order

def actorSize(spec: ObsSpec, numSlots: int) -> int:
  return spec.self + (spec.mates + spec.opponents) * spec.car + spec.ball + 2 * spec.goal + spec.game + numSlots

def flattenObs(obs: TensorDict, numSlots: int, keys=ENTITY_KEYS) -> Tensor:
  """Entity TensorDict (B, ...) -> one flat vector per agent (B, F), with the slot as a one-hot."""
  slot = nn.functional.one_hot(obs["slot"], numSlots).float()
  return torch.cat([obs[k].flatten(1) for k in keys] + [slot], -1)

def orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
  nn.init.orthogonal_(layer.weight, gain)
  nn.init.zeros_(layer.bias)
  return layer

def mlp(inSize: int, hidden: tuple[int, ...]) -> nn.Sequential:
  layers, size = [], inSize
  for h in hidden:
    layers += [orthogonal(nn.Linear(size, h), math.sqrt(2)), nn.Tanh()]
    size = h
  return nn.Sequential(*layers)

class FlatMlp(nn.Module):
  """Concatenate every entity, then an MLP. (B, F) -> (B, hidden[-1])."""

  def __init__(self, inSize: int, hidden: tuple[int, ...] = (256, 256)):
    super().__init__()
    self.net, self.outSize = mlp(inSize, hidden), hidden[-1]

  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)

ENCODERS = {"flatMlp": FlatMlp}

def registerEncoder(name: str):
  def wrap(cls):
    ENCODERS[name] = cls
    return cls
  return wrap
