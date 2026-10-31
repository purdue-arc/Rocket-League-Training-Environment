from torch import Tensor
from carSoccer.components.ctx import Ctx

class ActionParser:
  shape: tuple[int, ...]                   # one agent's raw action

  def parse(self, raw: Tensor, ctx: Ctx) -> Tensor:
    """Policy output -> carActions (N, 2P, 2) = [throttle, steer] in [-1, 1], blue first."""
    raise NotImplementedError

class ContinuousCarActions(ActionParser):
  """car scheme: one agent per car, raw (N, 2P, 2)."""
  shape = (2,)

  def parse(self, raw: Tensor, ctx: Ctx) -> Tensor:
    return raw.clamp(-1, 1)

class ContinuousTeamActions(ActionParser):
  """team scheme: one agent per team, raw (N, 2, 2P) = [throttle, steer] for each of its P cars."""

  def __init__(self, carsPerTeam: int):
    self.shape = (2 * carsPerTeam,)

  def parse(self, raw: Tensor, ctx: Ctx) -> Tensor:
    return raw.clamp(-1, 1).reshape(raw.shape[0], -1, 2)
