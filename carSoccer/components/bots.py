import math, torch
from torch import Tensor
from carSoccer.components.ctx import Ctx
from carSoccer.engine import quat
from carSoccer.engine.base import GameState

def chaseBall(state: GameState, ctx: Ctx) -> Tensor:
  """Scripted bot for every car: drive to a point just behind the ball on the line to the opponent goal. (N, 2P, 2)."""
  car, ball = state["measured", "car"], state["measured", "ball"]
  goalX = ctx.teamSign() * ctx.arena.length / 2                                         # (2P,) opponent goal line
  toGoal = torch.stack([goalX - ball["pos"][:, None, 0], -ball["pos"][:, None, 1].expand(-1, goalX.shape[0])], -1)
  toGoal = toGoal / toGoal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  aim = ball["pos"][:, None, :2] - toGoal * (ctx.ballRadius + 0.3)
  d = aim - car["pos"][..., :2]
  bearing = torch.remainder(torch.atan2(d[..., 1], d[..., 0]) - quat.yaw(car["quat"]) + math.pi, 2 * math.pi) - math.pi
  steer = (2 * bearing).clamp(-1, 1)
  throttle = 0.6 + 0.4 * torch.cos(bearing)
  return torch.stack([throttle, steer], -1)

class RandomDriver:
  """Random policy that actually moves: each car holds a random [throttle, steer] and redraws it with probability
  redraw per decision, so commands last about 1 / redraw decisions instead of cancelling out at 17 Hz."""

  def __init__(self, numWorlds: int, carsPerTeam: int, ctx: Ctx, redraw: float = 0.08, throttleBias: float = 0.3):
    self.ctx, self.redraw, self.throttleBias = ctx, redraw, throttleBias
    self.action = self.sample(numWorlds, 2 * carsPerTeam)

  def sample(self, N: int, C: int) -> Tensor:
    a = self.ctx.randUniform((N, C, 2), -1, 1)
    a[..., 0] = (a[..., 0] + self.throttleBias).clamp(-1, 1)
    return a

  def __call__(self, state: GameState) -> Tensor:
    N, C, _ = self.action.shape
    redraw = self.ctx.rand(N, C, 1) < self.redraw
    self.action = torch.where(redraw, self.sample(N, C), self.action)
    return self.action
