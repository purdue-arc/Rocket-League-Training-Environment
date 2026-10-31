import math, torch, carSoccer.config as cfg
from tensordict import TensorDict
from torch import Tensor
from carSoccer.components.ctx import Ctx
from carSoccer.engine import quat
from carSoccer.engine.base import GameState, maskedAssign
from carSoccer.engine.sim.simEngine import nominalParams

class StateMutator:
  """Writes the next start state, in place, for worlds where mask is True."""
  on = "kickoff"                           # "kickoff": every kickoff, including match start; "match": match start only
  setsParams = False                       # True if it writes state.params, which the engine then applies to the model

  def apply(self, state: GameState, mask: Tensor, ctx: Ctx) -> None:
    raise NotImplementedError

def standardFormation(ctx: Ctx) -> tuple[Tensor, Tensor]:
  """Kickoff spots, (2P, 2) positions and (2P,) headings. Blue is on -x facing +x; orange is blue rotated 180 degrees,
  so both teams see the same kickoff in their own frame."""
  P = ctx.carsPerTeam
  spots = [(-1.2, 0.0)] + [(-2.6, y) for y in torch.linspace(1.6, -1.6, max(P - 1, 1)).tolist()][:P - 1]
  blue = torch.tensor(spots, device=ctx.device)
  yaw = torch.tensor([0.0] * P + [math.pi] * P, device=ctx.device)
  return torch.cat([blue, -blue]), yaw

def scatteredStarts(ctx: Ctx, numWorlds: int) -> tuple[Tensor, Tensor, Tensor]:
  """Random car positions (N, 2P, 2), headings (N, 2P) and ball position (N, 3). Every object takes its own cell of a
  coarse grid, jittered, so nothing starts overlapping."""
  C, cell, jitter = 2 * ctx.carsPerTeam, 1.4, 0.3
  nx, ny = int(ctx.arena.length * 0.9 // cell), int(ctx.arena.width * 0.9 // cell)
  gx = (torch.arange(nx, device=ctx.device) - (nx - 1) / 2) * cell
  gy = (torch.arange(ny, device=ctx.device) - (ny - 1) / 2) * cell
  centres = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).view(-1, 2)            # (cells, 2)
  pick = ctx.rand(numWorlds, centres.shape[0]).argsort(-1)[:, :C + 1]                      # (N, 2P + 1)
  xy = centres[pick] + ctx.randUniform((numWorlds, C + 1, 2), -jitter, jitter)
  ballPos = torch.cat([xy[:, C], torch.full((numWorlds, 1), ctx.ballRadius, device=ctx.device)], -1)
  return xy[:, :C], ctx.randUniform((numWorlds, C), -math.pi, math.pi), ballPos

class Kickoff(StateMutator):
  """Standard formation, or with probability scatterProb a scattered start (random kickoffs, guide Section 3)."""

  def __init__(self, scatterProb: float = 0.0):
    self.scatterProb = scatterProb

  def apply(self, state: GameState, mask: Tensor, ctx: Ctx) -> None:
    N, C = mask.shape[0], 2 * ctx.carsPerTeam
    pos, yaw = standardFormation(ctx)
    pos, yaw = pos.expand(N, C, 2), yaw.expand(N, C)
    ballPos = torch.tensor([0.0, 0.0, ctx.ballRadius], device=ctx.device).expand(N, 3)
    scatter = ctx.rand(N) < self.scatterProb
    sPos, sYaw, sBall = scatteredStarts(ctx, N)
    pos = torch.where(scatter[:, None, None], sPos, pos)
    yaw = torch.where(scatter[:, None], sYaw, yaw)
    ballPos = torch.where(scatter[:, None], sBall, ballPos)
    height = ctx.spawnHeights[state["params", "variant"]]                  # each world's car variant, (N,)
    pos = torch.cat([pos, height[:, None, None].expand(N, C, 1)], -1)

    zeros3 = torch.zeros(N, C, 3, device=ctx.device)
    car = TensorDict({
      "pos": pos, "quat": quat.fromYaw(yaw), "vel": zeros3, "angvel": zeros3,
      "steer": torch.zeros(N, C, device=ctx.device), "uprightTime": torch.zeros(N, C, device=ctx.device),
    }, batch_size=[N])
    ball = TensorDict({"pos": ballPos, "vel": torch.zeros(N, 3, device=ctx.device), "angvel": torch.zeros(N, 3, device=ctx.device)}, batch_size=[N])
    maskedAssign(state["car"], car, mask)
    maskedAssign(state["ball"], ball, mask)

class NewMatch(StateMutator):
  """Score and match clock back to zero."""
  on = "match"

  def apply(self, state: GameState, mask: Tensor, ctx: Ctx) -> None:
    state["score"] = torch.where(mask[:, None], torch.zeros_like(state["score"]), state["score"])
    state["clock"] = torch.where(mask, torch.zeros_like(state["clock"]), state["clock"])

class SampleCarVariant(StateMutator):
  """Each match drives a car variant drawn uniformly from the scene's variants (car-model invariance)."""
  on, setsParams = "match", True

  def apply(self, state: GameState, mask: Tensor, ctx: Ctx) -> None:
    variant = (ctx.rand(mask.shape[0]) * ctx.numVariants).long().clamp(max=ctx.numVariants - 1)
    state["params", "variant"] = torch.where(mask, variant, state["params", "variant"])

class RandomPhysics(StateMutator):
  """Per-match physics randomisation: floor friction, ball mass, each pairing's restitution and friction, each car's
  motor strength and steering trim, and command latency."""
  on, setsParams = "match", True

  def __init__(self, sceneCfg: cfg.SceneConfig, ranges: cfg.RandomizationConfig = cfg.RandomizationConfig()):
    self.nominal, self.ranges = nominalParams(sceneCfg), ranges

  def apply(self, state: GameState, mask: Tensor, ctx: Ctx) -> None:
    r, nominal, N, C = self.ranges, self.nominal, mask.shape[0], 2 * ctx.carsPerTeam
    restitution = torch.tensor(nominal["restitution"], device=ctx.device)
    friction = torch.tensor(nominal["contactFriction"], device=ctx.device)
    lo, hi = r.latencyTicks
    drawn = {
      "floorFriction": nominal["floorFriction"] * ctx.randUniform((N,), *r.floorFriction),
      "ballMass": nominal["ballMass"] * ctx.randUniform((N,), *r.ballMass),
      "restitution": restitution + ctx.randUniform((N, restitution.shape[0]), -r.restitution, r.restitution),
      "contactFriction": friction * ctx.randUniform((N, friction.shape[0]), *r.contactFriction),
      "motorStrength": ctx.randUniform((N, C), *r.motorStrength),
      "steerTrim": ctx.randUniform((N, C), -r.steerTrim, r.steerTrim),
      "latency": lo + (ctx.rand(N) * (hi - lo + 1)).long().clamp(max=hi - lo),
    }
    params = state["params"]
    for key, value in drawn.items():
      old = params[key]
      params[key] = torch.where(mask.view(-1, *[1] * (old.dim() - 1)), value.to(old.dtype), old)
