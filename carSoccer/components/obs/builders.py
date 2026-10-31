import torch
from dataclasses import dataclass
from tensordict import TensorDict
from torch import Tensor
from carSoccer.components.ctx import Ctx
from carSoccer.components.obs.frames import egoFrame, teamFrame, teamYaw
from carSoccer.engine import quat
from carSoccer.engine.base import GameState, emptyGameState

@dataclass(frozen=True)
class ObsSpec:
  """Entity names and feature sizes, so models never hard-code input sizes."""
  scheme: str
  self: int
  car: int
  ball: int
  goal: int
  game: int
  mates: int
  opponents: int
  critic: int

class ObsBuilder:
  spec: ObsSpec

  def reset(self, mask: Tensor) -> None:
    """Clear per-world history (none yet)."""

  def build(self, state: GameState, ctx: Ctx) -> TensorDict:
    raise NotImplementedError

class EntityCarObs(ObsBuilder):
  """car scheme: one observation per car, B = N * 2P (world-major), every feature in the car's team frame and,
  where relative, its own frame. The actor reads state.measured; the critic entry is the same features from the truth."""

  def __init__(self, ctx: Ctx):
    P, C = ctx.carsPerTeam, 2 * ctx.carsPerTeam
    team = torch.arange(C) // P
    others = [[j for j in range(C) if j != i] for i in range(C)]
    self.mateIdx = torch.tensor([[j for j in others[i] if team[j] == team[i]] for i in range(C)], device=ctx.device)
    self.oppIdx = torch.tensor([[j for j in range(C) if team[j] != team[i]] for i in range(C)], device=ctx.device)
    self.slot = (torch.arange(C, device=ctx.device) % P)
    self.sign = ctx.teamSign()
    self.scale = torch.tensor([ctx.arena.length / 2, ctx.arena.width / 2], device=ctx.device)
    feats = self.features(_dummyState(ctx), ctx)
    self.spec = ObsSpec(
      scheme="car", self=feats["self"].shape[-1], car=feats["mates"].shape[-1], ball=feats["ball"].shape[-1],
      goal=feats["goals"].shape[-1], game=feats["game"].shape[-1], mates=P - 1, opponents=P,
      critic=sum(v.flatten(2).shape[-1] for v in feats.values()),
    )

  def features(self, state: GameState, ctx: Ctx, source: str = "measured") -> dict[str, Tensor]:
    """Entity features shaped (N, 2P, ...) for every car as the observer."""
    car, ball = (state[source, "car"], state[source, "ball"]) if source else (state["car"], state["ball"])
    sign = self.sign                                                              # (2P,) observer's team
    N, C = car["pos"].shape[:2]

    # every car and the ball seen in each observer's team frame: (N, observer, entity, 2)
    pos = teamFrame(car["pos"][:, None, :, :2], sign[None, :, None])
    vel = teamFrame(car["vel"][:, None, :, :2], sign[None, :, None])
    yaw = teamYaw(quat.yaw(car["quat"])[:, None, :].expand(N, C, C), sign[None, :, None])
    ballPos = teamFrame(ball["pos"][:, None, :2], sign[None, :])                 # (N, observer, 2)
    ballVel = teamFrame(ball["vel"][:, None, :2], sign[None, :])

    eye = torch.arange(C, device=sign.device)
    myPos, myVel, myYaw = pos[:, eye, eye], vel[:, eye, eye], yaw[:, eye, eye]    # (N, C, 2), (N, C)

    def relative(otherPos, otherVel, otherYaw=None):
      d = otherPos - myPos[..., None, :] if otherPos.dim() == 4 else otherPos - myPos
      v = otherVel - myVel[..., None, :] if otherVel.dim() == 4 else otherVel - myVel
      yawB = myYaw[..., None] if otherPos.dim() == 4 else myYaw
      out = [egoFrame(d, yawB), egoFrame(v, yawB), d.norm(dim=-1, keepdim=True)]
      if otherYaw is not None:
        out += [torch.cos(otherYaw - yawB)[..., None], torch.sin(otherYaw - yawB)[..., None]]
      return torch.cat(out, -1)

    carsRel = relative(pos, vel, yaw)                                             # (N, C, C, 7)
    goalsXY = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=sign.device) * self.scale[0]   # own, opponent
    goals = torch.cat([egoFrame(goalsXY - myPos[..., None, :], myYaw[..., None]),
                       (goalsXY - myPos[..., None, :]).norm(dim=-1, keepdim=True)], -1)       # (N, C, 2, 3)
    ownVel = egoFrame(myVel, myYaw)
    selfFeat = torch.cat([
      ownVel,                                                                     # forward, lateral speed
      car["angvel"][..., 2:],                                                     # yaw rate: same in both team frames
      car["steer"][..., None],
      quat.upZ(car["quat"])[..., None],
      myPos / self.scale, torch.cos(myYaw)[..., None], torch.sin(myYaw)[..., None],
    ], -1)
    score = state["score"].float()
    diff = (score[:, 0] - score[:, 1])[:, None] * sign[None, :]                   # (N, C) my team minus theirs
    remaining = 1 - state["clock"].float() / ctx.matchTicks

    return {
      "self": selfFeat,
      "mates": carsRel[:, eye[:, None], self.mateIdx],
      "opponents": carsRel[:, eye[:, None], self.oppIdx],
      "ball": torch.cat([relative(ballPos, ballVel), ball["pos"][:, None, 2:].expand(N, C, 1)], -1),
      "goals": goals,
      "game": torch.stack([diff, remaining[:, None].expand(N, C)], -1),
    }

  def build(self, state: GameState, ctx: Ctx) -> TensorDict:
    N, C = state.batch_size[0], self.sign.shape[0]
    B = N * C
    actor = self.features(state, ctx, "measured")
    critic = self.features(state, ctx, None)
    flat = {k: v.reshape(B, *v.shape[2:]) for k, v in actor.items()}
    ones = lambda n: torch.ones(B, n, dtype=torch.bool, device=self.sign.device)
    return TensorDict({
      **flat,
      "slot": self.slot.expand(N, C).reshape(B),
      "mask": TensorDict({"mates": ones(self.spec.mates), "opponents": ones(self.spec.opponents)}, batch_size=[B]),
      "critic": torch.cat([v.reshape(B, -1) for v in critic.values()], -1),
    }, batch_size=[B])

def _dummyState(ctx: Ctx) -> GameState:
  state = emptyGameState(1, ctx.carsPerTeam, ctx.device)
  state["car", "quat"][..., 0] = 1
  state["measured", "car", "quat"][..., 0] = 1
  return state
